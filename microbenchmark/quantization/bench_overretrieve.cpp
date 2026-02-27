// bench_overretrieve.cpp
//
// Quality microbenchmark: Exact vs FastScan vs RaBitQ vs TQ-4bit vs Byte TQ vs TQ-PQ-4bit.
// Reports average number of candidates (M) needed to achieve target recall@K (e.g. 90%, 95%).
//
// For each query vector:
//  1) Brute-force exact distances to ALL db vectors; obtain exact top-K ids.
//  2) For each quantized method: compute approximate distances to ALL db vectors; sort by score.
//  3) For each K and recall target r: find minimal M such that top-M approx contains
//     at least ceil(r*K) of the exact top-K ids.
//
// Args (parse_command_line.h style):
//   Synthetic mode:
//     -N <u32>      (default 1000000)   // number of db vectors
//     -Q <u32>      (default 1000)      // number of query vectors
//     -D <u32>      (default 128)
//     -seed_db <u64>   (default 12345)
//     -seed_q  <u64>   (default 999)
//
//   Common:
//     -dist_func <L2|IP> (default IP)
//     -fs_block <u32>    (default 8)    // FastScan block size
//     -rbits <u32>       (default 2)
//     -tqpq_block <u32>  (default 4)    // TQ-PQ-4bit block size (1, 2, 4, 8, or 16)
//     -Kmax <u32>        (default 100)  // largest K evaluated; K grid is derived from this
//     -rec99 <f>         (default 0.99)
//
// Notes:
// - K grid: {1, 5, 10, 20, 50, 100} intersected with [1..Kmax].
// - This benchmark is meant for *quality* not speed; it sorts full N lists.
// - Same scan functions as bench_pq_fastscan, but per query we also: (1) compute exact GT
//   (full float scan over N), (2) sort N distances after each method (4 sorts). So it's slower.
// - Queries are processed in parallel where possible to use all cores.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"

#include "mvsic/core/distance_measures/one_to_one.h"
#include "mvsic/core/quantization/fastscan.h"
#include "mvsic/core/quantization/rabitq.h"
#include "mvsic/core/quantization/turboquant_4bit.h"
#include "mvsic/core/quantization/turboquant_byte.h"
#include "mvsic/core/quantization/turboquant_pq_4bit.h"

#include "mvsic/core/utils/parse_command_line.h"

using namespace mvsic;

// Synthetic PointRange adapter (from bench_pq_fastscan)
struct DensePointRange {
  std::vector<float> buf;  // contiguous [n * dim]
  size_t n = 0;
  uint32_t dim = 0;

  DensePointRange() = default;
  DensePointRange(size_t n_, uint32_t d_) : buf(n_ * size_t(d_)), n(n_), dim(d_) {}

  size_t size() const { return n; }
  uint32_t get_dims() const { return dim; }

  const uint8_t* location(size_t i) const {
    return reinterpret_cast<const uint8_t*>(buf.data() + i * size_t(dim));
  }

  const float* data() const { return buf.data(); }
  float* data() { return buf.data(); }
};

// Fill with N(0,1) and optionally L2-normalize.
static void fill_random(DensePointRange& r, uint64_t seed, bool l2_normalize) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> nd(0.0f, 1.0f);

  for (size_t i = 0; i < r.n; ++i) {
    float* v = r.data() + i * size_t(r.dim);
    float ss = 0.0f;
    for (uint32_t j = 0; j < r.dim; ++j) {
      float x = nd(rng);
      v[j] = x;
      ss += x * x;
    }
    if (l2_normalize) {
      float inv = 1.0f / std::sqrt(std::max(ss, 1e-12f));
      for (uint32_t j = 0; j < r.dim; ++j)
        v[j] *= inv;
    }
  }
}

static std::vector<uint32_t> default_K_grid(uint32_t Kmax) {
  std::vector<uint32_t> ks = {1, 5, 10, 20, 50, 100};
  std::vector<uint32_t> out;
  out.reserve(ks.size());
  for (uint32_t k : ks) {
    if (k <= Kmax) out.push_back(k);
  }
  if (out.empty()) out.push_back(std::max<uint32_t>(1, Kmax));
  return out;
}

// Returns minimal M (1..N) such that top-M approx contains >= need hits from exact_topK_set.
static size_t min_M_for_recall(const std::vector<uint32_t>& approx_ranked_ids,
                               const std::unordered_set<uint32_t>& exact_topK_set, size_t need) {
  if (need == 0) return 0;
  size_t hits = 0;
  for (size_t m = 0; m < approx_ranked_ids.size(); ++m) {
    if (exact_topK_set.find(approx_ranked_ids[m]) != exact_topK_set.end()) {
      ++hits;
      if (hits >= need) return m + 1;  // M is 1-based count
    }
  }
  return approx_ranked_ids.size();
}

template<bool Metric>
static int run_benchmark(size_t N, size_t Q, uint32_t D, uint64_t seed_db, uint64_t seed_q,
                         uint32_t fs_block, uint32_t rbits, uint32_t tqpq_block, uint32_t Kmax,
                         float rec99) {
  DensePointRange db(N, D);
  DensePointRange queries(Q, D);

  fill_random(db, seed_db, /*l2_normalize=*/true);
  fill_random(queries, seed_q, /*l2_normalize=*/true);

  if (D % fs_block != 0) {
    std::cerr << "ERROR: D must be divisible by fs_block.\n";
    return 1;
  }
  if (tqpq_block != 1 && tqpq_block != 2 && tqpq_block != 4 && tqpq_block != 8 &&
      tqpq_block != 16) {
    std::cerr << "ERROR: tqpq_block must be 1, 2, 4, 8, or 16.\n";
    return 1;
  }

  const auto Kgrid = default_K_grid(Kmax);

  std::cout << "DB: vectors=" << N << " dims=" << D << std::endl;
  std::cout << "Q : vectors=" << Q << " dims=" << D << std::endl;
  std::cout << "fs_block=" << fs_block << " rbits=" << rbits << " tqpq_block=" << tqpq_block
            << " dist=" << (Metric ? "L2" : "IP") << std::endl;
  std::cout << "Kgrid: ";
  for (auto k : Kgrid)
    std::cout << k << " ";
  std::cout << std::endl;
  std::cout << "Recall target: " << rec99 << std::endl;

  fastscan::Model<Metric> fs_model;
  fs_model.train(db, fs_block);
  auto fs_db = fs_model.encode(db);

  rabitq::Model<Metric> rq_model;
  rq_model.train(db, rbits);
  auto rq_db = rq_model.encode(db);

  turboquant_4bit::Model<Metric> tq_model;
  tq_model.train(db);
  auto tq_db = tq_model.encode(db);

  turboquant_byte::Model<Metric> btq_model;
  btq_model.train(db);
  auto btq_db = btq_model.encode(db);

  std::vector<std::pair<uint32_t, float>> exact_scores(N);
  std::vector<std::pair<uint32_t, float>> approx_scores(N);
  std::vector<float> approx_distances(N);

  enum Method { FASTSCAN = 0, RABITQ = 1, TQ4BIT = 2, BYTETQ = 3, TQPQ4BIT = 4, NUM_METHODS = 5 };
  const char* method_names[NUM_METHODS] = {"FastScan", "RaBitQ", "TQ-4bit", "Byte TQ",
                                           "TQ-PQ-4bit"};

  std::vector<double> sum_M(NUM_METHODS * Kgrid.size(), 0.0);
  auto idx2 = [&](Method m, size_t k_i) { return static_cast<size_t>(m) * Kgrid.size() + k_i; };

  efanna2e::DistanceInnerProduct distfunc_ip;
  efanna2e::DistanceL2 distfunc_l2;

  std::vector<std::vector<std::unordered_set<uint32_t>>> exact_sets_per_query(
      Q, std::vector<std::unordered_set<uint32_t>>(Kgrid.size()));

  parlay::sequence<std::vector<double>> per_query_contrib(Q);
  parlay::parallel_for(0, Q, [&](size_t qi) {
    per_query_contrib[qi].resize(NUM_METHODS * Kgrid.size(), 0.0);
    const float* q = queries.data() + qi * size_t(D);

    static thread_local std::vector<std::pair<uint32_t, float>> tl_exact;
    static thread_local std::vector<std::pair<uint32_t, float>> tl_approx;
    static thread_local std::vector<float> tl_dists;
    tl_exact.resize(N);
    tl_approx.resize(N);
    tl_dists.resize(N);

    for (size_t i = 0; i < N; ++i) {
      const float* p = db.data() + i * size_t(D);
      float dist;
      if constexpr (Metric) {
        dist = distfunc_l2.compare(q, p, D);
      } else {
        dist = -distfunc_ip.compare(q, p, D);
      }
      tl_exact[i] = {static_cast<uint32_t>(i), dist};
    }
    parlay::sort_inplace(tl_exact, [](const auto& a, const auto& b) { return a.second < b.second; });
    for (size_t k_i = 0; k_i < Kgrid.size(); ++k_i) {
      const uint32_t K = Kgrid[k_i];
      exact_sets_per_query[qi][k_i].clear();
      for (uint32_t j = 0; j < K && j < N; ++j)
        exact_sets_per_query[qi][k_i].insert(tl_exact[j].first);
    }

    auto add_M = [&](Method meth, const std::vector<uint32_t>& approx_ids) {
      for (size_t k_i = 0; k_i < Kgrid.size(); ++k_i) {
        const size_t K = static_cast<size_t>(Kgrid[k_i]);
        const size_t need = static_cast<size_t>(std::ceil(rec99 * static_cast<float>(K)));
        const size_t M = min_M_for_recall(approx_ids, exact_sets_per_query[qi][k_i], need);
        per_query_contrib[qi][idx2(meth, k_i)] = static_cast<double>(M);
      }
    };

    auto run_method = [&](Method meth) {
      parlay::sort_inplace(tl_approx, [](const auto& a, const auto& b) { return a.second < b.second; });
      std::vector<uint32_t> approx_ids;
      approx_ids.reserve(N);
      for (const auto& p : tl_approx) approx_ids.push_back(p.first);
      add_M(meth, approx_ids);
    };

    {
      auto qq = fs_model.quantize_query(q);
      qq.distances_all(fs_db, tl_dists.data());
      for (size_t i = 0; i < N; ++i) tl_approx[i] = {static_cast<uint32_t>(i), tl_dists[i]};
      run_method(FASTSCAN);
    }
    {
      auto qq = rq_model.quantize_query(q);
      qq.distances_all(rq_db, tl_dists.data());
      for (size_t i = 0; i < N; ++i) tl_approx[i] = {static_cast<uint32_t>(i), tl_dists[i]};
      run_method(RABITQ);
    }
    {
      auto qq = tq_model.quantize_query(q);
      qq.distances_contiguous(tq_db.packed_codes.data(), tq_db.norm_scaling_factors.data(),
                              tq_db.unquantized_squared_norms.data(), tq_db.stride, N,
                              tl_dists.data());
      for (size_t i = 0; i < N; ++i) tl_approx[i] = {static_cast<uint32_t>(i), tl_dists[i]};
      run_method(TQ4BIT);
    }
    {
      auto qq = btq_model.quantize_query(q);
      for (size_t i = 0; i < N; ++i) tl_dists[i] = qq.distance(btq_db[i]);
      for (size_t i = 0; i < N; ++i) tl_approx[i] = {static_cast<uint32_t>(i), tl_dists[i]};
      run_method(BYTETQ);
    }
  });

  for (size_t qi = 0; qi < Q; ++qi)
    for (size_t i = 0; i < sum_M.size(); ++i)
      sum_M[i] += per_query_contrib[qi][i];

#if defined(__AVX512F__)
  {
    auto run_tqpq_queries = [&](auto& tqpq_m, auto& tqpq_enc) {
      for (size_t qi = 0; qi < Q; ++qi) {
        const float* q = queries.data() + qi * size_t(D);
        auto qq = tqpq_m.quantize_query(q);
        qq.distances_all(tqpq_enc, approx_distances.data());
        for (size_t i = 0; i < N; ++i)
          approx_scores[i] = {static_cast<uint32_t>(i), approx_distances[i]};
        parlay::sort_inplace(approx_scores, [](const auto& a, const auto& b) { return a.second < b.second; });
        std::vector<uint32_t> approx_ids;
        approx_ids.reserve(N);
        for (const auto& p : approx_scores)
          approx_ids.push_back(p.first);
        for (size_t k_i = 0; k_i < Kgrid.size(); ++k_i) {
          const size_t K = static_cast<size_t>(Kgrid[k_i]);
          const size_t need = static_cast<size_t>(std::ceil(rec99 * static_cast<float>(K)));
          const size_t M = min_M_for_recall(approx_ids, exact_sets_per_query[qi][k_i], need);
          sum_M[idx2(TQPQ4BIT, k_i)] += static_cast<double>(M);
        }
      }
    };
    if (tqpq_block == 1) {
      turboquant_pq_4bit::Model<Metric, 1> tqpq_m;
      tqpq_m.train(db);
      auto tqpq_enc = tqpq_m.encode(db);
      run_tqpq_queries(tqpq_m, tqpq_enc);
    } else if (tqpq_block == 2) {
      turboquant_pq_4bit::Model<Metric, 2> tqpq_m;
      tqpq_m.train(db);
      auto tqpq_enc = tqpq_m.encode(db);
      run_tqpq_queries(tqpq_m, tqpq_enc);
    } else if (tqpq_block == 4) {
      turboquant_pq_4bit::Model<Metric, 4> tqpq_m;
      tqpq_m.train(db);
      auto tqpq_enc = tqpq_m.encode(db);
      run_tqpq_queries(tqpq_m, tqpq_enc);
    } else if (tqpq_block == 8) {
      turboquant_pq_4bit::Model<Metric, 8> tqpq_m;
      tqpq_m.train(db);
      auto tqpq_enc = tqpq_m.encode(db);
      run_tqpq_queries(tqpq_m, tqpq_enc);
    } else {
      turboquant_pq_4bit::Model<Metric, 16> tqpq_m;
      tqpq_m.train(db);
      auto tqpq_enc = tqpq_m.encode(db);
      run_tqpq_queries(tqpq_m, tqpq_enc);
    }
  }
#endif

  std::cout << "\n=== Number of candidates (M) to reach recall@K ===" << std::endl;
  std::cout << "Averages over Q=" << Q << " query vectors." << std::endl;

  for (int meth = 0; meth < NUM_METHODS; ++meth) {
    std::cout << std::endl << method_names[meth] << ":\n";
    for (size_t k_i = 0; k_i < Kgrid.size(); ++k_i) {
      const uint32_t K = Kgrid[k_i];
      const double avg_M = sum_M[idx2(static_cast<Method>(meth), k_i)] / std::max<size_t>(1, Q);
      std::cout << "  K=" << std::setw(4) << K << "  rec" << int(rec99 * 100 + 0.5f)
                << "%: " << std::setw(10) << std::fixed << std::setprecision(1) << avg_M
                << std::endl;
    }
  }

  return 0;
}

int main(int argc, char** argv) {
  commandLine P(argc, argv,
                "[-N <n>] [-Q <q>] [-D <d>] [-seed_db <s>] [-seed_q <s>] "
                "[-dist_func <L2|IP>] [-fs_block <b>] [-rbits <b>] [-tqpq_block <b>] "
                "[-Kmax <k>] [-rec99 <f>]");

  std::string df = P.getOptionValue("-dist_func", "IP");
  size_t N = static_cast<size_t>(P.getOptionIntValue("-N", 1000000));
  size_t Q = static_cast<size_t>(P.getOptionIntValue("-Q", 1000));
  uint32_t D = static_cast<uint32_t>(P.getOptionIntValue("-D", 128));
  uint32_t fs_block = static_cast<uint32_t>(P.getOptionIntValue("-fs_block", 8));
  uint32_t rbits = static_cast<uint32_t>(P.getOptionIntValue("-rbits", 2));
  uint32_t tqpq_block = static_cast<uint32_t>(P.getOptionIntValue("-tqpq_block", 4));
  uint32_t Kmax = static_cast<uint32_t>(P.getOptionIntValue("-Kmax", 100));
  float rec99 = std::stof(P.getOptionValue("-rec99", "0.99"));

  uint64_t seed_db = 12345ULL;
  uint64_t seed_q = 999ULL;
  if (char* s = P.getOptionValue("-seed_db")) seed_db = static_cast<uint64_t>(std::stoull(s));
  if (char* s = P.getOptionValue("-seed_q")) seed_q = static_cast<uint64_t>(std::stoull(s));

  if (df == "L2" || df == "l2") {
    return run_benchmark<true>(N, Q, D, seed_db, seed_q, fs_block, rbits, tqpq_block, Kmax, rec99);
  }
  return run_benchmark<false>(N, Q, D, seed_db, seed_q, fs_block, rbits, tqpq_block, Kmax, rec99);
}
