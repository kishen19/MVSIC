// bench_overretrieve.cpp
//
// Quality microbenchmark: Exact vs PQ vs FastScan vs RaBitQ vs TurboQuant.
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
//     -pq_block <u32>    (default 8)
//     -pq_k <u32>        (default 16)    // PQ clusters per block
//     -fs_block <u32>    (default 8)
//     -rbits <u32>       (default 2)
//     -Kmax <u32>        (default 100)   // largest K evaluated; K grid is derived from this
//     -rec99 <f>         (default 0.99)
//
// Notes:
// - K grid: {1, 5, 10, 20, 50, 100} intersected with [1..Kmax].
// - This benchmark is meant for *quality* not speed; it sorts full N lists.

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

#include "mvsic/core/distance_measures/one_to_one.h"
#include "mvsic/core/quantization/fastscan.h"
#include "mvsic/core/quantization/pq.h"
#include "mvsic/core/quantization/rabitq.h"
#include "mvsic/core/quantization/turboquant.h"

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
                         uint32_t pq_block, uint32_t pq_k, uint32_t fs_block, uint32_t rbits,
                         uint32_t Kmax, float rec99) {
  DensePointRange db(N, D);
  DensePointRange queries(Q, D);

  // Always L2-normalize vectors for consistent comparison
  fill_random(db, seed_db, /*l2_normalize=*/true);
  fill_random(queries, seed_q, /*l2_normalize=*/true);

  if (D % pq_block != 0) {
    std::cerr << "ERROR: D must be divisible by pq_block.\n";
    return 1;
  }
  if (D % fs_block != 0) {
    std::cerr << "ERROR: D must be divisible by fs_block.\n";
    return 1;
  }

  const auto Kgrid = default_K_grid(Kmax);

  std::cout << "DB: vectors=" << N << " dims=" << D << "\n";
  std::cout << "Q : vectors=" << Q << " dims=" << D << "\n";
  std::cout << "pq_block=" << pq_block << " pq_k=" << pq_k << " fs_block=" << fs_block
            << " rbits=" << rbits << " dist=" << (Metric ? "L2" : "IP") << "\n";
  std::cout << "Kgrid: ";
  for (auto k : Kgrid) std::cout << k << " ";
  std::cout << "\n";
  std::cout << "Recall target: " << rec99 << "\n";

  // Train + Encode quantized DBs
  const uint32_t PQ_S = 20;

  pq::Model<Metric> pq_model;
  pq_model.train(db, pq_block, pq_k, PQ_S);
  auto pq_db = pq_model.encode(db);

  fastscan::Model<Metric> fs_model;
  fs_model.train(db, fs_block);
  auto fs_db = fs_model.encode(db);

  rabitq::Model<Metric> rq_model;
  rq_model.train(db, rbits);
  auto rq_db = rq_model.encode(db);

  turboquant::Model<Metric> tq_model;
  tq_model.train(db);
  auto tq_db = tq_model.encode(db);

  std::vector<std::pair<uint32_t, float>> exact_scores(N);
  std::vector<std::pair<uint32_t, float>> approx_scores(N);
  std::vector<float> approx_distances(N);  // Temporary buffer for TurboQuant

  enum Method { PQ = 0, FASTSCAN = 1, RABITQ = 2, TURBOQUANT = 3, NUM_METHODS = 4 };
  const char* method_names[NUM_METHODS] = {"PQ", "FastScan", "RaBitQ", "TurboQuant"};

  std::vector<double> sum_M(NUM_METHODS * Kgrid.size(), 0.0);
  auto idx2 = [&](Method m, size_t k_i) {
    return static_cast<size_t>(m) * Kgrid.size() + k_i;
  };

  // Compute exact distances using efanna2e distance functions
  efanna2e::DistanceInnerProduct distfunc_ip;
  efanna2e::DistanceL2 distfunc_l2;

  for (size_t qi = 0; qi < Q; ++qi) {
    const float* q = queries.data() + qi * size_t(D);

    // Compute exact distances
    parlay::parallel_for(0, N, [&](size_t i) {
      const float* p = db.data() + i * size_t(D);
      float dist;
      if constexpr (Metric) {
        dist = distfunc_l2.compare(q, p, D);
      } else {
        dist = -distfunc_ip.compare(q, p, D);  // IP: distance = -inner_product
      }
      exact_scores[i] = {static_cast<uint32_t>(i), dist};
    });

    std::sort(exact_scores.begin(), exact_scores.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });

    std::vector<std::unordered_set<uint32_t>> exact_sets;
    exact_sets.reserve(Kgrid.size());
    for (uint32_t K : Kgrid) {
      std::unordered_set<uint32_t> s;
      s.reserve(static_cast<size_t>(K) * 2);
      for (uint32_t j = 0; j < K && j < N; ++j) s.insert(exact_scores[j].first);
      exact_sets.emplace_back(std::move(s));
    }

    auto eval_method = [&](Method meth) {
      std::sort(approx_scores.begin(), approx_scores.end(),
                [](const auto& a, const auto& b) { return a.second < b.second; });
      std::vector<uint32_t> approx_ids;
      approx_ids.reserve(N);
      for (const auto& p : approx_scores) approx_ids.push_back(p.first);

      for (size_t k_i = 0; k_i < Kgrid.size(); ++k_i) {
        const size_t K = static_cast<size_t>(Kgrid[k_i]);
        const size_t need = static_cast<size_t>(std::ceil(rec99 * static_cast<float>(K)));
        const size_t M = min_M_for_recall(approx_ids, exact_sets[k_i], need);
        sum_M[idx2(meth, k_i)] += static_cast<double>(M);
      }
    };

    {
      auto qq = pq_model.quantize_query(q);
      qq.distances_all(pq_db, approx_distances.data());
      for (size_t i = 0; i < N; ++i) {
        approx_scores[i] = {static_cast<uint32_t>(i), approx_distances[i]};
      }
      eval_method(PQ);
    }
    {
      auto qq = fs_model.quantize_query(q);
      qq.distances_all(fs_db, approx_distances.data());
      for (size_t i = 0; i < N; ++i) {
        approx_scores[i] = {static_cast<uint32_t>(i), approx_distances[i]};
      }
      eval_method(FASTSCAN);
    }
    {
      auto qq = rq_model.quantize_query(q);
      qq.distances_all(rq_db, approx_distances.data());
      for (size_t i = 0; i < N; ++i) {
        approx_scores[i] = {static_cast<uint32_t>(i), approx_distances[i]};
      }
      eval_method(RABITQ);
    }
    {
      auto qq = tq_model.quantize_query(q);
      qq.distances_all(tq_db, approx_distances.data());
      for (size_t i = 0; i < N; ++i) {
        approx_scores[i] = {static_cast<uint32_t>(i), approx_distances[i]};
      }
      eval_method(TURBOQUANT);
    }
  }

  std::cout << "\n=== Number of candidates (M) to reach recall@K ===\n";
  std::cout << "Averages over Q=" << Q << " query vectors.\n";

  for (int meth = 0; meth < NUM_METHODS; ++meth) {
    std::cout << "\n" << method_names[meth] << ":\n";
    for (size_t k_i = 0; k_i < Kgrid.size(); ++k_i) {
      const uint32_t K = Kgrid[k_i];
      const double avg_M = sum_M[idx2(static_cast<Method>(meth), k_i)] / std::max<size_t>(1, Q);
      std::cout << "  K=" << std::setw(4) << K << "  rec" << int(rec99 * 100 + 0.5f)
                << "%: " << std::setw(10) << std::fixed << std::setprecision(1) << avg_M << "\n";
    }
  }

  return 0;
}

int main(int argc, char** argv) {
  commandLine P(argc, argv,
                "[-N <n>] [-Q <q>] [-D <d>] [-seed_db <s>] [-seed_q <s>] "
                "[-dist_func <L2|IP>] [-pq_block <b>] [-pq_k <k>] [-fs_block <b>] [-rbits <b>] "
                "[-Kmax <k>] [-rec99 <f>]");

  std::string df = P.getOptionValue("-dist_func", "IP");
  size_t N = static_cast<size_t>(P.getOptionIntValue("-N", 1000000));
  size_t Q = static_cast<size_t>(P.getOptionIntValue("-Q", 1000));
  uint32_t D = static_cast<uint32_t>(P.getOptionIntValue("-D", 128));
  uint32_t pq_block = static_cast<uint32_t>(P.getOptionIntValue("-pq_block", 8));
  uint32_t pq_k = static_cast<uint32_t>(P.getOptionIntValue("-pq_k", 16));
  uint32_t fs_block = static_cast<uint32_t>(P.getOptionIntValue("-fs_block", 8));
  uint32_t rbits = static_cast<uint32_t>(P.getOptionIntValue("-rbits", 2));
  uint32_t Kmax = static_cast<uint32_t>(P.getOptionIntValue("-Kmax", 100));
  float rec99 = std::stof(P.getOptionValue("-rec99", "0.99"));

  uint64_t seed_db = 12345ULL;
  uint64_t seed_q = 999ULL;
  if (char* s = P.getOptionValue("-seed_db")) seed_db = static_cast<uint64_t>(std::stoull(s));
  if (char* s = P.getOptionValue("-seed_q")) seed_q = static_cast<uint64_t>(std::stoull(s));

  if (df == "L2" || df == "l2") {
    return run_benchmark<true>(N, Q, D, seed_db, seed_q, pq_block, pq_k, fs_block, rbits, Kmax,
                               rec99);
  }
  return run_benchmark<false>(N, Q, D, seed_db, seed_q, pq_block, pq_k, fs_block, rbits, Kmax,
                              rec99);
}
