// bench_overretrieve.cpp
//
// Quality microbenchmark: Exact vs FastScan vs RaBitQ vs TQ-4bit vs Byte TQ vs TQ-PQ-4bit.
// Reports average number of candidates (M) needed to retrieve *all* K true neighbors
// (full recall@K) across queries.
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
//     -rec99 <f>         (ignored; always full recall@K)
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
#include <type_traits>
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
#include "mvsic/core/quantization/turboquant_pq_4bit_scalar.h"

#include "mvsic/core/types/ip_point.h"
#include "mvsic/core/types/l2_point.h"
#include "mvsic/core/types/point_range.h"
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

struct IbinGroundTruth {
  uint32_t num_queries = 0;
  uint32_t k = 0;
  std::vector<uint32_t> ids;     // [num_queries * k]
  std::vector<float> distances;  // [num_queries * k]
};

// ParlayANN ground-truth format ("ibin", used by big-ann-benchmarks):
// int32 num_queries, int32 k, then num_queries*k int32 ids, then num_queries*k float distances.
static IbinGroundTruth load_ibin_ground_truth(const char* path) {
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    std::cerr << "ERROR: cannot open ground truth file: " << path << "\n";
    std::exit(1);
  }

  int32_t nq_i32 = 0;
  int32_t k_i32 = 0;
  in.read(reinterpret_cast<char*>(&nq_i32), sizeof(int32_t));
  in.read(reinterpret_cast<char*>(&k_i32), sizeof(int32_t));
  if (!in || nq_i32 <= 0 || k_i32 <= 0) {
    std::cerr << "ERROR: invalid ibin header in " << path << "\n";
    std::exit(1);
  }

  IbinGroundTruth gt;
  gt.num_queries = static_cast<uint32_t>(nq_i32);
  gt.k = static_cast<uint32_t>(k_i32);
  const size_t total = static_cast<size_t>(gt.num_queries) * static_cast<size_t>(gt.k);
  gt.ids.resize(total);
  gt.distances.resize(total);
  in.read(reinterpret_cast<char*>(gt.ids.data()), total * sizeof(uint32_t));
  in.read(reinterpret_cast<char*>(gt.distances.data()), total * sizeof(float));
  if (!in) {
    std::cerr << "ERROR: ground truth file too short: " << path << "\n";
    std::exit(1);
  }
  return gt;
}

template<bool Metric, typename DBRange, typename QRange>
static int run_benchmark(const DBRange& db, const QRange& queries, uint32_t fs_block,
                         uint32_t rbits, uint32_t tqpq_block, uint32_t Kmax, float rec99,
                         const IbinGroundTruth* gt = nullptr, bool run_pq = true,
                         bool run_rabitq = true) {
  const size_t N = db.size();
  const size_t Q = queries.size();
  const uint32_t D = static_cast<uint32_t>(db.get_dims());

  if (D % fs_block != 0) {
    std::cerr << "ERROR: D must be divisible by fs_block.\n";
    return 1;
  }
  if (tqpq_block != 1 && tqpq_block != 2 && tqpq_block != 4 && tqpq_block != 8 &&
      tqpq_block != 16) {
    std::cerr << "ERROR: tqpq_block must be 1, 2, 4, 8, or 16.\n";
    return 1;
  }

  // Use a compact K grid; we care most about K = 1, 10, 100.
  std::vector<uint32_t> Kgrid;
  for (uint32_t k : {1u, 10u, 100u}) {
    if (k <= Kmax) Kgrid.push_back(k);
  }
  if (Kgrid.empty()) Kgrid.push_back(std::min<uint32_t>(1u, Kmax));

  std::cout << "DB: vectors=" << N << " dims=" << D << std::endl;
  std::cout << "Q : vectors=" << Q << " dims=" << D << std::endl;
  std::cout << "fs_block=" << fs_block << " rbits=" << rbits << " tqpq_block=" << tqpq_block
            << " dist=" << (Metric ? "L2" : "IP") << std::endl;
  std::cout << "Kgrid: ";
  for (auto k : Kgrid)
    std::cout << k << " ";
  std::cout << std::endl;
  std::cout << "Metric: full recall@K (all K true neighbors)\n";

  fastscan::Model<Metric> fs_model;
  fs_model.train(db, fs_block);
  auto fs_db = fs_model.encode(db);

  rabitq::Model<Metric> rq_model;
  using RQ_DB = decltype(rq_model.encode(db));
  RQ_DB rq_db;
  if (run_rabitq) {
    rq_model.train(db, rbits);
    rq_db = rq_model.encode(db);
  }

  turboquant_4bit::Model<Metric> tq_model;
  tq_model.train(db);
  auto tq_db = tq_model.encode(db);

  turboquant_byte::Model<Metric> btq_model;
  btq_model.train(db);
  auto btq_db = btq_model.encode(db);

  std::vector<std::pair<uint32_t, float>> approx_scores(N);
  std::vector<float> approx_distances(N);

  enum Method {
    EXACT = 0,
    FASTSCAN = 0,
    RABITQ = 1,
    TQ4BIT = 2,
    BYTETQ = 3,
    TQPQ16_1 = 4,
    TQPQ16_2 = 5,
    TQPQ16_4 = 6,
    TQPQ16_8 = 7,
    TQPQ_SCALAR_1 = 8,
    TQPQ_SCALAR_2 = 9,
    TQPQ_SCALAR_4 = 10,
    TQPQ_SCALAR_8 = 11,
    NUM_METHODS = 12
  };
  const char* method_names[NUM_METHODS] = {
      "FastScan",        "RaBitQ",          "TQ-4bit",        "Byte TQ",
      "TQ-PQ-16-1",      "TQ-PQ-16-2",      "TQ-PQ-16-4",     "TQ-PQ-16-8",
      "TQ-PQ-scalar-B1", "TQ-PQ-scalar-B2", "TQ-PQ-scalar-B4","TQ-PQ-scalar-B8"};

  std::vector<double> sum_M(NUM_METHODS * Kgrid.size(), 0.0);
  auto idx2 = [&](Method m, size_t k_i) { return static_cast<size_t>(m) * Kgrid.size() + k_i; };

  std::vector<std::vector<std::unordered_set<uint32_t>>> exact_sets_per_query(
      Q, std::vector<std::unordered_set<uint32_t>>(Kgrid.size()));

  parlay::sequence<std::vector<double>> per_query_contrib(Q);
  parlay::parallel_for(0, Q, [&](size_t qi) {
    per_query_contrib[qi].resize(NUM_METHODS * Kgrid.size(), 0.0);
    const float* q = reinterpret_cast<const float*>(queries.location(qi));

    static thread_local std::vector<std::pair<uint32_t, float>> tl_exact;
    static thread_local std::vector<std::pair<uint32_t, float>> tl_approx;
    static thread_local std::vector<float> tl_dists;
    tl_exact.resize(N);
    tl_approx.resize(N);
    tl_dists.resize(N);

    if (gt) {
      for (size_t k_i = 0; k_i < Kgrid.size(); ++k_i) {
        const uint32_t K = Kgrid[k_i];
        exact_sets_per_query[qi][k_i].clear();
        for (uint32_t j = 0; j < K; ++j) {
          const uint32_t id = gt->ids[qi * static_cast<size_t>(gt->k) + static_cast<size_t>(j)];
          exact_sets_per_query[qi][k_i].insert(id);
        }
      }
    } else {
      static thread_local efanna2e::DistanceInnerProduct distfunc_ip;
      static thread_local efanna2e::DistanceL2 distfunc_l2;
      for (size_t i = 0; i < N; ++i) {
        const float* p = reinterpret_cast<const float*>(db.location(i));
        float dist;
        if constexpr (Metric) {
          dist = distfunc_l2.compare(q, p, D);
        } else {
          dist = -distfunc_ip.compare(q, p, D);
        }
        tl_exact[i] = {static_cast<uint32_t>(i), dist};
      }
      parlay::sort_inplace(tl_exact,
                           [](const auto& a, const auto& b) { return a.second < b.second; });
      for (size_t k_i = 0; k_i < Kgrid.size(); ++k_i) {
        const uint32_t K = Kgrid[k_i];
        exact_sets_per_query[qi][k_i].clear();
        for (uint32_t j = 0; j < K && j < N; ++j)
          exact_sets_per_query[qi][k_i].insert(tl_exact[j].first);
      }
    }

    auto add_M = [&](Method meth, const std::vector<uint32_t>& approx_ids) {
      for (size_t k_i = 0; k_i < Kgrid.size(); ++k_i) {
        const size_t K = static_cast<size_t>(Kgrid[k_i]);
        // Full recall@K: need all K true neighbors.
        const size_t need = K;
        const size_t M = min_M_for_recall(approx_ids, exact_sets_per_query[qi][k_i], need);
        per_query_contrib[qi][idx2(meth, k_i)] = static_cast<double>(M);
      }
    };

    auto run_method = [&](Method meth) {
      parlay::sort_inplace(tl_approx,
                           [](const auto& a, const auto& b) { return a.second < b.second; });
      std::vector<uint32_t> approx_ids;
      approx_ids.reserve(N);
      for (const auto& p : tl_approx)
        approx_ids.push_back(p.first);
      add_M(meth, approx_ids);
    };

    {
      auto qq = fs_model.quantize_query(q);
      qq.distances_all(fs_db, tl_dists.data());
      for (size_t i = 0; i < N; ++i)
        tl_approx[i] = {static_cast<uint32_t>(i), tl_dists[i]};
      run_method(FASTSCAN);
    }
    if (run_rabitq) {
      auto qq = rq_model.quantize_query(q);
      qq.distances_all(rq_db, tl_dists.data());
      for (size_t i = 0; i < N; ++i)
        tl_approx[i] = {static_cast<uint32_t>(i), tl_dists[i]};
      run_method(RABITQ);
    }
    {
      auto qq = tq_model.quantize_query(q);
      qq.distances_contiguous(tq_db.packed_codes.data(), tq_db.norm_scaling_factors.data(),
                              tq_db.unquantized_squared_norms.data(), tq_db.stride, N,
                              tl_dists.data());
      for (size_t i = 0; i < N; ++i)
        tl_approx[i] = {static_cast<uint32_t>(i), tl_dists[i]};
      run_method(TQ4BIT);
    }
    {
      auto qq = btq_model.quantize_query(q);
      for (size_t i = 0; i < N; ++i)
        tl_dists[i] = qq.distance(btq_db[i]);
      for (size_t i = 0; i < N; ++i)
        tl_approx[i] = {static_cast<uint32_t>(i), tl_dists[i]};
      run_method(BYTETQ);
    }
  });

  for (size_t qi = 0; qi < Q; ++qi)
    for (size_t i = 0; i < sum_M.size(); ++i)
      sum_M[i] += per_query_contrib[qi][i];

  // TQ-PQ and scalar: train ONCE per B and run both with the same model so scalar is a
  // true reference (same encoding; only distance differs: LUT vs float). Avoids rotator
  // non-determinism giving different quality between the two.
  {
    auto run_tqpq_queries = [&](auto& tqpq_m, auto& tqpq_enc, Method meth) {
      for (size_t qi = 0; qi < Q; ++qi) {
        const float* q = reinterpret_cast<const float*>(queries.location(qi));
        auto qq = tqpq_m.quantize_query(q);
        qq.distances_all(tqpq_enc, approx_distances.data());
        for (size_t i = 0; i < N; ++i)
          approx_scores[i] = {static_cast<uint32_t>(i), approx_distances[i]};
        parlay::sort_inplace(approx_scores,
                             [](const auto& a, const auto& b) { return a.second < b.second; });
        std::vector<uint32_t> approx_ids;
        approx_ids.reserve(N);
        for (const auto& p : approx_scores)
          approx_ids.push_back(p.first);
        for (size_t k_i = 0; k_i < Kgrid.size(); ++k_i) {
          const size_t K = static_cast<size_t>(Kgrid[k_i]);
          const size_t need = K;
          const size_t M = min_M_for_recall(approx_ids, exact_sets_per_query[qi][k_i], need);
          sum_M[idx2(meth, k_i)] += static_cast<double>(M);
        }
      }
    };
    // auto run_scalar_tqpq = [&](auto& tqpq_m, Method meth) {
    //   const size_t pdim = tqpq_m.padded_dim;
    //   std::vector<turboquant_pq_scalar::EncodedVec> scalar_db(N);
    //   std::vector<float> ws;
    //   for (size_t i = 0; i < N; ++i) {
    //     const float* p = reinterpret_cast<const float*>(db.location(i));
    //     scalar_db[i] = turboquant_pq_scalar::encode_single<
    //         std::decay_t<decltype(tqpq_m)>::block_size>(tqpq_m, p, ws);
    //   }
    //   for (size_t qi = 0; qi < Q; ++qi) {
    //     const float* q = reinterpret_cast<const float*>(queries.location(qi));
    //     auto qq = turboquant_pq_scalar::prepare_query<
    //         std::decay_t<decltype(tqpq_m)>::block_size>(tqpq_m, q);
    //     for (size_t i = 0; i < N; ++i) {
    //       approx_scores[i] = {
    //           static_cast<uint32_t>(i),
    //           turboquant_pq_scalar::distance<std::decay_t<decltype(tqpq_m)>::block_size>(
    //               scalar_db[i], qq, pdim, Metric)};
    //     }
    //     parlay::sort_inplace(approx_scores,
    //                         [](const auto& a, const auto& b) { return a.second < b.second; });
    //     std::vector<uint32_t> approx_ids;
    //     approx_ids.reserve(N);
    //     for (const auto& p : approx_scores)
    //       approx_ids.push_back(p.first);
    //     for (size_t k_i = 0; k_i < Kgrid.size(); ++k_i) {
    //       const size_t K = static_cast<size_t>(Kgrid[k_i]);
    //       const size_t need = static_cast<size_t>(std::ceil(rec99 * static_cast<float>(K)));
    //       const size_t M = min_M_for_recall(approx_ids, exact_sets_per_query[qi][k_i], need);
    //       sum_M[idx2(meth, k_i)] += static_cast<double>(M);
    //     }
    //   }
    // };

    if (D >= 1 && (D % 1 == 0)) {
      turboquant_pq_4bit::Model<Metric, 1> tqpq_m;
      tqpq_m.train(db);
      auto tqpq_enc = tqpq_m.encode(db);
#if defined(__AVX512F__)
      run_tqpq_queries(tqpq_m, tqpq_enc, TQPQ16_1);
#endif
      // run_scalar_tqpq(tqpq_m, TQPQ_SCALAR_1);
    }
    if (D >= 2 && (D % 2 == 0)) {
      turboquant_pq_4bit::Model<Metric, 2> tqpq_m;
      tqpq_m.train(db);
      auto tqpq_enc = tqpq_m.encode(db);
#if defined(__AVX512F__)
      run_tqpq_queries(tqpq_m, tqpq_enc, TQPQ16_2);
#endif
      // run_scalar_tqpq(tqpq_m, TQPQ_SCALAR_2);
    }
    if (D >= 4 && (D % 4 == 0)) {
      turboquant_pq_4bit::Model<Metric, 4> tqpq_m;
      tqpq_m.train(db);
      auto tqpq_enc = tqpq_m.encode(db);
#if defined(__AVX512F__)
      run_tqpq_queries(tqpq_m, tqpq_enc, TQPQ16_4);
#endif
      // run_scalar_tqpq(tqpq_m, TQPQ_SCALAR_4);
    }
    if (D >= 8 && (D % 8 == 0)) {
      turboquant_pq_4bit::Model<Metric, 8> tqpq_m;
      tqpq_m.train(db);
      auto tqpq_enc = tqpq_m.encode(db);
#if defined(__AVX512F__)
      run_tqpq_queries(tqpq_m, tqpq_enc, TQPQ16_8);
#endif
      // run_scalar_tqpq(tqpq_m, TQPQ_SCALAR_8);
    }
  }

  std::cout << "\n=== Avg M (candidates) to reach recall@K ===\n";
  std::cout << "Averages over Q=" << Q << " query vectors.\n\n";

  constexpr int method_w = 16;
  constexpr int col_w = 12;

  std::cout << std::left << std::setw(method_w) << "Method";
  for (size_t k_i = 0; k_i < Kgrid.size(); ++k_i) {
    std::string col = "K=" + std::to_string(Kgrid[k_i]);
    std::cout << std::right << std::setw(col_w) << col;
  }
  std::cout << "\n";
  std::cout << std::string(method_w + int(col_w * Kgrid.size()), '-') << "\n";

  for (int meth = 0; meth < NUM_METHODS; ++meth) {
    std::cout << std::left << std::setw(method_w) << method_names[meth];
    for (size_t k_i = 0; k_i < Kgrid.size(); ++k_i) {
      const double avg_M = sum_M[idx2(static_cast<Method>(meth), k_i)] / std::max<size_t>(1, Q);
      std::cout << std::right << std::setw(col_w) << std::fixed << std::setprecision(1) << avg_M;
    }
    std::cout << "\n";
  }

  return 0;
}

int main(int argc, char** argv) {
  commandLine P(argc, argv,
                "[-i <dbFile>] [-q <qFile>] [-gt <gtFile>] "
                "[-N <n>] [-Q <q>] [-D <d>] [-seed_db <s>] [-seed_q <s>] "
                "[-dist_func <L2|IP>] [-fs_block <b>] [-rbits <b>] [-tqpq_block <b>] "
                "[-Kmax <k>] [-rec99 <f>] [-pq] [-rabitq]");

  std::string df = P.getOptionValue("-dist_func", "IP");
  uint32_t fs_block = static_cast<uint32_t>(P.getOptionIntValue("-fs_block", 8));
  uint32_t rbits = static_cast<uint32_t>(P.getOptionIntValue("-rbits", 2));
  uint32_t tqpq_block = static_cast<uint32_t>(P.getOptionIntValue("-tqpq_block", 4));
  uint32_t Kmax = static_cast<uint32_t>(P.getOptionIntValue("-Kmax", 100));
  float rec99 = std::stof(P.getOptionValue("-rec99", "0.99"));
  bool run_pq = P.getOption("-pq");
  bool run_rabitq = P.getOption("-rabitq");

  const bool file_mode = (P.getOptionValue("-i") != nullptr) || (P.getOptionValue("-q") != nullptr);
  const char* gt_file = P.getOptionValue("-gt");

  if (file_mode) {
    if (P.getOptionValue("-i") == nullptr || P.getOptionValue("-q") == nullptr) {
      std::cerr << "ERROR: Provide both -i <dbFile> and -q <qFile> for file mode.\n";
      return 1;
    }

    const char* db_file = P.getOptionValue("-i");
    const char* q_file = P.getOptionValue("-q");

    // Optional ParlayANN ground truth (.ibin).
    IbinGroundTruth gt;
    const IbinGroundTruth* gt_ptr = nullptr;
    if (gt_file) {
      gt = load_ibin_ground_truth(gt_file);
      gt_ptr = &gt;
    }

    if (df == "L2" || df == "l2") {
      mvsic::PointRange<float, mvsic::L2_Point<float>> db(const_cast<char*>(db_file));
      mvsic::PointRange<float, mvsic::L2_Point<float>> queries(const_cast<char*>(q_file));
      if (queries.get_dims() != db.get_dims()) {
        std::cerr << "ERROR: DB dims != query dims.\n";
        return 1;
      }
      if (gt_ptr) {
        if (gt_ptr->num_queries != queries.size()) {
          std::cerr << "ERROR: GT num_queries (" << gt_ptr->num_queries << ") != Q ("
                    << queries.size() << ").\n";
          return 1;
        }
        if (gt_ptr->k < Kmax) {
          std::cerr << "ERROR: GT k (" << gt_ptr->k << ") < Kmax (" << Kmax << ").\n";
          return 1;
        }
      }
      return run_benchmark<true>(db, queries, fs_block, rbits, tqpq_block, Kmax, rec99, gt_ptr,
                                 run_pq, run_rabitq);
    }

    mvsic::PointRange<float, mvsic::IP_Point<float>> db(const_cast<char*>(db_file));
    mvsic::PointRange<float, mvsic::IP_Point<float>> queries(const_cast<char*>(q_file));
    if (queries.get_dims() != db.get_dims()) {
      std::cerr << "ERROR: DB dims != query dims.\n";
      return 1;
    }
    if (gt_ptr) {
      if (gt_ptr->num_queries != queries.size()) {
        std::cerr << "ERROR: GT num_queries (" << gt_ptr->num_queries << ") != Q ("
                  << queries.size() << ").\n";
        return 1;
      }
      if (gt_ptr->k < Kmax) {
        std::cerr << "ERROR: GT k (" << gt_ptr->k << ") < Kmax (" << Kmax << ").\n";
        return 1;
      }
    }
    return run_benchmark<false>(db, queries, fs_block, rbits, tqpq_block, Kmax, rec99, gt_ptr,
                                run_pq, run_rabitq);
  }

  // Synthetic mode
  size_t N = static_cast<size_t>(P.getOptionIntValue("-N", 1000000));
  size_t Q = static_cast<size_t>(P.getOptionIntValue("-Q", 1000));
  uint32_t D = static_cast<uint32_t>(P.getOptionIntValue("-D", 128));
  uint64_t seed_db = 12345ULL;
  uint64_t seed_q = 999ULL;
  if (char* s = P.getOptionValue("-seed_db")) seed_db = static_cast<uint64_t>(std::stoull(s));
  if (char* s = P.getOptionValue("-seed_q")) seed_q = static_cast<uint64_t>(std::stoull(s));

  DensePointRange db(N, D);
  DensePointRange queries(Q, D);
  fill_random(db, seed_db, /*l2_normalize=*/true);
  fill_random(queries, seed_q, /*l2_normalize=*/true);

  if (df == "L2" || df == "l2") {
    return run_benchmark<true>(db, queries, fs_block, rbits, tqpq_block, Kmax, rec99, nullptr,
                               run_pq, run_rabitq);
  }
  return run_benchmark<false>(db, queries, fs_block, rbits, tqpq_block, Kmax, rec99, nullptr,
                              run_pq, run_rabitq);
}
