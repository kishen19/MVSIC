// bench_sym_tqpq_allpairs.cpp
//
// Synthetic quality microbenchmark:
//   - Generate N random D-dimensional vectors.
//   - Treat each vector as both a DB point and a query.
//   - For each query, compute exact distances to all DB points (excluding self)
//     and obtain exact top-K neighbors.
//   - Compare recall@K of:
//       * TurboQuant-4bit (strip-interleaved implementation)
//       * Symmetric scalar TQ-PQ-4bit (both DB and queries PQ-coded; centroid-to-centroid)
//
// This is an O(N^2) benchmark intended for *quality*, not speed. Choose N
// accordingly (defaults are modest).
//
// Args (parse_command_line.h style, synthetic only):
//   -N <u32>        (default 5000)     // number of vectors (DB = queries = N)
//   -D <u32>        (default 128)
//   -K <u32>        (default 10)       // recall@K
//   -seed <u64>     (default 12345)
//   -dist_func <L2|IP>  (default IP)
//

#include <algorithm>
#include <array>
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
#include "mvsic/core/quantization/other_methods/turboquant_4bit.h"
#include "mvsic/core/quantization/other_methods/turboquant_pq_4bit.h"
#include "mvsic/core/quantization/other_methods/turboquant_pq_sym_scalar.h"

#include "mvsic/core/types/ip_point.h"
#include "mvsic/core/types/l2_point.h"
#include "mvsic/core/types/point_range.h"
#include "mvsic/core/utils/parse_command_line.h"

using namespace mvsic;

// Simple dense synthetic container (matches bench_overretrieve.cpp style).
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

// Optional external point source for run_allpairs: when set, the benchmark
// uses this range instead of generating random synthetic data.
static const DensePointRange* g_external_pts = nullptr;

// Fill with N(0,1) and L2-normalize (good default for IP; harmless for L2).
static void fill_random(DensePointRange& r, uint64_t seed) {
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
    float inv = 1.0f / std::sqrt(std::max(ss, 1e-12f));
    for (uint32_t j = 0; j < r.dim; ++j)
      v[j] *= inv;
  }
}

template<bool Metric>
static int run_allpairs(const DensePointRange& db, const DensePointRange& queries, uint32_t K,
                        bool self_mode) {
  const size_t N = db.size();
  const size_t Q = queries.size();
  const uint32_t D = db.get_dims();

  if (N < 2) {
    std::cerr << "ERROR: N must be >= 2.\n";
    return 1;
  }
  if (Q == 0) {
    std::cerr << "ERROR: Q must be >= 1.\n";
    return 1;
  }
  if (queries.get_dims() != D) {
    std::cerr << "ERROR: DB dims != query dims.\n";
    return 1;
  }
  if (K == 0 || K >= N) {
    std::cerr << "ERROR: K must be in [1, N-1].\n";
    return 1;
  }

  std::cout << "Symmetric TQ-PQ scalar vs TQ-4bit all-pairs recall benchmark\n";
  std::cout << "N(db)=" << N << " Q=" << Q << " D=" << D << " K=" << K
            << " dist=" << (Metric ? "L2" : "IP") << "\n";

  // --------------------------------------------------------------------------
  // 1) Exact ground truth: for each query, exact top-K over DB.
  // --------------------------------------------------------------------------
  std::vector<std::vector<uint32_t>> exact_topk(Q);
  std::vector<std::unordered_set<uint32_t>> exact_sets(Q);
  parlay::parallel_for(0, Q, [&](size_t qi) {
    const float* q = reinterpret_cast<const float*>(queries.location(qi));

    static thread_local std::vector<std::pair<uint32_t, float>> tl_exact;
    tl_exact.clear();
    tl_exact.reserve(N);

    static thread_local efanna2e::DistanceInnerProduct distfunc_ip;
    static thread_local efanna2e::DistanceL2 distfunc_l2;

    for (size_t j = 0; j < N; ++j) {
      if (self_mode && j == qi) continue;
      const float* p = reinterpret_cast<const float*>(db.location(j));
      float dist;
      if constexpr (Metric) {
        dist = distfunc_l2.compare(q, p, D);
      } else {
        dist = -distfunc_ip.compare(q, p, D);
      }
      tl_exact.emplace_back(static_cast<uint32_t>(j), dist);
    }

    std::nth_element(tl_exact.begin(), tl_exact.begin() + K, tl_exact.end(),
                     [](const auto& a, const auto& b) { return a.second < b.second; });
    std::sort(tl_exact.begin(), tl_exact.begin() + K,
              [](const auto& a, const auto& b) { return a.second < b.second; });

    exact_topk[qi].resize(K);
    for (uint32_t r = 0; r < K; ++r)
      exact_topk[qi][r] = tl_exact[r].first;

    auto& truth = exact_sets[qi];
    truth.clear();
    truth.reserve(K * 2);
    for (uint32_t r = 0; r < K; ++r)
      truth.insert(exact_topk[qi][r]);
  });

  // Candidate budgets M' in terms of K: {K, 2K, 4K, 8K, 16K}, clipped to [1, N-1].
  static constexpr std::array<uint32_t, 5> kBudgetMultipliers = {1, 2, 4, 8, 16};
  std::array<size_t, kBudgetMultipliers.size()> budgets_M;
  for (size_t i = 0; i < kBudgetMultipliers.size(); ++i) {
    const uint64_t raw = static_cast<uint64_t>(kBudgetMultipliers[i]) * static_cast<uint64_t>(K);
    size_t M = static_cast<size_t>(std::min<uint64_t>(raw, static_cast<uint64_t>(N - 1)));
    if (M == 0) M = 1;
    budgets_M[i] = M;
  }

  // --------------------------------------------------------------------------
  // 2) TurboQuant-4bit: train, encode, compute recall at M' = {K, 2K, 4K, 8K, 16K}.
  // --------------------------------------------------------------------------
  turboquant_4bit::Model<Metric> tq_model;
  tq_model.train(db);
  auto tq_db = tq_model.encode(db);

  std::vector<std::array<double, kBudgetMultipliers.size()>> tq_recall_per_query(Q);

  parlay::parallel_for(0, Q, [&](size_t qi) {
    const float* q = reinterpret_cast<const float*>(queries.location(qi));
    auto qq = tq_model.quantize_query(q);

    static thread_local std::vector<float> tl_tq_dists;
    static thread_local std::vector<std::pair<uint32_t, float>> tl_approx;
    tl_tq_dists.resize(N);
    tl_approx.clear();
    tl_approx.reserve(N);

    // Full distances to all points.
    qq.distances_contiguous(tq_db.packed_codes.data(), tq_db.norm_scaling_factors.data(),
                            tq_db.unquantized_squared_norms.data(), tq_db.stride, N,
                            tl_tq_dists.data());

    for (size_t j = 0; j < N; ++j) {
      tl_approx.emplace_back(static_cast<uint32_t>(j), tl_tq_dists[j]);
    }

    std::sort(tl_approx.begin(), tl_approx.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });

    std::vector<uint32_t> approx_ids;
    approx_ids.reserve(tl_approx.size());
    for (const auto& p : tl_approx)
      approx_ids.push_back(p.first);

    const auto& truth = exact_sets[qi];
    std::array<double, kBudgetMultipliers.size()> rec{};
    for (size_t bi = 0; bi < budgets_M.size(); ++bi) {
      const size_t M = std::min(budgets_M[bi], approx_ids.size());
      size_t hits = 0;
      for (size_t m = 0; m < M; ++m) {
        if (truth.find(approx_ids[m]) != truth.end()) ++hits;
      }
      rec[bi] = static_cast<double>(hits) / static_cast<double>(K);
    }
    tq_recall_per_query[qi] = rec;
  });

  std::array<double, kBudgetMultipliers.size()> tq_recall_sum{};
  for (size_t qi = 0; qi < Q; ++qi)
    for (size_t bi = 0; bi < tq_recall_sum.size(); ++bi)
      tq_recall_sum[bi] += tq_recall_per_query[qi][bi];

  // --------------------------------------------------------------------------
  // 3) Symmetric scalar TQ-PQ-4bit for all supported block sizes B.
  // --------------------------------------------------------------------------
  struct SymResult {
    std::string name;
    std::array<double, kBudgetMultipliers.size()> recall_sum;
    bool active = false;
  };

  std::array<SymResult, 4> sym_results = {{
      {"TQ-PQ-sym-B1", {}, false},
      {"TQ-PQ-sym-B2", {}, false},
      {"TQ-PQ-sym-B4", {}, false},
      {"TQ-PQ-sym-B8", {}, false},
  }};

  auto run_sym_B1 = [&](SymResult& res) {
    if (D < 1 || (D % 1) != 0) return;
    using SymModel1 = turboquant_pq_4bit::Model<Metric, 1>;
    SymModel1 sym_model;
    sym_model.train(db);
    const size_t pdim = sym_model.padded_dim;

    std::vector<turboquant_pq_sym_scalar::EncodedVec> sym_db(N);
    std::vector<float> ws;
    for (size_t i = 0; i < N; ++i) {
      const float* p = reinterpret_cast<const float*>(db.location(i));
      sym_db[i] = turboquant_pq_sym_scalar::encode_single<1>(sym_model, p, ws);
    }

    std::vector<std::array<double, kBudgetMultipliers.size()>> rec_per_query(Q);
    parlay::parallel_for(0, Q, [&](size_t qi) {
      const float* q = reinterpret_cast<const float*>(queries.location(qi));
      static thread_local std::vector<float> ws_q;
      static thread_local std::vector<std::pair<uint32_t, float>> sym_buf;

      auto qq = turboquant_pq_sym_scalar::prepare_query<1>(sym_model, q, ws_q);

      sym_buf.clear();
      sym_buf.reserve(N);

      for (size_t j = 0; j < N; ++j) {
        float dist = turboquant_pq_sym_scalar::distance<1>(sym_db[j], qq, pdim, Metric);
        sym_buf.emplace_back(static_cast<uint32_t>(j), dist);
      }

      std::sort(sym_buf.begin(), sym_buf.end(),
                [](const auto& a, const auto& b) { return a.second < b.second; });

      std::vector<uint32_t> approx_ids;
      approx_ids.reserve(sym_buf.size());
      for (const auto& p : sym_buf)
        approx_ids.push_back(p.first);

      const auto& truth = exact_sets[qi];
      std::array<double, kBudgetMultipliers.size()> rec{};
      for (size_t bi = 0; bi < budgets_M.size(); ++bi) {
        const size_t M = std::min(budgets_M[bi], approx_ids.size());
        size_t hits = 0;
        for (size_t m = 0; m < M; ++m) {
          if (truth.find(approx_ids[m]) != truth.end()) ++hits;
        }
        rec[bi] = static_cast<double>(hits) / static_cast<double>(K);
      }
      rec_per_query[qi] = rec;
    });

    res.recall_sum = {};
    for (size_t qi = 0; qi < Q; ++qi)
      for (size_t bi = 0; bi < res.recall_sum.size(); ++bi)
        res.recall_sum[bi] += rec_per_query[qi][bi];
    res.active = true;
  };

  auto run_sym_B2 = [&](SymResult& res) {
    if (D < 2 || (D % 2) != 0) return;
    using SymModel2 = turboquant_pq_4bit::Model<Metric, 2>;
    SymModel2 sym_model;
    sym_model.train(db);
    const size_t pdim = sym_model.padded_dim;

    std::vector<turboquant_pq_sym_scalar::EncodedVec> sym_db(N);
    std::vector<float> ws;
    for (size_t i = 0; i < N; ++i) {
      const float* p = reinterpret_cast<const float*>(db.location(i));
      sym_db[i] = turboquant_pq_sym_scalar::encode_single<2>(sym_model, p, ws);
    }

    std::vector<std::array<double, kBudgetMultipliers.size()>> rec_per_query(Q);
    parlay::parallel_for(0, Q, [&](size_t qi) {
      const float* q = reinterpret_cast<const float*>(queries.location(qi));
      static thread_local std::vector<float> ws_q;
      static thread_local std::vector<std::pair<uint32_t, float>> sym_buf;

      auto qq = turboquant_pq_sym_scalar::prepare_query<2>(sym_model, q, ws_q);

      sym_buf.clear();
      sym_buf.reserve(N);

      for (size_t j = 0; j < N; ++j) {
        float dist = turboquant_pq_sym_scalar::distance<2>(sym_db[j], qq, pdim, Metric);
        sym_buf.emplace_back(static_cast<uint32_t>(j), dist);
      }

      std::sort(sym_buf.begin(), sym_buf.end(),
                [](const auto& a, const auto& b) { return a.second < b.second; });

      std::vector<uint32_t> approx_ids;
      approx_ids.reserve(sym_buf.size());
      for (const auto& p : sym_buf)
        approx_ids.push_back(p.first);

      const auto& truth = exact_sets[qi];
      std::array<double, kBudgetMultipliers.size()> rec{};
      for (size_t bi = 0; bi < budgets_M.size(); ++bi) {
        const size_t M = std::min(budgets_M[bi], approx_ids.size());
        size_t hits = 0;
        for (size_t m = 0; m < M; ++m) {
          if (truth.find(approx_ids[m]) != truth.end()) ++hits;
        }
        rec[bi] = static_cast<double>(hits) / static_cast<double>(K);
      }
      rec_per_query[qi] = rec;
    });

    res.recall_sum = {};
    for (size_t qi = 0; qi < Q; ++qi)
      for (size_t bi = 0; bi < res.recall_sum.size(); ++bi)
        res.recall_sum[bi] += rec_per_query[qi][bi];
    res.active = true;
  };

  auto run_sym_B4 = [&](SymResult& res) {
    if (D < 4 || (D % 4) != 0) return;
    using SymModel4 = turboquant_pq_4bit::Model<Metric, 4>;
    SymModel4 sym_model;
    sym_model.train(db);
    const size_t pdim = sym_model.padded_dim;

    std::vector<turboquant_pq_sym_scalar::EncodedVec> sym_db(N);
    std::vector<float> ws;
    for (size_t i = 0; i < N; ++i) {
      const float* p = reinterpret_cast<const float*>(db.location(i));
      sym_db[i] = turboquant_pq_sym_scalar::encode_single<4>(sym_model, p, ws);
    }

    std::vector<std::array<double, kBudgetMultipliers.size()>> rec_per_query(Q);
    parlay::parallel_for(0, Q, [&](size_t qi) {
      const float* q = reinterpret_cast<const float*>(queries.location(qi));
      static thread_local std::vector<float> ws_q;
      static thread_local std::vector<std::pair<uint32_t, float>> sym_buf;

      auto qq = turboquant_pq_sym_scalar::prepare_query<4>(sym_model, q, ws_q);

      sym_buf.clear();
      sym_buf.reserve(N);

      for (size_t j = 0; j < N; ++j) {
        float dist = turboquant_pq_sym_scalar::distance<4>(sym_db[j], qq, pdim, Metric);
        sym_buf.emplace_back(static_cast<uint32_t>(j), dist);
      }

      std::sort(sym_buf.begin(), sym_buf.end(),
                [](const auto& a, const auto& b) { return a.second < b.second; });

      std::vector<uint32_t> approx_ids;
      approx_ids.reserve(sym_buf.size());
      for (const auto& p : sym_buf)
        approx_ids.push_back(p.first);

      const auto& truth = exact_sets[qi];
      std::array<double, kBudgetMultipliers.size()> rec{};
      for (size_t bi = 0; bi < budgets_M.size(); ++bi) {
        const size_t M = std::min(budgets_M[bi], approx_ids.size());
        size_t hits = 0;
        for (size_t m = 0; m < M; ++m) {
          if (truth.find(approx_ids[m]) != truth.end()) ++hits;
        }
        rec[bi] = static_cast<double>(hits) / static_cast<double>(K);
      }
      rec_per_query[qi] = rec;
    });

    res.recall_sum = {};
    for (size_t qi = 0; qi < Q; ++qi)
      for (size_t bi = 0; bi < res.recall_sum.size(); ++bi)
        res.recall_sum[bi] += rec_per_query[qi][bi];
    res.active = true;
  };

  auto run_sym_B8 = [&](SymResult& res) {
    if (D < 8 || (D % 8) != 0) return;
    using SymModel8 = turboquant_pq_4bit::Model<Metric, 8>;
    SymModel8 sym_model;
    sym_model.train(db);
    const size_t pdim = sym_model.padded_dim;

    std::vector<turboquant_pq_sym_scalar::EncodedVec> sym_db(N);
    std::vector<float> ws;
    for (size_t i = 0; i < N; ++i) {
      const float* p = reinterpret_cast<const float*>(db.location(i));
      sym_db[i] = turboquant_pq_sym_scalar::encode_single<8>(sym_model, p, ws);
    }

    std::vector<std::array<double, kBudgetMultipliers.size()>> rec_per_query(Q);
    parlay::parallel_for(0, Q, [&](size_t qi) {
      const float* q = reinterpret_cast<const float*>(queries.location(qi));
      static thread_local std::vector<float> ws_q;
      static thread_local std::vector<std::pair<uint32_t, float>> sym_buf;

      auto qq = turboquant_pq_sym_scalar::prepare_query<8>(sym_model, q, ws_q);

      sym_buf.clear();
      sym_buf.reserve(N);

      for (size_t j = 0; j < N; ++j) {
        float dist = turboquant_pq_sym_scalar::distance<8>(sym_db[j], qq, pdim, Metric);
        sym_buf.emplace_back(static_cast<uint32_t>(j), dist);
      }

      std::sort(sym_buf.begin(), sym_buf.end(),
                [](const auto& a, const auto& b) { return a.second < b.second; });

      std::vector<uint32_t> approx_ids;
      approx_ids.reserve(sym_buf.size());
      for (const auto& p : sym_buf)
        approx_ids.push_back(p.first);

      const auto& truth = exact_sets[qi];
      std::array<double, kBudgetMultipliers.size()> rec{};
      for (size_t bi = 0; bi < budgets_M.size(); ++bi) {
        const size_t M = std::min(budgets_M[bi], approx_ids.size());
        size_t hits = 0;
        for (size_t m = 0; m < M; ++m) {
          if (truth.find(approx_ids[m]) != truth.end()) ++hits;
        }
        rec[bi] = static_cast<double>(hits) / static_cast<double>(K);
      }
      rec_per_query[qi] = rec;
    });

    res.recall_sum = {};
    for (size_t qi = 0; qi < Q; ++qi)
      for (size_t bi = 0; bi < res.recall_sum.size(); ++bi)
        res.recall_sum[bi] += rec_per_query[qi][bi];
    res.active = true;
  };

  run_sym_B1(sym_results[0]);
  run_sym_B2(sym_results[1]);
  run_sym_B4(sym_results[2]);
  run_sym_B8(sym_results[3]);

  // --------------------------------------------------------------------------
  // 4) Report average recall for all methods and budgets.
  // --------------------------------------------------------------------------
  std::cout << std::fixed << std::setprecision(4);
  std::cout << "Candidate budgets (M'): "
            << "K=" << budgets_M[0] << ", "
            << "2K=" << budgets_M[1] << ", "
            << "4K=" << budgets_M[2] << ", "
            << "8K=" << budgets_M[3] << ", "
            << "16K=" << budgets_M[4] << "\n";
  std::cout << "Average recall over N=" << N << " queries:\n\n";

  auto print_row = [&](const std::string& name,
                       const std::array<double, kBudgetMultipliers.size()>& sums) {
    std::cout << std::left << std::setw(20) << name;
    for (size_t bi = 0; bi < sums.size(); ++bi) {
      double avg = sums[bi] / static_cast<double>(Q);
      std::cout << " " << std::setw(10) << avg;
    }
    std::cout << "\n";
  };

  std::cout << std::left << std::setw(20) << "Method"
            << " " << std::setw(10) << "K"
            << " " << std::setw(10) << "2K"
            << " " << std::setw(10) << "4K"
            << " " << std::setw(10) << "8K"
            << " " << std::setw(10) << "16K"
            << "\n";
  std::cout << std::string(20 + 1 + 5 * 10, '-') << "\n";

  print_row("TQ-4bit", tq_recall_sum);
  for (const auto& res : sym_results) {
    if (res.active)
      print_row(res.name, res.recall_sum);
  }

  return 0;
}

int main(int argc, char** argv) {
  commandLine P(
      argc, argv,
      "[-i <dbFile>] [-q <qFile>] "
      "[-N <n>] [-D <d>] [-K <k>] [-seed <s>] "
      "[-block_N <n>] [-max_blocks <b>] "
      "[-dist_func <L2|IP>]");

  uint32_t K = static_cast<uint32_t>(P.getOptionIntValue("-K", 10));
  std::string df = P.getOptionValue("-dist_func", "IP");

  const char* db_file = P.getOptionValue("-i");
  const char* q_file = P.getOptionValue("-q");
  const bool file_mode = (db_file != nullptr);

  if (file_mode) {
    uint32_t block_N = static_cast<uint32_t>(P.getOptionIntValue("-block_N", 1000));
    uint32_t max_blocks = static_cast<uint32_t>(P.getOptionIntValue("-max_blocks", 10));

    if (df == "L2" || df == "l2") {
      mvsic::PointRange<float, mvsic::L2_Point<float>> db(const_cast<char*>(db_file));
      const size_t Ndb = db.size();
      const uint32_t D = db.get_dims();
      if (Ndb < 2) {
        std::cerr << "ERROR: DB has fewer than 2 points.\n";
        return 1;
      }

      DensePointRange query_pts;
      if (q_file) {
        mvsic::PointRange<float, mvsic::L2_Point<float>> queries_file(const_cast<char*>(q_file));
        if (queries_file.get_dims() != D) {
          std::cerr << "ERROR: DB dims != query dims.\n";
          return 1;
        }
        const size_t Qfull = queries_file.size();
        const size_t Qsub = std::min<size_t>(1000, Qfull);
        if (Qsub == 0) {
          std::cerr << "ERROR: Query file has zero vectors.\n";
          return 1;
        }
        query_pts = DensePointRange(Qsub, D);
        for (size_t i = 0; i < Qsub; ++i) {
          const float* src =
              reinterpret_cast<const float*>(queries_file.location(i));
          float* dst = query_pts.data() + i * static_cast<size_t>(D);
          std::memcpy(dst, src, sizeof(float) * D);
        }
      }

      const size_t total_blocks = (Ndb + block_N - 1) / block_N;
      const size_t blocks_to_run = std::min<size_t>(total_blocks, max_blocks);

      std::cout << "File mode: DB=" << db_file << " vectors=" << Ndb << " dims=" << D
                << " K=" << K << " dist=L2\n";
      std::cout << "Processing up to " << blocks_to_run << " blocks of size " << block_N
                << " (last block may be smaller).\n";

      for (size_t b = 0; b < blocks_to_run; ++b) {
        const size_t start = b * static_cast<size_t>(block_N);
        if (start >= Ndb) break;
        const size_t remaining = Ndb - start;
        const size_t cur_N = std::min<size_t>(remaining, block_N);
        if (cur_N < 2) continue;

        DensePointRange block_pts(cur_N, D);
        for (size_t i = 0; i < cur_N; ++i) {
          const float* src =
              reinterpret_cast<const float*>(db.location(start + i));
          float* dst = block_pts.data() + i * static_cast<size_t>(D);
          std::memcpy(dst, src, sizeof(float) * D);
        }

        std::cout << "\n=== Block " << (b + 1) << "/" << blocks_to_run << " (start=" << start
                  << ", N=" << cur_N << ") ===\n";
        if (q_file) {
          int ret = run_allpairs<true>(block_pts, query_pts, K, /*self_mode=*/false);
          if (ret != 0) return ret;
        } else {
          int ret = run_allpairs<true>(block_pts, block_pts, K, /*self_mode=*/true);
          if (ret != 0) return ret;
        }
      }
      return 0;
    }

    // IP mode
    mvsic::PointRange<float, mvsic::IP_Point<float>> db(const_cast<char*>(db_file));
    const size_t Ndb = db.size();
    const uint32_t D = db.get_dims();
    if (Ndb < 2) {
      std::cerr << "ERROR: DB has fewer than 2 points.\n";
      return 1;
    }

    DensePointRange query_pts;
    if (q_file) {
      mvsic::PointRange<float, mvsic::IP_Point<float>> queries_file(const_cast<char*>(q_file));
      if (queries_file.get_dims() != D) {
        std::cerr << "ERROR: DB dims != query dims.\n";
        return 1;
      }
      const size_t Qfull = queries_file.size();
      const size_t Qsub = std::min<size_t>(1000, Qfull);
      if (Qsub == 0) {
        std::cerr << "ERROR: Query file has zero vectors.\n";
        return 1;
      }
      query_pts = DensePointRange(Qsub, D);
      for (size_t i = 0; i < Qsub; ++i) {
        const float* src =
            reinterpret_cast<const float*>(queries_file.location(i));
        float* dst = query_pts.data() + i * static_cast<size_t>(D);
        std::memcpy(dst, src, sizeof(float) * D);
      }
    }

    const size_t total_blocks = (Ndb + block_N - 1) / block_N;
    const size_t blocks_to_run = std::min<size_t>(total_blocks, max_blocks);

    std::cout << "File mode: DB=" << db_file << " vectors=" << Ndb << " dims=" << D
              << " K=" << K << " dist=IP\n";
    std::cout << "Processing up to " << blocks_to_run << " blocks of size " << block_N
              << " (last block may be smaller).\n";

    for (size_t b = 0; b < blocks_to_run; ++b) {
      const size_t start = b * static_cast<size_t>(block_N);
      if (start >= Ndb) break;
      const size_t remaining = Ndb - start;
      const size_t cur_N = std::min<size_t>(remaining, block_N);
      if (cur_N < 2) continue;

      DensePointRange block_pts(cur_N, D);
      for (size_t i = 0; i < cur_N; ++i) {
        const float* src =
            reinterpret_cast<const float*>(db.location(start + i));
        float* dst = block_pts.data() + i * static_cast<size_t>(D);
        std::memcpy(dst, src, sizeof(float) * D);
      }

      std::cout << "\n=== Block " << (b + 1) << "/" << blocks_to_run << " (start=" << start
                << ", N=" << cur_N << ") ===\n";
      if (q_file) {
        int ret = run_allpairs<false>(block_pts, query_pts, K, /*self_mode=*/false);
        if (ret != 0) return ret;
      } else {
        int ret = run_allpairs<false>(block_pts, block_pts, K, /*self_mode=*/true);
        if (ret != 0) return ret;
      }
    }
    return 0;
  }

  // Synthetic mode (no -i provided).
  uint32_t N = static_cast<uint32_t>(P.getOptionIntValue("-N", 5000));
  uint32_t D = static_cast<uint32_t>(P.getOptionIntValue("-D", 128));
  uint64_t seed = 12345ULL;
  if (char* s = P.getOptionValue("-seed")) seed = static_cast<uint64_t>(std::stoull(s));

  if (df == "L2" || df == "l2") {
    DensePointRange pts(N, D);
    fill_random(pts, seed);
    return run_allpairs<true>(pts, pts, K, /*self_mode=*/true);
  }
  DensePointRange pts(N, D);
  fill_random(pts, seed);
  return run_allpairs<false>(pts, pts, K, /*self_mode=*/true);
}

