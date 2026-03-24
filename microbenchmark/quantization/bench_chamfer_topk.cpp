// bench_chamfer_topk.cpp
//
// Leaf-style Chamfer baseline benchmark.
// - Partitions the DB into uniform leaves of size leaf_size.
// - For each query and for powers-of-two numbers of leaves L
//   (1, 2, 4, ..., <= num_leaf_blocks), it simulates MVIVF-style
//   load by:
//     * Baseline: per-leaf distances_all + global sort over all candidates.
// - File mode:
//     -i <dbFile> -q <qFile> [-mm]
//
// - Synthetic mode (default if -i/-q not provided):
//     -N_db <u32>      (default 20000)
//     -N_q  <u32>      (default 200)
//     -K_db <u32>      (default 64)
//     -D    <u32>      (default 128)
//     -seed_db <u64>   (default 12345)
//     -seed_q  <u64>   (default 999)
//     NOTE: query vectors-per-cloud K_q is FIXED to 32.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "parlay/primitives.h"
#include "parlay/parallel.h"

#include "mvsic/core/quantization/fastscan_mv.h"
#include "mvsic/core/quantization/turboquant_mv.h"
#include "mvsic/core/quantization/other_methods/turboquant_pq_4bit.h"
#include "mvsic/core/quantization/other_methods/wrapper.h"

#include "mvsic/core/types/chamfer_ip_point.h"
#include "mvsic/core/types/chamfer_l2_point.h"
#include "mvsic/core/types/point_cloud_set.h"

#include "mvsic/core/utils/parse_command_line.h"

using namespace mvsic;

struct TimerTopk {
  using clock = std::chrono::steady_clock;
  clock::time_point t0;
  void start() { t0 = clock::now(); }
  double sec() const {
    auto t1 = clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
  }
};

struct MethodResults {
  std::string name;                // e.g., "FastScan (K=16)"
  std::vector<size_t> leaves;      // L values
  std::vector<double> baseline_s;  // per L
};

// ---------------------------
// Synthetic data gen for PointCloudSet (matches bench_chamfer_leaf)
// ---------------------------
template<typename PC>
static void fill_random_point_cloud_set(PC& pcs, uint64_t seed, bool l2_normalize_vectors) {
  std::normal_distribution<float> nd(0.0f, 1.0f);

  const uint32_t D = pcs.get_dims();
  const size_t total_vecs = pcs.total_size();
  float* base = pcs.data();

  // Deterministic per-(vi,j) RNG so parallel_for is stable.
  parlay::parallel_for(0, total_vecs, [&](size_t vi) {
    float* v = base + vi * size_t(D);
    float ss = 0.0f;
    for (uint32_t j = 0; j < D; ++j) {
      uint64_t s = seed ^ (vi * 0x9e3779b97f4a7c15ULL) ^ (uint64_t(j) << 32);
      std::mt19937_64 rlocal(s);
      float x = nd(rlocal);
      v[j] = x;
      ss += x * x;
    }
    if (l2_normalize_vectors) {
      float inv = 1.0f / std::sqrt(std::max(ss, 1e-12f));
      for (uint32_t j = 0; j < D; ++j) v[j] *= inv;
    }
  });
}

static void print_speedup_table(const MethodResults& exact, const MethodResults& fs,
                                const MethodResults& tq, const MethodResults& tqpq4,
                                const MethodResults& tqpq8, const char* title) {
  if (exact.leaves.empty() || exact.baseline_s.empty()) return;
  std::cout << "\n=== " << title << " (Exact / Method) ===\n";
  std::cout << std::left << std::setw(10) << "#leaves" << std::right;
  if (!fs.leaves.empty()) std::cout << std::setw(14) << "FS [x]";
  if (!tq.leaves.empty()) std::cout << std::setw(14) << "TQ [x]";
  if (!tqpq4.leaves.empty()) std::cout << std::setw(14) << "TQPQ4 [x]";
  if (!tqpq8.leaves.empty()) std::cout << std::setw(14) << "TQPQ8 [x]";
  std::cout << "\n";

  int num_methods = (!fs.leaves.empty()) + (!tq.leaves.empty()) + (!tqpq4.leaves.empty()) +
                    (!tqpq8.leaves.empty());
  std::cout << std::string(10 + 14 * num_methods, '-') << "\n";

  auto get_time = [&](const MethodResults& mr, size_t leaf) -> double {
    for (size_t i = 0; i < mr.leaves.size(); ++i) {
      if (mr.leaves[i] == leaf) {
        if (i < mr.baseline_s.size()) return mr.baseline_s[i];
        break;
      }
    }
    return 0.0;
  };

  for (size_t i = 0; i < exact.leaves.size(); ++i) {
    const size_t L = exact.leaves[i];
    const double t_exact = exact.baseline_s[i];
    std::cout << std::left << std::setw(10) << L << std::right;
    auto print_ratio = [&](double t_m) {
      double r = (t_m > 0.0) ? (t_exact / t_m) : 0.0;
      std::cout << std::setw(14) << std::fixed << std::setprecision(3) << r;
    };
    if (!fs.leaves.empty()) print_ratio(get_time(fs, L));
    if (!tq.leaves.empty()) print_ratio(get_time(tq, L));
    if (!tqpq4.leaves.empty()) print_ratio(get_time(tqpq4, L));
    if (!tqpq8.leaves.empty()) print_ratio(get_time(tqpq8, L));
    std::cout << "\n";
  }
  std::cout << "\n";
}

static void print_combined_table(const MethodResults& exact, const MethodResults& fs,
                                 const MethodResults& tq, const MethodResults& tqpq4,
                                 const MethodResults& tqpq8, const char* title) {
  const auto& ref_leaves = !exact.leaves.empty()
                               ? exact.leaves
                               : (!fs.leaves.empty()
                                      ? fs.leaves
                                      : (!tq.leaves.empty()
                                             ? tq.leaves
                                             : (!tqpq4.leaves.empty() ? tqpq4.leaves
                                                                      : tqpq8.leaves)));
  if (ref_leaves.empty()) return;

  std::cout << "\n=== " << title << " ===\n";
  std::cout << std::left << std::setw(10) << "#leaves" << std::right;
  if (!exact.leaves.empty()) std::cout << std::setw(14) << "Exact [s]";
  if (!fs.leaves.empty()) std::cout << std::setw(14) << "FS [s]";
  if (!tq.leaves.empty()) std::cout << std::setw(14) << "TQ [s]";
  if (!tqpq4.leaves.empty()) std::cout << std::setw(14) << "TQPQ4 [s]";
  if (!tqpq8.leaves.empty()) std::cout << std::setw(14) << "TQPQ8 [s]";
  std::cout << "\n";

  int num_methods = (!exact.leaves.empty()) + (!fs.leaves.empty()) + (!tq.leaves.empty()) +
                    (!tqpq4.leaves.empty()) + (!tqpq8.leaves.empty());
  std::cout << std::string(10 + 14 * num_methods, '-') << "\n";

  auto get_val = [&](const MethodResults& mr, size_t leaf) {
    for (size_t i = 0; i < mr.leaves.size(); ++i) {
      if (mr.leaves[i] == leaf) {
        if (i < mr.baseline_s.size()) return mr.baseline_s[i];
        break;
      }
    }
    return 0.0;
  };

  for (size_t leaf : ref_leaves) {
    std::cout << std::left << std::setw(10) << leaf << std::right;
    if (!exact.leaves.empty()) {
      double v = get_val(exact, leaf);
      std::cout << std::setw(14) << std::fixed << std::setprecision(6) << v;
    }
    if (!fs.leaves.empty()) {
      double v = get_val(fs, leaf);
      std::cout << std::setw(14) << std::fixed << std::setprecision(6) << v;
    }
    if (!tq.leaves.empty()) {
      double v = get_val(tq, leaf);
      std::cout << std::setw(14) << std::fixed << std::setprecision(6) << v;
    }
    if (!tqpq4.leaves.empty()) {
      double v = get_val(tqpq4, leaf);
      std::cout << std::setw(14) << std::fixed << std::setprecision(6) << v;
    }
    if (!tqpq8.leaves.empty()) {
      double v = get_val(tqpq8, leaf);
      std::cout << std::setw(14) << std::fixed << std::setprecision(6) << v;
    }
    std::cout << "\n";
  }
  std::cout << "\n";
}

// Core runner: given DB/Q sets, run powers-of-two leaf baseline experiments.
template<typename ChPoint>
static int run_from_sets_topk(const PointCloudSet<ChPoint>& db,
                              const PointCloudSet<ChPoint>& queries,
                              uint32_t leaf_size, int reps) {
  using PC = PointCloudSet<ChPoint>;
  constexpr bool Metric = ChPoint::is_metric();

  const uint32_t D = db.get_dims();
  if (queries.get_dims() != D) {
    std::cerr << "ERROR: DB dims (" << D << ") != Query dims (" << queries.get_dims() << ")\n";
    return 1;
  }

  std::cout << "DB: clouds=" << db.size() << "  dims=" << D << "  total_vecs=" << db.total_size()
            << "  avg_k=" << std::fixed << std::setprecision(2) << db.average_size() << std::endl;
  std::cout << "Q : clouds=" << queries.size() << "  dims=" << D
            << "  total_vecs=" << queries.total_size() << "  avg_k=" << std::fixed
            << std::setprecision(2) << queries.average_size() << std::endl;
  std::cout << "leaf_size=" << leaf_size << "  dist=" << (Metric ? "L2" : "IP")
            << "  reps=" << reps << std::endl;

  const size_t N = db.size();
  if (N == 0) {
    std::cerr << "ERROR: empty DB.\n";
    return 1;
  }

  // Light warmup: touch quantization / Chamfer kernels before timed work.
  if (queries.size() > 0) {
    const auto& q0 = queries[0];
    // Simple Chamfer warmup on unquantized data.
    std::vector<std::pair<uint32_t, float>> tmp(db.size());
    db.distances(q0, tmp.data());
  }

  const size_t leaf_n = std::min<size_t>(leaf_size, N);
  size_t num_leaf_blocks = N / leaf_n;
  if (num_leaf_blocks == 0) num_leaf_blocks = 1;
  const size_t used_N = leaf_n * num_leaf_blocks;

  std::cout << "Partitioning first " << used_N << " DB clouds into " << num_leaf_blocks
            << " leaf blocks of size " << leaf_n << " (tail ignored if any)." << std::endl;

  // Build true leaf PointCloudSets (no duplication).
  std::vector<PC> leaves;
  leaves.reserve(num_leaf_blocks);
  auto db_offsets = db.get_offsets();
  float* db_vals = db.data();

  for (size_t b = 0; b < num_leaf_blocks; ++b) {
    const size_t start = b * leaf_n;
    std::vector<size_t> leaf_offsets(leaf_n + 1);
    leaf_offsets[0] = 0;
    for (size_t i = 0; i < leaf_n; ++i) {
      const size_t src_begin = db_offsets[start + i];
      const size_t src_end = db_offsets[start + i + 1];
      const size_t len = src_end - src_begin;
      leaf_offsets[i + 1] = leaf_offsets[i] + len;
    }
    const size_t total_floats = leaf_offsets[leaf_n];
    std::vector<float> leaf_values(total_floats);
    for (size_t i = 0; i < leaf_n; ++i) {
      const size_t src_begin = db_offsets[start + i];
      const size_t src_end = db_offsets[start + i + 1];
      const size_t len = src_end - src_begin;
      std::memcpy(leaf_values.data() + leaf_offsets[i], db_vals + src_begin, len * sizeof(float));
    }
    std::vector<uint32_t> leaf_ids(leaf_n);
    for (size_t i = 0; i < leaf_n; ++i) {
      leaf_ids[i] = db.get_id(start + i);
    }
    PC leaf_pc(static_cast<uint32_t>(leaf_n), D, leaf_values.data(), leaf_offsets.data(),
               leaf_ids.data());
    leaves.emplace_back(std::move(leaf_pc));
  }

  // ---------------------------
  // Quantized models (FastScan, TQ4, TQ-PQ 4/8)
  // ---------------------------
  TimerTopk t;

#if defined(__AVX512F__) || defined(__AVX2__)
  fastscan_mv::Model<Metric> fs_model;
  double fs_train_s = 0.0;
  double fs_encode_s = 0.0;
  std::vector<fastscan_mv::Quantized_Point_Cloud_Set<Metric>> fs_leaf_dbs;
  t.start();
  fs_model.train(db, /*block_size=*/8);
  fs_train_s = t.sec();
  t.start();
  fs_leaf_dbs.reserve(num_leaf_blocks);
  for (size_t b = 0; b < num_leaf_blocks; ++b) {
    fs_leaf_dbs.emplace_back(fs_model.encode(leaves[b]));
  }
  fs_encode_s = t.sec();
#endif

  turboquant_mv::Model<Metric> tq_model;
  double tq_train_s = 0.0;
  double tq_encode_s = 0.0;
  std::vector<turboquant_mv::Quantized_Point_Cloud_Set<Metric>> tq_leaf_dbs;
  t.start();
  tq_model.train(db);
  tq_train_s = t.sec();
  t.start();
  tq_leaf_dbs.reserve(num_leaf_blocks);
  for (size_t b = 0; b < num_leaf_blocks; ++b) {
    tq_leaf_dbs.emplace_back(tq_model.encode(leaves[b]));
  }
  tq_encode_s = t.sec();

  MultiVecQuantizer<turboquant_pq_4bit::Model<Metric, 4>, Metric> tqpq4_model;
  MultiVecQuantizer<turboquant_pq_4bit::Model<Metric, 8>, Metric> tqpq8_model;

  double tqpq4_train_s = 0.0;
  double tqpq4_encode_s = 0.0;
  using TQPQ4_DB = decltype(tqpq4_model.encode(db));
  std::vector<TQPQ4_DB> tqpq4_leaf_dbs;
  t.start();
  tqpq4_model.train(db);
  tqpq4_train_s = t.sec();
  t.start();
  tqpq4_leaf_dbs.reserve(num_leaf_blocks);
  for (size_t b = 0; b < num_leaf_blocks; ++b) {
    tqpq4_leaf_dbs.emplace_back(tqpq4_model.encode(leaves[b]));
  }
  tqpq4_encode_s = t.sec();

  double tqpq8_train_s = 0.0;
  double tqpq8_encode_s = 0.0;
  using TQPQ8_DB = decltype(tqpq8_model.encode(db));
  std::vector<TQPQ8_DB> tqpq8_leaf_dbs;
  t.start();
  tqpq8_model.train(db);
  tqpq8_train_s = t.sec();
  t.start();
  tqpq8_leaf_dbs.reserve(num_leaf_blocks);
  for (size_t b = 0; b < num_leaf_blocks; ++b) {
    tqpq8_leaf_dbs.emplace_back(tqpq8_model.encode(leaves[b]));
  }
  tqpq8_encode_s = t.sec();

  std::cout << "\n=== Train / Encode (FastScan_mv / TurboQuant_mv / TQ-PQ) ===\n";
#if defined(__AVX512F__) || defined(__AVX2__)
  std::cout << "FastScan_mv train    : " << fs_train_s << " s\n";
  std::cout << "FastScan_mv encode   : " << fs_encode_s << " s (leaves)\n";
#endif
  std::cout << "TurboQuant_mv train  : " << tq_train_s << " s\n";
  std::cout << "TurboQuant_mv encode : " << tq_encode_s << " s (leaves)\n";
  std::cout << "TQ-PQ (B=4)          : " << (tqpq4_train_s + tqpq4_encode_s) << " s\n";
  std::cout << "TQ-PQ (B=8)          : " << (tqpq8_train_s + tqpq8_encode_s) << " s\n";

  // ---------------------------
  // Top-k helpers (same allocation + sort as MVIVF in mvivf.h)
  // MVIVF: sizes = parlay::delayed_tabulate, scan → offsets, total_size;
  //        visited = parlay::sequence<...>::uninitialized(total_size);
  //        fill per leaf into &visited[offsets[i]]; then
  //        parlay::sort_inplace(visited, [](a,b){ return a.second < b.second; });
  // ---------------------------
  auto cmp_pair = [](const auto& a, const auto& b) { return a.second < b.second; };

  // Exact baseline: per-leaf PointCloudSet::distances + global sort (mirrors MVIVF exact path).
  auto exact_baseline = [&]() -> MethodResults {
    MethodResults out;
    out.name = "Exact";
    for (size_t L = 1; L <= num_leaf_blocks; L <<= 1) {
      auto sizes = parlay::delayed_tabulate(L, [&](size_t) { return leaf_n; });
      auto scan_result = parlay::scan(sizes);
      const auto& offsets = scan_result.first;
      const size_t total_size = scan_result.second;

      out.leaves.push_back(L);
      TimerTopk m_t;
      double best = 1e100;
      for (int r = 0; r < reps; ++r) {
        m_t.start();
        for (size_t qi = 0; qi < queries.size(); ++qi) {
          auto visited = parlay::sequence<std::pair<uint32_t, float>>::uninitialized(total_size);
          for (size_t b = 0; b < L; ++b) {
            leaves[b].distances(queries[qi], &visited[offsets[b]]);
          }
          parlay::sort_inplace(visited, cmp_pair);
        }
        best = std::min(best, m_t.sec());
      }
      out.baseline_s.push_back(best);
    }
    return out;
  };

  auto baseline_topk_for_method = [&](auto& model, const auto& leaf_dbs_for_model,
                                      MethodResults& out_rows) {
    for (size_t L = 1; L <= num_leaf_blocks; L <<= 1) {
      auto sizes = parlay::delayed_tabulate(L, [&](size_t) { return leaf_n; });
      auto scan_result = parlay::scan(sizes);
      const auto& offsets = scan_result.first;
      const size_t total_size = scan_result.second;

      out_rows.leaves.push_back(L);
      TimerTopk m_t;
      double best = 1e100;
      for (int r = 0; r < reps; ++r) {
        m_t.start();
        for (size_t qi = 0; qi < queries.size(); ++qi) {
          auto qq = model.quantize_query(queries[qi]);
          auto visited = parlay::sequence<std::pair<uint32_t, float>>::uninitialized(total_size);
          for (size_t b = 0; b < L; ++b) {
            leaf_dbs_for_model[b].distances_all(qq, &visited[offsets[b]]);
          }
          parlay::sort_inplace(visited, cmp_pair);
        }
        best = std::min(best, m_t.sec());
      }
      out_rows.baseline_s.push_back(best);
    }
  };

  // ---------------------------
  // Run experiments for each method
  // ---------------------------
  MethodResults rows_exact = exact_baseline();
  MethodResults rows_fs;
  MethodResults rows_tq4;
  MethodResults rows_tqpq4;
  MethodResults rows_tqpq8;

#if defined(__AVX512F__) || defined(__AVX2__)
  rows_fs.name = "FastScan_mv (K=16)";
  baseline_topk_for_method(fs_model, fs_leaf_dbs, rows_fs);
#endif

  rows_tq4.name = "TurboQuant_mv";
  baseline_topk_for_method(tq_model, tq_leaf_dbs, rows_tq4);

  rows_tqpq4.name = "TQ-PQ (K=16,B=4)";
  baseline_topk_for_method(tqpq4_model, tqpq4_leaf_dbs, rows_tqpq4);

  rows_tqpq8.name = "TQ-PQ (K=16,B=8)";
  baseline_topk_for_method(tqpq8_model, tqpq8_leaf_dbs, rows_tqpq8);

  print_combined_table(rows_exact, rows_fs, rows_tq4, rows_tqpq4, rows_tqpq8, "Baseline times");

  // Keep times table as-is (quantized methods). Print speedups vs the exact baseline separately.
  print_speedup_table(rows_exact, rows_fs, rows_tq4, rows_tqpq4, rows_tqpq8,
                      "Baseline speedup");

  return 0;
}

// File mode only.
template<typename ChPoint>
static int run_files_topk(commandLine& P, uint32_t leaf_size, int reps) {
  using PC = PointCloudSet<ChPoint>;

  char* dbFile = P.getOptionValue("-i");
  if (dbFile == nullptr) {
    std::cerr << "ERROR: file mode requires -i <dbFile>\n";
    return 1;
  }
  char* qFile = P.getOptionValue("-q");
  if (qFile == nullptr) {
    std::cerr << "ERROR: file mode requires -q <qFile>\n";
    return 1;
  }

  bool mm = P.getOption("-mm");
  auto db = PC(dbFile, mm);
  auto queries = PC(qFile, /*is_mmap=*/false);

  std::cout << "Mode: file (baseline)" << std::endl;
  std::cout << "  db=" << dbFile << (mm ? " (mmap)" : "") << std::endl;
  std::cout << "  q =" << qFile << std::endl;

  return run_from_sets_topk<ChPoint>(db, queries, leaf_size, reps);
}

// Synthetic mode.
template<typename ChPoint>
static int run_synth_topk(uint32_t N_db, uint32_t N_q, uint32_t K_db, uint32_t D, uint64_t seed_db,
                          uint64_t seed_q, uint32_t leaf_size, int reps) {
  constexpr bool Metric = ChPoint::is_metric();
  using PC = PointCloudSet<ChPoint>;
  const uint32_t K_q = 32;

  PC db(N_db, K_db, D);
  PC queries(N_q, K_q, D);

  const bool l2_normalize_vectors = !Metric;  // IP case
  fill_random_point_cloud_set(db, seed_db, l2_normalize_vectors);
  fill_random_point_cloud_set(queries, seed_q, l2_normalize_vectors);

  std::cout << "Mode: synthetic (K_q fixed to 32)" << std::endl;
  std::cout << "  N_db=" << N_db << "  N_q=" << N_q << "  K_db=" << K_db << "  D=" << D
            << "  seed_db=" << seed_db << "  seed_q=" << seed_q << std::endl;
  return run_from_sets_topk<ChPoint>(db, queries, leaf_size, reps);
}

int main(int argc, char** argv) {
  commandLine P(argc, argv,
                "[-i <dbFile>] [-q <qFile>] [-mm] "
                "[-N_db <n>] [-N_q <n>] [-K_db <k>] [-D <d>] [-seed_db <s>] [-seed_q <s>] "
                "[-leaf_size <b>] [-reps <r>] "
                "[-dist_func <L2|IP>]");

  std::string df = P.getOptionValue("-dist_func", "IP");
  uint32_t leaf_size = static_cast<uint32_t>(P.getOptionIntValue("-leaf_size", 500));
  int reps = std::max(1, P.getOptionIntValue("-reps", 1));

  // Synthetic mode args
  uint32_t N_db = static_cast<uint32_t>(P.getOptionIntValue("-N_db", 20000));
  uint32_t N_q = static_cast<uint32_t>(P.getOptionIntValue("-N_q", 200));
  uint32_t K_db = static_cast<uint32_t>(P.getOptionIntValue("-K_db", 64));
  uint32_t D = static_cast<uint32_t>(P.getOptionIntValue("-D", 128));
  uint64_t seed_db = 12345ULL;
  uint64_t seed_q = 999ULL;
  if (char* s = P.getOptionValue("-seed_db")) seed_db = static_cast<uint64_t>(std::stoull(s));
  if (char* s = P.getOptionValue("-seed_q")) seed_q = static_cast<uint64_t>(std::stoull(s));

  const bool has_files = (P.getOptionValue("-i") != nullptr && P.getOptionValue("-q") != nullptr);

  if (df == "IP" || df == "ip") {
    if (has_files) return run_files_topk<ChamferIP_Point>(P, leaf_size, reps);
    return run_synth_topk<ChamferIP_Point>(N_db, N_q, K_db, D, seed_db, seed_q, leaf_size, reps);
  }
  if (has_files) return run_files_topk<ChamferL2_Point>(P, leaf_size, reps);
  return run_synth_topk<ChamferL2_Point>(N_db, N_q, K_db, D, seed_db, seed_q, leaf_size, reps);
}
