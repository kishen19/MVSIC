// bench_chamfer_leaf.cpp
//
// Point-cloud benchmark for leaf-sized Chamfer (one-to-many) blocks.
//
// Similar interface to bench_chamfer_pq_fastscan:
// - File mode:
//     -i <dbFile>      (PointCloudSet binary)
//     -q <qFile>       (PointCloudSet binary)
//     -mm              (mmap-load DB; queries loaded normally)
//
// - Synthetic mode (default if -i/-q not provided):
//     -N_db <u32>      (default 20000)
//     -N_q  <u32>      (default 200)
//     -K_db <u32>      (default 64)
//     -D    <u32>      (default 128)
//     -seed_db <u64>   (default 12345)
//     -seed_q  <u64>   (default 999)
//     NOTE: query vectors-per-cloud K_q is FIXED to 32.
//
// - Common:
//     -dist_func <L2|IP> (default IP)
//     -leaf_size <u32>   (default 500)
//     -reps <u32>        (default 3)
//
// The benchmark stresses leaf-sized Chamfer blocks using the same API as
// other benchmarks: `PointCloudSet::distances(query_cloud, results)` which
// computes one-to-many Chamfer distances from a query point cloud to all
// clouds in a `PointCloudSet`.
//
//   1) Sequential leaf mode:
//        - Partition the DB into consecutive leaves of size `leaf_size`
//          (ignoring any remainder).
//        - Materialize a `PointCloudSet` per leaf (true partition, no
//          duplication).
//        - For each query and each leaf, call `leaf.distances(query, results)`
//          (running leaves one after another).
//        - Report average time per leaf block.
//
//   2) All-at-once mode:
//        - Build a `PointCloudSet` `all_leafs` as the concatenation of all
//          leaves (i.e., the first `leaf_size * num_leaves` DB clouds).
//        - For each query, call `all_leafs.distances(query, results)` once
//          (all leaves together).
//        - Report average time per leaf block (total time divided by
//          number of logical leaf blocks).

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

#include "parlay/parallel.h"
#include "parlay/primitives.h"

#ifdef __AVX512F__
#include "mvsic/core/quantization/rabitq_mv.h"
#endif
#include "mvsic/core/quantization/fastscan_mv.h"
#include "mvsic/core/quantization/pq_mv.h"
#include "mvsic/core/quantization/turboquant_mv.h"
#include "mvsic/core/quantization/other_methods/turboquant_pq_4bit.h"
#include "mvsic/core/quantization/other_methods/wrapper.h"

#include "mvsic/core/types/chamfer_ip_point.h"
#include "mvsic/core/types/chamfer_l2_point.h"
#include "mvsic/core/types/point_cloud_set.h"

#include "mvsic/core/utils/parse_command_line.h"

using namespace mvsic;

// ---------------------------
// Timing helper
// ---------------------------
struct Timer {
  using clock = std::chrono::steady_clock;
  clock::time_point t0;
  void start() { t0 = clock::now(); }
  double sec() const {
    auto t1 = clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
  }
};

static inline double ns_per_op(double seconds, uint64_t ops) {
  return (seconds * 1e9) / double(ops);
}

struct BenchRow {
  std::string name;
  double best_s = 0.0;
};

static void print_leaf_table(const char* title, const std::vector<BenchRow>& rows,
                             uint64_t logical_blocks, double exact_best) {
  if (rows.empty() || logical_blocks == 0) return;
  std::cout << "\n=== " << title << " ===\n";
  std::cout << std::left << std::setw(28) << "Method" << std::right << std::setw(12) << "time [s]"
            << std::setw(16) << "ns / leaf" << std::setw(12) << "speedup\n";
  std::cout << std::string(28 + 12 + 16 + 12, '-') << "\n";
  for (const auto& r : rows) {
    const double t = r.best_s;
    const double ns = ns_per_op(t, logical_blocks);
    const double speedup = (exact_best > 0.0) ? (exact_best / t) : 0.0;
    std::cout << std::left << std::setw(28) << r.name << std::right << std::setw(12) << std::fixed
              << std::setprecision(4) << t << std::setw(16) << std::setprecision(1) << ns
              << std::setw(12) << std::setprecision(2) << speedup << "\n";
  }
  std::cout << "\n";
}

// ---------------------------
// Synthetic data gen for PointCloudSet
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
      for (uint32_t j = 0; j < D; ++j)
        v[j] *= inv;
    }
  });
}

// ---------------------------
// Benchmark helpers
// ---------------------------
template<typename ChPoint>
static double bench_leaf_sequential(const std::vector<PointCloudSet<ChPoint>>& leaves,
                                    const PointCloudSet<ChPoint>& queries, int reps,
                                    uint64_t& logical_blocks, volatile double& sink) {
  if (leaves.empty() || queries.size() == 0) return 0.0;

  const size_t num_leaf_blocks = leaves.size();
  const size_t leaf_n = leaves[0].size();

  logical_blocks = num_leaf_blocks * static_cast<uint64_t>(queries.size());

  std::vector<std::pair<uint32_t, float>> dists(leaf_n);

  Timer t;
  double best = 1e100;

  // Warmup on first leaf and few queries.
  for (size_t qi = 0; qi < std::min<size_t>(queries.size(), 2); ++qi) {
    leaves[0].distances(queries[qi], dists.data());
    sink += dists[0].second;
  }

  for (int r = 0; r < reps; ++r) {
    t.start();
    for (size_t qi = 0; qi < queries.size(); ++qi) {
      for (size_t b = 0; b < num_leaf_blocks; ++b) {
        leaves[b].distances(queries[qi], dists.data());
        sink += dists[qi % leaf_n].second;
      }
    }
    best = std::min(best, t.sec());
  }

  return best;
}

template<typename ChPoint>
static double bench_leaf_all(const PointCloudSet<ChPoint>& db_full,
                             const PointCloudSet<ChPoint>& queries, size_t num_leaf_blocks,
                             int reps, uint64_t& logical_blocks, volatile double& sink) {
  const size_t used_N = db_full.size();
  if (used_N == 0 || queries.size() == 0 || num_leaf_blocks == 0) return 0.0;

  logical_blocks = num_leaf_blocks * static_cast<uint64_t>(queries.size());

  std::vector<std::pair<uint32_t, float>> dists(used_N);

  Timer t;
  double best = 1e100;

  // Warmup on subset of queries.
  for (size_t qi = 0; qi < std::min<size_t>(queries.size(), 2); ++qi) {
    db_full.distances(queries[qi], dists.data());
    sink += dists[0].second;
  }

  for (int r = 0; r < reps; ++r) {
    t.start();
    for (size_t qi = 0; qi < queries.size(); ++qi) {
      db_full.distances(queries[qi], dists.data());
      sink += dists[qi % used_N].second;
    }
    best = std::min(best, t.sec());
  }

  return best;
}

// ---------------------------
// Core runner given concrete DB/Q sets
// ---------------------------
template<typename ChPoint>
static int run_from_sets(const PointCloudSet<ChPoint>& db, const PointCloudSet<ChPoint>& queries,
                         uint32_t leaf_size, int reps, bool run_pq, bool run_rabitq) {
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
  std::cout << "leaf_size=" << leaf_size << "  dist=" << (Metric ? "L2" : "IP") << "  reps=" << reps
            << std::endl;
  const size_t N = db.size();
  if (N == 0) {
    std::cerr << "ERROR: empty DB.\n";
    return 1;
  }

  const size_t leaf_n = std::min<size_t>(leaf_size, N);
  // Number of uniform leaf blocks; ignore any remainder.
  size_t num_leaf_blocks = N / leaf_n;
  if (num_leaf_blocks == 0) num_leaf_blocks = 1;
  const size_t used_N = leaf_n * num_leaf_blocks;

  std::cout << "Partitioning first " << used_N << " DB clouds into " << num_leaf_blocks
            << " leaf blocks of size " << leaf_n << " (tail ignored if any)." << std::endl;

  const uint64_t total_leaf_blocks = num_leaf_blocks * static_cast<uint64_t>(queries.size());
  std::cout << "Total logical leaf blocks per full sweep (all queries): " << total_leaf_blocks
            << std::endl;

  uint64_t blocks_seq = 0, blocks_all = 0;
  volatile double sink = 0.0;

  // Build true leaf PointCloudSets (no duplication) and an all-leaves PointCloudSet.
  std::vector<PointCloudSet<ChPoint>> leaves;
  leaves.reserve(num_leaf_blocks);
  for (size_t b = 0; b < num_leaf_blocks; ++b) {
    const size_t start = b * leaf_n;

    // Build a compact values/offsets/id view for this uniform leaf.
    auto db_offsets = db.get_offsets();
    float* db_vals = db.data();

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

    PointCloudSet<ChPoint> leaf_pc(static_cast<uint32_t>(leaf_n), D, leaf_values.data(),
                                   leaf_offsets.data(), leaf_ids.data());
    leaves.emplace_back(std::move(leaf_pc));
  }

  // Concatenation of all leaf blocks (first used_N DB clouds) as a single PointCloudSet.
  auto db_offsets = db.get_offsets();
  float* db_vals = db.data();

  std::vector<size_t> all_offsets(used_N + 1);
  all_offsets[0] = 0;
  for (size_t i = 0; i < used_N; ++i) {
    const size_t src_begin = db_offsets[i];
    const size_t src_end = db_offsets[i + 1];
    const size_t len = src_end - src_begin;
    all_offsets[i + 1] = all_offsets[i] + len;
  }

  const size_t all_total_floats = all_offsets[used_N];
  std::vector<float> all_values(all_total_floats);
  for (size_t i = 0; i < used_N; ++i) {
    const size_t src_begin = db_offsets[i];
    const size_t src_end = db_offsets[i + 1];
    const size_t len = src_end - src_begin;
    std::memcpy(all_values.data() + all_offsets[i], db_vals + src_begin, len * sizeof(float));
  }

  std::vector<uint32_t> all_ids(used_N);
  for (size_t i = 0; i < used_N; ++i) {
    all_ids[i] = db.get_id(i);
  }

  PointCloudSet<ChPoint> all_leafs(static_cast<uint32_t>(used_N), D, all_values.data(),
                                   all_offsets.data(), all_ids.data());

  std::cout << "\n=== Leaf benchmark (sequential vs all-at-once) ===" << std::endl;

  double t_seq = bench_leaf_sequential(leaves, queries, reps, blocks_seq, sink);
  double t_all = bench_leaf_all(all_leafs, queries, num_leaf_blocks, reps, blocks_all, sink);

  if (blocks_seq == 0 || blocks_all == 0) {
    std::cerr << "ERROR: zero logical blocks encountered.\n";
    return 1;
  }

  std::cout << "Sequential leaves:" << std::endl;
  std::cout << "  total_time (s): " << std::fixed << std::setprecision(6) << t_seq << std::endl
            << std::endl;

  std::cout << "All leaves together:" << std::endl;
  std::cout << "  total_time (s): " << std::fixed << std::setprecision(6) << t_all << std::endl;

  // ---------------------------
  // Quantized leaf benchmarks
  // ---------------------------
  std::cout << "\n=== Quantized leaf benchmarks ===" << std::endl;

  Timer t;

  const uint32_t pq_block = 8;
  constexpr uint32_t PQ_K = 16;
  constexpr uint32_t PQ_S = 20;
  const uint32_t fs_block = 8;
  const uint32_t rbits = 4;

  pq_mv::Model<Metric> pq_model;
  double pq_train_s = 0.0, pq_encode_s = 0.0;
  std::vector<pq_mv::Quantized_Point_Cloud_Set<Metric>> pq_leaf_dbs;
  pq_mv::Quantized_Point_Cloud_Set<Metric> pq_all_db;
  if (run_pq) {
    t.start();
    pq_model.train(db, pq_block, PQ_K, PQ_S);
    pq_train_s = t.sec();

    t.start();
    pq_leaf_dbs.reserve(num_leaf_blocks);
    for (size_t b = 0; b < num_leaf_blocks; ++b) {
      pq_leaf_dbs.emplace_back(pq_model.encode(leaves[b]));
    }
    pq_all_db = pq_model.encode(all_leafs);
    pq_encode_s = t.sec();
  }

#ifdef __AVX512F__
  rabitq_mv::Model<Metric> rq_model;
  double rq_train_s = 0.0, rq_encode_s = 0.0;
  std::vector<rabitq_mv::Quantized_Point_Cloud_Set<Metric>> rq_leaf_dbs;
  rabitq_mv::Quantized_Point_Cloud_Set<Metric> rq_all_db;
  if (run_rabitq) {
    t.start();
    rq_model.train(db, rbits);
    rq_train_s = t.sec();

    t.start();
    rq_leaf_dbs.reserve(num_leaf_blocks);
    for (size_t b = 0; b < num_leaf_blocks; ++b) {
      rq_leaf_dbs.emplace_back(rq_model.encode(leaves[b]));
    }
    rq_all_db = rq_model.encode(all_leafs);
    rq_encode_s = t.sec();
  }
#endif

#if defined(__AVX512F__) || defined(__AVX2__)
  fastscan_mv::Model<Metric> fs_model;
  t.start();
  fs_model.train(db, fs_block);
  double fs_train_s = t.sec();

  t.start();
  std::vector<fastscan_mv::Quantized_Point_Cloud_Set<Metric>> fs_leaf_dbs;
  fs_leaf_dbs.reserve(num_leaf_blocks);
  for (size_t b = 0; b < num_leaf_blocks; ++b) {
    fs_leaf_dbs.emplace_back(fs_model.encode(leaves[b]));
  }
  auto fs_all_db = fs_model.encode(all_leafs);
  double fs_encode_s = t.sec();
#endif

  turboquant_mv::Model<Metric> tq_model;
  t.start();
  tq_model.train(db);
  double tq_train_s = t.sec();

  t.start();
  std::vector<turboquant_mv::Quantized_Point_Cloud_Set<Metric>> tq_leaf_dbs;
  tq_leaf_dbs.reserve(num_leaf_blocks);
  for (size_t b = 0; b < num_leaf_blocks; ++b) {
    tq_leaf_dbs.emplace_back(tq_model.encode(leaves[b]));
  }
  auto tq_all_db = tq_model.encode(all_leafs);
  double tq_encode_s = t.sec();

  // TurboQuant PQ 4-bit (B=1/2/4/8)
  MultiVecQuantizer<turboquant_pq_4bit::Model<Metric, 1>, Metric> tqpq1_model;
  MultiVecQuantizer<turboquant_pq_4bit::Model<Metric, 2>, Metric> tqpq2_model;
  MultiVecQuantizer<turboquant_pq_4bit::Model<Metric, 4>, Metric> tqpq4_model;
  MultiVecQuantizer<turboquant_pq_4bit::Model<Metric, 8>, Metric> tqpq8_model;

  t.start();
  tqpq1_model.train(db);
  double tqpq1_train_s = t.sec();
  t.start();
  using TQPQ1_DB = decltype(tqpq1_model.encode(db));
  std::vector<TQPQ1_DB> tqpq1_leaf_dbs;
  tqpq1_leaf_dbs.reserve(num_leaf_blocks);
  for (size_t b = 0; b < num_leaf_blocks; ++b) {
    tqpq1_leaf_dbs.emplace_back(tqpq1_model.encode(leaves[b]));
  }
  auto tqpq1_all_db = tqpq1_model.encode(all_leafs);
  double tqpq1_encode_s = t.sec();

  t.start();
  tqpq2_model.train(db);
  double tqpq2_train_s = t.sec();
  t.start();
  using TQPQ2_DB = decltype(tqpq2_model.encode(db));
  std::vector<TQPQ2_DB> tqpq2_leaf_dbs;
  tqpq2_leaf_dbs.reserve(num_leaf_blocks);
  for (size_t b = 0; b < num_leaf_blocks; ++b) {
    tqpq2_leaf_dbs.emplace_back(tqpq2_model.encode(leaves[b]));
  }
  auto tqpq2_all_db = tqpq2_model.encode(all_leafs);
  double tqpq2_encode_s = t.sec();

  t.start();
  tqpq4_model.train(db);
  double tqpq4_train_s = t.sec();
  t.start();
  using TQPQ4_DB = decltype(tqpq4_model.encode(db));
  std::vector<TQPQ4_DB> tqpq4_leaf_dbs;
  tqpq4_leaf_dbs.reserve(num_leaf_blocks);
  for (size_t b = 0; b < num_leaf_blocks; ++b) {
    tqpq4_leaf_dbs.emplace_back(tqpq4_model.encode(leaves[b]));
  }
  auto tqpq4_all_db = tqpq4_model.encode(all_leafs);
  double tqpq4_encode_s = t.sec();

  t.start();
  tqpq8_model.train(db);
  double tqpq8_train_s = t.sec();
  t.start();
  using TQPQ8_DB = decltype(tqpq8_model.encode(db));
  std::vector<TQPQ8_DB> tqpq8_leaf_dbs;
  tqpq8_leaf_dbs.reserve(num_leaf_blocks);
  for (size_t b = 0; b < num_leaf_blocks; ++b) {
    tqpq8_leaf_dbs.emplace_back(tqpq8_model.encode(leaves[b]));
  }
  auto tqpq8_all_db = tqpq8_model.encode(all_leafs);
  double tqpq8_encode_s = t.sec();

  std::cout << "\n=== Train / Encode (quantized) ===\n";
  if (run_pq) {
    std::cout << "PQ(K=16) train       : " << pq_train_s << " s\n";
    std::cout << "PQ(K=16) encode      : " << pq_encode_s << " s (leaves + all_leafs)\n";
  } else {
    std::cout << "PQ(K=16)             : (skipped, pass -pq to enable)\n";
  }
#ifdef __AVX512F__
  if (run_rabitq) {
    std::cout << "RaBitQ train         : " << rq_train_s << " s\n";
    std::cout << "RaBitQ encode        : " << rq_encode_s << " s (leaves + all_leafs)\n";
  } else {
    std::cout << "RaBitQ               : (skipped, pass -rabitq to enable)\n";
  }
#endif
#if defined(__AVX512F__) || defined(__AVX2__)
  std::cout << "FastScan train   : " << fs_train_s << " s\n";
  std::cout << "FastScan encode  : " << fs_encode_s << " s (leaves + all_leafs)\n";
#endif
  std::cout << "TurboQuant_mv train : " << tq_train_s << " s\n";
  std::cout << "TurboQuant_mv encode: " << tq_encode_s << " s (leaves + all_leafs)\n";
  std::cout << "TQ-PQ(K=16,B=1)       : " << (tqpq1_train_s + tqpq1_encode_s) << " s\n";
  std::cout << "TQ-PQ(K=16,B=2)       : " << (tqpq2_train_s + tqpq2_encode_s) << " s\n";
  std::cout << "TQ-PQ(K=16,B=4)       : " << (tqpq4_train_s + tqpq4_encode_s) << " s\n";
  std::cout << "TQ-PQ(K=16,B=8)       : " << (tqpq8_train_s + tqpq8_encode_s) << " s\n";

  const uint64_t leaf_blocks_total = num_leaf_blocks * static_cast<uint64_t>(queries.size());

  // ---------------------------
  // Quantized: sequential leaves
  // ---------------------------
  std::vector<std::pair<uint32_t, float>> q_results(leaf_n);
  volatile double sink_q = 0.0;

  auto bench_seq = [&](auto& model, const auto& leaf_dbs_for_model) -> double {
    Timer m_t;
    double best = 1e100;
    for (int r = 0; r < reps; ++r) {
      m_t.start();
      for (size_t qi = 0; qi < queries.size(); ++qi) {
        auto qq = model.quantize_query(queries[qi]);
        for (size_t b = 0; b < num_leaf_blocks; ++b) {
          leaf_dbs_for_model[b].distances_all(qq, q_results.data());
          sink_q += q_results[qi % leaf_n].second;
        }
      }
      best = std::min(best, m_t.sec());
    }
    return best;
  };

  std::vector<BenchRow> seq_rows;
  seq_rows.push_back({"Exact", t_seq});
  if (run_pq) {
    seq_rows.push_back({"PQ_mv (K=16)", bench_seq(pq_model, pq_leaf_dbs)});
  }
#if defined(__AVX512F__) || defined(__AVX2__)
  seq_rows.push_back({"FastScan_mv (K=16)", bench_seq(fs_model, fs_leaf_dbs)});
#endif
#ifdef __AVX512F__
  if (run_rabitq) {
    seq_rows.push_back({"RaBitQ_mv", bench_seq(rq_model, rq_leaf_dbs)});
  }
#endif
  seq_rows.push_back({"TurboQuant_mv", bench_seq(tq_model, tq_leaf_dbs)});
  seq_rows.push_back({"TQ-PQ (K=16,B=1)", bench_seq(tqpq1_model, tqpq1_leaf_dbs)});
  seq_rows.push_back({"TQ-PQ (K=16,B=2)", bench_seq(tqpq2_model, tqpq2_leaf_dbs)});
  seq_rows.push_back({"TQ-PQ (K=16,B=4)", bench_seq(tqpq4_model, tqpq4_leaf_dbs)});
  seq_rows.push_back({"TQ-PQ (K=16,B=8)", bench_seq(tqpq8_model, tqpq8_leaf_dbs)});

  print_leaf_table("Sequential leaves (quantized)", seq_rows, leaf_blocks_total, t_seq);

  // ---------------------------
  // Quantized: all leaves together
  // ---------------------------
  const size_t used_N_all = all_leafs.size();
  std::vector<std::pair<uint32_t, float>> q_results_all(used_N_all);
  auto bench_all = [&](auto& model, const auto& all_db_for_model) -> double {
    Timer m_t;
    double best = 1e100;
    for (int r = 0; r < reps; ++r) {
      m_t.start();
      for (size_t qi = 0; qi < queries.size(); ++qi) {
        auto qq = model.quantize_query(queries[qi]);
        all_db_for_model.distances_all(qq, q_results_all.data());
        sink_q += q_results_all[qi % used_N].second;
      }
      best = std::min(best, m_t.sec());
    }
    return best;
  };

  std::vector<BenchRow> all_rows;
  all_rows.push_back({"Exact", t_all});
  if (run_pq) {
    all_rows.push_back({"PQ_mv (K=16)", bench_all(pq_model, pq_all_db)});
  }
#if defined(__AVX512F__) || defined(__AVX2__)
  all_rows.push_back({"FastScan_mv (K=16)", bench_all(fs_model, fs_all_db)});
#endif
#ifdef __AVX512F__
  if (run_rabitq) {
    all_rows.push_back({"RaBitQ_mv", bench_all(rq_model, rq_all_db)});
  }
#endif
  all_rows.push_back({"TurboQuant_mv", bench_all(tq_model, tq_all_db)});
  all_rows.push_back({"TQ-PQ (K=16,B=1)", bench_all(tqpq1_model, tqpq1_all_db)});
  all_rows.push_back({"TQ-PQ (K=16,B=2)", bench_all(tqpq2_model, tqpq2_all_db)});
  all_rows.push_back({"TQ-PQ (K=16,B=4)", bench_all(tqpq4_model, tqpq4_all_db)});
  all_rows.push_back({"TQ-PQ (K=16,B=8)", bench_all(tqpq8_model, tqpq8_all_db)});

  print_leaf_table("All leaves together (quantized)", all_rows, leaf_blocks_total, t_all);

  std::cout << "\n(sink=" << sink << ")" << std::endl;
  std::cout << "(sink_quant=" << sink_q << ")" << std::endl;
  return 0;
}

// ---------------------------
// Synthetic mode
// ---------------------------
template<typename ChPoint>
static int run_synth(uint32_t N_db, uint32_t N_q, uint32_t K_db, uint32_t D, uint64_t seed_db,
                     uint64_t seed_q, uint32_t leaf_size, int reps, bool run_pq, bool run_rabitq) {
  constexpr bool Metric = ChPoint::is_metric();
  using PC = PointCloudSet<ChPoint>;

  const uint32_t K_q = 32;  // fixed per request

  PC db(N_db, K_db, D);
  PC queries_full(N_q, K_q, D);
  auto samples = parlay::delayed_tabulate(200, [&](size_t i) { return queries_full[i]; });
  PC queries(samples, D);

  const bool l2_normalize_vectors = !Metric;  // IP case
  fill_random_point_cloud_set(db, seed_db, l2_normalize_vectors);
  fill_random_point_cloud_set(queries, seed_q, l2_normalize_vectors);

  std::cout << "Mode: synthetic (K_q fixed to 32)" << std::endl;
  return run_from_sets<ChPoint>(db, queries, leaf_size, reps, run_pq, run_rabitq);
}

// ---------------------------
// File mode
// ---------------------------
template<typename ChPoint>
static int run_files(commandLine& P, uint32_t leaf_size, int reps, bool run_pq, bool run_rabitq) {
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

  std::cout << "Mode: file" << std::endl;
  std::cout << "  db=" << dbFile << (mm ? " (mmap)" : "") << std::endl;
  std::cout << "  q =" << qFile << std::endl;
  // For file mode, respect -pq and -rabitq flags as in synthetic mode.
  return run_from_sets<ChPoint>(db, queries, leaf_size, reps, run_pq, run_rabitq);
}

int main(int argc, char** argv) {
  commandLine P(argc, argv,
                "[-i <dbFile>] [-q <qFile>] [-mm] "
                "[-N_db <n>] [-N_q <n>] [-K_db <k>] [-D <d>] [-seed_db <s>] [-seed_q <s>] "
                "[-dist_func <L2|IP>] [-leaf_size <b>] [-reps <r>] [-pq] [-rabitq]");

  // Common
  std::string df = P.getOptionValue("-dist_func", "IP");
  uint32_t leaf_size = static_cast<uint32_t>(P.getOptionIntValue("-leaf_size", 500));
  int reps = std::max(1, P.getOptionIntValue("-reps", 1));
  bool run_pq = P.getOption("-pq");
  bool run_rabitq = P.getOption("-rabitq");

  // Decide mode: if both -i and -q are present => file mode, else synthetic
  const bool file_mode = (P.getOptionValue("-i") != nullptr) || (P.getOptionValue("-q") != nullptr);
  if (file_mode) {
    // Require both.
    if (P.getOptionValue("-i") == nullptr || P.getOptionValue("-q") == nullptr) {
      std::cerr << "ERROR: Provide both -i <dbFile> and -q <qFile> for file mode.\n";
      return 1;
    }

    if (df == "IP" || df == "ip")
      return run_files<ChamferIP_Point>(P, leaf_size, reps, run_pq, run_rabitq);
    return run_files<ChamferL2_Point>(P, leaf_size, reps, run_pq, run_rabitq);
  }

  // Synthetic mode args
  uint32_t N_db = static_cast<uint32_t>(P.getOptionIntValue("-N_db", 20000));
  uint32_t N_q = static_cast<uint32_t>(P.getOptionIntValue("-N_q", 200));
  uint32_t K_db = static_cast<uint32_t>(P.getOptionIntValue("-K_db", 64));
  uint32_t D = static_cast<uint32_t>(P.getOptionIntValue("-D", 128));

  // Seeds (as u64)
  uint64_t seed_db = 12345ULL;
  uint64_t seed_q = 999ULL;
  if (char* s = P.getOptionValue("-seed_db")) seed_db = static_cast<uint64_t>(std::stoull(s));
  if (char* s = P.getOptionValue("-seed_q")) seed_q = static_cast<uint64_t>(std::stoull(s));

  if (df == "IP" || df == "ip") {
    return run_synth<ChamferIP_Point>(N_db, N_q, K_db, D, seed_db, seed_q, leaf_size, reps, run_pq,
                                      run_rabitq);
  }
  return run_synth<ChamferL2_Point>(N_db, N_q, K_db, D, seed_db, seed_q, leaf_size, reps, run_pq,
                                    run_rabitq);
}
