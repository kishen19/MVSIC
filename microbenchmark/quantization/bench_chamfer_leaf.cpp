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
//     -dist_func <L2|IP> (default L2)
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

#include "mvsic/core/quantization/fastscan.h"
#include "mvsic/core/quantization/turboquant_4bit.h"
#include "mvsic/core/quantization/turboquant_byte.h"
#include "mvsic/core/quantization/turboquant_low_bit.h"
#include "mvsic/core/quantization/wrapper.h"

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
static double bench_leaf_all(const PointCloudSet<ChPoint>& all_leafs,
                             const PointCloudSet<ChPoint>& queries, size_t num_leaf_blocks,
                             int reps, uint64_t& logical_blocks, volatile double& sink) {
  const size_t used_N = all_leafs.size();
  if (used_N == 0 || queries.size() == 0 || num_leaf_blocks == 0) return 0.0;

  logical_blocks = num_leaf_blocks * static_cast<uint64_t>(queries.size());

  std::vector<std::pair<uint32_t, float>> dists(used_N);

  Timer t;
  double best = 1e100;

  // Warmup on subset of queries.
  for (size_t qi = 0; qi < std::min<size_t>(queries.size(), 2); ++qi) {
    all_leafs.distances(queries[qi], dists.data());
    sink += dists[0].second;
  }

  for (int r = 0; r < reps; ++r) {
    t.start();
    for (size_t qi = 0; qi < queries.size(); ++qi) {
      all_leafs.distances(queries[qi], dists.data());
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
static int run_from_sets(const PointCloudSet<ChPoint>& db,
                         const PointCloudSet<ChPoint>& queries, uint32_t leaf_size, int reps) {
  using PC = PointCloudSet<ChPoint>;
  constexpr bool Metric = ChPoint::is_metric();

  const uint32_t D = db.get_dims();
  if (queries.get_dims() != D) {
    std::cerr << "ERROR: DB dims (" << D << ") != Query dims (" << queries.get_dims() << ")\n";
    return 1;
  }

  std::cout << "DB: clouds=" << db.size() << "  dims=" << D << "  total_vecs=" << db.total_size()
            << "  avg_k=" << std::fixed << std::setprecision(2) << db.average_size() << "\n";
  std::cout << "Q : clouds=" << queries.size() << "  dims=" << D
            << "  total_vecs=" << queries.total_size() << "  avg_k=" << std::fixed
            << std::setprecision(2) << queries.average_size() << "\n";
  std::cout << "leaf_size=" << leaf_size << "  dist=" << (Metric ? "L2" : "IP")
            << "  reps=" << reps << "\n";
  const size_t N = db.size();
  if (N == 0) {
    std::cerr << "ERROR: empty DB.\n";
    return 1;
  }

  const size_t leaf_n = std::min<size_t>(leaf_size, N);
  size_t num_leaf_blocks = N / leaf_n;  // ignore remainder to get uniform leaves
  if (num_leaf_blocks == 0) num_leaf_blocks = 1;
  const size_t used_N = leaf_n * num_leaf_blocks;

  std::cout << "Partitioning first " << used_N << " DB clouds into " << num_leaf_blocks
            << " leaf blocks of size " << leaf_n << " (tail ignored if any).\n";

  const uint64_t total_leaf_blocks = num_leaf_blocks * static_cast<uint64_t>(queries.size());
  std::cout << "Total logical leaf blocks per full sweep (all queries): " << total_leaf_blocks
            << "\n";

  uint64_t blocks_seq = 0, blocks_all = 0;
  volatile double sink = 0.0;

  // Build true leaf PointCloudSets (no duplication) and an all-leaves PointCloudSet.
  std::vector<PointCloudSet<ChPoint>> leaves;
  leaves.reserve(num_leaf_blocks);
  for (size_t b = 0; b < num_leaf_blocks; ++b) {
    parlay::sequence<ChPoint> leaf_clouds(leaf_n);
    const size_t start = b * leaf_n;
    for (size_t i = 0; i < leaf_n; ++i) {
      leaf_clouds[i] = db[start + i];
    }
    leaves.emplace_back(leaf_clouds, D);
  }

  parlay::sequence<ChPoint> all_clouds(used_N);
  for (size_t i = 0; i < used_N; ++i) all_clouds[i] = db[i];
  PointCloudSet<ChPoint> all_leafs(all_clouds, D);

  std::cout << "\n=== Leaf benchmark (sequential vs all-at-once) ===\n";

  double t_seq = bench_leaf_sequential(leaves, queries, reps, blocks_seq, sink);
  double t_all = bench_leaf_all(all_leafs, queries, num_leaf_blocks, reps, blocks_all, sink);

  if (blocks_seq == 0 || blocks_all == 0) {
    std::cerr << "ERROR: zero logical blocks encountered.\n";
    return 1;
  }

  double ns_per_block_seq = ns_per_op(t_seq, blocks_seq);
  double ns_per_block_all = ns_per_op(t_all, blocks_all);

  std::cout << "Sequential leaves:\n";
  std::cout << "  total_time : " << t_seq << " s\n";
  std::cout << "  ns / leaf  : " << std::fixed << std::setprecision(3) << ns_per_block_seq
            << "\n";

  std::cout << "All leaves together:\n";
  std::cout << "  total_time : " << t_all << " s\n";
  std::cout << "  ns / leaf  : " << std::fixed << std::setprecision(3) << ns_per_block_all
            << "\n";

  // ---------------------------
  // Quantized leaf benchmarks
  // ---------------------------
  std::cout << "\n=== Quantized leaf benchmarks ===\n";

  Timer t;

#if defined(__AVX512F__) || defined(__AVX2__)
  // FastScan
  MultiVecQuantizer<fastscan::Model<Metric>, Metric> fs_model;
  t.start();
  fs_model.train(db, /*fs_block=*/8);
  double fs_train_s = t.sec();

  t.start();
  using FS_DB = decltype(fs_model.encode(db));
  std::vector<FS_DB> fs_leaf_dbs;
  fs_leaf_dbs.reserve(num_leaf_blocks);
  for (size_t b = 0; b < num_leaf_blocks; ++b) {
    fs_leaf_dbs.emplace_back(fs_model.encode(leaves[b]));
  }
  auto fs_all_db = fs_model.encode(all_leafs);
  double fs_encode_s = t.sec();
#endif

  // TurboQuant (4-bit)
  MultiVecQuantizer<turboquant_4bit::Model<Metric>, Metric> tq_model;
  t.start();
  tq_model.train(db);
  double tq_train_s = t.sec();

  t.start();
  using TQ4_DB = decltype(tq_model.encode(db));
  std::vector<TQ4_DB> tq_leaf_dbs;
  tq_leaf_dbs.reserve(num_leaf_blocks);
  for (size_t b = 0; b < num_leaf_blocks; ++b) {
    tq_leaf_dbs.emplace_back(tq_model.encode(leaves[b]));
  }
  auto tq_all_db = tq_model.encode(all_leafs);
  double tq_encode_s = t.sec();

  // Byte TurboQuant (int8)
  MultiVecQuantizer<turboquant_byte::Model<Metric>, Metric> btq_model;
  t.start();
  btq_model.train(db);
  double btq_train_s = t.sec();

  t.start();
  using BTQ_DB = decltype(btq_model.encode(db));
  std::vector<BTQ_DB> btq_leaf_dbs;
  btq_leaf_dbs.reserve(num_leaf_blocks);
  for (size_t b = 0; b < num_leaf_blocks; ++b) {
    btq_leaf_dbs.emplace_back(btq_model.encode(leaves[b]));
  }
  auto btq_all_db = btq_model.encode(all_leafs);
  double btq_encode_s = t.sec();

  // TQ-1bit and TQ-2bit (scalar low-bit Chamfer using same rotator as TQ4)
  using EncodedVec = turboquant_low_bit::EncodedVec;
  using PreparedQuery = turboquant_low_bit::PreparedQuery;
  const auto& tq4_vec_model = tq_model.vec_model;
  const size_t pdim = tq4_vec_model.padded_dim;

  // Encode DB leaves (per-cloud, per-vector) for 1-bit and 2-bit.
  std::vector<std::vector<std::vector<EncodedVec>>> enc_1bit_leaves(num_leaf_blocks);
  std::vector<std::vector<std::vector<EncodedVec>>> enc_2bit_leaves(num_leaf_blocks);
  t.start();
  for (size_t b = 0; b < num_leaf_blocks; ++b) {
    const auto& leaf_pc = leaves[b];
    enc_1bit_leaves[b].resize(leaf_n);
    enc_2bit_leaves[b].resize(leaf_n);
    for (size_t i = 0; i < leaf_n; ++i) {
      const uint32_t nv = leaf_pc.get_size(i);
      enc_1bit_leaves[b][i].resize(nv);
      enc_2bit_leaves[b][i].resize(nv);
      const float* base = leaf_pc.data(i);
      std::vector<float> ws;
      for (uint32_t j = 0; j < nv; ++j) {
        enc_1bit_leaves[b][i][j] =
            turboquant_low_bit::encode_1bit(tq4_vec_model, base + j * D, ws);
        enc_2bit_leaves[b][i][j] =
            turboquant_low_bit::encode_2bit(tq4_vec_model, base + j * D, ws);
      }
    }
  }
  double tq_lowbit_encode_s = t.sec();

  // Prepare queries for 1-bit and 2-bit.
  std::vector<std::vector<PreparedQuery>> pqs_1bit(queries.size());
  std::vector<std::vector<PreparedQuery>> pqs_2bit(queries.size());
  for (size_t qi = 0; qi < queries.size(); ++qi) {
    const uint32_t nv = queries.get_size(qi);
    pqs_1bit[qi].resize(nv);
    pqs_2bit[qi].resize(nv);
    const float* base = queries.data(qi);
    for (uint32_t j = 0; j < nv; ++j) {
      pqs_1bit[qi][j] =
          turboquant_low_bit::prepare_query(tq4_vec_model, base + j * D);
      pqs_2bit[qi][j] =
          turboquant_low_bit::prepare_query(tq4_vec_model, base + j * D);
    }
  }

  std::cout << "\n=== Train / Encode (quantized) ===\n";
#if defined(__AVX512F__) || defined(__AVX2__)
  std::cout << "FastScan train   : " << fs_train_s << " s\n";
  std::cout << "FastScan encode  : " << fs_encode_s << " s (leaves + all_leafs)\n";
#endif
  std::cout << "TurboQuant-4bit train : " << tq_train_s << " s\n";
  std::cout << "TurboQuant-4bit encode: " << tq_encode_s << " s (leaves + all_leafs)\n";
  std::cout << "ByteTQ-int8 train     : " << btq_train_s << " s\n";
  std::cout << "ByteTQ-int8 encode    : " << btq_encode_s << " s (leaves + all_leafs)\n";
  std::cout << "TQ-lowbit encode (1+2bit, all leaves): " << tq_lowbit_encode_s << " s\n";

  const uint64_t leaf_blocks_total = num_leaf_blocks * static_cast<uint64_t>(queries.size());

  // ---------------------------
  // Quantized: sequential leaves
  // ---------------------------
  std::cout << "\n--- Sequential leaves (quantized) ---\n";

  std::vector<std::pair<uint32_t, float>> q_results(leaf_n);
  double t_seq_tq4 = 0.0, t_seq_btq = 0.0;
#if defined(__AVX512F__) || defined(__AVX2__)
  double t_seq_fs = 0.0;
#endif
  volatile double sink_q = 0.0;

  // TQ-4bit
  {
    Timer tq_t;
    double best = 1e100;
    for (int r = 0; r < reps; ++r) {
      tq_t.start();
      for (size_t qi = 0; qi < queries.size(); ++qi) {
        auto qq = tq_model.quantize_query(queries[qi]);
        for (size_t b = 0; b < num_leaf_blocks; ++b) {
          tq_leaf_dbs[b].distances_all(qq, q_results.data());
          sink_q += q_results[qi % leaf_n].second;
        }
      }
      best = std::min(best, tq_t.sec());
    }
    t_seq_tq4 = best;
  }

  // ByteTQ-int8
  {
    Timer btq_t;
    double best = 1e100;
    for (int r = 0; r < reps; ++r) {
      btq_t.start();
      for (size_t qi = 0; qi < queries.size(); ++qi) {
        auto qq = btq_model.quantize_query(queries[qi]);
        for (size_t b = 0; b < num_leaf_blocks; ++b) {
          btq_leaf_dbs[b].distances_all(qq, q_results.data());
          sink_q += q_results[qi % leaf_n].second;
        }
      }
      best = std::min(best, btq_t.sec());
    }
    t_seq_btq = best;
  }

#if defined(__AVX512F__) || defined(__AVX2__)
  // FastScan
  {
    Timer fs_t;
    double best = 1e100;
    for (int r = 0; r < reps; ++r) {
      fs_t.start();
      for (size_t qi = 0; qi < queries.size(); ++qi) {
        auto qq = fs_model.quantize_query(queries[qi]);
        for (size_t b = 0; b < num_leaf_blocks; ++b) {
          fs_leaf_dbs[b].distances_all(qq, q_results.data());
          sink_q += q_results[qi % leaf_n].second;
        }
      }
      best = std::min(best, fs_t.sec());
    }
    t_seq_fs = best;
  }
#endif

  // TQ-1bit and TQ-2bit (scalar Chamfer per leaf)
  auto chamfer_leaf_lowbit = [&](bool two_bit) {
    Timer t_lb;
    double best = 1e100;
    for (int r = 0; r < reps; ++r) {
      t_lb.start();
      for (size_t qi = 0; qi < queries.size(); ++qi) {
        const auto& q_pre_1 = pqs_1bit[qi];
        const auto& q_pre_2 = pqs_2bit[qi];
        for (size_t b = 0; b < num_leaf_blocks; ++b) {
          const auto& leaf_pc = leaves[b];
          double leaf_sum = 0.0;
          for (uint32_t c = 0; c < leaf_n; ++c) {
            float dist = 0.0f;
            // Chamfer: average over query vectors of min distance to vectors in cloud c.
            const uint32_t nv_q = queries.get_size(qi);
            for (uint32_t qv = 0; qv < nv_q; ++qv) {
              const auto& pq = two_bit ? q_pre_2[qv] : q_pre_1[qv];
              float best_v = std::numeric_limits<float>::max();
              const auto& enc_vecs =
                  two_bit ? enc_2bit_leaves[b][c] : enc_1bit_leaves[b][c];
              for (const auto& ev : enc_vecs) {
                float dv = two_bit ? turboquant_low_bit::distance_2bit(ev, pq, pdim, Metric)
                                   : turboquant_low_bit::distance_1bit(ev, pq, pdim, Metric);
                if (dv < best_v) best_v = dv;
              }
              leaf_sum += best_v;
            }
          }
        }
      }
      best = std::min(best, t_lb.sec());
    }
    return best;
  };

  double t_seq_tq1 = chamfer_leaf_lowbit(false);
  double t_seq_tq2 = chamfer_leaf_lowbit(true);

  std::cout << "  TQ4   seq ns/leaf : " << ns_per_op(t_seq_tq4, leaf_blocks_total) << "\n";
  std::cout << "  ByteTQ seq ns/leaf: " << ns_per_op(t_seq_btq, leaf_blocks_total) << "\n";
#if defined(__AVX512F__) || defined(__AVX2__)
  std::cout << "  FastScan seq ns/leaf: " << ns_per_op(t_seq_fs, leaf_blocks_total) << "\n";
#endif
  std::cout << "  TQ-1bit seq ns/leaf: " << ns_per_op(t_seq_tq1, leaf_blocks_total) << "\n";
  std::cout << "  TQ-2bit seq ns/leaf: " << ns_per_op(t_seq_tq2, leaf_blocks_total) << "\n";

  // ---------------------------
  // Quantized: all leaves together
  // ---------------------------
  std::cout << "\n--- All leaves together (quantized) ---\n";

  std::vector<std::pair<uint32_t, float>> q_results_all(used_N);
  double t_all_tq4 = 0.0, t_all_btq = 0.0;
#if defined(__AVX512F__) || defined(__AVX2__)
  double t_all_fs = 0.0;
#endif

  // TQ-4bit
  {
    Timer tq_t;
    double best = 1e100;
    for (int r = 0; r < reps; ++r) {
      tq_t.start();
      for (size_t qi = 0; qi < queries.size(); ++qi) {
        auto qq = tq_model.quantize_query(queries[qi]);
        tq_all_db.distances_all(qq, q_results_all.data());
        sink_q += q_results_all[qi % used_N].second;
      }
      best = std::min(best, tq_t.sec());
    }
    t_all_tq4 = best;
  }

  // ByteTQ-int8
  {
    Timer btq_t;
    double best = 1e100;
    for (int r = 0; r < reps; ++r) {
      btq_t.start();
      for (size_t qi = 0; qi < queries.size(); ++qi) {
        auto qq = btq_model.quantize_query(queries[qi]);
        btq_all_db.distances_all(qq, q_results_all.data());
        sink_q += q_results_all[qi % used_N].second;
      }
      best = std::min(best, btq_t.sec());
    }
    t_all_btq = best;
  }

#if defined(__AVX512F__) || defined(__AVX2__)
  // FastScan
  {
    Timer fs_t;
    double best = 1e100;
    for (int r = 0; r < reps; ++r) {
      fs_t.start();
      for (size_t qi = 0; qi < queries.size(); ++qi) {
        auto qq = fs_model.quantize_query(queries[qi]);
        fs_all_db.distances_all(qq, q_results_all.data());
        sink_q += q_results_all[qi % used_N].second;
      }
      best = std::min(best, fs_t.sec());
    }
    t_all_fs = best;
  }
#endif

  std::cout << "  TQ4   all ns/leaf : " << ns_per_op(t_all_tq4, leaf_blocks_total) << "\n";
  std::cout << "  ByteTQ all ns/leaf: " << ns_per_op(t_all_btq, leaf_blocks_total) << "\n";
#if defined(__AVX512F__) || defined(__AVX2__)
  std::cout << "  FastScan all ns/leaf: " << ns_per_op(t_all_fs, leaf_blocks_total) << "\n";
#endif

  std::cout << "\n(sink=" << sink << ")\n";
  std::cout << "(sink_quant=" << sink_q << ")\n";
  return 0;
}

// ---------------------------
// Synthetic mode
// ---------------------------
template<typename ChPoint>
static int run_synth(uint32_t N_db, uint32_t N_q, uint32_t K_db, uint32_t D, uint64_t seed_db,
                     uint64_t seed_q, uint32_t leaf_size, int reps) {
  constexpr bool Metric = ChPoint::is_metric();
  using PC = PointCloudSet<ChPoint>;

  const uint32_t K_q = 32;  // fixed per request

  PC db(N_db, K_db, D);
  PC queries(N_q, K_q, D);

  const bool l2_normalize_vectors = !Metric;  // IP case
  fill_random_point_cloud_set(db, seed_db, l2_normalize_vectors);
  fill_random_point_cloud_set(queries, seed_q, l2_normalize_vectors);

  std::cout << "Mode: synthetic (K_q fixed to 32)\n";
  return run_from_sets<ChPoint>(db, queries, leaf_size, reps);
}

// ---------------------------
// File mode
// ---------------------------
template<typename ChPoint>
static int run_files(commandLine& P, uint32_t leaf_size, int reps) {
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

  std::cout << "Mode: file\n";
  std::cout << "  db=" << dbFile << (mm ? " (mmap)\n" : "\n");
  std::cout << "  q =" << qFile << "\n";
  return run_from_sets<ChPoint>(db, queries, leaf_size, reps);
}

int main(int argc, char** argv) {
  commandLine P(argc, argv,
                "[-i <dbFile>] [-q <qFile>] [-mm] "
                "[-N_db <n>] [-N_q <n>] [-K_db <k>] [-D <d>] [-seed_db <s>] [-seed_q <s>] "
                "[-dist_func <L2|IP>] [-leaf_size <b>] [-reps <r>]");

  // Common
  std::string df = P.getOptionValue("-dist_func", "L2");
  uint32_t leaf_size = static_cast<uint32_t>(P.getOptionIntValue("-leaf_size", 500));
  int reps = std::max(1, P.getOptionIntValue("-reps", 3));

  // Decide mode: if both -i and -q are present => file mode, else synthetic
  const bool file_mode = (P.getOptionValue("-i") != nullptr) || (P.getOptionValue("-q") != nullptr);
  if (file_mode) {
    // Require both.
    if (P.getOptionValue("-i") == nullptr || P.getOptionValue("-q") == nullptr) {
      std::cerr << "ERROR: Provide both -i <dbFile> and -q <qFile> for file mode.\n";
      return 1;
    }

    if (df == "IP" || df == "ip")
      return run_files<ChamferIP_Point>(P, leaf_size, reps);
    return run_files<ChamferL2_Point>(P, leaf_size, reps);
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
    return run_synth<ChamferIP_Point>(N_db, N_q, K_db, D, seed_db, seed_q, leaf_size, reps);
  }
  return run_synth<ChamferL2_Point>(N_db, N_q, K_db, D, seed_db, seed_q, leaf_size, reps);
}

