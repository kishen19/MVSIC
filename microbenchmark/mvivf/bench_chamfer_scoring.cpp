// Chamfer scoring microbenchmark for MVIVF leaf nodes.
//
// Builds (or loads) an MVIVF hierarchy on a real dataset and times Chamfer
// distance scoring of pointcloud-vs-pointcloud against the *full* set of
// pointclouds packed into each chosen leaf -- mirroring exactly what happens
// in mvivf greedy/flat search when probing leaves.
//
// Four implementations are timed against the same inputs:
//
//   (1) float baseline       -- PointCloudSet::distances(query, ...)
//   (2) legacy TQ one-to-many -- pre-39427e2 strip-layout TQ + chamfer_vnni_gemm,
//                                 called once per query (legacy_tq_mv)
//   (3) current TQ one-to-many -- post-39427e2 panel-layout TQ + chamfer_panels,
//                                 called once per query (turboquant_mv::Set::distances_all)
//   (4) current TQ many-to-many -- post-39427e2 fused panel kernel
//                                  (turboquant_mv::ManyToMany::TopKIntoUninitialized)
//
// Usage:
//   bazel run -c opt --copt="-march=native" --copt="-DHOMEGROWN" \
//       //microbenchmark/mvivf:bench_chamfer_scoring -- \
//       -i data/beir/arguana/arguana_points.pcs \
//       -q data/beir/arguana/arguana_queries.pcs \
//       [-x out_arguana/mvivf/mvivf_500_tq/index_149ec46c.bin] \
//       [-num_leaves 32] [-num_q_clouds 64] [-q_block 8] [-reps 5] [-seed 0]
//
// If -x is omitted, the benchmark builds an MVIVF index in-process with the
// same params as out_arguana/mvivf/mvivf_500_tq (k_per_level=0, max_leaf_size=500,
// quantize_centers=false, pq.method=TurboQuant).

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "parlay/parallel.h"
#include "parlay/primitives.h"

#include "mvsic/core/types/chamfer_ip_point.h"
#include "mvsic/core/types/point_cloud_set.h"
#include "mvsic/core/utils/parse_command_line.h"
#include "mvsic/mvivf/mvivf.h"

#include "microbenchmark/mvivf/legacy_tq_mv.h"

using ChPoint = mvsic::ChamferIP_Point;
using PCS = mvsic::PointCloudSet<ChPoint>;
using IndexT = mvsic::IndexMVIVFIP;
using NodeT = IndexT::node_t;

namespace tq_mv = mvsic::turboquant_mv;
namespace lq_mv = mvsic::legacy_tq_mv;

constexpr bool kMetric = false;  // IP

using CurTQSet = tq_mv::Quantized_Point_Cloud_Set<kMetric>;
using CurTQQry = tq_mv::Quantized_Query_Point_Cloud<kMetric>;
using CurTQModel = tq_mv::Model<kMetric>;
using CurTQM2M = tq_mv::ManyToMany<CurTQSet>;

using LegTQSet = lq_mv::Quantized_Point_Cloud_Set<kMetric>;
using LegTQQry = lq_mv::Quantized_Query_Point_Cloud<kMetric>;
using LegTQModel = lq_mv::Model<kMetric>;

// =========================================================================
// Timing utilities
// =========================================================================
struct Timer {
  using clock = std::chrono::steady_clock;
  clock::time_point t0;
  void start() { t0 = clock::now(); }
  double sec() const { return std::chrono::duration<double>(clock::now() - t0).count(); }
};

static double median(std::vector<double> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

template<typename F>
static double bench(F&& fn, int reps) {
  std::vector<double> ts;
  ts.reserve(reps);
  // Warmup
  fn();
  for (int r = 0; r < reps; ++r) {
    Timer t;
    t.start();
    fn();
    ts.push_back(t.sec());
  }
  return median(std::move(ts));
}

// =========================================================================
// Tree walk: gather all leaves of the index
// =========================================================================
static void gather_leaves(NodeT* node, std::vector<NodeT*>& out) {
  if (node == nullptr) return;
  if (node->children.empty()) {
    out.push_back(node);
    return;
  }
  for (auto* c : node->children) gather_leaves(c, out);
}

// =========================================================================
// main
// =========================================================================
int main(int argc, char** argv) {
  mvsic::commandLine P(argc, argv,
                       "-i <db.pcs> -q <queries.pcs> "
                       "[-x <index.bin>] "
                       "[-num_leaves N] [-num_q_clouds Q] [-q_block B] [-reps R] [-seed S]");

  char* dbFile = P.getOptionValue("-i");
  char* qFile = P.getOptionValue("-q");
  if (dbFile == nullptr || qFile == nullptr) {
    std::cerr << "ERROR: require -i <db.pcs> and -q <queries.pcs>\n";
    return 1;
  }
  char* indexFile = P.getOptionValue("-x");

  const size_t num_leaves_to_use =
      static_cast<size_t>(P.getOptionLongValue("-num_leaves", 32));
  const size_t num_q_clouds = static_cast<size_t>(P.getOptionLongValue("-num_q_clouds", 64));
  const size_t q_block = static_cast<size_t>(P.getOptionLongValue("-q_block", 8));
  const int reps = P.getOptionIntValue("-reps", 5);
  const uint32_t seed = static_cast<uint32_t>(P.getOptionIntValue("-seed", 0));

  // -------------------------------------------------------------------------
  // Load data
  // -------------------------------------------------------------------------
  std::cout << "Loading DB:      " << dbFile << "\n";
  PCS db(dbFile, /*is_mmap=*/false);
  std::cout << "Loading Queries: " << qFile << "\n";
  PCS queries(qFile, /*is_mmap=*/false);

  if (db.size() == 0 || queries.size() == 0) {
    std::cerr << "ERROR: empty DB or queries.\n";
    return 1;
  }
  if (db.get_dims() != queries.get_dims()) {
    std::cerr << "ERROR: dim mismatch DB(" << db.get_dims() << ") vs Q(" << queries.get_dims()
              << ")\n";
    return 1;
  }

  std::cout << "DB     : clouds=" << db.size() << " dims=" << db.get_dims()
            << " avg_k=" << db.average_size() << "\n";
  std::cout << "Queries: clouds=" << queries.size() << " avg_k=" << queries.average_size() << "\n";

  // -------------------------------------------------------------------------
  // Build (or load) MVIVF index. Same params as out_arguana/mvivf/mvivf_500_tq:
  //   k_per_level=0, max_leaf_size=500, quantize_centers=false, pq=TurboQuant.
  // -------------------------------------------------------------------------
  mvsic::IndexParams params = mvsic::IndexParams::mvivf(
      /*k_per_level=*/0,
      /*max_leaf_size=*/500);
  params.pq.method = mvsic::IndexParams::QuantizerType::TurboQuant;
  params.quantize_centers = false;
  params.verbose = 1;

  IndexT index(db.get_dims(), params);
  if (indexFile != nullptr) {
    std::cout << "Loading index:   " << indexFile << "\n";
    std::string idx_path(indexFile);
    index.load(idx_path, db);
  } else {
    std::cout << "Building MVIVF index in-process...\n";
    Timer bt;
    bt.start();
    index.build(db);
    std::cout << "  build time: " << bt.sec() << " s\n";
  }

  // -------------------------------------------------------------------------
  // Gather leaves and pick a sample
  // -------------------------------------------------------------------------
  std::vector<NodeT*> leaves;
  gather_leaves(index.root, leaves);
  std::cout << "Total leaves in tree: " << leaves.size() << "\n";

  if (leaves.empty()) {
    std::cerr << "ERROR: no leaves\n";
    return 1;
  }

  std::mt19937 gen(seed);
  std::shuffle(leaves.begin(), leaves.end(), gen);
  const size_t use_leaves = std::min(num_leaves_to_use, leaves.size());
  leaves.resize(use_leaves);

  // Stats over chosen leaves
  size_t total_leaf_clouds = 0;
  size_t min_lc = SIZE_MAX, max_lc = 0;
  for (NodeT* L : leaves) {
    const size_t s = L->data.size();
    total_leaf_clouds += s;
    if (s < min_lc) min_lc = s;
    if (s > max_lc) max_lc = s;
  }
  std::cout << "Sampled leaves: " << use_leaves << "  total_clouds=" << total_leaf_clouds
            << "  per-leaf min=" << min_lc << " max=" << max_lc
            << " avg=" << static_cast<double>(total_leaf_clouds) / use_leaves << "\n";

  // -------------------------------------------------------------------------
  // Sample query clouds and pre-quantize them once with both TQ models.
  // The legacy and current TQ models share the same params (training is from
  // the dataset's points), so we use the index's TQ model where possible and
  // train a separate legacy model for the legacy path.
  // -------------------------------------------------------------------------
  std::uniform_int_distribution<size_t> q_dist(0, queries.size() - 1);
  std::vector<size_t> q_indices(num_q_clouds);
  for (size_t i = 0; i < num_q_clouds; ++i) q_indices[i] = q_dist(gen);

  // Materialize the float query clouds (so we can re-use them for all paths).
  std::vector<ChPoint> q_clouds_float;
  q_clouds_float.reserve(num_q_clouds);
  for (size_t i = 0; i < num_q_clouds; ++i) q_clouds_float.push_back(queries[q_indices[i]]);

  // Train a legacy-TQ model on the same point cloud set so the encoding has
  // a comparable rotation/codebook to the current TQ.
  std::cout << "Training legacy TQ model on DB...\n";
  LegTQModel leg_model;
  leg_model.train(db);

  // Quantize queries with the legacy model.
  std::vector<LegTQQry> leg_qs(num_q_clouds);
  parlay::parallel_for(0, num_q_clouds,
                       [&](size_t i) { leg_qs[i] = leg_model.quantize_query(q_clouds_float[i]); });

  // Quantize queries with the current TQ model. We grab the index's center
  // model only when quantize_centers=true; since we built with =false, we
  // train our own current-TQ model from the same DB instead.
  std::cout << "Training current TQ model on DB...\n";
  CurTQModel cur_model;
  cur_model.train(db);
  std::vector<CurTQQry> cur_qs(num_q_clouds);
  parlay::parallel_for(0, num_q_clouds,
                       [&](size_t i) { cur_qs[i] = cur_model.quantize_query(q_clouds_float[i]); });

  // -------------------------------------------------------------------------
  // Build per-leaf quantized payloads.
  //   - leg_leaf_set[L] : legacy strip-layout TQ encoding of leaf L's points
  //   - cur_leaf_set[L] : current panel-layout TQ encoding of leaf L's points
  // The index already has a leaf->quantized_data of (current) TQ_Set type
  // because we built with pq.method=TurboQuant; we re-encode anyway so the
  // legacy and current models are paired and use independently-trained
  // rotations/codebooks consistently.
  // -------------------------------------------------------------------------
  std::cout << "Encoding " << use_leaves << " leaves with both TQ formats...\n";
  std::vector<LegTQSet> leg_leaf_set(use_leaves);
  std::vector<CurTQSet> cur_leaf_set(use_leaves);
  parlay::parallel_for(0, use_leaves, [&](size_t L) {
    leg_leaf_set[L] = leg_model.encode(leaves[L]->data);
    cur_leaf_set[L] = cur_model.encode(leaves[L]->data);
  });

  // -------------------------------------------------------------------------
  // Result buffers (one per leaf, sized to that leaf's cloud count for
  // distances_all, or num_q_clouds*k for the m2m path).
  // -------------------------------------------------------------------------
  using Result = std::pair<uint32_t, float>;
  std::vector<std::vector<Result>> results_per_leaf(use_leaves);
  for (size_t L = 0; L < use_leaves; ++L) {
    results_per_leaf[L].resize(leaves[L]->data.size());
  }

  // For m2m: each query keeps top-k results across the leaf.
  const uint32_t k = 10;
  std::vector<std::vector<Result>> m2m_results_per_leaf(use_leaves);
  for (size_t L = 0; L < use_leaves; ++L) {
    m2m_results_per_leaf[L].resize(num_q_clouds * k);
  }

  // Pointers vector reused by m2m calls (independent of leaf).
  std::vector<const CurTQQry*> cur_q_ptrs(num_q_clouds);
  for (size_t i = 0; i < num_q_clouds; ++i) cur_q_ptrs[i] = &cur_qs[i];

  // -------------------------------------------------------------------------
  // Pre-fused query workspace for the new fused path. Built ONCE outside the
  // timed region so the per-query memcpy fusion overhead is amortized across
  // every (leaf, repetition) the workspace participates in -- exactly the
  // assumption the user wanted us to exploit.
  // -------------------------------------------------------------------------
  std::cout << "Building pre-fused query workspace once (parlay::num_workers="
            << parlay::num_workers() << ")...\n";
  tq_mv::FusedQueryBatch<kMetric> fused_qs;
  {
    Timer ft;
    ft.start();
    fused_qs.Build(cur_q_ptrs);
    std::cout << "  fused workspace build: " << ft.sec() << " s  ("
              << fused_qs.total_embeddings << " embeddings, q_stride=" << fused_qs.q_stride
              << ")\n";
  }
  // Reusable per-thread scratch buffers for the fused-multi inner loop.
  std::vector<std::vector<float>> fused_workspaces(parlay::num_workers());

  // Result buffer for the fused fast path: per leaf, num_q_clouds × leaf_size
  // raw distances (no top-k inside the kernel; the caller can sort externally).
  std::vector<std::vector<Result>> fast_results_per_leaf(use_leaves);
  for (size_t L = 0; L < use_leaves; ++L) {
    fast_results_per_leaf[L].resize(num_q_clouds * leaves[L]->data.size());
  }

  // Sanity check: stride consistency.
  if (num_q_clouds > 0) {
    for (size_t i = 1; i < num_q_clouds; ++i) {
      if (cur_qs[i].q_stride != cur_qs[0].q_stride) {
        std::cerr << "WARNING: cur_qs[" << i << "].q_stride=" << cur_qs[i].q_stride
                  << " != cur_qs[0].q_stride=" << cur_qs[0].q_stride << "\n";
      }
    }
  }

  // -------------------------------------------------------------------------
  // 1. Float baseline: outer parallel over leaves; inner sequential per query.
  //    We mirror what the float-Chamfer probe path would do: each (query,leaf)
  //    pair invokes leaf->data.distances(...).
  // -------------------------------------------------------------------------
  auto run_float = [&]() {
    parlay::parallel_for(
        0, use_leaves,
        [&](size_t L) {
          auto& leaf_pcs = leaves[L]->data;
          for (size_t qi = 0; qi < num_q_clouds; ++qi) {
            // Repeated calls overwrite results_per_leaf[L]; that's fine for
            // timing -- we are stress-testing the kernel, not collecting top-k.
            leaf_pcs.distances(q_clouds_float[qi], results_per_leaf[L].data());
          }
        },
        1);
  };

  // -------------------------------------------------------------------------
  // 2. Legacy TQ one-to-many: same call shape, but using legacy strip layout.
  // -------------------------------------------------------------------------
  auto run_legacy_one_to_many = [&]() {
    parlay::parallel_for(
        0, use_leaves,
        [&](size_t L) {
          for (size_t qi = 0; qi < num_q_clouds; ++qi) {
            leg_leaf_set[L].distances_all(leg_qs[qi], results_per_leaf[L].data());
          }
        },
        1);
  };

  // -------------------------------------------------------------------------
  // 3. Current TQ one-to-many: panel layout, called once per query.
  // -------------------------------------------------------------------------
  auto run_current_one_to_many = [&]() {
    parlay::parallel_for(
        0, use_leaves,
        [&](size_t L) {
          for (size_t qi = 0; qi < num_q_clouds; ++qi) {
            cur_leaf_set[L].distances_all(cur_qs[qi], results_per_leaf[L].data());
          }
        },
        1);
  };

  // -------------------------------------------------------------------------
  // 4. Current TQ many-to-many: a single fused call per leaf.
  // -------------------------------------------------------------------------
  auto run_current_many_to_many = [&]() {
    parlay::parallel_for(
        0, use_leaves,
        [&](size_t L) {
          CurTQM2M::TopKIntoUninitialized(
              cur_q_ptrs, cur_leaf_set[L], k, m2m_results_per_leaf[L].data(),
              /*q_block=*/q_block, /*parallel_query_blocks=*/false);
        },
        1);
  };

  // -------------------------------------------------------------------------
  // 5. Fused fast path: pre-fused queries + parallel-over-DB-clouds inner
  //    loop, no per-call memcpy fusion, no heap. Outer parallel_for over
  //    leaves composes with the inner DB-cloud parallel_for so the workload
  //    saturates all cores -- mirroring the nested parallelism that lets the
  //    o2m path scale (distances_all parallelizes over DB clouds internally).
  // -------------------------------------------------------------------------
  auto run_fast_fused_many_to_many = [&]() {
    parlay::parallel_for(
        0, use_leaves,
        [&](size_t L) {
          tq_mv::chamfer_score_all_fused<kMetric>(
              fused_qs, cur_leaf_set[L], fast_results_per_leaf[L].data(), &fused_workspaces,
              /*parallel_db=*/true);
        },
        1);
  };

  // -------------------------------------------------------------------------
  // Sanity check: the fused fast path must produce the same per (q, db)
  // distances as the o2m path on the same inputs (modulo float reordering).
  // We compare a single sampled leaf -- enough to catch indexing/aggregation
  // bugs without slowing the benchmark down.
  // -------------------------------------------------------------------------
  {
    const size_t L = 0;
    auto& leaf = cur_leaf_set[L];
    const size_t leaf_clouds = leaf.num_clouds();
    std::vector<Result> ref(leaf_clouds);
    std::vector<Result> fast(num_q_clouds * leaf_clouds);
    tq_mv::chamfer_score_all_fused<kMetric>(fused_qs, leaf, fast.data(), &fused_workspaces,
                                            /*parallel_db=*/true);
    double max_rel_err = 0.0;
    size_t mismatches = 0;
    for (size_t qi = 0; qi < num_q_clouds; ++qi) {
      leaf.distances_all(cur_qs[qi], ref.data());
      for (size_t c = 0; c < leaf_clouds; ++c) {
        const float a = ref[c].second;
        const float b = fast[qi * leaf_clouds + c].second;
        const float denom = std::max(1e-12f, std::max(std::abs(a), std::abs(b)));
        const double rel = std::abs(a - b) / denom;
        if (rel > max_rel_err) max_rel_err = rel;
        if (rel > 1e-4) ++mismatches;
      }
    }
    std::cout << "Sanity check (leaf 0, " << num_q_clouds << " queries × " << leaf_clouds
              << " db clouds): max_rel_err=" << max_rel_err << " mismatches(>1e-4)=" << mismatches
              << "\n";
  }

  // -------------------------------------------------------------------------
  // Run!
  // -------------------------------------------------------------------------
  std::cout << "\nBenchmark configuration:\n"
            << "  num_leaves       : " << use_leaves << "\n"
            << "  num_q_clouds     : " << num_q_clouds << "\n"
            << "  q_block          : " << q_block << "\n"
            << "  reps             : " << reps << "\n"
            << "  total work units : " << use_leaves * num_q_clouds
            << " (query x leaf pairs)\n\n";

  std::cout << "Running float baseline...\n" << std::flush;
  const double t_float = bench(run_float, reps);
  std::cout << "Running legacy TQ (one-to-many, strip layout)...\n" << std::flush;
  const double t_leg = bench(run_legacy_one_to_many, reps);
  std::cout << "Running current TQ (one-to-many, panel layout)...\n" << std::flush;
  const double t_cur = bench(run_current_one_to_many, reps);
  std::cout << "Running current TQ (many-to-many, fused panel kernel)...\n" << std::flush;
  const double t_m2m = bench(run_current_many_to_many, reps);
  std::cout << "Running fused fast path (pre-fused queries, parallel DB clouds)...\n" << std::flush;
  const double t_fast = bench(run_fast_fused_many_to_many, reps);

  // -------------------------------------------------------------------------
  // Report
  // -------------------------------------------------------------------------
  const size_t pairs = use_leaves * num_q_clouds;
  auto qps = [&](double t) { return pairs / t; };

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "\n=== Median timings (s) ===\n";
  std::cout << "                                       median(s)        pairs/s    speedup_vs_float "
               " speedup_vs_legacy_o2m\n";
  std::cout << "  (1) float baseline                 : " << std::setw(10) << t_float << "  "
            << std::setw(13) << qps(t_float) << "  " << std::setw(16) << "1.00x"
            << "  " << std::setw(20) << "-"
            << "\n";
  std::cout << "  (2) legacy TQ one-to-many          : " << std::setw(10) << t_leg << "  "
            << std::setw(13) << qps(t_leg) << "  " << std::setw(15) << (t_float / t_leg) << "x"
            << "  " << std::setw(19) << "1.00x"
            << "\n";
  std::cout << "  (3) current TQ one-to-many         : " << std::setw(10) << t_cur << "  "
            << std::setw(13) << qps(t_cur) << "  " << std::setw(15) << (t_float / t_cur) << "x"
            << "  " << std::setw(19) << (t_leg / t_cur) << "x\n";
  std::cout << "  (4) current TQ many-to-many fused  : " << std::setw(10) << t_m2m << "  "
            << std::setw(13) << qps(t_m2m) << "  " << std::setw(15) << (t_float / t_m2m) << "x"
            << "  " << std::setw(19) << (t_leg / t_m2m) << "x\n";
  std::cout << "  (5) fused fast path (parallel db)  : " << std::setw(10) << t_fast << "  "
            << std::setw(13) << qps(t_fast) << "  " << std::setw(15) << (t_float / t_fast) << "x"
            << "  " << std::setw(19) << (t_leg / t_fast) << "x\n";

  std::cout << "\n=== Speedups ===\n";
  std::cout << "  legacy_o2m -> current_o2m (storage rewrite alone) : "
            << (t_leg / t_cur) << "x\n";
  std::cout << "  current_o2m -> current_m2m (fusion alone)         : "
            << (t_cur / t_m2m) << "x\n";
  std::cout << "  legacy_o2m -> current_m2m (storage + fusion)      : "
            << (t_leg / t_m2m) << "x\n";
  std::cout << "  current_o2m -> fused_fast (the new fast path)     : "
            << (t_cur / t_fast) << "x\n";
  std::cout << "  legacy_o2m  -> fused_fast                          : "
            << (t_leg / t_fast) << "x\n";

  return 0;
}
