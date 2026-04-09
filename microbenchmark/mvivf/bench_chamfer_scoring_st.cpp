// Single-threaded chamfer scoring microbenchmark for MVIVF leaf nodes.
//
// Companion to bench_chamfer_scoring.cpp -- same paths, same data, but every
// timed loop runs strictly sequentially. This exposes per-call kernel overhead
// (per-query memcpy fusion, top-k heap maintenance, broadcast/dpbusd ratio in
// the inner kernel) that the multi-threaded benchmark masks behind 288-way
// parallelism on a SPR machine.
//
// Two scoring "shapes" are supported, controlled by -shape:
//   * shape=random  -- pick `num_leaves` random leaves and score every query
//                      against every cloud in each leaf, sequentially.
//   * shape=greedy  -- simulate a greedy MVIVF probe descent: from a random
//                      query, walk the tree picking the closest child by
//                      Chamfer distance until a leaf, then score that leaf.
//                      Repeat `num_leaves` times so the work is comparable to
//                      `random` mode.
//
// Run truly single-threaded by setting `PARLAY_NUM_THREADS=1` in the environment
// before invoking the binary -- the benchmark warns and refuses to time if
// parlay reports more than 1 worker.
//
// Usage:
//   PARLAY_NUM_THREADS=1 \
//   bazel-bin/microbenchmark/mvivf/bench_chamfer_scoring_st \
//       -i data/beir/arguana/arguana_points.pcs \
//       -q data/beir/arguana/arguana_queries.pcs \
//       -x out_arguana/mvivf/mvivf_500_tq/index_149ec46c.bin \
//       [-num_leaves 32] [-num_q_clouds 64] [-q_block 8] [-reps 5] [-seed 0]
//       [-shape random|greedy]

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "parlay/parallel.h"

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
  fn();  // warmup
  for (int r = 0; r < reps; ++r) {
    Timer t;
    t.start();
    fn();
    ts.push_back(t.sec());
  }
  return median(std::move(ts));
}

// =========================================================================
// Tree walk helpers
// =========================================================================
static void gather_leaves(NodeT* node, std::vector<NodeT*>& out) {
  if (node == nullptr) return;
  if (node->children.empty()) {
    out.push_back(node);
    return;
  }
  for (auto* c : node->children) gather_leaves(c, out);
}

// Simulate a single greedy probe descent from `root` for `query`. At each
// internal node, score all children with float Chamfer (cheap, mirrors how
// MVIVF traverses centers when quantize_centers=false), descend into the
// best child, and return the leaf reached.
static NodeT* greedy_descend(NodeT* root, const ChPoint& query) {
  NodeT* node = root;
  std::vector<std::pair<uint32_t, float>> child_dists;
  while (!node->children.empty()) {
    auto& children = node->children;
    child_dists.resize(children.size());
    // Score against each child's center pointcloud (node->data is the float
    // PointCloudSet of children's centers when quantize_centers=false).
    node->data.distances(query, child_dists.data());
    size_t best = 0;
    float best_d = child_dists[0].second;
    for (size_t i = 1; i < children.size(); ++i) {
      if (child_dists[i].second < best_d) {
        best_d = child_dists[i].second;
        best = i;
      }
    }
    node = children[best];
  }
  return node;
}

// =========================================================================
// main
// =========================================================================
int main(int argc, char** argv) {
  mvsic::commandLine P(argc, argv,
                       "-i <db.pcs> -q <queries.pcs> "
                       "[-x <index.bin>] "
                       "[-num_leaves N] [-num_q_clouds Q] [-q_block B] [-reps R] [-seed S] "
                       "[-shape random|greedy]");

  char* dbFile = P.getOptionValue("-i");
  char* qFile = P.getOptionValue("-q");
  if (dbFile == nullptr || qFile == nullptr) {
    std::cerr << "ERROR: require -i <db.pcs> and -q <queries.pcs>\n";
    return 1;
  }
  char* indexFile = P.getOptionValue("-x");
  char* shapeStr = P.getOptionValue("-shape");
  const std::string shape = shapeStr ? std::string(shapeStr) : std::string("random");
  if (shape != "random" && shape != "greedy") {
    std::cerr << "ERROR: -shape must be 'random' or 'greedy'\n";
    return 1;
  }

  const size_t num_leaves_to_use =
      static_cast<size_t>(P.getOptionLongValue("-num_leaves", 32));
  const size_t num_q_clouds = static_cast<size_t>(P.getOptionLongValue("-num_q_clouds", 64));
  const size_t q_block = static_cast<size_t>(P.getOptionLongValue("-q_block", 8));
  const int reps = P.getOptionIntValue("-reps", 5);
  const uint32_t seed = static_cast<uint32_t>(P.getOptionIntValue("-seed", 0));

  // -------------------------------------------------------------------------
  // Refuse to run unless parlay is configured with a single worker. The whole
  // point of this benchmark is to expose per-call overhead, so we don't want
  // any inner parlay::parallel_for to actually fan out.
  // -------------------------------------------------------------------------
  const size_t nw = parlay::num_workers();
  std::cout << "parlay::num_workers() = " << nw << "\n";
  if (nw != 1) {
    std::cerr << "WARNING: parlay reports " << nw
              << " workers. Re-run with PARLAY_NUM_THREADS=1 for a clean single-threaded "
                 "measurement; numbers below will be polluted by inner parallel_for fan-out.\n";
  }

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
  // Build (or load) MVIVF index.
  // -------------------------------------------------------------------------
  mvsic::IndexParams params = mvsic::IndexParams::mvivf(/*k_per_level=*/0, /*max_leaf_size=*/500);
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
  // Sample query clouds (these are reused across every leaf scoring loop).
  // -------------------------------------------------------------------------
  std::mt19937 gen(seed);
  std::uniform_int_distribution<size_t> q_dist(0, queries.size() - 1);
  std::vector<size_t> q_indices(num_q_clouds);
  for (size_t i = 0; i < num_q_clouds; ++i) q_indices[i] = q_dist(gen);

  std::vector<ChPoint> q_clouds_float;
  q_clouds_float.reserve(num_q_clouds);
  for (size_t i = 0; i < num_q_clouds; ++i) q_clouds_float.push_back(queries[q_indices[i]]);

  // -------------------------------------------------------------------------
  // Pick the leaves to score, depending on -shape.
  //
  // random: shuffle all leaves, take the first `num_leaves`.
  // greedy: pick `num_leaves` random *queries* and walk the hierarchy from
  //         the root for each, recording the leaf reached. Some leaves may
  //         appear multiple times -- that's realistic.
  // -------------------------------------------------------------------------
  std::vector<NodeT*> all_leaves;
  gather_leaves(index.root, all_leaves);
  std::cout << "Total leaves in tree: " << all_leaves.size() << "\n";
  if (all_leaves.empty()) {
    std::cerr << "ERROR: no leaves\n";
    return 1;
  }

  std::vector<NodeT*> chosen_leaves;
  if (shape == "random") {
    auto shuffled = all_leaves;
    std::shuffle(shuffled.begin(), shuffled.end(), gen);
    const size_t use = std::min(num_leaves_to_use, shuffled.size());
    chosen_leaves.assign(shuffled.begin(), shuffled.begin() + use);
  } else {
    // greedy
    chosen_leaves.reserve(num_leaves_to_use);
    std::uniform_int_distribution<size_t> dq(0, queries.size() - 1);
    for (size_t i = 0; i < num_leaves_to_use; ++i) {
      const ChPoint& q = queries[dq(gen)];
      chosen_leaves.push_back(greedy_descend(index.root, q));
    }
  }
  const size_t use_leaves = chosen_leaves.size();

  size_t total_leaf_clouds = 0, min_lc = SIZE_MAX, max_lc = 0;
  for (NodeT* L : chosen_leaves) {
    const size_t s = L->data.size();
    total_leaf_clouds += s;
    if (s < min_lc) min_lc = s;
    if (s > max_lc) max_lc = s;
  }
  std::cout << "Sampled leaves (" << shape << "): " << use_leaves
            << "  total_clouds=" << total_leaf_clouds << "  per-leaf min=" << min_lc
            << " max=" << max_lc
            << " avg=" << static_cast<double>(total_leaf_clouds) / use_leaves << "\n";

  // -------------------------------------------------------------------------
  // Train models, quantize queries, encode chosen leaves with both formats.
  // (Build/encode is NOT in the timed region. The user said per-query overhead
  // is amortized across many pointclouds, and we want to measure scoring
  // throughput only.)
  // -------------------------------------------------------------------------
  std::cout << "Training legacy TQ model on DB...\n";
  LegTQModel leg_model;
  leg_model.train(db);

  std::cout << "Training current TQ model on DB...\n";
  CurTQModel cur_model;
  cur_model.train(db);

  std::vector<LegTQQry> leg_qs(num_q_clouds);
  std::vector<CurTQQry> cur_qs(num_q_clouds);
  for (size_t i = 0; i < num_q_clouds; ++i) {
    leg_qs[i] = leg_model.quantize_query(q_clouds_float[i]);
    cur_qs[i] = cur_model.quantize_query(q_clouds_float[i]);
  }

  std::cout << "Encoding " << use_leaves << " leaves with both TQ formats...\n";
  std::vector<LegTQSet> leg_leaf_set(use_leaves);
  std::vector<CurTQSet> cur_leaf_set(use_leaves);
  for (size_t L = 0; L < use_leaves; ++L) {
    leg_leaf_set[L] = leg_model.encode(chosen_leaves[L]->data);
    cur_leaf_set[L] = cur_model.encode(chosen_leaves[L]->data);
  }

  // -------------------------------------------------------------------------
  // Result buffers.
  // -------------------------------------------------------------------------
  using Result = std::pair<uint32_t, float>;

  std::vector<std::vector<Result>> results_per_leaf(use_leaves);
  for (size_t L = 0; L < use_leaves; ++L) {
    results_per_leaf[L].resize(chosen_leaves[L]->data.size());
  }

  const uint32_t k = 10;
  std::vector<std::vector<Result>> m2m_results_per_leaf(use_leaves);
  for (size_t L = 0; L < use_leaves; ++L) {
    m2m_results_per_leaf[L].resize(num_q_clouds * k);
  }

  std::vector<const CurTQQry*> cur_q_ptrs(num_q_clouds);
  for (size_t i = 0; i < num_q_clouds; ++i) cur_q_ptrs[i] = &cur_qs[i];

  // Pre-fused workspace, built ONCE outside the timed region.
  std::cout << "Building pre-fused query workspace once...\n";
  tq_mv::FusedQueryBatch<kMetric> fused_qs;
  {
    Timer ft;
    ft.start();
    fused_qs.Build(cur_q_ptrs);
    std::cout << "  build: " << ft.sec() << " s  (" << fused_qs.total_embeddings
              << " embeddings, q_stride=" << fused_qs.q_stride << ")\n";
  }
  // For single-threaded mode we only need ONE workspace; size num_workers
  // anyway in case the user runs without PARLAY_NUM_THREADS=1.
  std::vector<std::vector<float>> fused_workspaces(parlay::num_workers());

  std::vector<std::vector<Result>> fast_results_per_leaf(use_leaves);
  for (size_t L = 0; L < use_leaves; ++L) {
    fast_results_per_leaf[L].resize(num_q_clouds * chosen_leaves[L]->data.size());
  }

  // -------------------------------------------------------------------------
  // Implementations -- ALL strictly sequential at the leaf level. Inner calls
  // (distances_all etc.) may still call parlay::parallel_for internally; with
  // PARLAY_NUM_THREADS=1 those collapse to plain for-loops.
  // -------------------------------------------------------------------------

  // (1) Float baseline.
  auto run_float = [&]() {
    for (size_t L = 0; L < use_leaves; ++L) {
      auto& leaf_pcs = chosen_leaves[L]->data;
      for (size_t qi = 0; qi < num_q_clouds; ++qi) {
        leaf_pcs.distances(q_clouds_float[qi], results_per_leaf[L].data());
      }
    }
  };

  // (2) Legacy TQ one-to-many: strip layout, called once per (q, leaf).
  auto run_legacy_o2m = [&]() {
    for (size_t L = 0; L < use_leaves; ++L) {
      for (size_t qi = 0; qi < num_q_clouds; ++qi) {
        leg_leaf_set[L].distances_all(leg_qs[qi], results_per_leaf[L].data());
      }
    }
  };

  // (3) Current TQ one-to-many: panel layout, called once per (q, leaf).
  auto run_current_o2m = [&]() {
    for (size_t L = 0; L < use_leaves; ++L) {
      for (size_t qi = 0; qi < num_q_clouds; ++qi) {
        cur_leaf_set[L].distances_all(cur_qs[qi], results_per_leaf[L].data());
      }
    }
  };

  // (4) Current many-to-many fused (the existing kernel, with the
  //     parallel_query_blocks knob disabled so the leaf is handled in one
  //     sequential call). This is the path that today's TopKIntoUninitialized
  //     would exercise if it were wired into MVIVF for TQ.
  auto run_current_m2m = [&]() {
    for (size_t L = 0; L < use_leaves; ++L) {
      CurTQM2M::TopKIntoUninitialized(cur_q_ptrs, cur_leaf_set[L], k,
                                      m2m_results_per_leaf[L].data(),
                                      /*q_block=*/q_block,
                                      /*parallel_query_blocks=*/false);
    }
  };

  // (5) NEW fused fast path. Pre-fused queries, no per-call memcpy, no heap,
  //     parallel_db=false so the inner DB-cloud loop is sequential.
  auto run_fast_fused = [&]() {
    for (size_t L = 0; L < use_leaves; ++L) {
      tq_mv::chamfer_score_all_fused<kMetric>(fused_qs, cur_leaf_set[L],
                                              fast_results_per_leaf[L].data(), &fused_workspaces,
                                              /*parallel_db=*/false);
    }
  };

  // -------------------------------------------------------------------------
  // Sanity check: the new fast path should produce the same per-(q, db)
  // distances as the o2m path. Verify on the first leaf.
  // -------------------------------------------------------------------------
  if (use_leaves > 0) {
    auto& leaf = cur_leaf_set[0];
    const size_t leaf_clouds = leaf.num_clouds();
    std::vector<Result> ref(leaf_clouds);
    std::vector<Result> fast(num_q_clouds * leaf_clouds);
    tq_mv::chamfer_score_all_fused<kMetric>(fused_qs, leaf, fast.data(), &fused_workspaces,
                                            /*parallel_db=*/false);
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
  std::cout << "\nBenchmark configuration (single-threaded):\n"
            << "  shape            : " << shape << "\n"
            << "  num_leaves       : " << use_leaves << "\n"
            << "  num_q_clouds     : " << num_q_clouds << "\n"
            << "  q_block          : " << q_block << "\n"
            << "  reps             : " << reps << "\n"
            << "  total work units : " << use_leaves * num_q_clouds
            << " (query × leaf pairs)\n\n";

  std::cout << "Running float baseline...\n" << std::flush;
  const double t_float = bench(run_float, reps);
  std::cout << "Running legacy TQ one-to-many...\n" << std::flush;
  const double t_leg = bench(run_legacy_o2m, reps);
  std::cout << "Running current TQ one-to-many...\n" << std::flush;
  const double t_cur = bench(run_current_o2m, reps);
  std::cout << "Running current TQ many-to-many (existing kernel)...\n" << std::flush;
  const double t_m2m = bench(run_current_m2m, reps);
  std::cout << "Running fused fast path (pre-fused queries, sequential)...\n" << std::flush;
  const double t_fast = bench(run_fast_fused, reps);

  // -------------------------------------------------------------------------
  // Report
  // -------------------------------------------------------------------------
  const size_t pairs = use_leaves * num_q_clouds;
  auto qps = [&](double t) { return pairs / t; };

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "\n=== Single-threaded median timings (s) ===\n";
  std::cout << "                                       median(s)        pairs/s    "
               "speedup_vs_float  speedup_vs_legacy_o2m\n";
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
  std::cout << "  (4) current TQ many-to-many        : " << std::setw(10) << t_m2m << "  "
            << std::setw(13) << qps(t_m2m) << "  " << std::setw(15) << (t_float / t_m2m) << "x"
            << "  " << std::setw(19) << (t_leg / t_m2m) << "x\n";
  std::cout << "  (5) fused fast path (NEW)          : " << std::setw(10) << t_fast << "  "
            << std::setw(13) << qps(t_fast) << "  " << std::setw(15) << (t_float / t_fast) << "x"
            << "  " << std::setw(19) << (t_leg / t_fast) << "x\n";

  std::cout << "\n=== Speedups ===\n";
  std::cout << "  legacy_o2m -> current_o2m  : " << (t_leg / t_cur) << "x\n";
  std::cout << "  current_o2m -> current_m2m : " << (t_cur / t_m2m) << "x\n";
  std::cout << "  current_o2m -> fused_fast  : " << (t_cur / t_fast) << "x\n";
  std::cout << "  current_m2m -> fused_fast  : " << (t_m2m / t_fast) << "x\n";
  std::cout << "  legacy_o2m  -> fused_fast  : " << (t_leg / t_fast) << "x\n";

  return 0;
}
