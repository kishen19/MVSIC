#pragma once

// =============================================================================
// Multi-Vector IVF (MVIVF) — templated index.
//
// IndexMVIVF<metric, CompressCenters, LeafModel> builds a kmeans tree over
// multi-vector point clouds and searches it with optional center-side and
// leaf-side quantization.
//
//   * `metric`           — false = IP (Chamfer max-IP), true = L2 (Chamfer L2)
//   * `CompressCenters`  — if true, internal-node centers and leaf centers are
//                          stored using the 4-bit TurboQuant multi-vector
//                          kernel (`turboquant_mv::Model`). Greedy / flat
//                          search then score through the quantized centers.
//   * `LeafModel`        — the concrete multi-vector quantizer `Model` class
//                          used to encode each leaf point cloud for fast
//                          probing.  `NoQuantizer<metric>` disables leaf
//                          encoding and falls back to raw PointCloudSet kernels
//                          for probe scoring.
//
// Concrete aliases are declared at the bottom of this file:
//
//   IndexMVIVFIP / IndexMVIVFL2               — raw centers, raw leaves
//   IndexMVIVFCompressIP / …L2                — TQ centers, raw leaves
//   IndexMVIVF{PQ,FastScan,RaBitQ,TQ,
//              SPQTQ,OneBitTQ}{IP,L2}          — raw centers, quantized leaves
//   IndexMVIVFCompress{PQ,…}{IP,L2}            — TQ centers + quantized leaves
//
// The kmeans tree topology (cluster assignments, children pointers, leaf point
// ids) is independent of quantization — the same skeleton can be reused across
// quantizers by re-training the leaf model on load.  See `save()` / `load()`.
// =============================================================================

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "mvsic/core/index.h"
#include "mvsic/core/mvclustering/mvclustering.h"
#include "mvsic/core/mvclustering/mvclustering_8bit.h"
#include "mvsic/core/query_compression.h"
#include "mvsic/core/utils/util.h"

namespace mvsic {

// Empty marker used with [[no_unique_address]] to collapse optional members to
// zero bytes when their template slot is unused.
namespace mvivf_internal {
struct Empty {};
}  // namespace mvivf_internal

template<bool metric, bool CompressCenters = false, class LeafModel = NoQuantizer<metric>>
class IndexMVIVF : public Index<metric> {
 public:
  using ChPoint = typename Index<metric>::ChPoint;
  using QT = typename Index<metric>::QT;
  using Index<metric>::d;
  using Index<metric>::params;
  using Index<metric>::quantization_mode;  // kept for bench_utils / stats compat

  // Center quantization is always 4-bit TurboQuant when enabled.
  using CenterModel = turboquant_mv::Model<metric>;
  using CenterSet = typename CenterModel::EncodedSet;
  using CenterQuery = typename CenterModel::EncodedQuery;

  using LeafSet = typename LeafModel::EncodedSet;
  using LeafQuery = typename LeafModel::EncodedQuery;
  using LeafParams = typename LeafModel::Params;

  static constexpr bool kHasLeafQuant = !std::is_same_v<LeafModel, NoQuantizer<metric>>;
  static constexpr bool kHasCenterQuant = CompressCenters;

  // ---------------------------------------------------------------------------
  // Tree node:
  //
  // During build we always populate `data` (raw centers for internal, raw
  // points for leaves).  At the end of build, if `CompressCenters` is set, we
  // encode internal-node `data` into `compressed_centers` and clear the raw
  // centers to save memory.  Leaf nodes always keep raw `data` (required for
  // rerank) and optionally also hold an encoded leaf.
  // ---------------------------------------------------------------------------
  struct node_t {
    parlay::sequence<node_t*> children;
    PointCloudSet<ChPoint> data;
    [[no_unique_address]]
    std::conditional_t<kHasCenterQuant, CenterSet, mvivf_internal::Empty> compressed_centers;
    [[no_unique_address]]
    std::conditional_t<kHasLeafQuant, LeafSet, mvivf_internal::Empty> encoded_leaf;
    node_t() noexcept = default;
    ~node_t() noexcept = default;
    inline size_t get_size() const noexcept { return data.size(); }
  };

  // Tree root + flat leaf shortcuts.
  node_t* root = nullptr;
  parlay::sequence<node_t*> leaves_flat;  // leaves in DFS order
  PointCloudSet<ChPoint> leaf_centers;    // one raw center per leaf (only when !CompressCenters)
  [[no_unique_address]]
  std::conditional_t<kHasCenterQuant, CenterSet, mvivf_internal::Empty> leaf_centers_compressed;

  // Trained models.  Kept as members so they persist across build / search /
  // save / load.  Empty marker for unused slots.
  [[no_unique_address]]
  std::conditional_t<kHasCenterQuant, CenterModel, mvivf_internal::Empty> center_model_;
  [[no_unique_address]]
  std::conditional_t<kHasLeafQuant, LeafModel, mvivf_internal::Empty> leaf_model_;
  [[no_unique_address]]
  std::conditional_t<kHasLeafQuant, LeafParams, mvivf_internal::Empty> leaf_params_;

  size_t kmeanstree_height = 0;
  std::vector<uint32_t> leaf_to_root_child_;

  // ---------------------------------------------------------------------------
  // Build-time 8BTQ cache. When `params.build_with_8btq` is set, we train the
  // 8BTQ rotator and pre-encode every input point cloud as a query *once* at
  // the top of `build()`. Each MVClustering8BTQ call in `recursive_build_`
  // then reads encoded queries directly out of `build_8btq_q_clouds_` via the
  // per-subtree `orig_indices` mapping, instead of re-training a rotator and
  // re-encoding its training points.
  //
  // The rotator is data-independent (depends only on dim + a fixed seed) so
  // sharing it across calls is correctness-equivalent. The encoded queries are
  // a per-PC byproduct of `model.quantize_query(points[i])`.
  //
  // Both members are populated only for the lifetime of `build()`; cleared
  // afterwards to avoid retaining ~6 N×padded_dim bytes for the lifetime of
  // the index.
  // ---------------------------------------------------------------------------
  using BTQModel = ::mvsic::turboquant_8bit_mv::Model<metric>;
  using BTQEncQuery = typename BTQModel::EncodedQuery;
  std::unique_ptr<BTQModel> build_8btq_model_;
  std::vector<BTQEncQuery> build_8btq_q_clouds_;

  // Flat per-individual-vector mirror of `build_8btq_q_clouds_` (all vectors
  // across all input PCs concatenated). Plumbed through MVClustering8BTQ to
  // TQ8LloydsBackend so the inner Lloyd's k-means can skip its rotate+
  // quantize step (gather-from-cache instead).  pc_vec_offsets[i] gives the
  // starting row of root PC i in the flat buffers.
  size_t build_8btq_vec_q_stride_ = 0;
  parlay::sequence<int8_t> build_8btq_vec_data_;
  parlay::sequence<float> build_8btq_vec_nsf_;
  parlay::sequence<float> build_8btq_vec_sqn_;
  parlay::sequence<int32_t> build_8btq_vec_bsum_;
  parlay::sequence<size_t> build_8btq_pc_vec_offsets_;

  // ---------------------------------------------------------------------------
  // Per-level build timing instrumentation. Compile-time gated by
  // MVIVF_BUILD_STATS (define to 1 to enable). Off by default so build()
  // pays no atomic-add or chrono cost on the hot path.
  //
  // Build with -DMVIVF_BUILD_STATS=1 to re-enable the per-depth `sum_cpu`
  // counters and the per-level wall window printed at -v >= 2.
  // ---------------------------------------------------------------------------
#ifndef MVIVF_BUILD_STATS
#define MVIVF_BUILD_STATS 0
#endif

#if MVIVF_BUILD_STATS
  static constexpr size_t kMaxBuildLevels = 16;
  mutable std::array<std::atomic<uint64_t>, kMaxBuildLevels> level_us_clustering_{};
  mutable std::array<std::atomic<uint64_t>, kMaxBuildLevels> level_us_split_{};
  mutable std::array<std::atomic<uint64_t>, kMaxBuildLevels> level_us_leaf_encode_{};
  mutable std::array<std::atomic<uint64_t>, kMaxBuildLevels> level_n_internal_{};
  mutable std::array<std::atomic<uint64_t>, kMaxBuildLevels> level_n_leaves_{};
  mutable std::array<std::atomic<uint64_t>, kMaxBuildLevels> level_total_points_{};
  // Wall-clock window covered by all recursive_build_ calls at a given depth:
  // first entry timestamp = min(t_in), last exit timestamp = max(t_out). The
  // wall time at level d is therefore (last - first), regardless of how many
  // parallel workers contributed. Stored as us-since-build-start.
  mutable std::array<std::atomic<uint64_t>, kMaxBuildLevels> level_first_in_us_{};
  mutable std::array<std::atomic<uint64_t>, kMaxBuildLevels> level_last_out_us_{};
  uint64_t build_t0_us_ = 0;

  static inline uint64_t now_us_() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
  }

  void reset_level_timings_() {
    build_t0_us_ = now_us_();
    for (size_t d = 0; d < kMaxBuildLevels; ++d) {
      level_us_clustering_[d].store(0, std::memory_order_relaxed);
      level_us_split_[d].store(0, std::memory_order_relaxed);
      level_us_leaf_encode_[d].store(0, std::memory_order_relaxed);
      level_n_internal_[d].store(0, std::memory_order_relaxed);
      level_n_leaves_[d].store(0, std::memory_order_relaxed);
      level_total_points_[d].store(0, std::memory_order_relaxed);
      level_first_in_us_[d].store(UINT64_MAX, std::memory_order_relaxed);
      level_last_out_us_[d].store(0, std::memory_order_relaxed);
    }
  }

  static inline void atomic_min_(std::atomic<uint64_t>& dst, uint64_t v) {
    uint64_t cur = dst.load(std::memory_order_relaxed);
    while (v < cur && !dst.compare_exchange_weak(cur, v, std::memory_order_relaxed)) {}
  }
  static inline void atomic_max_(std::atomic<uint64_t>& dst, uint64_t v) {
    uint64_t cur = dst.load(std::memory_order_relaxed);
    while (v > cur && !dst.compare_exchange_weak(cur, v, std::memory_order_relaxed)) {}
  }

  void print_level_timings_() const {
    auto fmt_s = [](uint64_t us) {
      std::ostringstream os;
      os << std::fixed << std::setprecision(3) << (static_cast<double>(us) * 1e-6);
      return os.str();
    };
    std::cout << "[MVIVF] Per-level build breakdown:\n";
    std::cout << "[MVIVF]   sum_cpu = sum across all workers; wall = (last_exit - first_entry) at level\n";
    std::cout << "[MVIVF]   "
              << "  d   internal     leaves     n_pts"
              << "  cpu_clus(s)  cpu_split(s)  cpu_leafenc(s)   wall(s)\n";
    for (size_t d = 0; d < kMaxBuildLevels; ++d) {
      uint64_t ni = level_n_internal_[d].load(std::memory_order_relaxed);
      uint64_t nl = level_n_leaves_[d].load(std::memory_order_relaxed);
      uint64_t np = level_total_points_[d].load(std::memory_order_relaxed);
      uint64_t uc = level_us_clustering_[d].load(std::memory_order_relaxed);
      uint64_t us = level_us_split_[d].load(std::memory_order_relaxed);
      uint64_t ue = level_us_leaf_encode_[d].load(std::memory_order_relaxed);
      uint64_t fi = level_first_in_us_[d].load(std::memory_order_relaxed);
      uint64_t lo = level_last_out_us_[d].load(std::memory_order_relaxed);
      uint64_t wall = (fi == UINT64_MAX || lo <= fi) ? 0 : (lo - fi);
      if (ni == 0 && nl == 0 && uc == 0 && us == 0 && ue == 0) continue;
      std::cout << "[MVIVF]   " << std::setw(3) << d
                << std::setw(11) << ni
                << std::setw(11) << nl
                << std::setw(10) << np
                << std::setw(13) << fmt_s(uc)
                << std::setw(14) << fmt_s(us)
                << std::setw(16) << fmt_s(ue)
                << std::setw(10) << fmt_s(wall) << "\n";
    }
    std::cout.flush();
  }
#endif  // MVIVF_BUILD_STATS

  // ---------------------------------------------------------------------------
  // Construction.
  // ---------------------------------------------------------------------------
  IndexMVIVF(size_t d_) noexcept {
    d = d_;
    params = IndexParams::mvivf();
    set_quant_mode_();
  }
  IndexMVIVF(size_t d_, const IndexParams& p) noexcept {
    d = d_;
    params = p;
    set_quant_mode_();
  }
  template<class LP = LeafParams,
           std::enable_if_t<kHasLeafQuant && std::is_same_v<LP, LeafParams>, int> = 0>
  IndexMVIVF(size_t d_, const IndexParams& p, const LP& lp) noexcept {
    d = d_;
    params = p;
    if constexpr (kHasLeafQuant) leaf_params_ = lp;
    set_quant_mode_();
  }

  ~IndexMVIVF() {
    if (root != nullptr) {
      traverse_and_delete_(root);
      delete root;
      root = nullptr;
    }
  }

  // ---------------------------------------------------------------------------
  // Back-compat: populate `Index<metric>::quantization_mode` so that
  // `bench_utils::print_compression_stats` and legacy stat reporting keep
  // working.  This field is no longer used for runtime dispatch in this class.
  // ---------------------------------------------------------------------------
  void set_quant_mode_() {
    if constexpr (!kHasLeafQuant) {
      quantization_mode = QT::None;
    } else {
      using L = LeafModel;
      if constexpr (std::is_same_v<L, pq_mv::Model<metric>>)
        quantization_mode = QT::PQ;
      else if constexpr (std::is_same_v<L, rabitq_mv::Model<metric>>)
        quantization_mode = QT::RaBitQ;
      else if constexpr (std::is_same_v<L, fastscan_mv::Model<metric>>)
        quantization_mode = QT::FastScan;
      else if constexpr (std::is_same_v<L, turboquant_mv::Model<metric>>)
        quantization_mode = QT::TurboQuant;
      else if constexpr (std::is_same_v<L, pqtq_mv::Model<metric>>)
        quantization_mode = QT::SPQTQ;
      else if constexpr (std::is_same_v<L, turboquant_1bit_mv::Model<metric>>)
        quantization_mode = QT::OneBitTQ;
      else if constexpr (std::is_same_v<L, turboquant_8bit_mv::Model<metric>>)
        quantization_mode = QT::EightBitTQ;
      else
        quantization_mode = QT::None;
    }
  }

  // ---------------------------------------------------------------------------
  // Build.
  // ---------------------------------------------------------------------------
  void build(const PointCloudSet<ChPoint>& points) override {
    parlay::internal::timer t;
    if (root != nullptr) {
      traverse_and_delete_(root);
      delete root;
      root = nullptr;
    }
    root = new node_t();
    // Propagate the build-time 8BTQ flag into MVClusteringConfig so the inner
    // Lloyd's k-means picks the TQ8 backend in kmeans_subsample/weighted.
    params.mvclus.build_with_8btq = params.build_with_8btq;
#if MVIVF_BUILD_STATS
    reset_level_timings_();
#endif

    t.start();
    if constexpr (kHasCenterQuant) center_model_.train(points);
    if constexpr (kHasLeafQuant) leaf_model_.train(points, leaf_params_);
    if (params.verbose >= 1 && (kHasCenterQuant || kHasLeafQuant)) {
      std::cout << "[MVIVF] Quantizers Trained: " << t.stop() << " sec" << std::endl;
    }
    t.reset();

    // Pre-encode every input PC as an 8BTQ query for reuse across recursion.
    // The rotator (dim-only) and the encoded queries are then handed to every
    // MVClustering8BTQ call in `recursive_build_` via points_to_root mappings.
    //
    // In the same pass we also concatenate every per-vector encoded state into
    // a flat (build_8btq_vec_*) buffer so the inner Lloyd's k-means inside
    // each MVClustering8BTQ centroid update can gather-from-cache instead of
    // re-rotating + re-quantizing its ~hundreds of vectors per call.
    if (params.build_with_8btq) {
      t.start();
      build_8btq_model_ = std::make_unique<BTQModel>();
      build_8btq_model_->train(points);
      const size_t n = points.size();
      build_8btq_q_clouds_.assign(n, BTQEncQuery{});
      parlay::parallel_for(0, n, [&](size_t i) {
        build_8btq_q_clouds_[i] = build_8btq_model_->quantize_query(points[i]);
      });

      // Build the per-vector flat cache from the per-PC EncodedQuery objects.
      // q_stride is constant across all per-PC EncQueries (a Model property),
      // so we can grab it from the first non-empty one (or compute from the
      // model directly).
      const size_t stride = (build_8btq_model_->encoder.padded_dim + 3) & ~3;
      build_8btq_vec_q_stride_ = stride;
      build_8btq_pc_vec_offsets_ = parlay::sequence<size_t>(n + 1, 0);
      // Sequential prefix sum -- n is small enough (<= a few M) that the
      // serial scan is dominated by the parallel encode step that follows.
      for (size_t i = 0; i < n; ++i) {
        build_8btq_pc_vec_offsets_[i + 1] =
            build_8btq_pc_vec_offsets_[i] + build_8btq_q_clouds_[i].num_queries;
      }
      const size_t total_vecs = build_8btq_pc_vec_offsets_[n];
      build_8btq_vec_data_.assign(total_vecs * stride, 0);
      build_8btq_vec_nsf_.assign(total_vecs, 0.0f);
      build_8btq_vec_sqn_.assign(total_vecs, 0.0f);
      build_8btq_vec_bsum_.assign(total_vecs, 0);
      parlay::parallel_for(0, n, [&](size_t i) {
        const auto& q = build_8btq_q_clouds_[i];
        if (q.num_queries == 0) return;
        const size_t off = build_8btq_pc_vec_offsets_[i];
        std::memcpy(build_8btq_vec_data_.data() + off * stride, q.flat_query_data.data(),
                    q.num_queries * stride * sizeof(int8_t));
        std::memcpy(build_8btq_vec_nsf_.data() + off, q.norm_scaling_factors.data(),
                    q.num_queries * sizeof(float));
        std::memcpy(build_8btq_vec_sqn_.data() + off, q.unquantized_squared_norms.data(),
                    q.num_queries * sizeof(float));
        std::memcpy(build_8btq_vec_bsum_.data() + off, q.byte_sums.data(),
                    q.num_queries * sizeof(int32_t));
      });

      if (params.verbose >= 1) {
        std::cout << "[MVIVF] 8BTQ pre-encode (" << n << " PCs, " << total_vecs
                  << " vecs): " << t.stop() << " sec" << std::endl;
      }
      t.reset();
    }

    t.start();
    parlay::sequence<uint32_t> root_orig_indices(points.size());
    parlay::parallel_for(0, points.size(),
                         [&](size_t i) { root_orig_indices[i] = static_cast<uint32_t>(i); });
    recursive_build_(root, points, 0, root_orig_indices);
    if (params.verbose >= 1) {
      std::cout << "[MVIVF] MV-Kmeans-Tree Built: " << t.stop() << " sec" << std::endl;
    }
    t.reset();

    t.start();
    compute_leaf_flat_();
    double t_compute_leaf_flat = t.stop();
    t.reset();
    t.start();
    compress_internal_centers_();
    double t_compress_centers = t.stop();
    if (params.verbose >= 1) {
      std::cout << "[MVIVF] Leaf-data computed: "
                << (t_compute_leaf_flat + t_compress_centers) << " sec ("
                << "compute_leaf_flat=" << t_compute_leaf_flat << "s, "
                << "compress_internal_centers=" << t_compress_centers << "s)" << std::endl;
    }
#if MVIVF_BUILD_STATS
    if (params.verbose >= 2) {
      print_level_timings_();
    }
#endif

    // Cache exists only for the lifetime of build(); release it now so the
    // index doesn't carry the per-PC encoded queries (~ N * padded_dim bytes)
    // nor the per-vector flat buffer (~ total_vecs * padded_dim bytes).
    build_8btq_model_.reset();
    build_8btq_q_clouds_.clear();
    build_8btq_q_clouds_.shrink_to_fit();
    build_8btq_vec_data_ = parlay::sequence<int8_t>{};
    build_8btq_vec_nsf_ = parlay::sequence<float>{};
    build_8btq_vec_sqn_ = parlay::sequence<float>{};
    build_8btq_vec_bsum_ = parlay::sequence<int32_t>{};
    build_8btq_pc_vec_offsets_ = parlay::sequence<size_t>{};
    build_8btq_vec_q_stride_ = 0;
  }

 private:
  // ---------------------------------------------------------------------------
  // Recursive kmeans tree builder.  Topology only — quantization happens after
  // the tree is fully built (see `compress_internal_centers_`).  Leaves, however,
  // are encoded eagerly so we don't have to walk the tree a second time just
  // for leaf encoding.
  // ---------------------------------------------------------------------------
  void recursive_build_(node_t* node, const PointCloudSet<ChPoint>& points, uint32_t depth,
                        const parlay::sequence<uint32_t>& orig_indices) {
    const size_t n = points.size();
    const size_t auto_nc = (params.k_per_level > 0)
                               ? params.k_per_level
                               : static_cast<size_t>(std::ceil(std::sqrt(static_cast<double>(n))));
    const size_t small_nc =
        4 * (n + params.max_leaf_size - 1) / std::max<uint32_t>(params.max_leaf_size, 1);
    const size_t num_clusters = std::max(static_cast<size_t>(4), std::min({auto_nc, small_nc, n}));
    if (params.verbose >= 1) {
      std::cout << "[MVIVF] Building with " << n << " points, num_clusters: " << num_clusters
                << std::endl;
    }
#if MVIVF_BUILD_STATS
    const size_t bucket = std::min<size_t>(depth, kMaxBuildLevels - 1);
    const uint64_t t_entry = now_us_();
    atomic_min_(level_first_in_us_[bucket], t_entry - build_t0_us_);
    level_n_internal_[bucket].fetch_add(1, std::memory_order_relaxed);
    level_total_points_[bucket].fetch_add(n, std::memory_order_relaxed);
#endif

    PointCloudSet<ChPoint> centers;
    parlay::sequence<uint32_t> cluster_ids;
    auto run_clus = [&](auto& Clus) {
      Clus.train(points);
      cluster_ids = Clus.get_clustering(points);
      centers = std::move(Clus.get_centers());
    };
#if MVIVF_BUILD_STATS
    const uint64_t t_clus_start = now_us_();
#endif
    if (params.build_with_8btq) {
      MVClustering8BTQ<metric> Clus(d, num_clusters, params.s, params.mvclus);
      // Reuse the build-top-level rotator and pre-encoded queries instead of
      // training/encoding per call. orig_indices is the local-to-root mapping
      // for this subtree's points.
      if (build_8btq_model_ && !build_8btq_q_clouds_.empty()) {
        Clus.set_shared_cache(*build_8btq_model_, build_8btq_q_clouds_, orig_indices);
        if (build_8btq_vec_q_stride_ > 0 && !build_8btq_vec_data_.empty()) {
          ::mvsic::lloyds::TQ8VectorCache<metric> vc;
          vc.model = build_8btq_model_.get();
          vc.q_stride = build_8btq_vec_q_stride_;
          vc.q_data = build_8btq_vec_data_.data();
          vc.q_nsf = build_8btq_vec_nsf_.data();
          vc.q_sqn = build_8btq_vec_sqn_.data();
          vc.q_bsum = build_8btq_vec_bsum_.data();
          Clus.set_shared_vector_cache(vc, build_8btq_pc_vec_offsets_);
        }
      }
      run_clus(Clus);
    } else {
      MVClustering<metric> Clus(d, num_clusters, params.s, params.mvclus);
      run_clus(Clus);
    }
#if MVIVF_BUILD_STATS
    level_us_clustering_[bucket].fetch_add(now_us_() - t_clus_start, std::memory_order_relaxed);
    const uint64_t t_split_start = now_us_();
#endif
    auto id_pt = parlay::tabulate(n, [&](uint32_t i) { return std::make_pair(cluster_ids[i], i); });
    auto grouped = group_by_key_inplace(id_pt);
    node->children.resize(grouped.size());
    if (grouped.size() < centers.size()) {
      auto active = parlay::delayed_seq<uint32_t>(grouped.size(),
                                                  [&](size_t i) { return grouped[i][0].first; });
      node->data = PointCloudSet<ChPoint>(centers.filter(active), d);
    } else {
      node->data = std::move(centers);
    }
#if MVIVF_BUILD_STATS
    level_us_split_[bucket].fetch_add(now_us_() - t_split_start, std::memory_order_relaxed);
    const size_t child_bucket = std::min<size_t>(depth + 1, kMaxBuildLevels - 1);
#endif

    parlay::parallel_for(
        0, grouped.size(),
        [&](size_t i) {
#if MVIVF_BUILD_STATS
          const uint64_t t_child_split_start = now_us_();
#endif
          auto group = parlay::delayed_seq<uint32_t>(
              grouped[i].size(), [&](size_t j) { return grouped[i][j].second; });
          PointCloudSet<ChPoint> child_points(points.filter(group), d);
          // Map this child's local indices back to root indices so that any
          // recursive MVClustering8BTQ call can pull encoded queries out of
          // the build-top-level cache.
          parlay::sequence<uint32_t> child_orig_indices(grouped[i].size());
          for (size_t j = 0; j < grouped[i].size(); ++j) {
            child_orig_indices[j] = orig_indices[grouped[i][j].second];
          }
          node_t* child = new node_t();
          node->children[i] = child;
          const bool depth_exhausted = (params.max_depth > 0) && (depth + 1 >= params.max_depth);
          const bool will_be_leaf =
              depth_exhausted || child_points.size() <= params.max_leaf_size;
#if MVIVF_BUILD_STATS
          level_us_split_[bucket].fetch_add(now_us_() - t_child_split_start,
                                            std::memory_order_relaxed);
#endif
          if (!will_be_leaf) {
            recursive_build_(child, child_points, depth + 1, child_orig_indices);
          } else {
#if MVIVF_BUILD_STATS
            const uint64_t t_leaf_in = now_us_();
            atomic_min_(level_first_in_us_[child_bucket], t_leaf_in - build_t0_us_);
            level_n_leaves_[child_bucket].fetch_add(1, std::memory_order_relaxed);
#endif
            child->data = std::move(child_points);
            if constexpr (kHasLeafQuant) {
#if MVIVF_BUILD_STATS
              const uint64_t t_enc_start = now_us_();
#endif
              child->encoded_leaf = leaf_model_.encode(child->data);
#if MVIVF_BUILD_STATS
              level_us_leaf_encode_[child_bucket].fetch_add(now_us_() - t_enc_start,
                                                            std::memory_order_relaxed);
#endif
            }
#if MVIVF_BUILD_STATS
            atomic_max_(level_last_out_us_[child_bucket], now_us_() - build_t0_us_);
#endif
          }
        },
        1);
#if MVIVF_BUILD_STATS
    atomic_max_(level_last_out_us_[bucket], now_us_() - build_t0_us_);
#endif
  }

  // Extract a flat list of leaves plus one representative raw center per leaf.
  // Populates `leaves_flat`, `leaf_centers` (when !CompressCenters) or
  // `leaf_centers_compressed` (when CompressCenters), and `leaf_to_root_child_`.
  void compute_leaf_flat_() {
    leaves_flat.clear();
    leaf_to_root_child_.clear();
    if (!root) return;
    std::vector<ChPoint> center_points;
    std::vector<uint32_t> root_child_for_leaf;
    std::function<void(node_t*, node_t*, size_t, uint32_t)> visit =
        [&](node_t* node, node_t* parent, size_t child_idx, uint32_t root_child_idx) {
          if (node->children.empty()) {
            leaves_flat.push_back(node);
            auto& centers_pc = parent->data;
            center_points.push_back(centers_pc[child_idx]);
            root_child_for_leaf.push_back(root_child_idx);
          } else {
            for (size_t i = 0; i < node->children.size(); ++i) {
              visit(node->children[i], node, i,
                    (parent == root) ? static_cast<uint32_t>(i) : root_child_idx);
            }
          }
        };
    visit(root, nullptr, 0, 0);
    leaf_to_root_child_ = std::move(root_child_for_leaf);
    if (center_points.empty()) return;
    PointCloudSet<ChPoint> raw_centers(center_points, d);
    if constexpr (kHasCenterQuant) {
      leaf_centers_compressed = center_model_.encode(raw_centers);
    } else {
      leaf_centers = std::move(raw_centers);
    }
  }

  // Encode all internal-node centers with the center model.  Keeps the raw
  // `node->data` floats alongside the encoded form so that save() has the
  // exact float centers used during build.  No-op when !CompressCenters.
  void compress_internal_centers_() {
    if constexpr (kHasCenterQuant) {
      if (!root) return;
      std::function<void(node_t*)> visit = [&](node_t* n) {
        if (!n) return;
        if (n->children.empty()) return;  // leaves keep raw data
        if (n->data.size() > 0) {
          n->compressed_centers = center_model_.encode(n->data);
        }
        for (node_t* c : n->children)
          visit(c);
      };
      visit(root);
    }
  }

  void traverse_and_delete_(node_t* node) {
    for (node_t* c : node->children) {
      traverse_and_delete_(c);
      delete c;
    }
  }

 public:
  // ---------------------------------------------------------------------------
  // Greedy beam search over internal nodes.
  // ---------------------------------------------------------------------------
  struct GreedySearchResult {
    parlay::sequence<std::pair<float, node_t*>> probe_list;
    size_t bytes_accessed = 0;
    std::vector<double> stats = {};
  };

  GreedySearchResult greedy_search(const ChPoint& query, const CenterQuery& q_center,
                                   size_t nprobes) const {
    using score_node = std::pair<float, node_t*>;
    auto less = [](const score_node& a, const score_node& b) {
      return a.first < b.first || (a.first == b.first && a.second < b.second);
    };
    (void)less;
    const size_t beam_length = 2 * nprobes;
    parlay::internal::timer t;
    double t_dists = 0.0, t_beam = 0.0, t_rest = 0.0;

    t.start();
    size_t dist_cmps = 0;
    size_t bytes_accessed = 0;
    std::set<score_node> beam;
    parlay::sequence<score_node> top_probes;
    top_probes.reserve(nprobes + 1);
    std::vector<std::pair<uint32_t, float>> child_dists;
    child_dists.reserve(root->children.size());
    t_rest += t.stop();
    t.reset();

    t.start();
    beam.insert({0.0f, root});
    t_beam += t.stop();
    t.reset();

    while (!beam.empty()) {
      t.start();
      auto it = beam.begin();
      score_node best = *it;
      beam.erase(it);
      t_beam += t.stop();
      t.reset();

      t.start();
      node_t* current = best.second;
      if (top_probes.size() == nprobes && best.first >= top_probes.front().first) {
        break;
      }
      auto& children = current->children;
      child_dists.resize(children.size());
      if constexpr (kHasCenterQuant) {
        current->compressed_centers.distances_all(q_center, child_dists.data());
        bytes_accessed += current->compressed_centers.num_bytes();
      } else {
        auto& centers = current->data;
        centers.distances(query, child_dists.data());
        bytes_accessed += centers.num_bytes();
      }
      dist_cmps += children.size();
      t_dists += t.stop();
      t.reset();

      for (size_t i = 0; i < children.size(); ++i) {
        float cd = child_dists[i].second;
        node_t* child = children[i];
        if (child->children.empty()) {
          t.start();
          if (top_probes.size() < nprobes || cd < top_probes.front().first) {
            top_probes.push_back({cd, child});
            std::push_heap(top_probes.begin(), top_probes.end());
            if (top_probes.size() > nprobes) {
              std::pop_heap(top_probes.begin(), top_probes.end());
              top_probes.pop_back();
            }
          }
          t_rest += t.stop();
          t.reset();
        } else {
          t.start();
          const size_t bs = beam.size();
          if (bs < beam_length)
            beam.insert({cd, child});
          else {
            auto worst_it = std::prev(beam.end());
            if (cd < worst_it->first) {
              beam.erase(worst_it);
              beam.insert({cd, child});
            }
          }
          t_beam += t.stop();
          t.reset();
        }
      }
    }
    GreedySearchResult out;
    out.bytes_accessed = bytes_accessed;
    out.stats = {static_cast<double>(dist_cmps), t_dists, t_beam, t_rest};
    out.probe_list = std::move(top_probes);
    return out;
  }

  GreedySearchResult flat_leaf_search(const ChPoint& query, const CenterQuery& q_center,
                                      size_t nprobes) const {
    using score_node = std::pair<float, node_t*>;
    GreedySearchResult out;
    const size_t L = leaves_flat.size();
    if (L == 0 || nprobes == 0) return out;
    // When !CompressCenters, we need raw leaf centers.
    if constexpr (!kHasCenterQuant) {
      if (leaf_centers.size() == 0) return out;
    }
    const size_t use_nprobes = std::min(nprobes, L);

    parlay::internal::timer t;
    double t_dists = 0.0, t_rest = 0.0;
    t.start();
    size_t dist_cmps = L;
    size_t bytes_accessed = 0;
    auto centers_dists = parlay::sequence<std::pair<uint32_t, float>>::uninitialized(L);
    auto scores = parlay::sequence<score_node>::uninitialized(L);

    if constexpr (kHasCenterQuant) {
      leaf_centers_compressed.distances_all(q_center, centers_dists.data());
      bytes_accessed += leaf_centers_compressed.num_bytes();
    } else {
      leaf_centers.distances(query, centers_dists.data());
      bytes_accessed += leaf_centers.num_bytes();
    }
    parlay::parallel_for(0, L,
                         [&](size_t i) { scores[i] = {centers_dists[i].second, leaves_flat[i]}; });
    t_dists += t.stop();
    t.reset();

    t.start();
    if (use_nprobes < L) {
      std::nth_element(scores.begin(), scores.begin() + use_nprobes, scores.end(),
                       [](const score_node& a, const score_node& b) { return a.first < b.first; });
      scores.resize(use_nprobes);
    }
    t_rest += t.stop();
    t.reset();

    out.probe_list = std::move(scores);
    out.bytes_accessed = bytes_accessed;
    out.stats = {static_cast<double>(dist_cmps), t_dists, /*t_beam=*/0.0, t_rest};
    return out;
  }

  // ---------------------------------------------------------------------------
  // Per-query leaf probing (Step 2 of search).
  // ---------------------------------------------------------------------------
  inline std::pair<parlay::sequence<std::pair<uint32_t, float>>, size_t> process_probes(
      const ChPoint& query, const LeafQuery& q_leaf,
      parlay::sequence<std::pair<float, node_t*>>& probe_list) const {
    const size_t nprobes = probe_list.size();
    auto sizes = parlay::delayed_tabulate(
        nprobes, [&](size_t i) { return probe_list[i].second->get_size(); });
    auto scan_result = parlay::scan(sizes);
    auto& offsets = scan_result.first;
    size_t& total_size = scan_result.second;
    auto visited = parlay::sequence<std::pair<uint32_t, float>>::uninitialized(total_size);
    auto bytes_accessed = parlay::sequence<size_t>::uninitialized(nprobes);

    parlay::parallel_for(0, nprobes, [&](size_t i) {
      node_t* leaf = probe_list[i].second;
      if constexpr (kHasLeafQuant) {
        leaf->encoded_leaf.distances_all(q_leaf, &visited[offsets[i]]);
        bytes_accessed[i] = leaf->encoded_leaf.num_bytes();
      } else {
        (void)q_leaf;
        leaf->data.distances(query, &visited[offsets[i]]);
        bytes_accessed[i] = leaf->data.num_bytes();
      }
    });
    return std::make_pair(std::move(visited), parlay::reduce(bytes_accessed));
  }

  // ---------------------------------------------------------------------------
  // Single-query search with detailed stats.
  //
  // Timer labels (stats[0..]):
  //   0  search_cmps
  //   1  probe_cmps
  //   2  t_search_dists
  //   3  t_search_beam
  //   4  t_search_rest
  //   5  t_compress
  //   6  t_quantize
  //   7  t_distances
  //   8  t_rest
  //   9  t_rerank
  //  10  t_greedy
  // ---------------------------------------------------------------------------
  std::tuple<parlay::sequence<std::pair<uint32_t, float>>, size_t, std::vector<double>>
  search_with_stats(const ChPoint& query, const PointCloudSet<ChPoint>& points,
                    const SearchParams& search_params) override {
    parlay::internal::timer t;
    const size_t k = search_params.k;
    size_t nprobes = search_params.nprobes;
    size_t bytes_accessed = 0;
    size_t dist_cmps = 0;
    double t_compress = 0.0, t_quantize = 0.0, t_distances = 0.0;
    double t_rest = 0.0, t_rerank = 0.0;

    // Step -1: optional query compression.
    t.start();
    CompressedPointCloud<ChPoint> compressed_storage;
    ChPoint effective_query = query;
    if (search_params.query_compression != SearchParams::QueryCompression::None) {
      uint32_t ba = 1;
      if constexpr (kHasLeafQuant) ba = LeafModel::kBatchAlignment;
      compressed_storage = compress_query<ChPoint>(query, search_params.query_compression,
                                                   search_params.query_compression_threshold, ba);
      effective_query = compressed_storage.view();
    }
    t_compress = t.stop();
    t.reset();

    // Step 0: quantize query for leaf & center models.
    t.start();
    LeafQuery q_leaf{};
    CenterQuery q_center{};
    if constexpr (kHasLeafQuant) q_leaf = leaf_model_.quantize_query(effective_query);
    if constexpr (kHasCenterQuant) q_center = center_model_.quantize_query(effective_query);
    t_quantize = t.stop();
    t.reset();

    // Step 1: greedy / flat search for probe list.
    t.start();
    const size_t num_leaves = leaves_flat.size();
    const double alpha = 1.0;
    bool use_flat = (num_leaves > 0 && nprobes >= static_cast<size_t>(alpha * num_leaves));
    GreedySearchResult gs = use_flat ? flat_leaf_search(effective_query, q_center, nprobes)
                                     : greedy_search(effective_query, q_center, nprobes);
    double t_greedy = t.stop();
    t.reset();
    auto& probe_list = gs.probe_list;
    bytes_accessed += gs.bytes_accessed;
    nprobes = std::min(nprobes, probe_list.size());

    // Step 2: probe leaves.
    t.start();
    parlay::sequence<std::pair<uint32_t, float>> visited;
    size_t bytes_pp;
    std::tie(visited, bytes_pp) = process_probes(effective_query, q_leaf, probe_list);
    bytes_accessed += bytes_pp;
    dist_cmps += visited.size();
    t_distances = t.stop();
    t.reset();

    t.start();
    mvsic::sort_inplace_kv(visited);
    t_rest += t.stop();
    t.reset();

    // Step 3: rerank.
    const ChPoint& rerank_query = search_params.compress_rerank ? effective_query : query;
    t.start();
    auto final_results =
        parlay::sequence<std::pair<uint32_t, float>>::uninitialized(std::min(k, visited.size()));
    if (search_params.num_rerank > 0) {
      size_t num_rerank = std::min(search_params.num_rerank, visited.size());
      bytes_accessed += this->rerank(rerank_query, points, visited, num_rerank, final_results);
    } else {
      parlay::parallel_for(0, final_results.size(),
                           [&](size_t i) { final_results[i] = visited[i]; });
    }
    t_rerank = t.stop();
    t.reset();

    std::vector<double> stats;
    stats.push_back(gs.stats.empty() ? 0.0 : gs.stats[0]);  // search_cmps
    stats.push_back(static_cast<double>(dist_cmps));        // probe_cmps
    for (size_t i = 1; i < gs.stats.size(); ++i)
      stats.push_back(gs.stats[i]);
    stats.push_back(t_compress);
    stats.push_back(t_quantize);
    stats.push_back(t_distances);
    stats.push_back(t_rest);
    stats.push_back(t_rerank);
    stats.push_back(t_greedy);
    return std::make_tuple(std::move(final_results), bytes_accessed, std::move(stats));
  }

  // ---------------------------------------------------------------------------
  // Many-query cache-efficient search (search_all_new).
  //
  // For LeafModel types that provide a `ManyToMany::TopKIntoUninitialized`
  // entry point (FastScan, SPQTQ, 1-bit TQ), we batch queries grouped by leaf
  // and run the batched kernel per leaf.  Other quantizers fall through to
  // parallel 1-to-N scoring.
  // ---------------------------------------------------------------------------
 private:
  template<class M>
  struct has_many_to_many_ : std::false_type {};
  template<>
  struct has_many_to_many_<fastscan_mv::Model<metric>> : std::true_type {
    using type = fastscan_mv::ManyToMany<LeafSet>;
  };
  template<>
  struct has_many_to_many_<pqtq_mv::Model<metric>> : std::true_type {
    using type = pqtq_mv::ManyToMany<LeafSet>;
  };
  template<>
  struct has_many_to_many_<turboquant_1bit_mv::Model<metric>> : std::true_type {
    using type = turboquant_1bit_mv::ManyToMany<LeafSet>;
  };
  static constexpr bool kUseM2M = has_many_to_many_<LeafModel>::value;

 public:
  std::pair<parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>>, size_t> search_all_new(
      const PointCloudSet<ChPoint>& query_points, const PointCloudSet<ChPoint>& points,
      const SearchParams& search_params) {
    if (search_params.nprobes <= 16) {
      return Index<metric>::search_all(query_points, points, search_params);
    }
    parlay::internal::timer t;
    const size_t num_q = query_points.size();
    const size_t num_leaves = leaves_flat.size();
    const size_t k = search_params.k;
    size_t nprobes = std::min(num_leaves, search_params.nprobes);
    size_t bytes_accessed = 0;
    size_t dist_cmps = 0;

    // Step -1: optional query compression.
    PointCloudSet<ChPoint> compressed_storage;
    const PointCloudSet<ChPoint>* eff_ptr = &query_points;
    if (search_params.query_compression != SearchParams::QueryCompression::None) {
      t.start();
      uint32_t ba = 1;
      if constexpr (kHasLeafQuant) ba = LeafModel::kBatchAlignment;
      compressed_storage =
          compress_point_cloud_set<ChPoint>(query_points, search_params.query_compression,
                                            search_params.query_compression_threshold, ba);
      eff_ptr = &compressed_storage;
      t.stop();
      std::cout << "[MVIVF] Query Compression: " << t.total_time() << " sec" << std::endl;
      t.reset();
    }
    const auto& eff_queries = *eff_ptr;
    const auto& rerank_queries = search_params.compress_rerank ? eff_queries : query_points;

    // Step 0: pre-quantize queries.
    t.start();
    parlay::sequence<LeafQuery> q_leaves(num_q);
    if constexpr (kHasLeafQuant) {
      parlay::parallel_for(
          0, num_q, [&](size_t i) { q_leaves[i] = leaf_model_.quantize_query(eff_queries[i]); });
    }
    t.stop();
    std::cout << "[MVIVF] Query Quantization: " << t.total_time() << " sec" << std::endl;
    t.reset();

    // Step 1: parallel greedy search.
    t.start();
    auto leaf_query_pairs =
        parlay::sequence<std::pair<node_t*, std::pair<uint32_t, uint32_t>>>::uninitialized(nprobes *
                                                                                           num_q);
    auto dist_cmps_gs = parlay::sequence<size_t>::uninitialized(num_q);
    auto bytes_gs = parlay::sequence<size_t>::uninitialized(num_q);
    parlay::parallel_for(0, num_q, [&](uint32_t i) {
      CenterQuery q_center{};
      if constexpr (kHasCenterQuant) q_center = center_model_.quantize_query(eff_queries[i]);
      const double alpha = 1.0;
      bool use_flat = (num_leaves > 0 && nprobes >= static_cast<size_t>(alpha * num_leaves));
      GreedySearchResult gs = use_flat ? flat_leaf_search(eff_queries[i], q_center, nprobes)
                                       : greedy_search(eff_queries[i], q_center, nprobes);
      dist_cmps_gs[i] = static_cast<size_t>(gs.stats[0]);
      bytes_gs[i] = gs.bytes_accessed;
      parlay::parallel_for(0, gs.probe_list.size(), [&](uint32_t j) {
        leaf_query_pairs[i * nprobes + j] = {gs.probe_list[j].second, std::make_pair(i, j)};
      });
    });
    t.stop();
    std::cout << "[MVIVF] Greedy Search: " << t.total_time() << " sec" << std::endl;
    t.reset();
    bytes_accessed += parlay::reduce(bytes_gs);
    dist_cmps += parlay::reduce(dist_cmps_gs);

    // ---------------------------------------------------------------------
    // Sub-batching path (OBTQ-only). Reorganizes Step 2 + Step 3 around
    // query-major sub-batches: at any moment, each core only touches queries
    // from a single sub-batch of size B, so the broadcasted query working
    // set per core stays bounded by B * qbuf_size.
    // ---------------------------------------------------------------------
    if constexpr (std::is_same_v<LeafModel, turboquant_1bit_mv::Model<metric>>) {
      const char* sb_env = std::getenv("MVIVF_SUBBATCH");
      const size_t kSubbatchB = sb_env ? static_cast<size_t>(std::atoi(sb_env)) : 0;
      if (kSubbatchB > 0 && kSubbatchB <= 64) {
        using SetType = LeafSet;
        using QueryType = LeafQuery;
        using M2MType = typename has_many_to_many_<LeafModel>::type;

        // ----- Step 2': build per-sub-batch leaf-want lists -----
        t.start();
        const size_t B = kSubbatchB;
        const size_t num_subbatches = (num_q + B - 1) / B;
        const size_t num_rerank_sb = std::max(search_params.num_rerank, k);
        const size_t max_cands_per_query_sb = nprobes * num_rerank_sb;

        // Build leaf_ptr -> leaf_idx map once.
        std::unordered_map<node_t*, uint32_t> leaf_to_idx;
        leaf_to_idx.reserve(num_leaves * 2);
        for (uint32_t l = 0; l < num_leaves; ++l)
          leaf_to_idx[leaves_flat[l]] = l;

        struct LeafWant {
          uint32_t leaf_idx;
          uint64_t mask;
        };
        auto subbatch_lists = parlay::sequence<parlay::sequence<LeafWant>>(num_subbatches);

        parlay::parallel_for(0, num_subbatches, [&](size_t sb) {
          const size_t q_start = sb * B;
          const size_t q_end = std::min(q_start + B, num_q);
          const size_t b_size = q_end - q_start;
          // Accumulate (leaf_idx, q_off) entries, then sort+dedupe with mask OR.
          std::vector<uint64_t> entries;  // pack: (leaf_idx << 8) | q_off, since b_size <= 64
          entries.reserve(b_size * nprobes);
          for (size_t q = q_start; q < q_end; ++q) {
            const uint8_t q_off = static_cast<uint8_t>(q - q_start);
            for (size_t j = 0; j < nprobes; ++j) {
              auto* leaf_ptr = leaf_query_pairs[q * nprobes + j].first;
              if (!leaf_ptr) continue;
              auto it = leaf_to_idx.find(leaf_ptr);
              if (it == leaf_to_idx.end()) continue;
              entries.push_back((static_cast<uint64_t>(it->second) << 8) | q_off);
            }
          }
          std::sort(entries.begin(), entries.end());
          std::vector<LeafWant> result;
          result.reserve(entries.size());
          for (size_t i = 0; i < entries.size();) {
            const uint32_t lidx = static_cast<uint32_t>(entries[i] >> 8);
            uint64_t mask = 0;
            while (i < entries.size() && static_cast<uint32_t>(entries[i] >> 8) == lidx) {
              mask |= (1ULL << (entries[i] & 0xff));
              ++i;
            }
            result.push_back({lidx, mask});
          }
          subbatch_lists[sb] = parlay::sequence<LeafWant>(result.begin(), result.end());
        });
        t.stop();
        std::cout << "[MVIVF] SB Group: " << t.total_time() << " sec" << std::endl;
        t.reset();

        // ----- Step 3': process leaves in (sub-batch, leaf) parallel pairs -----
        t.start();
        auto all_candidates_sb = parlay::sequence<std::pair<uint32_t, float>>::uninitialized(
            num_q * max_cands_per_query_sb);
        auto cand_counts_sb =
            std::unique_ptr<std::atomic<size_t>[]>(new std::atomic<size_t>[num_q] {});

        auto safe_scatter_sb = [&](uint32_t q_id, const std::pair<uint32_t, float>* src,
                                   size_t count) {
          size_t slot = cand_counts_sb[q_id].fetch_add(count, std::memory_order_relaxed);
          if (slot + count > max_cands_per_query_sb) return;
          size_t base_idx = q_id * max_cands_per_query_sb + slot;
          for (size_t c = 0; c < count; ++c)
            all_candidates_sb[base_idx + c] = src[c];
        };

        parlay::parallel_for(0, num_subbatches, [&](size_t sb) {
          const size_t q_start = sb * B;
          const size_t q_end = std::min(q_start + B, num_q);
          const size_t b_size = q_end - q_start;
          std::vector<const QueryType*> sb_queries(b_size);
          for (size_t qi = 0; qi < b_size; ++qi) {
            sb_queries[qi] = &q_leaves[q_start + qi];
          }
          auto& leaves_for_sb = subbatch_lists[sb];

          parlay::parallel_for(0, leaves_for_sb.size(), [&](size_t li) {
            const uint32_t leaf_idx = leaves_for_sb[li].leaf_idx;
            const uint64_t mask = leaves_for_sb[li].mask;
            node_t* leaf = leaves_flat[leaf_idx];
            const SetType& leaf_data = leaf->encoded_leaf;
            const size_t C = std::min<size_t>(num_rerank_sb, leaf->get_size());

            // Gather wanted queries from sub-batch.
            std::vector<const QueryType*> wanted;
            wanted.reserve(b_size);
            std::vector<uint32_t> wanted_global;
            wanted_global.reserve(b_size);
            for (size_t qi = 0; qi < b_size; ++qi) {
              if (mask & (1ULL << qi)) {
                wanted.push_back(sb_queries[qi]);
                wanted_global.push_back(static_cast<uint32_t>(q_start + qi));
              }
            }
            if (wanted.empty()) return;

            std::vector<std::pair<uint32_t, float>> results(wanted.size() * num_rerank_sb);
            M2MType::TopKIntoUninitialized(wanted, leaf_data, num_rerank_sb, results.data(),
                                           /*q_block=*/4,
                                           /*parallel_query_blocks=*/false);
            for (size_t wi = 0; wi < wanted.size(); ++wi) {
              safe_scatter_sb(wanted_global[wi], results.data() + wi * num_rerank_sb, C);
            }
          });
        });
        t.stop();
        std::cout << "[MVIVF] SB Probing & Scattering: " << t.total_time() << " sec" << std::endl;
        t.reset();

        // ----- Aggregation + Rerank -----
        t.start();
        auto final_results = parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>>(num_q);
        auto bytes_accessed_rerank = parlay::sequence<size_t>::uninitialized(num_q);
        parlay::parallel_for(0, num_q, [&](size_t q_id) {
          size_t base_idx = q_id * max_cands_per_query_sb;
          size_t num_scattered = std::min(cand_counts_sb[q_id].load(std::memory_order_relaxed),
                                          max_cands_per_query_sb);
          auto* cands = all_candidates_sb.begin() + base_idx;
          size_t num_valid = 0;
          for (size_t c = 0; c < num_scattered; ++c) {
            if (cands[c].first != UINT32_MAX) {
              if (num_valid != c) cands[num_valid] = cands[c];
              ++num_valid;
            }
          }
          size_t take = std::min(num_rerank_sb, num_valid);
          if (take > 0 && take < num_valid) {
            std::nth_element(cands, cands + take, cands + num_valid,
                             [](const auto& a, const auto& b) { return a.second < b.second; });
          }
          parlay::sequence<std::pair<uint32_t, float>> top_cands;
          top_cands.reserve(take);
          for (size_t c = 0; c < take; ++c)
            top_cands.push_back(cands[c]);
          auto q_final = parlay::sequence<std::pair<uint32_t, float>>::uninitialized(
              std::min(k, top_cands.size()));
          bytes_accessed_rerank[q_id] = 0;
          if (search_params.num_rerank > 0) {
            size_t actual_rerank = std::min(num_rerank_sb, top_cands.size());
            bytes_accessed_rerank[q_id] =
                this->rerank(rerank_queries[q_id], points, top_cands, actual_rerank, q_final);
          } else {
            for (size_t c = 0; c < q_final.size(); ++c)
              q_final[c] = top_cands[c];
          }
          final_results[q_id] = std::move(q_final);
        });
        t.stop();
        std::cout << "[MVIVF] SB Aggregation and Re-ranking: " << t.total_time() << " sec"
                  << std::endl;
        t.reset();
        bytes_accessed += parlay::reduce(bytes_accessed_rerank);
        std::cout << "[MVIVF] Bytes Accessed: " << bytes_accessed << std::endl;
        std::cout << "[MVIVF] Dist Cmps: " << dist_cmps << std::endl;
        return std::make_pair(std::move(final_results), bytes_accessed);
      }
    }

    // Step 2: group by leaf.
    t.start();
    auto grouped = mvsic::group_by_key_inplace(leaf_query_pairs);
    t.stop();
    std::cout << "[MVIVF] Grouping: " << t.total_time() << " sec" << std::endl;
    t.reset();

    // Step 3: probe each leaf in parallel.
    t.start();
    const size_t num_rerank = std::max(search_params.num_rerank, k);
    const size_t max_cands = nprobes * num_rerank;
    auto all_candidates =
        parlay::sequence<std::pair<uint32_t, float>>::uninitialized(num_q * max_cands);
    auto cand_counts = std::unique_ptr<std::atomic<size_t>[]>(new std::atomic<size_t>[num_q] {});
    auto leaf_bytes = parlay::sequence<size_t>::uninitialized(grouped.size());
    auto leaf_cmps = parlay::sequence<size_t>::uninitialized(grouped.size());

    auto safe_scatter = [&](uint32_t q_id, const std::pair<uint32_t, float>* src, size_t count) {
      size_t slot = cand_counts[q_id].fetch_add(count, std::memory_order_relaxed);
      if (slot + count > max_cands) return;
      size_t base_idx = q_id * max_cands + slot;
      for (size_t c = 0; c < count; ++c)
        all_candidates[base_idx + c] = src[c];
    };

    parlay::parallel_for(0, grouped.size(), [&](size_t i) {
      auto& group = grouped[i];
      node_t* leaf = group[0].first;
      size_t C = std::min<size_t>(num_rerank, leaf->get_size());
      size_t nq_grp = group.size();
      leaf_bytes[i] = 0;
      leaf_cmps[i] = leaf->get_size() * nq_grp;
      if constexpr (!kHasLeafQuant) {
        auto query_ids =
            parlay::delayed_tabulate(nq_grp, [&](size_t j) { return group[j].second.first; });
        PointCloudSet<ChPoint> batched(eff_queries.filter(query_ids), d);
        auto leaf_results = leaf->data.distances(batched, num_rerank);
        parlay::parallel_for(0, nq_grp, [&](size_t j) {
          uint32_t q_id = group[j].second.first;
          safe_scatter(q_id, leaf_results.begin() + j * num_rerank, C);
        });
      } else if constexpr (kUseM2M) {
        using M2M = typename has_many_to_many_<LeafModel>::type;
        std::vector<const LeafQuery*> typed(nq_grp);
        parlay::parallel_for(0, nq_grp,
                             [&](size_t j) { typed[j] = &q_leaves[group[j].second.first]; });
        std::vector<std::pair<uint32_t, float>> batch_results(nq_grp * num_rerank);
        M2M::TopKIntoUninitialized(typed, leaf->encoded_leaf, num_rerank, batch_results.data(),
                                   /*q_block=*/4, /*parallel_query_blocks=*/true);
        parlay::parallel_for(0, nq_grp, [&](size_t j) {
          uint32_t q_id = group[j].second.first;
          safe_scatter(q_id, batch_results.data() + j * num_rerank, C);
        });
      } else {
        parlay::parallel_for(0, nq_grp, [&](size_t j) {
          uint32_t q_id = group[j].second.first;
          auto all_dists =
              parlay::sequence<std::pair<uint32_t, float>>::uninitialized(leaf->get_size());
          leaf->encoded_leaf.distances_all(q_leaves[q_id], all_dists.data());
          if (C < all_dists.size()) {
            std::nth_element(all_dists.begin(), all_dists.begin() + C, all_dists.end(),
                             [](const auto& a, const auto& b) { return a.second < b.second; });
          }
          safe_scatter(q_id, all_dists.data(), C);
        });
      }
    });
    t.stop();
    std::cout << "[MVIVF] Probing & Scattering: " << t.total_time() << " sec" << std::endl;
    t.reset();
    bytes_accessed += parlay::reduce(leaf_bytes);
    dist_cmps += parlay::reduce(leaf_cmps);

    // Step 4: aggregate + rerank.
    t.start();
    auto final_results = parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>>(num_q);
    auto bytes_accessed_rerank = parlay::sequence<size_t>::uninitialized(num_q);
    parlay::parallel_for(0, num_q, [&](size_t q_id) {
      size_t base_idx = q_id * max_cands;
      size_t num_scattered =
          std::min(cand_counts[q_id].load(std::memory_order_relaxed), max_cands);
      auto* cands = all_candidates.begin() + base_idx;

      // Compact: remove sentinel entries (UINT32_MAX) from TopKIntoUninitialized padding
      size_t num_valid = 0;
      for (size_t c = 0; c < num_scattered; ++c) {
        if (cands[c].first != UINT32_MAX) {
          if (num_valid != c) cands[num_valid] = cands[c];
          ++num_valid;
        }
      }

      size_t take = std::min(num_rerank, num_valid);
      if (take > 0 && take < num_valid) {
        std::nth_element(cands, cands + take, cands + num_valid,
                         [](const auto& a, const auto& b) { return a.second < b.second; });
      }

      parlay::sequence<std::pair<uint32_t, float>> top_cands;
      top_cands.reserve(take);
      for (size_t c = 0; c < take; ++c) {
        top_cands.push_back(cands[c]);
      }
      auto q_final = parlay::sequence<std::pair<uint32_t, float>>::uninitialized(
          std::min(k, top_cands.size()));
      bytes_accessed_rerank[q_id] = 0;

      if (search_params.num_rerank > 0) {
        size_t actual_rerank = std::min(num_rerank, top_cands.size());
        bytes_accessed_rerank[q_id] =
            this->rerank(rerank_queries[q_id], points, top_cands, actual_rerank, q_final);
      } else {
        for (size_t c = 0; c < q_final.size(); ++c) {
          q_final[c] = top_cands[c];
        }
      }

      final_results[q_id] = std::move(q_final);
    });
    t.stop();
    std::cout << "[MVIVF] Aggregation and Re-ranking: " << t.total_time() << " sec" << std::endl;
    t.reset();
    bytes_accessed += parlay::reduce(bytes_accessed_rerank);

    std::cout << "[MVIVF] Bytes Accessed: " << bytes_accessed << std::endl;
    std::cout << "[MVIVF] Dist Cmps: " << dist_cmps << std::endl;

    return std::make_pair(std::move(final_results), bytes_accessed);
  }

  // ---------------------------------------------------------------------------
  // Save / load (v4 skeleton format).
  //
  // The on-disk file is a quantization-agnostic "skeleton" describing only the
  // tree topology, the raw float internal-node centers, the leaf point ids,
  // and the IndexParams scalars used to build the tree.  No codebooks, no
  // encoded leaves, and no compressed centers are written.  Any templated
  // variant (e.g. IndexMVIVFCompressOneBitTQIP) can load any skeleton from the
  // same family and re-train / re-encode quantizers on load from the supplied
  // raw points; class_id is therefore a fixed constant per family rather than
  // a function of LeafModel / CompressCenters.
  //
  // Layout:
  //   magic        : uint32 = 'MVIF'
  //   version      : uint32 = 4
  //   class_id     : uint32 = 0 (reserved; always 0)
  //   IndexParams blob (POD subset)
  //   skeleton     : tree topology + raw float centers + leaf point ids
  //
  // save() requires raw centers to be present, i.e. it is only valid on the
  // skeleton variant IndexMVIVF<metric, /*CompressCenters=*/false,
  // NoQuantizer<metric>>.  Calling save() on a quantized / compressed-centers
  // variant aborts with a clear error: build the raw skeleton, save it once,
  // then load() into any templated variant that retrains on load.
  // ---------------------------------------------------------------------------
  static constexpr uint32_t kMagic = 0x4D564946u;  // 'MVIF'
  static constexpr uint32_t kVersion = 4u;         // v4: uniform skeleton format (no quant on disk)
  static constexpr uint32_t kClassId = 0u;         // reserved; same for every templated variant

 private:
  size_t traverse_tree_(node_t* node, parlay::sequence<node_t*>& ind_to_node,
                        std::unordered_map<node_t*, size_t>& node_to_ind,
                        parlay::sequence<size_t>& center_offsets,
                        parlay::sequence<size_t>& children_offsets,
                        parlay::sequence<size_t>& point_offsets, size_t height) {
    node_to_ind[node] = ind_to_node.size();
    ind_to_node.push_back(node);
    if (node->children.size() == 0) {
      point_offsets.push_back(node->data.size());
    } else {
      point_offsets.push_back(0);
      size_t dims = node->data.get_dims();
      for (size_t i = 0; i < node->children.size(); ++i) {
        center_offsets.push_back(node->data.get_size(i) * dims);
      }
    }
    children_offsets.push_back(node->children.size());
    size_t h = height + 1;
    for (node_t* child : node->children) {
      h = std::max(h, traverse_tree_(child, ind_to_node, node_to_ind, center_offsets,
                                     children_offsets, point_offsets, height + 1));
    }
    return h;
  }

  // Serialize the skeleton portion of IndexParams used by this family.
  // Quantization (leaf model + center compression) is encoded in the class id,
  // not the params blob, so there are no quantization fields here.
  void write_params_(std::ostream& out) const {
    auto w = [&](auto x) { out.write(reinterpret_cast<const char*>(&x), sizeof(x)); };
    w(params.k_per_level);
    w(params.max_leaf_size);
    w(params.max_depth);
    w(params.num_spill);
    w(params.s);
    w(params.mvclus.niters);
    w(params.mvclus.max_point_clouds_per_cluster);
    w(params.mvclus.max_points_per_centroid_inner_kmeans);
  }
  void read_params_(std::istream& in) {
    auto r = [&](auto& x) { in.read(reinterpret_cast<char*>(&x), sizeof(x)); };
    r(params.k_per_level);
    r(params.max_leaf_size);
    r(params.max_depth);
    r(params.num_spill);
    r(params.s);
    r(params.mvclus.niters);
    r(params.mvclus.max_point_clouds_per_cluster);
    r(params.mvclus.max_points_per_centroid_inner_kmeans);
  }

 public:
  void save(const std::string& filename) override {
    // Skeleton save persists raw internal-node centers + leaf point ids only.
    // Leaf encodings and compressed centers are never written; any templated
    // variant can load this file and re-derive its quantization on load using
    // the supplied raw points.  Quantized variants now keep `node->data`
    // populated alongside `compressed_centers` (see compress_internal_centers_),
    // so save() is valid for all variants.
    std::ofstream outfile(filename, std::ios::binary);
    std::cout << "Saving index to " << filename << std::endl;
    if (!outfile.is_open()) {
      std::cerr << "Error opening file for writing: " << filename << std::endl;
      return;
    }
    const uint32_t magic = kMagic;
    const uint32_t ver = kVersion;
    const uint32_t cid = kClassId;
    outfile.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    outfile.write(reinterpret_cast<const char*>(&ver), sizeof(ver));
    outfile.write(reinterpret_cast<const char*>(&cid), sizeof(cid));
    write_params_(outfile);

    parlay::sequence<node_t*> ind_to_node;
    std::unordered_map<node_t*, size_t> node_to_ind;
    parlay::sequence<size_t> center_offsets, children_offsets, point_offsets;
    size_t height = traverse_tree_(root, ind_to_node, node_to_ind, center_offsets, children_offsets,
                                   point_offsets, 0);
    kmeanstree_height = height;

    size_t total_center = parlay::scan_inplace(center_offsets);
    center_offsets.push_back(total_center);
    size_t total_children = parlay::scan_inplace(children_offsets);
    children_offsets.push_back(total_children);
    size_t total_points = parlay::scan_inplace(point_offsets);
    point_offsets.push_back(total_points);

    size_t num = ind_to_node.size();
    outfile.write(reinterpret_cast<const char*>(&num), sizeof(size_t));
    size_t num_center_offsets = center_offsets.size();
    outfile.write(reinterpret_cast<const char*>(&num_center_offsets), sizeof(size_t));
    outfile.write(reinterpret_cast<const char*>(center_offsets.begin()),
                  center_offsets.size() * sizeof(size_t));
    // Raw internal-node centers (always written; static_assert above guarantees
    // node->data holds the unencoded floats).
    for (size_t i = 0; i < num; ++i) {
      node_t* node = ind_to_node[i];
      if (!node->children.empty()) {
        auto coords = node->data.data();
        size_t nent = node->data.total_size() * node->data.get_dims();
        outfile.write(reinterpret_cast<const char*>(coords), nent * sizeof(float));
      }
    }

    outfile.write(reinterpret_cast<const char*>(children_offsets.begin()),
                  children_offsets.size() * sizeof(size_t));
    for (size_t i = 0; i < num; ++i) {
      node_t* node = ind_to_node[i];
      parlay::sequence<node_t*> children = node->children;
      for (size_t j = 0; j < children.size(); ++j) {
        size_t child_id = node_to_ind[children[j]];
        outfile.write(reinterpret_cast<const char*>(&child_id), sizeof(size_t));
      }
    }
    outfile.write(reinterpret_cast<const char*>(point_offsets.begin()),
                  point_offsets.size() * sizeof(size_t));
    for (size_t i = 0; i < num; ++i) {
      node_t* node = ind_to_node[i];
      if (node->children.empty()) {
        PointCloudSet<ChPoint>& pc = node->data;
        for (size_t j = 0; j < pc.size(); ++j) {
          uint32_t pid = pc.get_id(j);
          outfile.write(reinterpret_cast<const char*>(&pid), sizeof(uint32_t));
        }
      }
    }
    outfile.close();
  }

  void load(const std::string& filename, const PointCloudSet<ChPoint>& points) override {
    parlay::internal::timer t_io;
    t_io.start();
    std::ifstream infile(filename, std::ios::binary);
    std::cout << "Loading index from " << filename << std::endl;
    if (!infile.is_open()) {
      std::cerr << "Error opening file for reading: " << filename << std::endl;
      return;
    }
    uint32_t magic = 0, ver = 0, cid = 0;
    infile.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    infile.read(reinterpret_cast<char*>(&ver), sizeof(ver));
    infile.read(reinterpret_cast<char*>(&cid), sizeof(cid));
    if (magic != kMagic) {
      std::cerr << "[MVIVF] bad magic: file is not an MVIF skeleton index." << std::endl;
      return;
    }
    if (ver != kVersion) {
      std::cerr << "[MVIVF] MVIVF index file format changed in v" << kVersion << "; got v" << ver
                << ". Rebuild with current code." << std::endl;
      return;
    }
    if (cid != kClassId) {
      std::cerr << "[MVIVF] unexpected class_id " << cid << " (expected " << kClassId
                << " for the v" << kVersion << " skeleton)." << std::endl;
      return;
    }
    read_params_(infile);

    size_t num = 0;
    infile.read(reinterpret_cast<char*>(&num), sizeof(size_t));
    size_t num_center_offsets = 0;
    infile.read(reinterpret_cast<char*>(&num_center_offsets), sizeof(size_t));
    parlay::sequence<size_t> center_offsets(num_center_offsets);
    infile.read(reinterpret_cast<char*>(center_offsets.begin()),
                center_offsets.size() * sizeof(size_t));
    parlay::sequence<float> center_values(center_offsets[center_offsets.size() - 1]);
    infile.read(reinterpret_cast<char*>(center_values.begin()),
                center_values.size() * sizeof(float));
    parlay::sequence<size_t> children_offsets(num + 1);
    infile.read(reinterpret_cast<char*>(children_offsets.begin()),
                children_offsets.size() * sizeof(size_t));
    parlay::sequence<size_t> children_values(children_offsets[children_offsets.size() - 1]);
    infile.read(reinterpret_cast<char*>(children_values.begin()),
                children_values.size() * sizeof(size_t));
    parlay::sequence<size_t> point_offsets(num + 1);
    infile.read(reinterpret_cast<char*>(point_offsets.begin()),
                point_offsets.size() * sizeof(size_t));
    parlay::sequence<uint32_t> point_values(point_offsets[point_offsets.size() - 1]);
    infile.read(reinterpret_cast<char*>(point_values.begin()),
                point_values.size() * sizeof(uint32_t));
    infile.close();
    double t_io_ms = t_io.stop() * 1000.0;

    parlay::internal::timer t_retrain;
    t_retrain.start();
    if constexpr (kHasCenterQuant) center_model_.train(points);
    if constexpr (kHasLeafQuant) leaf_model_.train(points, leaf_params_);

    const size_t dim = points.get_dims();
    auto point_id_to_data_id = parlay::sequence<uint32_t>::uninitialized(points.size());
    parlay::parallel_for(0, points.size(),
                         [&](uint32_t i) { point_id_to_data_id[points.get_id(i)] = i; });
    parlay::sequence<size_t> children_sizes = parlay::sequence<size_t>::from_function(
        num, [&](size_t i) { return children_offsets[i + 1] - children_offsets[i]; });
    parlay::sequence<size_t> children_scan;
    size_t total_children;
    std::tie(children_scan, total_children) = parlay::scan(children_sizes);
    children_scan.push_back(total_children);
    parlay::sequence<size_t> point_sizes = parlay::sequence<size_t>::from_function(
        num, [&](size_t i) { return point_offsets[i + 1] - point_offsets[i]; });
    parlay::sequence<node_t*> ind_to_node =
        parlay::sequence<node_t*>::from_function(num, [&](size_t i) {
          node_t* node = new node_t();
          if (children_sizes[i] > 0) {
            size_t start = children_scan[i];
            parlay::sequence<size_t> local_offsets = parlay::sequence<size_t>::from_function(
                children_sizes[i] + 1,
                [&](size_t j) { return center_offsets[start + j] - center_offsets[start]; });
            node->data = PointCloudSet<ChPoint>(children_sizes[i], dim,
                                                center_values.data() + center_offsets[start],
                                                local_offsets.data(), nullptr);
          }
          node->children.resize(children_sizes[i]);
          if (point_sizes[i] > 0) {
            parlay::sequence<uint32_t> pg =
                parlay::sequence<uint32_t>::from_function(point_sizes[i], [&](size_t j) {
                  uint32_t pid = point_values[point_offsets[i] + j];
                  return point_id_to_data_id[pid];
                });
            node->data = PointCloudSet<ChPoint>(points.filter(pg), dim);
          }
          return node;
        });
    parlay::parallel_for(0, num, [&](size_t i) {
      node_t* node = ind_to_node[i];
      parlay::sequence<node_t*>& ch = node->children;
      parlay::parallel_for(0, ch.size(), [&](size_t j) {
        size_t cid2 = children_values[children_offsets[i] + j];
        ch[j] = ind_to_node[cid2];
      });
    });
    if (root != nullptr) {
      traverse_and_delete_(root);
      delete root;
    }
    root = ind_to_node[0];

    compute_leaf_flat_();
    compress_internal_centers_();

    if constexpr (kHasLeafQuant) {
      parlay::parallel_for(
          0, num,
          [&](size_t i) {
            node_t* node = ind_to_node[i];
            if (!node) return;
            if (node->children.empty() && node->data.size() > 0) {
              node->encoded_leaf = leaf_model_.encode(node->data);
            }
          },
          /*granularity=*/1);
    }
    double t_retrain_ms = t_retrain.stop() * 1000.0;
    std::cerr << "[MVIVF] load: tree_io=" << t_io_ms << "ms retrain=" << t_retrain_ms
              << "ms (compress_centers=" << (kHasCenterQuant ? 1 : 0)
              << ", leaf_quant=" << (kHasLeafQuant ? 1 : 0) << ")" << std::endl;
  }

  size_t get_height() const noexcept override { return kmeanstree_height; }

  // ---------------------------------------------------------------------------
  // Tree-quality stats (ported verbatim from the legacy implementation).
  // ---------------------------------------------------------------------------
  struct TreeStats {
    size_t num_internal_nodes = 0;
    size_t num_leaves = 0;
    double avg_leaf_size = 0.0;
    double avg_internal_node_size = 0.0;
    size_t total_point_clouds_internal = 0;
    size_t height = 0;
    double avg_child_fraction_imbalance = 0.0;
    double max_child_fraction_imbalance = 0.0;
    static constexpr double kBadImbalanceThreshold = 0.8;
    std::vector<std::pair<size_t, double>> bad_imbalance_entries;
  };

  TreeStats get_tree_stats() const {
    TreeStats s;
    if (!root) return s;
    size_t leaf_size_sum = 0, internal_size_sum = 0;
    double sum_frac_imbalance = 0.0;
    size_t num_imb = 0;
    std::function<size_t(const node_t*, size_t)> visit = [&](const node_t* node,
                                                             size_t depth) -> size_t {
      if (node->children.empty()) {
        s.num_leaves++;
        size_t lsz = node->get_size();
        leaf_size_sum += lsz;
        if (depth + 1 > s.height) s.height = depth + 1;
        return lsz;
      }
      s.num_internal_nodes++;
      size_t n_centers = node->get_size();
      s.total_point_clouds_internal += n_centers;
      internal_size_sum += n_centers;
      std::vector<size_t> sizes;
      sizes.reserve(node->children.size());
      size_t total = 0;
      for (const node_t* c : node->children) {
        size_t cs = visit(c, depth + 1);
        sizes.push_back(cs);
        total += cs;
      }
      if (sizes.size() >= 2 && total > 0) {
        auto [mn, mx] = std::minmax_element(sizes.begin(), sizes.end());
        double fi = static_cast<double>(*mx - *mn) / static_cast<double>(total);
        sum_frac_imbalance += fi;
        if (fi > s.max_child_fraction_imbalance) s.max_child_fraction_imbalance = fi;
        num_imb++;
        if (fi >= TreeStats::kBadImbalanceThreshold)
          s.bad_imbalance_entries.emplace_back(total, fi);
      }
      return total;
    };
    visit(root, 0);
    if (s.num_leaves > 0) s.avg_leaf_size = static_cast<double>(leaf_size_sum) / s.num_leaves;
    if (s.num_internal_nodes > 0)
      s.avg_internal_node_size = static_cast<double>(internal_size_sum) / s.num_internal_nodes;
    if (num_imb > 0)
      s.avg_child_fraction_imbalance = sum_frac_imbalance / static_cast<double>(num_imb);
    std::sort(s.bad_imbalance_entries.begin(), s.bad_imbalance_entries.end());
    return s;
  }

  parlay::sequence<uint32_t> get_flat_clustering() const {
    parlay::sequence<uint32_t> empty;
    if (!root) return empty;
    std::vector<std::pair<uint32_t, uint32_t>> assignments;
    assignments.reserve(1024);
    size_t max_id = 0, leaf_idx = 0;
    std::function<void(const node_t*)> visit = [&](const node_t* node) {
      if (node->children.empty()) {
        size_t n = node->data.size();
        for (size_t j = 0; j < n; ++j) {
          uint32_t id = node->data.get_id(j);
          assignments.emplace_back(id, static_cast<uint32_t>(leaf_idx));
          if (id > max_id) max_id = id;
        }
        ++leaf_idx;
      } else {
        for (const node_t* c : node->children)
          if (c) visit(c);
      }
    };
    visit(root);
    if (assignments.empty()) return empty;
    parlay::sequence<uint32_t> out(max_id + 1);
    parlay::parallel_for(0, out.size(), [&](size_t i) { out[i] = UINT32_MAX; });
    parlay::parallel_for(0, assignments.size(),
                         [&](size_t i) { out[assignments[i].first] = assignments[i].second; });
    return out;
  }

  parlay::sequence<uint32_t> get_root_child_clustering() const {
    parlay::sequence<uint32_t> empty;
    if (!root || root->children.empty()) return empty;
    std::vector<std::pair<uint32_t, uint32_t>> assignments;
    assignments.reserve(1024);
    size_t max_id = 0;
    std::function<void(const node_t*, uint32_t)> visit = [&](const node_t* node, uint32_t rc) {
      if (node->children.empty()) {
        size_t n = node->data.size();
        for (size_t j = 0; j < n; ++j) {
          uint32_t id = node->data.get_id(j);
          assignments.emplace_back(id, rc);
          if (id > max_id) max_id = id;
        }
      } else {
        for (size_t i = 0; i < node->children.size(); ++i) {
          uint32_t next = (node == root) ? static_cast<uint32_t>(i) : rc;
          visit(node->children[i], next);
        }
      }
    };
    for (size_t i = 0; i < root->children.size(); ++i) {
      if (root->children[i]) visit(root->children[i], static_cast<uint32_t>(i));
    }
    if (assignments.empty()) return empty;
    parlay::sequence<uint32_t> out(max_id + 1);
    parlay::parallel_for(0, out.size(), [&](size_t i) { out[i] = UINT32_MAX; });
    parlay::parallel_for(0, assignments.size(),
                         [&](size_t i) { out[assignments[i].first] = assignments[i].second; });
    return out;
  }

  std::vector<std::vector<uint32_t>> get_leaves_of_point() const {
    std::vector<std::vector<uint32_t>> out;
    if (!root) return out;
    std::vector<std::pair<uint32_t, uint32_t>> assignments;
    assignments.reserve(1024);
    size_t max_id = 0, leaf_idx = 0;
    std::function<void(const node_t*)> visit = [&](const node_t* node) {
      if (node->children.empty()) {
        size_t n = node->data.size();
        for (size_t j = 0; j < n; ++j) {
          uint32_t id = node->data.get_id(j);
          assignments.emplace_back(id, static_cast<uint32_t>(leaf_idx));
          if (id > max_id) max_id = id;
        }
        ++leaf_idx;
      } else {
        for (const node_t* c : node->children)
          if (c) visit(c);
      }
    };
    visit(root);
    if (assignments.empty()) return out;
    out.resize(max_id + 1);
    for (auto [pid, lid] : assignments)
      out[pid].push_back(lid);
    return out;
  }

  std::vector<std::vector<uint32_t>> get_root_children_of_point() const {
    auto lop = get_leaves_of_point();
    if (lop.empty() || leaf_to_root_child_.empty()) return {};
    std::vector<std::vector<uint32_t>> out(lop.size());
    for (size_t pid = 0; pid < lop.size(); ++pid) {
      std::unordered_set<uint32_t> rids;
      for (uint32_t lid : lop[pid]) {
        if (lid < leaf_to_root_child_.size()) rids.insert(leaf_to_root_child_[lid]);
      }
      out[pid].assign(rids.begin(), rids.end());
    }
    return out;
  }
};

// =============================================================================
// Concrete aliases.  These are what downstream code (benches, Python bindings,
// benchmark.py) should use.
//
// Naming scheme:
//   IndexMVIVF{Compress}?{Quantizer}?{IP|L2}
//
// Where CompressCenters => 4-bit TQ centers, and {Quantizer} picks the
// LeafModel.  Compress-only or Quantizer-only or both are all valid.
// =============================================================================

template<bool m>
using IndexMVIVFIP_ = IndexMVIVF<m, false, NoQuantizer<m>>;
using IndexMVIVFIP = IndexMVIVF<false, false, NoQuantizer<false>>;
using IndexMVIVFL2 = IndexMVIVF<true, false, NoQuantizer<true>>;

using IndexMVIVFCompressIP = IndexMVIVF<false, true, NoQuantizer<false>>;
using IndexMVIVFCompressL2 = IndexMVIVF<true, true, NoQuantizer<true>>;

using IndexMVIVFPQIP = IndexMVIVF<false, false, pq_mv::Model<false>>;
using IndexMVIVFPQL2 = IndexMVIVF<true, false, pq_mv::Model<true>>;
using IndexMVIVFCompressPQIP = IndexMVIVF<false, true, pq_mv::Model<false>>;
using IndexMVIVFCompressPQL2 = IndexMVIVF<true, true, pq_mv::Model<true>>;

using IndexMVIVFFastScanIP = IndexMVIVF<false, false, fastscan_mv::Model<false>>;
using IndexMVIVFFastScanL2 = IndexMVIVF<true, false, fastscan_mv::Model<true>>;
using IndexMVIVFCompressFastScanIP = IndexMVIVF<false, true, fastscan_mv::Model<false>>;
using IndexMVIVFCompressFastScanL2 = IndexMVIVF<true, true, fastscan_mv::Model<true>>;

using IndexMVIVFRaBitQIP = IndexMVIVF<false, false, rabitq_mv::Model<false>>;
using IndexMVIVFRaBitQL2 = IndexMVIVF<true, false, rabitq_mv::Model<true>>;
using IndexMVIVFCompressRaBitQIP = IndexMVIVF<false, true, rabitq_mv::Model<false>>;
using IndexMVIVFCompressRaBitQL2 = IndexMVIVF<true, true, rabitq_mv::Model<true>>;

using IndexMVIVFTQIP = IndexMVIVF<false, false, turboquant_mv::Model<false>>;
using IndexMVIVFTQL2 = IndexMVIVF<true, false, turboquant_mv::Model<true>>;
using IndexMVIVFCompressTQIP = IndexMVIVF<false, true, turboquant_mv::Model<false>>;
using IndexMVIVFCompressTQL2 = IndexMVIVF<true, true, turboquant_mv::Model<true>>;

using IndexMVIVFSPQTQIP = IndexMVIVF<false, false, pqtq_mv::Model<false>>;
using IndexMVIVFSPQTQL2 = IndexMVIVF<true, false, pqtq_mv::Model<true>>;
using IndexMVIVFCompressSPQTQIP = IndexMVIVF<false, true, pqtq_mv::Model<false>>;
using IndexMVIVFCompressSPQTQL2 = IndexMVIVF<true, true, pqtq_mv::Model<true>>;

using IndexMVIVFOneBitTQIP = IndexMVIVF<false, false, turboquant_1bit_mv::Model<false>>;
using IndexMVIVFOneBitTQL2 = IndexMVIVF<true, false, turboquant_1bit_mv::Model<true>>;
using IndexMVIVFCompressOneBitTQIP = IndexMVIVF<false, true, turboquant_1bit_mv::Model<false>>;
using IndexMVIVFCompressOneBitTQL2 = IndexMVIVF<true, true, turboquant_1bit_mv::Model<true>>;

using IndexMVIVFEightBitTQIP = IndexMVIVF<false, false, turboquant_8bit_mv::Model<false>>;
using IndexMVIVFEightBitTQL2 = IndexMVIVF<true, false, turboquant_8bit_mv::Model<true>>;
using IndexMVIVFCompressEightBitTQIP = IndexMVIVF<false, true, turboquant_8bit_mv::Model<false>>;
using IndexMVIVFCompressEightBitTQL2 = IndexMVIVF<true, true, turboquant_8bit_mv::Model<true>>;

}  // namespace mvsic
