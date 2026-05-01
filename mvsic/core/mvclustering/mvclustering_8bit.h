#pragma once

// 8-bit TurboQuant clustering: drop-in replacement for `MVClustering<metric>`
// that runs Lloyd's k-means with the assignment step computed by the int8
// VPDPBUSD panel kernel from `turboquant_8bit_mv.h`.
//
// Pipeline:
//   * Encode every input point cloud once as a Quantized_Query_Point_Cloud
//     (flat int8 + per-vector norm scaling factor) -- this is amortized over
//     all Lloyd iterations.
//   * Each iteration, encode the current float centers as a
//     Quantized_Point_Cloud_Set (pre-transposed panel form), then run
//     turboquant_8bit_mv::ManyToMany::TopKIntoUninitialized with k=1 to obtain
//     cluster ids. This is the same VNNI GEMM kernel the search path uses, so
//     assignment runs at int8×int8 -> int32 throughput.
//   * Centroid update keeps using the existing float `kmeans_subsample`/
//     `kmeans_weighted_subsample` helpers on the original float points -- the
//     per-cluster work is small relative to assignment and using float here
//     keeps the centers in the original (un-rotated) space, so the rest of the
//     index pipeline can consume them with no extra changes.

#include <Eigen/Dense>
#include <algorithm>
#include <cstring>
#include <optional>
#include <vector>
#include "parlay/primitives.h"

#include "mvsic/core/types/chamfer_ip_point.h"
#include "mvsic/core/types/chamfer_l2_point.h"
#include "mvsic/core/types/point_cloud_set.h"
#include "mvsic/core/quantization/turboquant_8bit_mv.h"
#include "mvsic/core/utils/kmeans_util.h"
#include "mvsic/core/utils/util.h"

#include "seeding/uniformlyrandom.h"
#include "mvclustering.h"
#include "mvclustering_config.h"

namespace mvsic {

template<bool metric>
class MVClustering8BTQ {
 public:
  using ChPoint = std::conditional_t<metric, ChamferL2_Point, ChamferIP_Point>;
  using QModel = ::mvsic::turboquant_8bit_mv::Model<metric>;
  using EncSet = typename QModel::EncodedSet;
  using EncQuery = typename QModel::EncodedQuery;

  uint32_t d;
  uint32_t k;
  uint32_t s = 0;
  MVClusteringConfig params;
  PointCloudSet<ChPoint> centers;
  parlay::sequence<uint32_t> cluster_ids;

  struct IterationStats {
    float centroid_update_time = 0.0;
    float assignment_time = 0.0;
    float cost = 0.0;

    float total_time() const { return centroid_update_time + assignment_time; }
  };
  std::vector<IterationStats> _iteration_stats;

  MVClustering8BTQ(uint32_t d, uint32_t k, MVClusteringConfig params) noexcept :
      d(d), k(k), s(0), params(params) {}
  MVClustering8BTQ(uint32_t d, uint32_t k, uint32_t s, MVClusteringConfig params) noexcept :
      d(d), k(k), s(s), params(params) {}

  // Optional shared 8BTQ cache: a model + pre-encoded queries for the *root*
  // input set, plus a mapping `points_to_root[i] = root_index_of_local_i`.
  // When set, train() / get_clustering() skip per-call rotator setup and
  // per-call point encoding entirely; the assignment kernel reads encoded
  // queries directly out of the cache.
  //
  // The cache is correct iff:
  //   * the local PointCloudSet's i-th PC was encoded (at build top-level)
  //     into shared_q_clouds_[points_to_root_[i]];
  //   * shared_model_ is the model that produced those queries.
  void set_shared_cache(const QModel& m, const std::vector<EncQuery>& q,
                        const parlay::sequence<uint32_t>& points_to_root) noexcept {
    shared_model_ = &m;
    shared_q_clouds_ = &q;
    points_to_root_ = &points_to_root;
  }

  // Optional second tier of the build-top-level cache: a flat per-vector
  // encoded-query buffer, sized total_root_vecs × stride, plus a per-PC
  // offset table (size: num_root_pcs + 1).  When set together with
  // set_shared_cache, the inner Lloyd's k-means inside the centroid update
  // skips its own rotate+quantize step too: the TQ8 backend gathers each
  // row's encoded state directly out of `vec_*` using indices computed from
  // `pc_vec_offsets[points_to_root_[local_pc]]`.
  void set_shared_vector_cache(::mvsic::lloyds::TQ8VectorCache<metric> vec_cache,
                               const parlay::sequence<size_t>& pc_vec_offsets) noexcept {
    vec_cache_ = vec_cache;
    pc_vec_offsets_ = &pc_vec_offsets;
  }

  void train(const PointCloudSet<ChPoint>& data);

  auto& get_centers() { return centers; }

  // Returns the final cluster id for each input point cloud. Mirrors the
  // float MVClustering API: callers pass the *full* point set (not the
  // possibly-subsampled training set) and we score them through the encoded
  // 8-bit kernel against the trained float centers.
  parlay::sequence<uint32_t> get_clustering(const PointCloudSet<ChPoint>& points) {
    const size_t n = points.size();
    parlay::sequence<uint32_t> out(n);
    if (n == 0 || centers.size() == 0) return out;
    parlay::sequence<std::pair<uint32_t, float>> topk = score_top_k_(points, /*top_k=*/1);
    // score_top_k_ writes pairs; first = cluster id (we set top_k=1).
    parlay::parallel_for(0, n, [&](size_t i) { out[i] = topk[i].first; });
    return out;
  }

  // Top-C centers per input point cloud. Returns flat array of size n*C in
  // best-first order, mirroring MVClustering<metric>::get_topC.
  parlay::sequence<uint32_t> get_topC(const PointCloudSet<ChPoint>& points, uint32_t C) {
    const size_t n = points.size();
    C = std::min(C, k);
    if (C == 0) {
      std::cerr << "[MVClustering8BTQ]: C is 0; need at least 1 in get_topC." << std::endl;
      abort();
    }
    parlay::sequence<uint32_t> out(n * C);
    if (n == 0 || centers.size() == 0) return out;
    auto topk = score_top_k_(points, C);
    // ManyToMany::TopKIntoUninitialized writes in worst-to-best (heap pop)
    // order at offset [i*k .. i*k+k-1]; reverse per-row so out is best-first.
    parlay::parallel_for(0, n, [&](size_t i) {
      for (uint32_t r = 0; r < C; ++r) out[i * C + r] = topk[i * C + r].first;
    });
    return out;
  }

 private:
  // Optional shared cache. See set_shared_cache().
  const QModel* shared_model_ = nullptr;
  const std::vector<EncQuery>* shared_q_clouds_ = nullptr;
  const parlay::sequence<uint32_t>* points_to_root_ = nullptr;

  // Optional per-vector cache for the *inner* kmeans. See
  // set_shared_vector_cache().
  ::mvsic::lloyds::TQ8VectorCache<metric> vec_cache_{};
  const parlay::sequence<size_t>* pc_vec_offsets_ = nullptr;

  inline bool has_shared_cache_() const noexcept {
    return shared_model_ != nullptr && shared_q_clouds_ != nullptr &&
           points_to_root_ != nullptr;
  }
  inline bool has_vector_cache_() const noexcept {
    return vec_cache_.model != nullptr && vec_cache_.q_data != nullptr &&
           pc_vec_offsets_ != nullptr;
  }


  // Encode `points` as queries (or pull from the shared cache via
  // points_to_root_), encode current `centers` as DB, run 8BTQ ManyToMany
  // top-k. Returns the n*top_k pair array in the order written by the kernel
  // (heap order = best-first, see ManyToMany::TopKIntoUninitialized).
  parlay::sequence<std::pair<uint32_t, float>> score_top_k_(const PointCloudSet<ChPoint>& points,
                                                            uint32_t top_k) {
    const size_t n = points.size();
    QModel local_model;
    const QModel* model_ptr = nullptr;
    std::vector<EncQuery> local_q_clouds;
    std::vector<const EncQuery*> q_ptrs(n);

    if (has_shared_cache_()) {
      model_ptr = shared_model_;
      // points_to_root_ is set against the local point set passed to train().
      // get_clustering() may be called with the same set (the common case in
      // recursive_build_), so we use the same mapping. If a different point
      // set is ever passed here, the size check below catches it.
      if (points_to_root_->size() != n) {
        std::cerr << "[MVClustering8BTQ] shared cache size mismatch ("
                  << points_to_root_->size() << " vs " << n << ")" << std::endl;
        std::abort();
      }
      for (size_t i = 0; i < n; ++i) {
        q_ptrs[i] = &(*shared_q_clouds_)[(*points_to_root_)[i]];
      }
    } else {
      local_model.train(points);
      model_ptr = &local_model;
      local_q_clouds.resize(n);
      parlay::parallel_for(
          0, n, [&](size_t i) { local_q_clouds[i] = model_ptr->quantize_query(points[i]); });
      for (size_t i = 0; i < n; ++i) q_ptrs[i] = &local_q_clouds[i];
    }
    EncSet center_db = model_ptr->encode(centers);
    parlay::sequence<std::pair<uint32_t, float>> out(n * top_k);
    ::mvsic::turboquant_8bit_mv::ManyToMany<EncSet>::TopKIntoUninitialized(
        q_ptrs, center_db, top_k, out.data());
    return out;
  }
};

template<bool metric>
void MVClustering8BTQ<metric>::train(const PointCloudSet<ChPoint>& points_) {
  // ----- Optional subsample (mirror MVClustering<metric>::train).
  // When a shared cache is provided, the subsample picks a subset of the
  // *local* indices into points_; we just propagate those local indices into
  // the cache via points_to_root_ at lookup time.
  std::optional<PointCloudSet<ChPoint>> subsampled_points;
  std::optional<parlay::sequence<uint32_t>> sampled_local_ids;
  const PointCloudSet<ChPoint>* points_ptr;
  if (params.max_point_clouds_per_cluster > 0 &&
      points_.size() > params.max_point_clouds_per_cluster * k && points_.size() > 2048) {
    if (params.verbose >= 1) {
      std::cout << "[MVClustering8BTQ] Subsampling from " << points_.size() << " to "
                << params.max_point_clouds_per_cluster * k << " point clouds for training"
                << std::endl;
    }
    parlay::internal::timer _subt;
    _subt.start();
    auto sampled_ids =
        parlay::sequence<uint32_t>::uninitialized(params.max_point_clouds_per_cluster * k);
    parlay::parallel_for(0, params.max_point_clouds_per_cluster * k, [&](size_t i) {
      sampled_ids[i] = parlay::hash32(params.seed + i) % points_.size();
    });
    auto sampled_pcs = parlay::delayed_tabulate(sampled_ids.size(),
                                                [&](size_t i) { return points_[sampled_ids[i]]; });
    subsampled_points = PointCloudSet<ChPoint>(sampled_pcs, d);
    points_ptr = &(*subsampled_points);
    sampled_local_ids = std::move(sampled_ids);
    _subt.stop();
    if (params.verbose >= 1) {
      std::cout << "[MVClustering8BTQ] Subsampling time: " << _subt.total_time() << " seconds"
                << std::endl;
    }
  } else {
    points_ptr = &points_;
  }
  const PointCloudSet<ChPoint>& points = *points_ptr;

  uint32_t n = points.size();
  if (s == 0) s = static_cast<uint32_t>(std::ceil(points.average_size()));
  if (params.verbose >= 1)
    std::cout << "[MVClustering8BTQ] Centroid-Point Cloud Size: " << s << std::endl;
  _iteration_stats.resize(params.niters + 1);
  cluster_ids.resize(n);

  // When a vector cache is available, precompute the local-PC -> root-PC map
  // so the per-cluster centroid-update path can build per-row vec indices in
  // O(1) lookups instead of re-deriving from the optional subsample.
  parlay::sequence<uint32_t> local_to_root_pc;
  const bool use_vec_cache = has_shared_cache_() && has_vector_cache_();
  if (use_vec_cache) {
    local_to_root_pc.resize(n);
    if (sampled_local_ids.has_value()) {
      parlay::parallel_for(0, n, [&](size_t i) {
        local_to_root_pc[i] = (*points_to_root_)[(*sampled_local_ids)[i]];
      });
    } else {
      parlay::parallel_for(0, n, [&](size_t i) {
        local_to_root_pc[i] = (*points_to_root_)[i];
      });
    }
  }

  // ----- Train the 8BTQ encoder (Hadamard rotator) and pre-encode the
  // training points as queries -- unless a shared cache is provided, in which
  // case we reuse the build-top-level rotator and queries verbatim.
  QModel local_model;
  const QModel* model_ptr = nullptr;
  std::vector<EncQuery> local_q_clouds;
  std::vector<const EncQuery*> q_ptrs(n);
  parlay::internal::timer _enct;
  _enct.start();
  if (has_shared_cache_()) {
    model_ptr = shared_model_;
    if (sampled_local_ids.has_value()) {
      // The subsample picked local-into-points_ indices. Translate to root.
      for (size_t i = 0; i < n; ++i) {
        const uint32_t local_idx = (*sampled_local_ids)[i];
        q_ptrs[i] = &(*shared_q_clouds_)[(*points_to_root_)[local_idx]];
      }
    } else {
      // No subsample: training set is the full points_ argument.
      if (points_to_root_->size() != n) {
        std::cerr << "[MVClustering8BTQ] shared cache size mismatch in train ("
                  << points_to_root_->size() << " vs " << n << ")" << std::endl;
        std::abort();
      }
      for (size_t i = 0; i < n; ++i) {
        q_ptrs[i] = &(*shared_q_clouds_)[(*points_to_root_)[i]];
      }
    }
  } else {
    local_model.train(points);
    model_ptr = &local_model;
    local_q_clouds.resize(n);
    parlay::parallel_for(
        0, n, [&](size_t i) { local_q_clouds[i] = model_ptr->quantize_query(points[i]); });
    for (size_t i = 0; i < n; ++i) q_ptrs[i] = &local_q_clouds[i];
  }
  _enct.stop();
  if (params.verbose >= 1) {
    std::cout << "[MVClustering8BTQ] Point encode time: " << _enct.total_time() << " seconds"
              << (has_shared_cache_() ? " (cached)" : "") << std::endl;
  }

  // ----- Initial seeding (uniform random sampling, same as MVClustering).
  parlay::internal::timer _st;
  _st.start();
  if (params.verbose >= 2)
    std::cout << "[MVClustering8BTQ] Seeding algorithm: " << params.init << std::endl;
  if (params.init == "Random") {
    centers = UniformlyRandomMV(points, k, params.seed);
  } else {
    std::cout << "[MVClustering8BTQ] Error: Invalid seeding algorithm." << std::endl;
    abort();
  }
  _st.stop();
  _iteration_stats[0].centroid_update_time = _st.total_time();
  if (params.verbose >= 1) {
    std::cout << "[MVClustering8BTQ] Seeding time: " << _iteration_stats[0].total_time() << " sec"
              << std::endl;
  }
  if (params.niters == 0) return;

  // ----- Lloyd's iterations.
  parlay::internal::timer _it_timer;
  for (long it = 1; it <= params.niters; ++it) {
    // Step A: Encode current centers and run 8BTQ ManyToMany top-1.
    _it_timer.start();
    parlay::sequence<std::pair<uint32_t, float>> top1(n);
    EncSet center_db = model_ptr->encode(centers);
    ::mvsic::turboquant_8bit_mv::ManyToMany<EncSet>::TopKIntoUninitialized(
        q_ptrs, center_db, /*k=*/1, top1.data());
    parlay::parallel_for(0, n, [&](size_t i) { cluster_ids[i] = top1[i].first; });
    _it_timer.stop();
    _iteration_stats[it].assignment_time = _it_timer.total_time();
    _it_timer.reset();

    if (it == 1) {  // Reshape centers to fixed-size point clouds for in-place updates.
      centers = PointCloudSet<ChPoint>(k, s, d);
    }

    // Step B: Centroid update on float points (cluster sizes are small, so
    // the per-cluster inner k-means dominates this step; staying in float
    // keeps centers in the original (un-rotated) space for downstream code).
    _it_timer.start();
    auto id_pt = parlay::tabulate(n, [&](uint32_t i) { return std::make_pair(cluster_ids[i], i); });
    auto grouped = group_by_key_inplace(id_pt);
    parlay::parallel_for(0, grouped.size(), [&](size_t i) {
      const size_t M = grouped[i].size();
      // Per-cluster prefix sum of vec counts: pc_offsets_in_data[j] = sum of
      // get_size(grouped[i][0..j-1]) (so [j] is the global vec index of the
      // first vec belonging to PC j within this cluster). M is small (~100-5K),
      // so a serial scan is fine and simpler than a parlay::scan.
      parlay::sequence<size_t> pc_offsets_in_data(M + 1);
      pc_offsets_in_data[0] = 0;
      for (size_t j = 0; j < M; ++j) {
        pc_offsets_in_data[j + 1] =
            pc_offsets_in_data[j] +
            static_cast<size_t>(points.get_size(grouped[i][j].second));
      }
      const size_t T_c = pc_offsets_in_data[M];

      // Pick the sample size. When T_c > m we sample m vecs in tuple space and
      // gather only those; this is statistically equivalent to the internal
      // sample inside kmeans_subsample but avoids materializing T_c rows we'd
      // throw away. When T_c <= m we gather all (current behavior).
      const size_t m = static_cast<size_t>(
          params.max_points_per_centroid_inner_kmeans) * static_cast<size_t>(s);
      const bool use_outer_sample = (T_c > m);
      const size_t row_count = use_outer_sample ? m : T_c;

      auto data = parlay::sequence<parlay::sequence<float>>(
          row_count, parlay::sequence<float>::uninitialized(d));

      const bool emit_vec_indices = use_vec_cache && row_count > 0;
      const bool need_weights = params.use_weighted_inner_kmeans;
      parlay::sequence<uint32_t> vec_root_indices;
      parlay::sequence<float> weights;
      if (emit_vec_indices) vec_root_indices.resize(row_count);
      if (need_weights) weights.resize(row_count);

      if (use_outer_sample) {
        // Sample m random global vec indices in [0, T_c), map each to
        // (pc, v_in_pc) via binary search of pc_offsets_in_data, then gather.
        const size_t* off = pc_offsets_in_data.data();
        parlay::parallel_for(0, m, [&](size_t r) {
          const size_t g =
              static_cast<size_t>(parlay::hash32(static_cast<uint32_t>(r))) % T_c;
          // upper_bound returns first off[k] > g; (k - 1) is the PC index.
          const size_t k_ub =
              static_cast<size_t>(std::upper_bound(off, off + (M + 1), g) - off);
          const size_t j = k_ub - 1;
          const uint32_t pc_local = grouped[i][j].second;
          const uint32_t v_in_pc = static_cast<uint32_t>(g - off[j]);
          std::memcpy(data[r].begin(), points.data(pc_local) + v_in_pc * d,
                      d * sizeof(float));
          if (emit_vec_indices) {
            const uint32_t pc_root = local_to_root_pc[pc_local];
            vec_root_indices[r] =
                static_cast<uint32_t>((*pc_vec_offsets_)[pc_root] + v_in_pc);
          }
          if (need_weights) {
            const uint32_t doc_size = points.get_size(pc_local);
            weights[r] = (doc_size > 0)
                             ? 1.0f / static_cast<float>(doc_size)
                             : 0.0f;
          }
        });
      } else {
        // Gather all T_c vecs (small/medium cluster: full inner k-means input).
        parlay::parallel_for(0, M, [&](size_t j) {
          const uint32_t pc_local = grouped[i][j].second;
          const size_t base = pc_offsets_in_data[j];
          const uint32_t doc_size = points.get_size(pc_local);
          const float* src = points.data(pc_local);
          for (uint32_t v = 0; v < doc_size; ++v) {
            std::memcpy(data[base + v].begin(), src + v * d, d * sizeof(float));
          }
          if (emit_vec_indices) {
            const uint32_t pc_root = local_to_root_pc[pc_local];
            const size_t pc_root_base = (*pc_vec_offsets_)[pc_root];
            for (uint32_t v = 0; v < doc_size; ++v) {
              vec_root_indices[base + v] =
                  static_cast<uint32_t>(pc_root_base + v);
            }
          }
          if (need_weights && doc_size > 0) {
            const float w = 1.0f / static_cast<float>(doc_size);
            for (uint32_t v = 0; v < doc_size; ++v) weights[base + v] = w;
          }
        });
      }

      const ::mvsic::lloyds::TQ8VectorCache<metric>* cache_ptr =
          emit_vec_indices ? &vec_cache_ : nullptr;
      const parlay::sequence<uint32_t>* idx_ptr =
          emit_vec_indices ? &vec_root_indices : nullptr;

      if (s >= row_count) {
        centers.set_point_cloud(i, data);
      } else if (need_weights) {
        auto new_centers = kmeans_weighted_subsample<metric>(
            data, weights, s, params.max_points_per_centroid_inner_kmeans,
            params.verbose >= 3, params.build_with_8btq, cache_ptr, idx_ptr);
        centers.set_point_cloud(i, new_centers);
      } else {
        auto new_centers = kmeans_subsample<metric>(
            data, s, params.max_points_per_centroid_inner_kmeans,
            params.verbose >= 3, params.build_with_8btq, cache_ptr, idx_ptr);
        centers.set_point_cloud(i, new_centers);
      }
    });

    // Sample from input for empty clusters (mirror MVClustering).
    if (k - grouped.size() > 0) {
      if (params.verbose >= 2) {
        std::cout << "[MVClustering8BTQ] " << k - grouped.size()
                  << " clusters empty, sampling from input" << std::endl;
      }
      parlay::parallel_for(grouped.size(), k, [&](size_t i) {
        parlay::sequence<size_t> id = {parlay::hash32(params.seed + i - grouped.size()) % n};
        auto data = points.filter_flattened(id);

        parlay::sequence<uint32_t> vec_root_indices;
        const bool emit_vec_indices = use_vec_cache && data.size() > 0;
        if (emit_vec_indices) {
          const uint32_t pc_local = static_cast<uint32_t>(id[0]);
          const uint32_t pc_root = local_to_root_pc[pc_local];
          const size_t base = (*pc_vec_offsets_)[pc_root];
          vec_root_indices.resize(data.size());
          for (size_t t = 0; t < data.size(); ++t) {
            vec_root_indices[t] = static_cast<uint32_t>(base + t);
          }
        }
        const ::mvsic::lloyds::TQ8VectorCache<metric>* cache_ptr =
            emit_vec_indices ? &vec_cache_ : nullptr;
        const parlay::sequence<uint32_t>* idx_ptr =
            emit_vec_indices ? &vec_root_indices : nullptr;

        if (s >= data.size()) {
          centers.set_point_cloud(i, data);
        } else if (params.use_weighted_inner_kmeans) {
          parlay::sequence<float> weights;
          uint32_t doc_size = points.get_size(id[0]);
          if (doc_size > 0) {
            float w = 1.0f / static_cast<float>(doc_size);
            weights = parlay::sequence<float>(data.size(), w);
          }
          auto new_centers = kmeans_weighted_subsample<metric>(
              data, weights, s, params.max_points_per_centroid_inner_kmeans, params.verbose >= 3,
              params.build_with_8btq, cache_ptr, idx_ptr);
          centers.set_point_cloud(i, new_centers);
        } else {
          auto new_centers = kmeans_subsample<metric>(
              data, s, params.max_points_per_centroid_inner_kmeans, params.verbose >= 3,
              params.build_with_8btq, cache_ptr, idx_ptr);
          centers.set_point_cloud(i, new_centers);
        }
      });
      params.seed += (k - grouped.size());
    }
    _it_timer.stop();
    _iteration_stats[it].centroid_update_time = _it_timer.total_time();
    _it_timer.reset();

    if (params.verbose >= 1) {
      std::cout << "[MVClustering8BTQ] Iteration " << it
                << ": assignment time = " << _iteration_stats[it].assignment_time
                << ", centroid update time = " << _iteration_stats[it].centroid_update_time
                << std::endl;
    }
  }
  if (params.verbose >= 1) {
    std::cout << "[MVClustering8BTQ] Completed: " << params.niters << " iterations" << std::endl;
  }
}

}  // namespace mvsic
