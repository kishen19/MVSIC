#pragma once

#include <Eigen/Dense>
#include <cstring>
#include "parlay/primitives.h"

#include "mvsic/core/types/chamfer_ip_point.h"
#include "mvsic/core/types/chamfer_l2_point.h"
#include "mvsic/core/types/point_cloud_set.h"
#include "mvsic/core/utils/kmeans_util.h"
#include "mvsic/core/utils/util.h"
#include "mvsic/core/distance_measures/many_to_many.h"

#include "seeding/uniformlyrandom.h"
#include "mvclustering_config.h"

namespace mvsic {

/* Multi-Vector Clustering Algorithm
 - Given a set of point clouds, cluster them into k clusters, where each cluster
   is represented by a centroid point cloud with s points each.
 - The clustering is done via a Lloyd's style algorithm, with an inner k-means
   algorithm to compute the centroid point clouds.
 - The distance function used is the Chamfer distance (L2 or IP).
*/

template<bool metric>
class MVClustering {
 public:
  using ChPoint = std::conditional_t<metric, ChamferL2_Point, ChamferIP_Point>;

  uint32_t d;  // Dimension of vectors
  uint32_t k;  // Number of centroid-point_clouds
  // Number of points clouds per centroid-point_cloud
  // Default 0: Computes the average number of points per point cloud in the input
  uint32_t s = 0;
  MVClusteringConfig params;
  PointCloudSet<ChPoint> centers;          // Centers of clusters
  parlay::sequence<uint32_t> cluster_ids;  // Cluster ids for each data point cloud

  struct IterationStats {
    float centroid_update_time = 0.0;
    float assignment_time = 0.0;
    float cost = 0.0;

    float total_time() const { return centroid_update_time + assignment_time; }
  };
  std::vector<IterationStats> _iteration_stats;

  MVClustering(uint32_t d, uint32_t k, MVClusteringConfig params) noexcept :
      d(d), k(k), s(0), params(params) {}
  MVClustering(uint32_t d, uint32_t k, uint32_t s, MVClusteringConfig params) noexcept :
      d(d), k(k), s(s), params(params) {}
  MVClustering(uint32_t d, uint32_t k, uint32_t s = 0, uint32_t niters = 5,
               uint32_t max_points_per_centroid_inner_kmeans = 20, uint32_t verbose = 0,
               std::string init = "Random", uint32_t random_seed = 0,
               bool use_weighted_inner_kmeans = false) :
      d(d),
      k(k),
      s(s),
      params(MVClusteringConfig(niters, max_points_per_centroid_inner_kmeans, verbose, init,
                                random_seed, use_weighted_inner_kmeans)) {}
  /* ------------------------------------------------------------------------------------------- */
  // Data given as a PointCloudSet Object: Main implementation
  void train(const PointCloudSet<ChPoint>& data);

  // Raw data given
  void train(uint32_t n, const float* data, const size_t* offsets, const uint32_t* ids) {
    PointCloudSet<ChPoint> points(n, d, data, offsets, ids);
    train(points);
  }
  // Data given as a range type: data[i][j][k]
  template<template<typename> class seqA, template<typename> class seqB,
           template<typename> class seqC>
  void train(const seqA<seqB<seqC<float>>>& data) {
    PointCloudSet<ChPoint> points(data, d, {});
    train(points);
  }

  // Computes cluster ids for each doc point cloud given centroid-point clouds
  void compute_cluster_ids(const PointCloudSet<ChPoint>& points,
                           parlay::sequence<uint32_t>& cluster_ids);
  void compute_cluster_ids_naive(const PointCloudSet<ChPoint>& points,
                                 parlay::sequence<uint32_t>& cluster_ids);
  void compute_cluster_ids_new(const PointCloudSet<ChPoint>& points,
                               parlay::sequence<uint32_t>& cluster_ids);
  // Compute the MV Kmedian cost
  float compute_cost(const PointCloudSet<ChPoint>& points,
                     const parlay::sequence<uint32_t>& cluster_ids) const;
};

/* =======================================Implementation======================================= */

// Data given as a PointCloudSet Object: Main Implementation
template<bool metric>
void MVClustering<metric>::train(const PointCloudSet<ChPoint>& points) {
  uint32_t n = points.size();
  if (s == 0) {  // Default
    auto pc_sizes = parlay::delayed_seq<size_t>(n, [&](size_t i) { return points.get_size(i); });
    s = static_cast<uint32_t>((parlay::reduce(pc_sizes) + n) / n);
  }
  if (params.verbose >= 1)
    std::cout << "[MVClustering] Centroid-Point Cloud Size: " << s << std::endl;
  _iteration_stats.resize(params.niters + 1);
  cluster_ids.resize(n);

  // Step 1: Initialization
  parlay::internal::timer _st;
  _st.start();
  std::cout << "[MVClustering] Seeding algorithm: " << params.init << std::endl;
  if (params.init == "Random") {
    centers = UniformlyRandomMV(points, k, params.seed);
  } else {
    std::cout << "[MVClustering] Error: Invalid seeding algorithm." << std::endl;
    abort();
  }
  _st.stop();
  _iteration_stats[0].centroid_update_time = _st.total_time();
  _st.reset();
  _st.start();
  compute_cluster_ids_new(points, cluster_ids);
  _st.stop();
  _iteration_stats[0].assignment_time = _st.total_time();
  if (params.verbose >= 1) {
    std::cout << "[MVClustering] Seeding time: " << _iteration_stats[0].total_time() << " seconds"
              << std::endl;
    if (params.verbose >= 2) {
      _iteration_stats[0].cost = compute_cost(points, cluster_ids);
      std::cout << "[MVClustering] Seeding cost: " << _iteration_stats[0].cost << std::endl;
    }
  }
  if (params.niters == 0) {
    if (params.verbose >= 1) std::cout << "[MVClustering] Completed: 0 iterations";
    return;
  }
  // Reset to uninitialized fixed size point clouds
  centers = PointCloudSet<ChPoint>(k, s, d);

  // Step 2: Lloyd's Iterations
  parlay::internal::timer _it_timer;
  for (long it = 1; it <= params.niters; it++) {
    // Step 2A: Compute new centers
    _it_timer.start();
    auto id_pt = parlay::tabulate(n, [&](uint32_t i) { return std::make_pair(cluster_ids[i], i); });
    auto grouped = group_by_key_inplace(id_pt);
    parlay::parallel_for(0, grouped.size(), [&](size_t i) {
      auto del_group = parlay::delayed_tabulate(grouped[i].size(),
                                                [&](size_t j) { return grouped[i][j].second; });
      auto data = points.filter_flattened(del_group);
      if (s >= data.size()) {
        centers.set_point_cloud(i, data);
      } else {
        auto new_centers = kmeans_subsample<metric>(
            data, s, params.max_points_per_centroid_inner_kmeans, params.verbose >= 3);
        centers.set_point_cloud(i, new_centers);
      }
    });
    // Sample from input for empty clusters
    if (k - grouped.size() > 0) {
      if (params.verbose >= 2) {
        std::cout << "[MVClustering] " << k - grouped.size()
                  << " clusters empty, sampling from input" << std::endl;
      }
      parlay::parallel_for(grouped.size(), k, [&](size_t i) {
        parlay::sequence<size_t> id = {parlay::hash32(params.seed + i - grouped.size()) % n};
        auto data = points.filter_flattened(id);
        if (s >= data.size()) {
          centers.set_point_cloud(i, data);
        } else {
          auto new_centers = kmeans_subsample<metric>(
              data, s, params.max_points_per_centroid_inner_kmeans, params.verbose >= 3);
          centers.set_point_cloud(i, new_centers);
        }
      });
      params.seed += (k - grouped.size());
    }
    _it_timer.stop();
    _iteration_stats[it].centroid_update_time = _it_timer.total_time();
    _it_timer.reset();
    // Step 2B: Reassign points
    _it_timer.start();
    compute_cluster_ids_new(points, cluster_ids);
    _it_timer.stop();
    _iteration_stats[it].assignment_time = _it_timer.total_time();
    if (params.verbose >= 1) {
      std::cout << "[MVClustering] Iteration " << it << ": "
                << "centroid update time = " << _iteration_stats[it].centroid_update_time
                << ", assignment time = " << _iteration_stats[it].assignment_time;
      if (params.verbose >= 2) {
        _iteration_stats[it].cost = compute_cost(points, cluster_ids);
        std::cout << ", cost = " << _iteration_stats[it].cost;
      }
      std::cout << std::endl;
    }
  }
  if (params.verbose >= 1) {
    std::cout << "[MVClustering] Completed: " << params.niters << " iterations" << std::endl;
  }
}

// Computes cluster ids for each doc point cloud given centroid-point clouds
// Naive Approach: Independently run one-to-one chamfer computation, and find best for each doc.
template<bool metric>
void MVClustering<metric>::compute_cluster_ids_naive(const PointCloudSet<ChPoint>& points,
                                                     parlay::sequence<uint32_t>& cluster_ids) {
  size_t n = points.size();
  parlay::parallel_for(0, n, [&](uint32_t i) {
    auto dist =
        parlay::delayed_tabulate(k, [&](uint32_t j) { return points[i].distance(centers[j]); });
    cluster_ids[i] = parlay::min_element(dist) - dist.begin();
  });
}

// Optimized (eigen)
// TODO: Need to auto optimize this. Useful function to have for many-to-one and many-to-many
// chamfer computation.
template<bool metric>
void MVClustering<metric>::compute_cluster_ids(const PointCloudSet<ChPoint>& points,
                                               parlay::sequence<uint32_t>& cluster_ids) {
  const size_t n = points.size();
  auto points_offsets = points.get_offsets();
  auto centers_offsets = centers.get_offsets();

  // Block Size: Want the entire block to fit in cache
  // TODO: Make generic based on cache size
  const size_t DOC_BLOCK_SIZE = static_cast<size_t>(std::ceil(128.0 / static_cast<float>(s)));

  parlay::parallel_for(0, (n + DOC_BLOCK_SIZE - 1) / DOC_BLOCK_SIZE, [&](size_t doc_block_idx) {
    const size_t doc_start_idx = doc_block_idx * DOC_BLOCK_SIZE;
    const size_t num_docs_in_block = std::min(DOC_BLOCK_SIZE, n - doc_start_idx);
    const size_t doc_end_idx = doc_start_idx + num_docs_in_block;

    const size_t p_float_start_offset = points_offsets[doc_start_idx];
    const size_t num_p_vectors_in_block = (points_offsets[doc_end_idx] - p_float_start_offset) / d;

    if (num_p_vectors_in_block == 0) return;

    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> P_block(
        points.data() + p_float_start_offset, num_p_vectors_in_block, d);
    Eigen::Matrix<float, Eigen::Dynamic, 1> P_block_sq_norms;
    if constexpr (metric) {  // Only for L2
      P_block_sq_norms = P_block.rowwise().squaredNorm();
    }

    // Create a matrix to store all chamfer distances for this block.
    Eigen::MatrixXf chamfer_dists_block(num_docs_in_block, k);

    parlay::parallel_for(0, k, [&](size_t j) {
      const size_t center_vec_count = (centers_offsets[j + 1] - centers_offsets[j]) / d;
      if (center_vec_count == 0) {
        // If a centroid is empty, set its distance to max for all docs in the block
        for (size_t i = 0; i < num_docs_in_block; i++) {
          chamfer_dists_block(i, j) = std::numeric_limits<float>::max();
        }
        return;
      }
      Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> C_j(
          centers.data() + centers_offsets[j], center_vec_count, d);
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> dist_matrix_block_j(
          num_p_vectors_in_block, center_vec_count);
      if constexpr (metric) {  // L2
        dist_matrix_block_j.noalias() = -2 * (P_block * C_j.transpose());
        dist_matrix_block_j.colwise() += P_block_sq_norms;
        Eigen::Matrix<float, 1, Eigen::Dynamic> c_j_sq_norms = C_j.rowwise().squaredNorm();
        dist_matrix_block_j.rowwise() += c_j_sq_norms;
        dist_matrix_block_j = dist_matrix_block_j.cwiseMax(0.0);
      } else {  // IP
        dist_matrix_block_j.noalias() = -1 * (P_block * C_j.transpose());
      }

      for (size_t i = 0; i < num_docs_in_block; ++i) {
        const size_t doc_vec_offset_in_block =
            (points_offsets[doc_start_idx + i] - p_float_start_offset) / d;
        const size_t doc_vec_count =
            (points_offsets[doc_start_idx + i + 1] - points_offsets[doc_start_idx + i]) / d;

        auto doc_dist_slice =
            dist_matrix_block_j.middleRows(doc_vec_offset_in_block, doc_vec_count);
        chamfer_dists_block(i, j) = doc_dist_slice.rowwise().minCoeff().mean();
      }
    });
    for (size_t i = 0; i < num_docs_in_block; ++i) {
      Eigen::Index best_cluster_for_doc;
      chamfer_dists_block.row(i).minCoeff(&best_cluster_for_doc);
      cluster_ids[doc_start_idx + i] = best_cluster_for_doc;
    }
  });
}

template<bool metric>
void MVClustering<metric>::compute_cluster_ids_new(const PointCloudSet<ChPoint>& points,
                                                   parlay::sequence<uint32_t>& cluster_ids) {
  auto results = ManyToMany<PointCloudSet<ChPoint>>::Top(points, centers);
  parlay::parallel_for(0, points.size(), [&](size_t i) { cluster_ids[i] = results[i].first; });
}

// Computes the k-median cost with chamfer distances, given cluster ids
// Not optimized with many-to-many computations, since not necessary for optimized runs.
template<bool metric>
float MVClustering<metric>::compute_cost(const PointCloudSet<ChPoint>& points,
                                         const parlay::sequence<uint32_t>& cluster_ids) const {
  auto distances = parlay::delayed_tabulate(
      points.size(), [&](size_t i) { return points[i].distance(centers[cluster_ids[i]]); });
  return parlay::reduce(distances);
}

using MVClusteringL2 = MVClustering<true>;   // Instantiates for L2 metric (metric = true)
using MVClusteringIP = MVClustering<false>;  // Instantiates for MIPS      (metric = false)

}  // namespace mvsic