#pragma once

#include <Eigen/Dense>
#include "parlay/primitives.h"

#include "src/utils/chamfer_ip_point.h"
#include "src/utils/chamfer_l2_point.h"
#include "src/utils/kmeans_util.h"
#include "src/utils/point_cloud_set.h"
#include "seeding/uniformlyrandom.h"
// #include "lower_bounds.h"

namespace mvivf {

/* Params Type */
struct MVClusteringParams {
  int iters = 5;                   // Number of Outer Lloyd's Iterations
  std::string seeding = "Random";  // Seeding Algorithm
  size_t os_rate = 20;             // Oversampling rate for Inner Kmeans
  bool verbose = false;            // Print debug statements
  bool comp_lb = false;            // Compute a naive lower bound (TODO: fix)
};

/* Main Multi-Vector Clustering Class */
template<bool metric>
class MVClustering : MVClusteringParams {
 public:
  using ChPoint = std::conditional_t<metric, ChamferL2_Point, ChamferIP_Point>;

  size_t d;  // Dimension of vectors
  size_t k;  // Number of centroid-sets
  double s;  // Number of points per centroid-set

  PointCloudSet<ChPoint> centers;
  parlay::sequence<size_t> cluster_ids;
  // TODO: stats type for each Lloyds iteration

  MVClustering(size_t d, size_t k) noexcept;
  MVClustering(size_t d, size_t k, double s) noexcept;
  MVClustering(size_t d, size_t k, const MVClusteringParams &params);
  MVClustering(size_t d, size_t k, double s, const MVClusteringParams &params);

  // Naive: Computes the cluster ids of each input doc, given centers
  void compute_cluster_ids(const PointCloudSet<ChPoint> &points,
                           parlay::sequence<size_t> &cluster_ids);
  // Optimized (eigen): Computes the cluster ids of each input doc, given centers
  void compute_cluster_ids_blocked(const PointCloudSet<ChPoint> &points,
                                   parlay::sequence<size_t> &cluster_ids);
  // Utility to compute the MV Kmeans cost
  float sum_of_squared_cost(const PointCloudSet<ChPoint> &points,
                            const parlay::sequence<size_t> &cluster_ids) const;
  // Raw data given
  void train(size_t n, const float *data, const size_t *offsets, const size_t *ids);
  // Data given as a range type
  template<template<typename> class seqA, template<typename> class seqB,
           template<typename> class seqC>
  void train(const seqA<seqB<seqC<float>>> &data);
  // Data given as a PointCloudSet Object
  void train(const PointCloudSet<ChPoint> &data);
};

/* -----------------------------------------Implementation-----------------------------------------*/

template<bool metric>
MVClustering<metric>::MVClustering(size_t d, size_t k) noexcept : d(d), k(k), s(0.0) {}
template<bool metric>
MVClustering<metric>::MVClustering(size_t d, size_t k, double s) noexcept : d(d), k(k), s(s) {}
template<bool metric>
MVClustering<metric>::MVClustering(size_t d, size_t k, const MVClusteringParams &params) :
    MVClusteringParams(params), d(d), k(k), s(0.0) {}
template<bool metric>
MVClustering<metric>::MVClustering(size_t d, size_t k, double s, const MVClusteringParams &params) :
    MVClusteringParams(params), d(d), k(k), s(s) {}

template<bool metric>
void MVClustering<metric>::compute_cluster_ids(const PointCloudSet<ChPoint> &points,
                                               parlay::sequence<size_t> &cluster_ids) {
  size_t n = points.size();
  parlay::parallel_for(0, n, [&](size_t i) {
    auto dist =
        parlay::delayed_tabulate(k, [&](size_t j) { return points[i].distance(centers[j]); });
    cluster_ids[i] = parlay::min_element(dist) - dist.begin();
  });
}

template<bool metric>
void MVClustering<metric>::compute_cluster_ids_blocked(const PointCloudSet<ChPoint> &points,
                                                       parlay::sequence<size_t> &cluster_ids) {
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
float MVClustering<metric>::sum_of_squared_cost(const PointCloudSet<ChPoint> &points,
                                                const parlay::sequence<size_t> &cluster_ids) const {
  auto distances = parlay::delayed_tabulate(
      points.size(), [&](size_t i) { return points[i].distance(centers[cluster_ids[i]]); });
  return parlay::reduce(distances);
}

template<bool metric>
void MVClustering<metric>::train(size_t n, const float *data, const size_t *offsets,
                                 const size_t *ids) {
  PointCloudSet<ChPoint> points(n, d, data, offsets, ids);
  train(points);
}

template<bool metric>
template<template<typename> class seqA, template<typename> class seqB,
         template<typename> class seqC>
void MVClustering<metric>::train(const seqA<seqB<seqC<float>>> &data) {
  PointCloudSet<ChPoint> points(data, d, {});
  train(points);
}

template<bool metric>
void MVClustering<metric>::train(const PointCloudSet<ChPoint> &points) {
  size_t n = points.size();
  auto pc_sizes = parlay::delayed_seq<size_t>(n, [&](size_t i) { return points.get_size(i); });
  size_t centroid_size = s * parlay::reduce(pc_sizes) / n;
  if (verbose)
    std::cout << "Average number of embeddings per point: " << centroid_size << std::endl;
  // if (comp_lb){
  //   auto lb = lowerbound<Range>(points, k, s);
  //   if (verbose)
  //     std::cout << "Naive Lower Bound: " << lb << std::endl;
  // }
  // Step 1: Initialization
  parlay::internal::timer st;
  st.start();
  cluster_ids.resize(n);
  if (seeding == "Random") {
    centers = UniformlyRandomMV(points, k);
  } else {
    std::cout << "Error: seeding algorithm not specified correctly" << std::endl;
    abort();
  }
  // compute_cluster_ids(points, cluster_ids);
  compute_cluster_ids_blocked(points, cluster_ids);
  st.stop();

  std::vector<float> lloyds_times;  // TODO: move this to a struct
  std::vector<float> costs;
  float seed_cost = sum_of_squared_cost(points, cluster_ids);
  costs.push_back(seed_cost);
  lloyds_times.push_back(st.total_time());
  if (verbose) {
    std::cout << "Seeding cost: " << seed_cost << std::endl;
    std::cout << "Seeding time: " << st.total_time() << " seconds" << std::endl;
  }

  // Step 2: Lloyd's Iteration
  float cost;
  parlay::internal::timer it_timer;
  for (long it = 0; it < iters; it++) {
    it_timer.start();
    // Step 2A: Compute new centers
    auto id_pt = parlay::delayed_seq<std::pair<size_t, size_t>>(
        n, [&](size_t i) { return std::make_pair(cluster_ids[i], i); });
    auto grouped = parlay::group_by_index(id_pt, k);
    parlay::sequence<parlay::sequence<parlay::sequence<float>>> new_centers(
        k, parlay::sequence<parlay::sequence<float>>(centroid_size, parlay::sequence<float>(d)));
    // static size_t seed = 42;
    // size_t current_seed = seed;
    parlay::parallel_for(0, k, [&](size_t i) {
      if (grouped[i].size() > 0) {
        auto data = points.filter_flattened(grouped[i]);
        if (centroid_size >= data.size()) {
          new_centers[i].resize(data.size());
          new_centers[i] = data;
        } else {
          // new_centers[i] = faiss_kmeans(data, d, centroid_size, metric, os_rate);
          new_centers[i] = kmeans_subsample<metric>(data, centroid_size, os_rate, verbose);
        }
      } else {  // Empty Cluster, sample from input
        if (verbose) {
          std::cout << "Cluster " << i << ": empty" << std::endl;
          std::cout << "Sampling from input" << std::endl;
        }
        // parlay::sequence<size_t> id = {parlay::hash32(current_seed+i) % n};
        parlay::sequence<size_t> id = {parlay::hash32(i) % n};
        auto data = points.filter_flattened(id);
        if (centroid_size >= data.size()) {
          new_centers[i].resize(data.size());
          new_centers[i] = data;
        } else {
          // new_centers[i] = faiss_kmeans(data, d, centroid_size, metric, os_rate);
          new_centers[i] = kmeans_subsample<metric>(data, centroid_size, os_rate, verbose);
        }
      }
    });
    // seed += k;
    centers = PointCloudSet<ChPoint>(new_centers, d, {});
    // Step 2B: Reassign points
    // compute_cluster_ids(points, cluster_ids);
    compute_cluster_ids_blocked(points, cluster_ids);
    it_timer.stop();
    double round_time = it_timer.total_time();
    lloyds_times.push_back(round_time);
    it_timer.reset();
    if (verbose) {
      cost = sum_of_squared_cost(points, cluster_ids);
      costs.push_back(cost);
      std::cout << "Lloyd's iteration " << it << ": cost = " << cost << ", time = " << round_time
                << " seconds" << std::endl;
    }
  }
}

template struct MVClustering<true>;   // Instantiates for L2 metric (metric = true)
template struct MVClustering<false>;  // Instantiates for MIPS      (metric = false)

}  // namespace mvivf