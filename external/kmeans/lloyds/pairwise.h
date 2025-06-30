#pragma once
#include <Eigen/Dense>
#include "parlay/delayed_sequence.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"

template<typename PointTy, typename DistTy = typename PointTy::distanceType, typename Range>
Range copyPoints(const Range &points, const parlay::sequence<uint32_t> &center_ids) {
  auto centers = parlay::delayed_seq<PointTy>(center_ids.size(),
                                              [&](size_t i) { return points[center_ids[i]]; });
  int d = points.get_dims();
  return Range(centers, d);
}

template<typename PointTy, typename Seq, typename Range>
void compute_mean(const Seq &points, Range &centers, uint32_t id, uint32_t d) {
  using T = typename PointTy::distanceType;
  assert(points.size() != 0);
  for (size_t i = 0; i < d; i++) {
    auto coords = parlay::delayed_seq<T>(points.size(), [&](size_t j) { return points[j][i]; });
    centers[id][i] = parlay::reduce(coords);
  }
  if (centers[0].is_metric()) {
    T num_points = static_cast<T>(points.size());
    for (size_t i = 0; i < d; i++) {
      centers[id][i] /= num_points;
    }
  } else {
    double sum_sqrs = 0.0;
    for (size_t i = 0; i < d; i++) {
      sum_sqrs += (centers[id][i] * centers[id][i]);
    }
    if (sum_sqrs != 0.0) {
      float sqrt_sum_sqrs = std::sqrt(sum_sqrs);
      for (size_t i = 0; i < d; i++) {
        centers[id][i] /= sqrt_sum_sqrs;
      }
    }
  }
}

template<typename PointTy, typename DistTy = typename PointTy::distanceType, typename Range>
auto compute_centroids_from_clusters(const Range &points, parlay::sequence<uint32_t> &cluster_ids,
                                     size_t k) {
  size_t n = points.size();
  size_t d = points.get_dims();

  auto min_distance_ids = parlay::sequence<std::pair<uint32_t, uint32_t>>::from_function(
      n, [&](size_t i) { return std::make_pair(cluster_ids[i], i); });
  auto grouped = parlay::group_by_index(min_distance_ids, k);
  // parlay::sort_inplace(make_slice(min_distance_ids));
  Range updated_centers(k, d);
  static uint32_t seed = 0;
  // auto sampled = parlay::sequence<uint32_t>(k, k);
  parlay::parallel_for(0, k, [&](size_t i) {
    if (grouped[i].size() > 0) {
      auto subset_points = parlay::delayed_seq<PointTy>(
          grouped[i].size(), [&](size_t j) { return points[grouped[i][j]]; });
      compute_mean<PointTy>(subset_points, updated_centers, i, d);
    } else {
      uint32_t id = parlay::hash32(seed + i) % n;
      // updated_centers[i] = points[id]; // TODO: fix this (Euclidian and Mips Point)
      for (size_t j = 0; j < d; j++) {
        updated_centers[i][j] = points[id][j];
      }
      cluster_ids[id] = i;
      // sampled[i] = i;
    }
  });
  // auto filtered_samples = parlay::filter(sampled, [&](uint32_t i) { return i < k; });
  // std::cout << "Sampled " << filtered_samples.size() << " centers from data." << std::endl;
  seed += k;
  return updated_centers;
}

template<typename PointTy, typename DistTy = typename PointTy::distanceType, typename Range>
auto compute_cluster_ids_pairwise(const Range &points, const Range &centers) {
  size_t n = points.size();
  size_t k = centers.size();
  parlay::sequence<uint32_t> updated_cluster_ids(n);
  parlay::parallel_for(0, n, [&](size_t i) {
    auto dist = parlay::tabulate(k, [&](size_t j) { return points[i].distance(centers[j]); });
    updated_cluster_ids[i] = parlay::min_element(dist) - dist.begin();
  });
  return updated_cluster_ids;
}

template<typename PointTy, typename DistTy = typename PointTy::distanceType, typename Range>
auto compute_cluster_ids_pairwise_blocked(Range &points, Range &centers) {
  const size_t n = points.size();
  const size_t k = centers.size();
  const size_t d = points.get_dims();

  const size_t BLOCK_SIZE = 256;

  Eigen::Map<const Eigen::Matrix<DistTy, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> C(
      &centers[0][0], k, d);
  Eigen::Matrix<DistTy, 1, Eigen::Dynamic> C_sq_norms;
  if (points[0].is_metric()) {
    C_sq_norms = C.rowwise().squaredNorm();
  }
  parlay::sequence<uint32_t> updated_cluster_ids(n);

  parlay::parallel_for(0, (n + BLOCK_SIZE - 1) / BLOCK_SIZE, [&](size_t block_idx) {
    size_t start_row = block_idx * BLOCK_SIZE;
    size_t num_rows = std::min(BLOCK_SIZE, n - start_row);
    Eigen::Map<const Eigen::Matrix<DistTy, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        P_block(&points[start_row][0], num_rows, d);
    Eigen::Matrix<DistTy, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> D_block(num_rows, k);
    if (points[0].is_metric()) {
      D_block.noalias() = -2 * (P_block * C.transpose());
      D_block.rowwise() += C_sq_norms;
    } else {
      D_block.noalias() = -1 * (P_block * C.transpose());
    }
    for (size_t i = 0; i < num_rows; ++i) {
      Eigen::Index min_index;
      D_block.row(i).minCoeff(&min_index);
      updated_cluster_ids[start_row + i] = static_cast<uint32_t>(min_index);
    }
  });
  return updated_cluster_ids;
}

template<typename PointTy, typename DistTy = typename PointTy::distanceType, typename Range>
auto lloyds_pairwise(Range &points, Range &centers, parlay::sequence<uint32_t> &cluster_ids) {
  size_t k = centers.size();

  // Step 1: Compute centroids of each cluster
  auto updated_centers = compute_centroids_from_clusters<PointTy>(points, cluster_ids, k);

  // Step 2: Assigning points to new clusters
  auto updated_cluster_ids = compute_cluster_ids_pairwise_blocked<PointTy>(points, updated_centers);
  // auto updated_cluster_ids = compute_cluster_ids_pairwise<PointTy>(points, updated_centers);

  return std::make_pair(updated_centers, updated_cluster_ids);
}
