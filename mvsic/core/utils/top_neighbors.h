#pragma once

#include "parlay/primitives.h"
#include "point_cloud_set.h"
#include "util.h"

namespace mvsic {

template<typename ChPoint>
std::pair<parlay::sequence<std::pair<uint32_t, float>>, size_t> get_knn(
    const ChPoint& q, const PointCloudSet<ChPoint>& points, size_t k) {
  size_t dist_cmps = 0;
  parlay::sequence<size_t> cmps(points.size());
  auto dists = parlay::tabulate(points.size(), [&](size_t i) {
    auto [dist, d_c] = q.distance_w_cmps(points[i]);
    cmps[i] = d_c;
    return std::pair(points.get_id(i), dist);
  });
  parlay::sort_inplace(dists, [](const auto& a, const auto& b) {
    return a.second < b.second;  // Sort by distance
  });
  auto knn = parlay::sequence<std::pair<uint32_t, float>>::from_function(
      std::min(k, dists.size()), [&](size_t i) { return dists[i]; });
  dist_cmps += parlay::reduce(cmps);
  return std::make_pair(knn, dist_cmps);
}

template<typename ChPoint>
size_t get_knn_into_uninitialized(const ChPoint& q, const PointCloudSet<ChPoint>& points,
                                  std::pair<uint32_t, float>* knn) {
  size_t dist_cmps = 0;
  parlay::sequence<size_t> cmps(points.size());
  auto dists = parlay::tabulate(points.size(), [&](size_t i) {
    auto [dist, d_c] = q.distance_w_cmps(points[i]);
    cmps[i] = d_c;
    return std::pair(points.get_id(i), dist);
  });
  parlay::parallel_for(0, dists.size(), [&](size_t i) { knn[i] = dists[i]; });
  dist_cmps += parlay::reduce(cmps);
  return dist_cmps;
}

// Cluster:
// - .centroids, .point_ids, .LUT
template<typename ChPoint, typename Cluster>
size_t get_knn_via_centroids_into_uninitialized(const ChPoint& q, const Cluster& cluster, size_t k,
                                                std::pair<size_t, float>* knn) {
  // Compute dot product matrix
  size_t dim = q.get_dims();
  auto& centroids = cluster.centroids;
  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      M_centroids(centroids.data(), centroids.size(), dim);
  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> M_q(
      q.data(), q.size(), dim);
  Eigen::MatrixXf inner_product_matrix = -1.0f * M_q * M_centroids.transpose();
  size_t dist_cmps = q.size() + centroids.size();
  // Compute Chamfer distance estimates
  auto& LUT = cluster.LUT;
  auto& point_ids = cluster.point_ids;
  auto estimates = parlay::tabulate(point_ids.size(), [&](size_t i) {
    auto qdists = parlay::sequence<float>::from_function(q.size(), [&](size_t j) {
      auto dots = parlay::delayed_seq<float>(
          LUT[i].size(), [&](size_t c) { return inner_product_matrix(j, LUT[i][c]); });
      return parlay::reduce(dots, parlay::minm<float>());
    });
    return std::pair(point_ids[i], parlay::reduce(qdists) / q.size());
  });
  parlay::sort_inplace(estimates, [](const auto& a, const auto& b) {
    return a.second < b.second;  // Sort by distance
  });
  parlay::parallel_for(0, std::min(k, estimates.size()), [&](size_t i) { knn[i] = estimates[i]; });
  return dist_cmps;
}

template<typename Point, typename Range>
std::pair<parlay::sequence<std::pair<size_t, float>>, size_t> get_knn_ids(
    const Point& q, const Range& points, const parlay::sequence<std::pair<size_t, size_t>>& ids,
    size_t k) {
  parlay::sequence<size_t> cmps(points.size());
  auto dists = parlay::tabulate(points.size(), [&](size_t i) {
    auto [dist, d_c] = q.distance_w_cmps(points[i]);
    cmps[i] = d_c;
    return std::pair(ids[i].first, dist);
  });
  size_t dist_cmps = parlay::reduce(cmps);
  parlay::sort_inplace(dists);
  // Pick only first copy of same id elements
  auto cutoff_indices = parlay::delayed_seq<size_t>(
      dists.size(), [&](size_t i) { return i == 0 || dists[i].first != dists[i - 1].first; });
  auto indices = parlay::pack_index(cutoff_indices);
  auto new_dists = parlay::sequence<std::pair<float, size_t>>::from_function(
      indices.size(),
      [&](size_t i) { return std::make_pair(dists[indices[i]].second, dists[indices[i]].first); });
  parlay::sort_inplace(new_dists);
  auto knn = parlay::sequence<std::pair<size_t, float>>::from_function(
      std::min(k, new_dists.size()),
      [&](size_t i) { return std::make_pair(new_dists[i].second, new_dists[i].first); });
  return std::make_pair(knn, dist_cmps);
}

// Versions using VQSort
template<typename ChPoint>
std::pair<parlay::sequence<std::pair<size_t, float>>, size_t> get_knn_hwy(
    const ChPoint& q, const PointCloudSet<ChPoint>& points, size_t k) {
  size_t dist_cmps = 0;
  parlay::sequence<size_t> cmps(points.size());
  auto dists = parlay::tabulate(points.size(), [&](size_t i) {
    auto [dist, d_c] = q.distance_w_cmps(points[i]);
    cmps[i] = d_c;
    return packFloatAndInt(dist, points.get_id(i));
  });
  if (k < 0.9 * dists.size()) {
    VQPartialSort(dists.begin(), dists.end(), k);
  } else {
    VQSort(dists.begin(), dists.end());
  }
  auto knn = parlay::sequence<std::pair<size_t, float>>::from_function(
      std::min(k, dists.size()), [&](size_t i) {
        auto [ext_float, ind] = unpackDouble2(dists[i]);
        return std::make_pair(ind, ext_float);
      });
  dist_cmps += parlay::reduce(cmps);
  return std::make_pair(knn, dist_cmps);
}

template<typename ChPoint>
size_t get_knn_hwy_into_uninitialized(const ChPoint& q, const PointCloudSet<ChPoint>& points,
                                      size_t k, std::pair<size_t, float>* knn) {
  size_t dist_cmps = 0;
  parlay::sequence<size_t> cmps(points.size());
  auto dists = parlay::tabulate(points.size(), [&](size_t i) {
    auto [dist, d_c] = q.distance_w_cmps(points[i]);
    cmps[i] = d_c;
    return packFloatAndInt(dist, points.get_id(i));
  });
  if (k < 0.9 * dists.size()) {
    VQPartialSort(dists.begin(), dists.end(), k);
  } else {
    VQSort(dists.begin(), dists.end());
  }
  parlay::parallel_for(0, std::min(k, dists.size()), [&](size_t i) {
    auto [ext_float, ind] = unpackDouble2(dists[i]);
    knn[i] = std::make_pair(ind, ext_float);
  });
  dist_cmps += parlay::reduce(cmps);
  return dist_cmps;
}

}  // namespace mvsic