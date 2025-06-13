#pragma once

#include "parlay/primitives.h"
#include "lloyds/kmeans.h"

namespace mvivf {

// Runs kmeans on a subsample of size os_rate*k
template<typename DistTy, typename PointTy, typename Range>
auto kmeans_subsample(Range& data, size_t k, size_t os_rate) {
  size_t n = data.size();
  Range centers;
  if (os_rate * k >= n) {
    centers = kmeans<DistTy, PointTy>(data, k);
  } else {
    auto sampled_points = parlay::delayed_tabulate(os_rate * k, [&](size_t i) {
      size_t id = parlay::hash32(static_cast<uint32_t>(i)) % n;
      return data[id];
    });
    auto sampled_data = Range(sampled_points, data.get_dims());
    centers = kmeans<DistTy, PointTy>(sampled_data, k);
  }
  // Convert centers_range to sequence of floats
  parlay::sequence<parlay::sequence<float>> final_centers(k);
  parlay::parallel_for(0, k, [&](size_t i) {
    parlay::sequence<float> center(data.get_dims());
    for (size_t j = 0; j < data.get_dims(); j++) {
      center[j] = centers[i][j];
    }
    final_centers[i] = std::move(center);
  });
  return final_centers;
}

// Runs kmeans on a subsample of size os_rate*k
// Returns centers, cluster_ids, and indices of "valid" clusters
// Validity: Not empty and Not containing more than maxsize many identical clusters
template<typename DistTy, typename Point, typename Range>
auto kmeans_subsample_assign(Range& data, size_t k, size_t os_rate, size_t maxsize, bool metric,
                             bool verbose) {
  size_t n = data.size();
  Range centers;
  if (os_rate * k >= n) {
    centers = kmeans<DistTy, Point>(data, k, "SequentialPlusPlus", "Pairwise", 10, verbose);
  } else {
    auto sampled_points = parlay::delayed_tabulate(os_rate * k, [&](size_t i) {
      size_t id = parlay::hash32(static_cast<uint32_t>(i)) % n;
      return data[id];
    });
    auto sampled_data = Range(sampled_points, data.get_dims());
    centers = kmeans<DistTy, Point>(sampled_data, k, "SequentialPlusPlus", "Pairwise", 10, verbose);
  }
  parlay::sequence<uint32_t> cluster_ids_ =
      compute_cluster_ids_pairwise_blocked<Point>(data, centers);
  auto cluster_ids = parlay::sequence<size_t>::from_function(
      cluster_ids_.size(), [&](size_t i) { return static_cast<size_t>(cluster_ids_[i]); });
  // Remove clusters with all almost-duplicates
  auto id_pt = parlay::delayed_tabulate(
      data.size(), [&](size_t i) { return std::make_pair(cluster_ids[i], i); });
  auto grouped = parlay::group_by_index(id_pt, k);
  parlay::sequence<bool> active(k, true);
  parlay::parallel_for(0, k, [&](size_t i) {
    if (grouped[i].size() == 0) {
      active[i] = false;
    } else {
      // average distance to center
      auto dists = parlay::delayed_tabulate(
          grouped[i].size(), [&](size_t j) { return centers[i].distance(data[grouped[i][j]]); });
      float avg_dist = parlay::reduce(dists);
      if (avg_dist == 0.0 && grouped[i].size() > maxsize) {
        active[i] = false;
      }
    }
  });
  auto active_indices = parlay::pack_index(active);
  return std::make_tuple(centers, cluster_ids, active_indices);
}

}  // namespace mvivf