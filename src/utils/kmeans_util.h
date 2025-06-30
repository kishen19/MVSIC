#pragma once

#include "parlay/primitives.h"
#include "algorithms/utils/euclidian_point.h"
#include "algorithms/utils/mips_point.h"
#include "algorithms/utils/point_range.h"
#include "lloyds/kmeans.h"

namespace mvivf {

// Runs kmeans on a subsample of size os_rate*k
template<bool metric>
auto kmeans_subsample(const parlay::sequence<parlay::sequence<float>>& data, size_t k,
                      size_t os_rate, bool verbose = false) {
  using PointTy = std::conditional_t<metric, Euclidian_Point<float>, Mips_Point<float>>;
  using Range = PointRange<float, PointTy>;
  size_t n = data.size();
  size_t dims = data[0].size();
  Range centers;
  if (os_rate * k >= n) {
    Range data_range(data, dims);
    centers = kmeans<float, PointTy>(data_range, k, "UniformlyRandom", "Pairwise", 10, verbose);
  } else {
    auto sampled_points = parlay::delayed_tabulate(os_rate * k, [&](size_t i) {
      size_t id = parlay::hash32(static_cast<uint32_t>(i)) % n;
      return data[id];
    });
    Range sampled_data_range = Range(sampled_points, dims);
    centers =
        kmeans<float, PointTy>(sampled_data_range, k, "UniformlyRandom", "Pairwise", 10, verbose);
  }
  // Convert centers_range to sequence of floats
  parlay::sequence<parlay::sequence<float>> final_centers(k);
  parlay::parallel_for(0, k, [&](size_t i) {
    parlay::sequence<float> center(dims);
    for (size_t j = 0; j < dims; j++) {
      center[j] = centers[i][j];
    }
    final_centers[i] = std::move(center);
  });
  return final_centers;
}

// Runs kmeans on a subsample of size os_rate*k
// Returns centers, cluster_ids, and indices of "valid" clusters
// Validity: Not empty and Not containing more than maxsize many identical clusters
template<bool metric>
auto kmeans_subsample_assign(const parlay::sequence<parlay::sequence<float>>& data, size_t k,
                             size_t os_rate, size_t maxsize, bool verbose = false) {
  using PointTy = std::conditional_t<metric, Euclidian_Point<float>, Mips_Point<float>>;
  using Range = PointRange<float, PointTy>;
  size_t n = data.size();
  size_t dims = data[0].size();
  Range data_range = Range(data, dims);
  Range centers;
  if (os_rate * k >= n) {
    centers = kmeans<float, PointTy>(data_range, k, "UniformlyRandom", "Pairwise", 10, verbose);
  } else {
    auto sampled_points = parlay::delayed_tabulate(os_rate * k, [&](size_t i) {
      size_t id = parlay::hash32(static_cast<uint32_t>(i)) % n;
      return data[id];
    });
    Range sampled_data_range = Range(sampled_points, dims);
    centers =
        kmeans<float, PointTy>(sampled_data_range, k, "UniformlyRandom", "Pairwise", 10, verbose);
  }
  parlay::sequence<uint32_t> cluster_ids_ =
      compute_cluster_ids_pairwise_blocked<PointTy>(data_range, centers);
  auto cluster_ids = parlay::sequence<size_t>::from_function(
      n, [&](size_t i) { return static_cast<size_t>(cluster_ids_[i]); });
  // Remove clusters with all almost-duplicates
  auto id_pt =
      parlay::delayed_tabulate(n, [&](size_t i) { return std::make_pair(cluster_ids[i], i); });
  auto grouped = parlay::group_by_index(id_pt, k);
  parlay::sequence<bool> active(k, true);
  parlay::parallel_for(0, k, [&](size_t i) {
    if (grouped[i].size() == 0) {
      active[i] = false;
    } else {
      auto is_duplicate = parlay::sequence<int>::from_function(grouped[i].size(), [&](size_t j) {
        if (data_range[grouped[i][j]] == data_range[grouped[i][0]]) {
          return 1;
        }
        return 0;
      });
      size_t num_duplicates = parlay::reduce(is_duplicate);
      if (num_duplicates >= grouped[i].size() - 2 && grouped[i].size() > maxsize) {
        active[i] = false;
      }
    }
  });
  auto active_indices = parlay::pack_index(active);
  return std::make_tuple(centers, cluster_ids, active_indices);
}

}  // namespace mvivf