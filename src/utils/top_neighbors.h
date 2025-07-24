#pragma once

#include "parlay/primitives.h"
#include "point_cloud_set.h"
#include "vqsort_utils.h"

namespace mvivf {

template<typename ChPoint>
std::pair<parlay::sequence<std::pair<size_t, float>>, size_t> get_knn(
    const ChPoint& q, const PointCloudSet<ChPoint>& points, size_t k) {
  size_t dist_cmps = 0;
  parlay::sequence<size_t> cmps(points.size());
#ifdef USE_HWY
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
#else
  auto dists = parlay::tabulate(points.size(), [&](size_t i) {
    auto [dist, d_c] = q.distance_w_cmps(points[i]);
    cmps[i] = d_c;
    return std::pair(points.get_id(i), dist);
  });
  parlay::sort_inplace(dists, [](const auto& a, const auto& b) {
    return a.second < b.second;  // Sort by distance
  });
  auto knn = parlay::sequence<std::pair<size_t, float>>::from_function(
      std::min(k, dists.size()), [&](size_t i) { return dists[i]; });
#endif
  dist_cmps += parlay::reduce(cmps);
  return std::make_pair(knn, dist_cmps);
}

template<typename ChPoint>
size_t get_knn_into_uninitialized(const ChPoint& q, const PointCloudSet<ChPoint>& points, size_t k,
                                  std::pair<size_t, float>* knn) {
  size_t dist_cmps = 0;
  parlay::sequence<size_t> cmps(points.size());

#ifdef USE_HWY
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
#else
  auto dists = parlay::tabulate(points.size(), [&](size_t i) {
    auto [dist, d_c] = q.distance_w_cmps(points[i]);
    cmps[i] = d_c;
    return std::pair(points.get_id(i), dist);
  });
  parlay::sort_inplace(dists, [](const auto& a, const auto& b) {
    return a.second < b.second;  // Sort by distance
  });
  parlay::parallel_for(0, std::min(k, dists.size()), [&](size_t i) { knn[i] = dists[i]; });
#endif
  dist_cmps += parlay::reduce(cmps);
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

}  // namespace mvivf