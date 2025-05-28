#pragma once

#include "parlay/primitives.h"
#include "mvc/utils/point_cloud_set.h"


namespace mvivf {

template <typename ChPoint>
std::pair<parlay::sequence<std::pair<size_t, float>>, size_t> get_knn(const ChPoint& q, 
    const PointCloudSet<ChPoint>& points, size_t k){
  size_t dist_cmps = 0;
  auto dists = parlay::tabulate(points.size(), [&](size_t i) {
    return std::pair(q.distance(points[i]), points.get_id(i));
  });
  dist_cmps += dists.size();
  parlay::sort_inplace(dists);
  auto knn = parlay::sequence<std::pair<size_t, float>>::from_function(
      std::min(k, dists.size()), [&](size_t i) { 
    return std::make_pair(dists[i].second, dists[i].first); });
  return std::make_pair(knn, dist_cmps);
}

template <typename Point, typename Range>
std::pair<parlay::sequence<std::pair<size_t, float>>, size_t> get_knn_ids(const Point& q, 
    const Range& points, const parlay::sequence<size_t>& ids, size_t k){
  size_t dist_cmps = 0;
  auto dists = parlay::tabulate(points.size(), [&](size_t i) {
    return std::pair(q.distance(points[i]), ids[i]);
  });
  dist_cmps += dists.size();
  parlay::sort_inplace(dists);
  auto knn = parlay::sequence<std::pair<size_t, float>>::from_function(
      std::min(k, dists.size()), [&](size_t i) { 
    return std::make_pair(dists[i].second, dists[i].first); });
  return std::make_pair(knn, dist_cmps);
}

} // namespace mvivf