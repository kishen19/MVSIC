#pragma once

#include "parlay/parallel.h"


namespace mvivf {

template <typename ChPoint, typename PointCloud>
auto get_knn(const ChPoint& q, const PointCloud& points, uint32_t k){
  using T = typename ChPoint::distance_type;
  auto dists = parlay::tabulate(points.size(), [&](size_t i) {
    return std::pair(q.distance(points[i]), points.get_id(i)); // Points should have ids
  });
  auto sorted = parlay::sort(dists, [](const auto& a, const auto& b) {
    return a.first < b.first;
  });
  auto knn = parlay::sequence<std::pair<uint32_t, T>>::from_function(std::min((size_t)k, sorted.size()), 
      [&](size_t i) { return std::make_pair(sorted[i].second, sorted[i].first); });
  return knn;
}

} // namespace mvivf