#pragma once
#include "../utils/point_cloud_set.h"
#include "parlay/primitives.h"

namespace mvivf {

template<typename ChPoint>
auto UniformlyRandomMV(const PointCloudSet<ChPoint> &points, size_t k) {
  size_t n = points.size();
  parlay::sequence<size_t> center_ids(k);
  parlay::parallel_for(0, k, [&](size_t i) {
    size_t random_center = parlay::hash32(i) % n;
    center_ids[i] = random_center;
  });
  auto center_points =
      parlay::delayed_seq<ChPoint>(k, [&](size_t i) { return points[center_ids[i]]; });
  return PointCloudSet<ChPoint>(center_points, points.get_dims());
}

}  // namespace mvivf