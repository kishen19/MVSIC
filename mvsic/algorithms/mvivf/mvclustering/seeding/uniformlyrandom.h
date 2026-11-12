#pragma once

#include "parlay/primitives.h"

#include "mvsic/core/types/point_cloud_set.h"

namespace mvsic {

template<typename ChPoint>
auto UniformlyRandomMV(const PointCloudSet<ChPoint>& points, uint32_t k, uint32_t& seed) {
  size_t n = points.size();
  parlay::sequence<uint32_t> center_ids(k);
  parlay::parallel_for(0, k, [&](uint32_t i) {
    uint32_t random_center = parlay::hash32(seed + i) % n;
    center_ids[i] = random_center;
  });
  seed += k;
  auto center_points =
      parlay::delayed_seq<ChPoint>(k, [&](size_t i) { return points[center_ids[i]]; });
  return PointCloudSet<ChPoint>(center_points, points.get_dims());
}

}  // namespace mvsic