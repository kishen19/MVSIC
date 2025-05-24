#pragma once
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"
#include "../utils/pointcloudset.h"

// TODO define namespace

template <typename ChPoint>
auto UniformlyRandomMV(PointCloudSet<ChPoint>& points, size_t k) {
  size_t n = points.size();
  parlay::sequence<size_t> center_ids(k);
  parlay::parallel_for(0, k, [&](size_t i) {
    size_t random_center = parlay::hash32(i) % n;
    center_ids[i] = random_center;
  });
  auto center_points = parlay::delayed_seq<ChPoint>(k,
    [&](size_t i) { return points[center_ids[i]]; });
  return PointCloudSet<ChPoint>(center_points, points.get_dims());
}