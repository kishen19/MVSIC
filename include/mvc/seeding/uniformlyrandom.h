#pragma once
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"
#include "mvc/utils/pointcloud.h"

template <typename T, typename PointCloud>
auto UniformlyRandom(const PointCloud &points, uint32_t k) {
  size_t n = points.size();
  parlay::sequence<uint32_t> center_ids(k);
  parlay::parallel_for(0, k, [&](size_t i) {
    uint32_t random_center = parlay::hash32(i) % n;
    center_ids[i] = random_center;
  });
  auto center_points = parlay::delayed_seq<T>(k, [&](size_t i) {
    return points[center_ids[i]];
  });
  return PointCloud(center_points);
}