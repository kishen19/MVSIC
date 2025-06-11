#pragma once
#include "algorithms/utils/point_range.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"

template<typename T, typename Range>
parlay::sequence<uint32_t> UniformlyRandom(const Range &points, uint32_t k) {
  size_t n = points.size();
  parlay::sequence<uint32_t> centers(k);
  parlay::parallel_for(0, k, [&](size_t i) {
    uint32_t random_center = parlay::hash32(i) % n;
    centers[i] = random_center;
  });
  return centers;
}