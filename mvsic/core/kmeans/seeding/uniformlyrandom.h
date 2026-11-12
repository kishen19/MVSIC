// Source: https://github.com/ParAlg/Parallel-KMeans, seeding/uniformlyrandom.h
// Revision: 6cdb5b8649ea254c31a329fcdbbf6537e01452b2 (mvsic branch).
// Paper: UltraKMeans: Fully Scalable and High Quality k-means, PPoPP 2027
// (to appear). https://kishen19.github.io/ (citation in ../README.md).
// See ../README.md for provenance and local modifications.

#pragma once
#include "algorithms/utils/point_range.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"

template <typename T, typename Range>
parlay::sequence<uint32_t> UniformlyRandom(const Range &points, uint32_t k) {
  size_t n = points.size();
  parlay::sequence<uint32_t> centers(k);
  parlay::parallel_for(0, k, [&](size_t i) {
    uint32_t random_center = parlay::hash32(i) % n;
    centers[i] = random_center;
  });
  return centers;
}