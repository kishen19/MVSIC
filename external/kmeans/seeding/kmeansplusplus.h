#pragma once
#include <random>

#include "algorithms/utils/point_range.h"
#include "kmeans_utils.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"

template <typename T, typename Range>
T UpdateDistances(const Range &points, parlay::sequence<T> &distances,
                  uint32_t new_center_id) {
  // for (size_t i = 0; i < points.size(); i++) {
  parlay::parallel_for(0, points.size(), [&](size_t i) {
    T new_distance = points[i].distance(points[new_center_id]);
    distances[i] = std::min(distances[i], new_distance);
    // }
  });
  return parlay::reduce(distances);
}

template <typename T, typename Range>
parlay::sequence<uint32_t> SequentialPlusPlus(
    const Range &points, uint32_t k,
    const parlay::sequence<T> &weights = parlay::sequence<T>{}) {
  size_t n = points.size();

  int seed = 0;
  std::mt19937 generator(seed);
  parlay::sequence<T> distances(n, std::numeric_limits<T>::max());

  parlay::sequence<uint32_t> centers(k);
  T sum_squared_distances = 0;
  for (size_t i = 0; i < k; i++) {
    uint32_t random_center;
    if (i == 0) {
      std::uniform_int_distribution<uint32_t> distribution(0, n - 1);
      random_center = distribution(generator);
    } else {
      std::uniform_real_distribution<double> distribution(0, sum_squared_distances);
      T random_number = distribution(generator);
      random_center = PickRandomCenter(distances, random_number);
    }
    centers[i] = random_center;
    sum_squared_distances = UpdateDistances(points, distances, random_center);
  }
  return centers;
}
