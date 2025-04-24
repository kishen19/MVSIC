#pragma once
#include "parlay/delayed_sequence.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"

template <typename R, typename T>
uint32_t PickRandomCenter(const R &distances, T cutoff) {
  size_t n = distances.size();
  for (size_t i = 0; i < n; i++) {
    if (distances[i] != 0) {
      cutoff -= distances[i];
      if (cutoff <= 0) {
        return i;
      }
    }
  }
  return n - 1;
}

template <typename T, typename Range>
T BatchUpdateDistances(const Range &points, parlay::sequence<T> &distances,
                       const parlay::sequence<uint32_t> &new_center_ids) {
  parlay::parallel_for(0, points.size(), [&](size_t i) {
    auto new_distances =
        parlay::delayed_seq<T>(new_center_ids.size(), [&](size_t j) {
          return points[i].distance(points[new_center_ids[j]]);
        });
    T smallest_new_distance = reduce(new_distances, parlay::minm<T>());
    distances[i] = std::min(distances[i], smallest_new_distance);
  });
  return parlay::reduce(distances);
}

template <typename T>
parlay::sequence<uint32_t> PickRandomCenters(
    const parlay::sequence<T> &distances, size_t round_id,
    T sum_squared_distances, size_t num_centers_to_add) {
  size_t n = distances.size();
  parlay::random_generator parallel_generator(round_id * n);
  auto pred = parlay::delayed_seq<bool>(n, [&](size_t i) {
    double sampling_probability =
        num_centers_to_add * distances[i] / sum_squared_distances;

    std::bernoulli_distribution bernoulli_generator(sampling_probability);
    // Get a random_seed from the parallel_generator
    auto random_seed = parallel_generator[i];
    if (sampling_probability > 1 ||
        (sampling_probability > 0 && bernoulli_generator(random_seed))) {
      return true;
    } else {
      return false;
    }
  });
  return pack_index<uint32_t>(pred);
}
