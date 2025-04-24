#pragma once
#include "kmeans_utils.h"
#include "kmeansplusplus.h"
#include "parlay/delayed_sequence.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"
#include "utils.h"

template <typename T, typename Range>
parlay::sequence<uint32_t> ParallelPlusPlus(const Range &points, uint32_t k) {
  size_t n = points.size();
  parlay::sequence<T> distances(n, std::numeric_limits<T>::max());

  parlay::sequence<uint32_t> centers;
  T sum_squared_distances = 0;
  size_t selected_centers = 0;

  // TODO: add num_iterations and oversampling_factor to command line options
  constexpr size_t num_iterations = 8;
  constexpr double oversampling_factor = 1;
  size_t num_centers_to_add = k * oversampling_factor;

  for (size_t round_id = 0; round_id < num_iterations || selected_centers < k;
       round_id++) {
    parlay::sequence<uint32_t> new_centers;
    if (round_id == 0) {
      uint32_t random_center = parlay::hash32(round_id) % n;
      new_centers.push_back(random_center);
    } else {
      new_centers = PickRandomCenters(
          distances, round_id, sum_squared_distances, num_centers_to_add);
    }
    centers.resize(centers.size() + new_centers.size());
    parlay::parallel_for(0, new_centers.size(), [&](size_t i) {
      uint32_t c = new_centers[selected_centers + i];
      centers[selected_centers + i] = c;
    });
    // resize if more than k centers are selected
    if (selected_centers + new_centers.size() > k) {
      new_centers.resize(k - selected_centers);
    }
    selected_centers += new_centers.size();
    sum_squared_distances =
        BatchUpdateDistances(points, distances, new_centers);
  }
  // parlay::sequence<T> weights(centers.size());
  // parlay::parallel_for(0, n, [&](size_t i) {
  //   uint32_t smallest_distance_id =
  //       parlay::reduce(parlay::iota(centers.size()),
  //                      parlay::make_monoid(
  //                          [&](uint32_t a, uint32_t b) {
  //                            return points[i].distance(points[a]) <
  //                                   points[i].distance(points[b]);
  //                          },
  //                          0));
  //   write_max(&weights[smallest_distance_id], (T)1);
  // });
  // centers = SequentialPlusPlus<T>(centers, k, weights);
  return centers;
}
