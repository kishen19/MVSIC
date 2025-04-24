#pragma once
#include "parlay/monoid.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"
#include "seeding/kmeans_utils.h"
#include "seeding/utils.h"

// Return if swap happens
template <typename DistTy, typename Range>
bool multi_swap(const Range &points, parlay::sequence<uint32_t> &center_ids,
                int max_swaps) {
  size_t n = points.size();
  size_t k = center_ids.size();
  bool swapped = false;
  for (size_t c = 0; c < k; c++) {
    auto dist = parlay::delayed_seq<DistTy>(n, [&](size_t i) {
      return reduce(parlay::delayed_seq<DistTy>(k,
                                                [&](size_t j) {
                                                  return points[i].distance(
                                                      points[center_ids[j]]);
                                                }),
                    parlay::minm<DistTy>());
    });
    DistTy best_cost = parlay::reduce(dist);
    std::vector<uint32_t> candidates(max_swaps);
    static int seed = 0;
    for (int i = 0; i < max_swaps; i++) {
      DistTy random_number =
          (double)best_cost * parlay::hash32(seed++) / UINT_MAX;
      candidates[i] = PickRandomCenter(dist, random_number);
    }
    for (auto candidate : candidates) {
      uint32_t tmp = center_ids[c];
      center_ids[c] = candidate;
      double new_cost = parlay::reduce(dist);
      if (new_cost < best_cost) {
        best_cost = new_cost;
        swapped = true;
      } else {
        center_ids[c] = tmp;
      }
    }
  }
  return swapped;
}

// Return if swap happens
template <typename DistTy, typename Range>
bool multi_swap_greedy(const Range &points,
                       parlay::sequence<uint32_t> &center_ids, int max_swaps) {
  size_t n = points.size();
  size_t k = center_ids.size();
  bool swapped = false;
  auto dist = parlay::delayed_seq<DistTy>(n, [&](size_t i) {
    return reduce(parlay::tabulate<DistTy>(k,
                                           [&](size_t j) {
                                             return points[i].distance(
                                                 points[center_ids[j]]);
                                           }),
                  parlay::minm<DistTy>());
  });
  DistTy best_cost = parlay::reduce(dist);
  for (size_t c = 0; c < max_swaps; c++) {
    parlay::sequence<DistTy> dist_increase(k);
    parlay::parallel_for(0, n, [&](size_t i) {
      DistTy closest = std::numeric_limits<DistTy>::max(),
             second_closest = closest;
      uint32_t closest_id = 0, second_closest_id = 0;
      for (size_t j = 0; j < k; j++) {
        DistTy dist = points[i].distance(points[center_ids[j]]);
        if (dist < closest) {
          second_closest = closest;
          second_closest_id = closest_id;
          closest = dist;
          closest_id = j;
        } else if (dist < second_closest) {
          second_closest = dist;
          second_closest_id = j;
        }
      }
      write_add(&dist_increase[closest_id], second_closest - closest);
    });

    uint32_t least_increase_id =
        parlay::min_element(dist_increase) - begin(dist_increase);
    uint32_t tmp = center_ids[least_increase_id];

    static int seed = 0;
    DistTy random_number =
        (double)best_cost * parlay::hash32(seed++) / UINT_MAX;
    uint32_t new_center = PickRandomCenter(dist, random_number);
    center_ids[least_increase_id] = new_center;

    double new_cost = parlay::reduce(dist);
    if (new_cost < best_cost) {
      best_cost = new_cost;
      swapped = true;
    } else {
      center_ids[least_increase_id] = tmp;
    }
  }
  return swapped;
}