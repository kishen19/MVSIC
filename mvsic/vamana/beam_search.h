// This code is part of the Problem Based Benchmark Suite (PBBS)
// Copyright (c) 2011 Guy Blelloch and the PBBS team
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights (to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once

#include <algorithm>
#include <functional>
#include <random>
#include <set>
#include <unordered_set>
#include <queue>

#include "parlay/io.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/random.h"
#include "mvsic/core/search_params.h"
// #include "graph.h"

namespace mvsic {
namespace vamana {

// main beam search
template<typename indexType, typename ChPoint, typename PC, class GT>
std::pair<std::pair<parlay::sequence<std::pair<indexType, typename ChPoint::distanceType>>,
                    parlay::sequence<std::pair<indexType, typename ChPoint::distanceType>>>,
          size_t>
beam_search_impl(const ChPoint p, GT &G, const PC &Points,
                 parlay::sequence<indexType> starting_points, const SearchParams &params);

template<typename indexType, typename ChPoint, typename PC, class GT>
std::pair<std::pair<parlay::sequence<std::pair<indexType, typename ChPoint::distanceType>>,
                    parlay::sequence<std::pair<indexType, typename ChPoint::distanceType>>>,
          indexType>
beam_search(const ChPoint p, GT &G, const PC &Points, indexType starting_point,
            const SearchParams &params) {
  parlay::sequence<indexType> start_points = {starting_point};
  return beam_search_impl<indexType>(p, G, Points, start_points, params);
}

template<typename indexType, typename ChPoint, typename PC, class GT>
std::pair<std::pair<parlay::sequence<std::pair<indexType, typename ChPoint::distanceType>>,
                    parlay::sequence<std::pair<indexType, typename ChPoint::distanceType>>>,
          size_t>
beam_search(const ChPoint p, GT &G, const PC &Points, parlay::sequence<indexType> starting_points,
            const SearchParams &params) {
  return beam_search_impl<indexType>(p, G, Points, starting_points, params);
}

// main beam search
template<typename indexType, typename ChPoint, typename PC, class GT>
std::pair<std::pair<parlay::sequence<std::pair<indexType, typename ChPoint::distanceType>>,
                    parlay::sequence<std::pair<indexType, typename ChPoint::distanceType>>>,
          size_t>
beam_search_impl(const ChPoint p, GT &G, const PC &Points,
                 parlay::sequence<indexType> starting_points, const SearchParams &params) {
  if (starting_points.size() == 0) {
    std::cout << "beam search expects at least one start point" << std::endl;
    abort();
  }

  // compare two (node_id,distance) pairs, first by distance and then id if
  // equal
  using distanceType = typename ChPoint::distanceType;
  auto less = [&](std::pair<indexType, distanceType> a, std::pair<indexType, distanceType> b) {
    return a.second < b.second || (a.second == b.second && a.first < b.first);
  };

  // used as a hash filter (can give false negative -- i.e. can say
  // not in table when it is)
  int bits = std::max<int>(10, std::ceil(std::log2(params.L * params.L)) - 2);
  std::vector<indexType> hash_filter(1 << bits, -1);
  auto has_been_seen = [&](indexType a) -> bool {
    int loc = parlay::hash64_2(a) & ((1 << bits) - 1);
    if (hash_filter[loc] == a) return true;
    hash_filter[loc] = a;
    return false;
  };

  // counters
  size_t dist_cmps = 0;
  int remain = 1;
  int num_visited = 0;
  double total;

  // Frontier maintains the closest points found so far and its size
  // is always at most L.  Each entry is a (id,distance) pair.
  // Initialized with starting points and kept sorted by distance.
  std::vector<std::pair<indexType, distanceType>> frontier;
  frontier.reserve(params.L);
  for (auto q : starting_points) {
    auto [dist, d_c] = p.distance_w_cmps(Points[q]);
    dist_cmps += d_c;
    frontier.push_back(std::pair<indexType, distanceType>(q, dist));
  }
  std::sort(frontier.begin(), frontier.end(), less);

  // The subset of the frontier that has not been visited
  // Use the first of these to pick next vertex to visit.
  std::vector<std::pair<indexType, distanceType>> unvisited_frontier(params.L);
  unvisited_frontier[0] = frontier[0];

  // maintains sorted set of visited vertices (id-distance pairs)
  std::vector<std::pair<indexType, distanceType>> visited;
  visited.reserve(2 * params.L);

  // used as temporaries in the loop
  std::vector<std::pair<indexType, distanceType>> new_frontier(
      std::max<size_t>(params.L, starting_points.size()) + G.max_degree());
  std::vector<std::pair<indexType, distanceType>> candidates;
  candidates.reserve(G.max_degree());
  std::vector<indexType> keep;
  keep.reserve(G.max_degree());

  // The main loop.  Terminate beam search when the entire frontier
  // has been visited or have reached max_visit.
  while (remain > 0 && num_visited < Points.size()) {
    // the next node to visit is the unvisited frontier node that is closest to
    // p
    std::pair<indexType, distanceType> current = unvisited_frontier[0];
    G[current.first].prefetch();
    // add to visited set
    visited.insert(std::upper_bound(visited.begin(), visited.end(), current, less), current);
    num_visited++;

    // keep neighbors that have not been visited (using approximate
    // hash). Note that if a visited node is accidentally kept due to
    // approximate hash it will be removed below by the union or will
    // not bump anyone else.
    candidates.clear();
    keep.clear();
    long num_elts = std::min<long>(G[current.first].size(), G.max_degree());
    for (indexType i = 0; i < num_elts; i++) {
      auto a = G[current.first][i];
      if (has_been_seen(a) /*|| Points[a].same_as(p)*/) continue;  // skip if already seen
      keep.push_back(a);
      // Points[a].prefetch(); // TODO: implement prefetch
    }

    // Further filter on whether distance is greater than current
    // furthest distance in current frontier (if full).
    distanceType cutoff =
        ((frontier.size() < params.L) ? (distanceType)std::numeric_limits<int>::max()
                                      : frontier[frontier.size() - 1].second);
    for (auto a : keep) {
      auto [dist, d_c] = p.distance_w_cmps(Points[a]);
      total += dist;
      dist_cmps += d_c;
      // skip if frontier not full and distance too large
      if (dist >= cutoff) continue;
      candidates.push_back(std::pair{a, dist});
    }

    // sort the candidates by distance from p
    std::sort(candidates.begin(), candidates.end(), less);

    // union the frontier and candidates into new_frontier, both are sorted
    auto new_frontier_size = std::set_union(frontier.begin(), frontier.end(), candidates.begin(),
                                            candidates.end(), new_frontier.begin(), less) -
                             new_frontier.begin();

    // trim to at most beam size
    new_frontier_size = std::min<size_t>(params.L, new_frontier_size);

    // if a k is given (i.e. k != 0) then trim off entries that have a
    // distance greater than cut * current-kth-smallest-distance.
    // Only used during query and not during build.
    if (params.k > 0 && new_frontier_size > params.k && ChPoint::is_metric())
      new_frontier_size =
          (std::upper_bound(new_frontier.begin(), new_frontier.begin() + new_frontier_size,
                            std::pair{0, params.cut * new_frontier[params.k].second}, less) -
           new_frontier.begin());

    // copy new_frontier back to the frontier
    frontier.clear();
    for (indexType i = 0; i < new_frontier_size; i++)
      frontier.push_back(new_frontier[i]);

    // get the unvisited frontier (we only care about the first one)
    remain = std::set_difference(frontier.begin(), frontier.end(), visited.begin(), visited.end(),
                                 unvisited_frontier.begin(), less) -
             unvisited_frontier.begin();
  }

  return std::make_pair(std::make_pair(parlay::to_sequence(frontier), parlay::to_sequence(visited)),
                        dist_cmps);
}

// // searches every element in q starting from a randomly selected point
// template<typename Point, typename PointRange, typename indexType>
// parlay::sequence<parlay::sequence<indexType>> beamSearchRandom(PointRange &Query_Points,
//                                                                Graph<indexType> &G,
//                                                                PointRange &Base_Points,
//                                                                stats<indexType> &QueryStats,
//                                                                QueryParams &QP) {
//   if (QP.k > QP.L) {
//     std::cout << "Error: beam search parameter Q = " << QP.L
//               << " same size or smaller than k = " << QP.k << std::endl;
//     abort();
//   }
//   // use a random shuffle to generate random starting points for each query
//   size_t n = G.size();

//   parlay::sequence<parlay::sequence<indexType>> all_neighbors(Query_Points.size());

//   parlay::random_generator gen;
//   std::uniform_int_distribution<long> dis(0, n - 1);
//   auto indices = parlay::tabulate(Query_Points.size(), [&](size_t i) {
//     auto r = gen[i];
//     return dis(r);
//   });

//   parlay::parallel_for(0, Query_Points.size(), [&](size_t i) {
//     parlay::sequence<indexType> neighbors = parlay::sequence<indexType>(QP.k);
//     indexType start = indices[i];
//     parlay::sequence<std::pair<indexType, typename Point::distanceType>> beamElts;
//     parlay::sequence<std::pair<indexType, typename Point::distanceType>> visitedElts;
//     auto [pairElts, dist_cmps] = beam_search(Query_Points[i], G, Base_Points, start, QP);
//     beamElts = pairElts.first;
//     visitedElts = pairElts.second;
//     for (indexType j = 0; j < QP.k; j++) {
//       neighbors[j] = beamElts[j].first;
//     }
//     all_neighbors[i] = neighbors;
//     QueryStats.increment_visited(i, visitedElts.size());
//     QueryStats.increment_dist(i, dist_cmps);
//   });
//   return all_neighbors;
// }

// template<typename Point, typename PointRange, typename indexType>
// parlay::sequence<parlay::sequence<indexType>> searchAll(PointRange &Query_Points,
//                                                         Graph<indexType> &G,
//                                                         PointRange &Base_Points,
//                                                         stats<indexType> &QueryStats,
//                                                         indexType starting_point, QueryParams
//                                                         &QP) {
//   parlay::sequence<indexType> start_points = {starting_point};
//   return searchAll<Point, PointRange, indexType>(Query_Points, G, Base_Points, QueryStats,
//                                                  start_points, QP);
// }

// template<typename Point, typename PointRange, typename indexType>
// parlay::sequence<parlay::sequence<indexType>> searchAll(
//     PointRange &Query_Points, Graph<indexType> &G, PointRange &Base_Points,
//     stats<indexType> &QueryStats, parlay::sequence<indexType> starting_points, QueryParams &QP) {
//   if (QP.k > QP.L) {
//     std::cout << "Error: beam search parameter Q = " << QP.L
//               << " same size or smaller than k = " << QP.k << std::endl;
//     abort();
//   }
//   parlay::sequence<parlay::sequence<indexType>> all_neighbors(Query_Points.size());
//   parlay::parallel_for(0, Query_Points.size(), [&](size_t i) {
//     parlay::sequence<indexType> neighbors = parlay::sequence<indexType>(QP.k);
//     auto [pairElts, dist_cmps] = beam_search(Query_Points[i], G, Base_Points, starting_points,
//     QP); auto [beamElts, visitedElts] = pairElts; for (indexType j = 0; j < QP.k; j++) {
//       neighbors[j] = beamElts[j].first;
//     }
//     all_neighbors[i] = neighbors;
//     QueryStats.increment_visited(i, visitedElts.size());
//     QueryStats.increment_dist(i, dist_cmps);
//   });

//   return all_neighbors;
// }

}  // namespace vamana
}  // namespace mvsic