#include "kmeans_utils.h"
#include "parlay/delayed_sequence.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"

std::pair<parlay::sequence<size_t>, parlay::sequence<std::pair<uint32_t, uint32_t>>> symmetrize(
    parlay::sequence<std::pair<uint32_t, uint32_t>> &edges, size_t n) {
  size_t m = edges.size();
  edges.resize(m * 2);
  parlay::parallel_for(0, m, [&](size_t i) { edges[i + m] = {edges[i].second, edges[i].first}; });
  sort_inplace(make_slice(edges));
  auto pred = parlay::delayed_seq<bool>(m * 2, [&](size_t i) {
    if (i != 0 && edges[i].first == edges[i - 1].first && edges[i].second == edges[i - 1].second) {
      return false;
    }
    return true;
  });
  edges = parlay::pack(make_slice(edges), pred);
  parlay::sequence<size_t> offsets(n + 1, m);
  parlay::parallel_for(0, m, [&](size_t i) {
    if (i == 0 || edges[i].first != edges[i - 1].first) {
      offsets[edges[i].first] = i;
    }
  });
  parlay::scan_inclusive_inplace(parlay::make_slice(offsets.rbegin(), offsets.rend()),
                                 parlay::minm<size_t>());
  return {offsets, edges};
}

template<typename T, typename PointTy, typename Range>
parlay::sequence<uint32_t> KSetCover(Range &points, uint32_t k, BuildParams &BP) {
  size_t n = points.size();
  constexpr int num_samples = 1000;

  Graph<uint32_t> G = Graph<uint32_t>(BP.R, n);
  knn_index<PointTy, Range, uint32_t> I(BP);
  stats<uint32_t> BuildStats(G.size());
  I.build_index(G, points, BuildStats);
  parlay::sequence<parlay::sequence<std::pair<uint32_t, T>>> id_and_dist(n);
  double cut = 1.35;
  auto QP = QueryParams(k, BP.L, cut, (long)G.size(), (long)G.max_degree());
  parlay::parallel_for(0, n, [&](size_t i) {
    parlay::sequence<uint32_t> start_points = {I.get_start()};
    auto [pairElts, dist_cmps] =
        beam_search<PointTy, Range, uint32_t>(points[i], G, points, start_points, QP);
    auto [beamElts, visitedElts] = pairElts;
    id_and_dist[i] = beamElts;
  });

  float min_dist = std::numeric_limits<float>::max();
  for (int i = 0; i < num_samples; i++) {
    int u = parlay::hash32(i) % n, v = parlay::hash32(i + n) % n;
    min_dist = std::min(min_dist, points[u].distance(points[v]));
  }
  float min_sse = std::numeric_limits<float>::max();
  parlay::sequence<uint32_t> centers;
  // Compute set cover for edges within dist_thres
  parlay::sequence<size_t> degree(n);
  int round = 0;
  for (T dist_thres = min_dist; round <= 15; dist_thres *= 2, round++) {
    parlay::parallel_for(0, n, [&](size_t i) {
      degree[i] = 0;
      for (size_t j = 0; j < id_and_dist[i].size(); j++) {
        if (id_and_dist[i][j].second <= dist_thres) {
          degree[i]++;
        } else {
          break;
        }
      }
    });

    size_t m;
    parlay::sequence<size_t> offsets;
    std::tie(offsets, m) = scan(degree);
    // printf("m: %zu\n", m);
    parlay::sequence<std::pair<uint32_t, uint32_t>> edges(m);
    parlay::parallel_for(0, n, [&](size_t i) {
      size_t o = offsets[i];
      for (size_t j = 0; j < id_and_dist[i].size(); j++) {
        if (id_and_dist[i][j].second <= dist_thres) {
          edges[o++] = {i, id_and_dist[i][j].first};
        } else {
          break;
        }
      }
      assert((i == n - 1 && o == m) || o == offsets[i + 1]);
    });

    std::tie(offsets, edges) = symmetrize(edges, n);
    std::priority_queue<std::pair<uint32_t, uint32_t>> pq;
    for (size_t i = 0; i < n; i++) {
      degree[i] = offsets[i + 1] - offsets[i];
      pq.push({degree[i], i});
    }
    parlay::sequence<uint32_t> selected_centers;
    parlay::sequence<uint32_t> selected(n);
    parlay::sequence<uint32_t> covered(n);
    uint32_t num_covers = 0;
    while (selected_centers.size() < k) {
      auto [deg, id] = pq.top();
      pq.pop();
      if (degree[id] != deg || selected[id]) {
        continue;
      }
      selected[id] = true;
      selected_centers.push_back(id);
      for (size_t i = offsets[id]; i < offsets[id + 1]; i++) {
        uint32_t u = edges[i].second;
        if (!covered[u]) {
          covered[u] = true;
          num_covers++;
        }
        if (!selected[u]) {
          degree[u]--;
          pq.push({degree[u], u});
        }
      }
    }
    parlay::sequence<float> distances(n, std::numeric_limits<T>::max());
    BatchUpdateDistances(points, distances, selected_centers);
    auto total_distance = reduce(distances);
    if (total_distance < min_sse) {
      min_sse = total_distance;
      centers = selected_centers;
    }
    // printf("num_covers: %u\n", num_covers);
    if (num_covers == n) {
      break;
    }
  }
  return centers;
}