#include <functional>
#include <unordered_set>
#include <vector>

#include "algorithms/utils/graph.h"
#include "algorithms/utils/stats.h"
#include "algorithms/utils/types.h"
#include "algorithms/vamana/index.h"
#include "parlay/delayed_sequence.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"

template<typename Grapht>
std::vector<uint32_t> compute_source_set_with_zero(Grapht &G) {
  const size_t n = G.size();

  // --- Step 1: Kosaraju's SCC computation ---
  std::vector<std::vector<uint32_t>> revG(n);
  for (size_t u = 0; u < n; ++u) {
    for (size_t i = 0; i < G[u].size(); i++) {
      uint32_t v = G[u][i];
      revG[v].push_back(u);
    }
  }

  std::vector<bool> visited(n, false);
  std::vector<uint32_t> order;
  std::function<void(uint32_t)> dfs1 = [&](uint32_t u) {
    visited[u] = true;
    for (size_t i = 0; i < G[u].size(); i++) {
      uint32_t v = G[u][i];
      if (!visited[v]) dfs1(v);
    }
    order.push_back(u);
  };
  for (uint32_t u = 0; u < n; ++u) {
    if (!visited[u]) dfs1(u);
  }

  std::fill(visited.begin(), visited.end(), false);
  std::vector<int> component(n, -1);
  int num_components = 0;
  std::function<void(uint32_t)> dfs2 = [&](uint32_t u) {
    visited[u] = true;
    component[u] = num_components;
    for (uint32_t v : revG[u]) {
      if (!visited[v]) dfs2(v);
    }
  };
  for (int i = n - 1; i >= 0; --i) {
    uint32_t u = order[i];
    if (!visited[u]) {
      dfs2(u);
      ++num_components;
    }
  }

  // --- Step 2: Build condensation DAG ---
  std::vector<std::unordered_set<int>> cond_edges(num_components);
  std::vector<int> rep(num_components, -1);  // representative node from each SCC
  for (size_t u = 0; u < n; ++u) {
    int cu = component[u];
    rep[cu] = u;
    for (size_t i = 0; i < G[u].size(); i++) {
      uint32_t v = G[u][i];
      int cv = component[v];
      if (cu != cv) cond_edges[cu].insert(cv);
    }
  }

  // --- Step 3: BFS from SCC(0) in condensation DAG ---
  int c0 = component[0];
  std::vector<bool> reachable(num_components, false);
  std::vector<int> q = {c0};
  reachable[c0] = true;
  for (size_t i = 0; i < q.size(); ++i) {
    int u = q[i];
    for (int v : cond_edges[u]) {
      if (!reachable[v]) {
        reachable[v] = true;
        q.push_back(v);
      }
    }
  }

  // --- Step 4: Add a node from each unreachable SCC ---
  std::vector<uint32_t> source_nodes = {0};
  for (int c = 0; c < num_components; ++c) {
    if (!reachable[c]) {
      source_nodes.push_back(rep[c]);
    }
  }

  return source_nodes;
}

template<typename PointTy, typename DistTy = typename PointTy::distanceType, typename Range>
auto compute_cluster_ids_anns(Range &points, Range &centers, BuildParams &BP,
                              parlay::sequence<uint32_t> cluster_ids = {}) {
  size_t n = points.size();
  size_t k = centers.size();
  // Build Index on new centers
  Graph<uint32_t> G = Graph<uint32_t>(BP.R, k);
  knn_index<PointTy, Range, uint32_t> I(BP);
  stats<uint32_t> BuildStats(G.size());
  I.build_index(G, centers, BuildStats);
#ifdef COMPUTE_RECALL
  // Sanity check starting point can reach every other point
  auto visited = parlay::sequence<uint32_t>(k, 0);
  parlay::sequence<uint32_t> q;
  q.reserve(k);
  q.push_back(I.get_start());
  visited[I.get_start()] = 1;
  size_t i = 0;
  while (i < q.size()) {
    uint32_t u = q[i++];
    for (size_t j = 0; j < G[u].size(); j++) {
      uint32_t v = G[u][j];
      if (visited[v] == 0) {
        visited[v] = 1;
        q.push_back(v);
      }
    }
  }
  std::cout << "Number of visited points: " << q.size() << std::endl;
#endif
  // Find the sources in the directed graph G
  std::vector<uint32_t> source_nodes = {0};  // compute_source_set_with_zero(G);
  std::cout << "Number of source nodes: " << source_nodes.size() << std::endl;
  // Compute Nearest Center for each point
  double cut = 1.35;
  auto QP = QueryParams(1, BP.L, cut, (long)G.size(), (long)G.max_degree());
  // stats<uint32_t> QueryStats(n);
  // QueryStats.clear();
  parlay::sequence<uint32_t> updated_cluster_ids(n);
  parlay::parallel_for(0, n, [&](size_t i) {
    parlay::sequence<uint32_t> start_points(source_nodes.size());
    std::copy(source_nodes.begin(), source_nodes.end(), start_points.begin());
    if (cluster_ids.size() > 0) {
      start_points.push_back(cluster_ids[i]);
    }
    auto [pairElts, dist_cmps] =
        beam_search<PointTy, Range, uint32_t>(points[i], G, centers, start_points, QP);
    auto [beamElts, visitedElts] = pairElts;
    uint32_t min_id = visitedElts[0].first;
    updated_cluster_ids[i] = min_id;
  });
#ifdef COMPUTE_RECALL
  auto exact_cluster_ids = compute_cluster_ids_pairwise<PointTy>(points, centers);
  // Compute Recall
  auto is_correct = parlay::sequence<int>::from_function(
      n, [&](size_t i) { return (updated_cluster_ids[i] == exact_cluster_ids[i]) ? 1 : 0; });
  size_t total_correct = parlay::reduce(is_correct);
  printf("Total correct: %d\n", total_correct);
  double recall = (double)total_correct / n;
  printf("Recall: %f\n", recall);
#endif
  return updated_cluster_ids;
}

template<typename PointTy, typename DistTy = typename PointTy::distanceType, typename Range>
auto lloyds_anns(Range &points, Range &centers, parlay::sequence<uint32_t> &cluster_ids,
                 BuildParams &BP) {

  size_t k = centers.size();
  // Step 1: Compute centroids of each cluster
  auto updated_centers = compute_centroids_from_clusters<PointTy, DistTy>(points, cluster_ids, k);

  // Step 2: Assigning points to new clusters
  auto updated_cluster_ids =
      compute_cluster_ids_anns<PointTy, DistTy>(points, updated_centers, BP, cluster_ids);
  return std::make_pair(updated_centers, updated_cluster_ids);
}