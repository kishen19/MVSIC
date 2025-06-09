#include "algorithms/utils/graph.h"
#include "algorithms/utils/stats.h"
#include "algorithms/utils/types.h"
#include "algorithms/vamana/index.h"
#include "parlay/delayed_sequence.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"

template <typename PointTy, typename DistTy = typename PointTy::distanceType,
					typename Range>
auto compute_cluster_ids_anns(const Range &points, Range &centers,
															BuildParams &BP,
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
	// Compute Nearest Center for each point
	double cut = 1.35;
	auto QP = QueryParams(1, BP.L, cut, (long)G.size(), (long)G.max_degree());
	// stats<uint32_t> QueryStats(n);
	// QueryStats.clear();
	parlay::sequence<uint32_t> updated_cluster_ids(n);
	parlay::parallel_for(0, n, [&](size_t i) {
		parlay::sequence<uint32_t> start_points = {I.get_start()};
		if (cluster_ids.size() > 0) {
			start_points.push_back(cluster_ids[i]);
		}
		auto [pairElts, dist_cmps] = beam_search<PointTy, Range, uint32_t>(
				points[i], G, centers, start_points, QP);
		auto [beamElts, visitedElts] = pairElts;
		uint32_t min_id = visitedElts[0].first;
		updated_cluster_ids[i] = min_id;
	});
#ifdef COMPUTE_RECALL
	auto exact_cluster_ids =
			compute_cluster_ids_pairwise<PointTy>(points, centers);
	// Compute Recall
	auto is_correct = parlay::sequence<int>::from_function(n, [&](size_t i) {
		return (updated_cluster_ids[i] == exact_cluster_ids[i]) ? 1 : 0;
	});
	size_t total_correct = parlay::reduce(is_correct);
	printf("Total correct: %d\n", total_correct);
	double recall = (double)total_correct / n;
	printf("Recall: %f\n", recall);
#endif
	return updated_cluster_ids;
}

template <typename PointTy, typename DistTy = typename PointTy::distanceType,
					typename Range>
auto lloyds_anns(const Range &points, Range &centers,
								 parlay::sequence<uint32_t> &cluster_ids, BuildParams &BP) {

	size_t k = centers.size();
	// Step 1: Compute centroids of each cluster
	auto updated_centers =
			compute_centroids_from_clusters<PointTy, DistTy>(points, cluster_ids, k);

	// Step 2: Assigning points to new clusters
	auto updated_cluster_ids = compute_cluster_ids_anns<PointTy, DistTy>(
			points, updated_centers, BP, cluster_ids);
	return std::make_pair(updated_centers, updated_cluster_ids);
}