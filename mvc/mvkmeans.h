#pragma once

#include "parlay/primitives.h"
#include "utils/chamfer_ip_point.h"
#include "utils/chamfer_l2_point.h"
// #include "utils/euclidian_point.h"
#include "algorithms/utils/euclidian_point.h"
// #include "utils/mips_point.h"
#include "algorithms/utils/mips_point.h"
#include "utils/point_cloud_set.h"
// #include "utils/point_range.h"
#include "algorithms/utils/point_range.h"
#include "lloyds/kmeans.h"
#include "lower_bounds.h"
#include "seeding/uniformlyrandom.h"
// #include "utils/faiss_kmeans.h"

namespace mvivf {

template <typename DistTy, typename Point, typename Range>
auto kmeans_subsample(const Range &data, size_t k, size_t os_rate) {
	size_t n = data.size();
	if (os_rate * k >= n) {
		return kmeans<DistTy, Point>(data, k);
	} else {
		auto sampled_points = parlay::delayed_tabulate(os_rate * k, [&](size_t i) {
			size_t id = parlay::hash32(static_cast<uint32_t>(i)) % n;
			return data[id];
		});
		auto sampled_data = Range(sampled_points, data.get_dims());
		return kmeans<DistTy, Point>(sampled_data, k);
	}
}

constexpr bool L2 = true;
constexpr bool IP = false;

struct MVClusteringParams {
	int iters = 5;
	std::string seeding = "Random";
	bool comp_lb = false;
	bool verbose = false;
	size_t os_rate = 20;
};

template <bool metric> struct MVClustering : MVClusteringParams {
	using ChPoint = std::conditional_t<metric, ChamferL2_Point, ChamferIP_Point>;
	using Point =
			std::conditional_t<metric, Euclidian_Point<float>, Mips_Point<float>>;
	using Range = PointRange<float, Point>;

	size_t d; // Dimension of vectors
	size_t k; // Number of centroid-sets
	size_t s; // Number of points per centroid-set

	PointCloudSet<ChPoint> centers;
	parlay::sequence<size_t> cluster_ids;
	// TODO: stats for each Lloyds iteration

	MVClustering(size_t d, size_t k) noexcept;
	MVClustering(size_t d, size_t k, size_t s) noexcept;
	MVClustering(size_t d, size_t k, const MVClusteringParams &params);
	MVClustering(size_t d, size_t k, size_t s, const MVClusteringParams &params);

	void compute_cluster_ids(const PointCloudSet<ChPoint> &points,
													 parlay::sequence<size_t> &cluster_ids);
	float sum_of_squared_cost(const PointCloudSet<ChPoint> &points,
														const parlay::sequence<size_t> &cluster_ids) const;

	void train(size_t n, const float *data, const size_t *offsets,
						 const size_t *ids);
	void train(const PointCloudSet<ChPoint> &data);
};

template <bool metric>
MVClustering<metric>::MVClustering(size_t d, size_t k) noexcept
		: d(d), k(k), s(0) {}
template <bool metric>
MVClustering<metric>::MVClustering(size_t d, size_t k, size_t s) noexcept
		: d(d), k(k), s(s) {}
template <bool metric>
MVClustering<metric>::MVClustering(size_t d, size_t k,
																	 const MVClusteringParams &params)
		: MVClusteringParams(params), d(d), k(k), s(0) {}
template <bool metric>
MVClustering<metric>::MVClustering(size_t d, size_t k, size_t s,
																	 const MVClusteringParams &params)
		: MVClusteringParams(params), d(d), k(k), s(s) {}

template <bool metric>
void MVClustering<metric>::compute_cluster_ids(
		const PointCloudSet<ChPoint> &points,
		parlay::sequence<size_t> &cluster_ids) {
	size_t n = points.size();
	size_t k = centers.size();
	parlay::parallel_for(0, n, [&](size_t i) {
		auto dist = parlay::delayed_tabulate(
				k, [&](size_t j) { return points[i].distance(centers[j]); });
		cluster_ids[i] = parlay::min_element(dist) - dist.begin();
	});
}

template <bool metric>
float MVClustering<metric>::sum_of_squared_cost(
		const PointCloudSet<ChPoint> &points,
		const parlay::sequence<size_t> &cluster_ids) const {
	auto distances = parlay::delayed_tabulate(points.size(), [&](size_t i) {
		return points[i].distance(centers[cluster_ids[i]]);
	});
	return parlay::reduce(distances);
}

template <bool metric>
void MVClustering<metric>::train(size_t n, const float *data,
																 const size_t *offsets, const size_t *ids) {
	PointCloudSet<ChPoint> points(n, d, data, offsets, ids);
	train(points);
}

template <bool metric>
void MVClustering<metric>::train(const PointCloudSet<ChPoint> &points) {
	size_t n = points.size();
	if (s == 0) {
		auto pc_sizes = parlay::delayed_seq<size_t>(
				points.size(), [&](size_t i) { return points.get_size(i); });
		s = parlay::reduce(pc_sizes) / n;
		if (verbose)
			std::cout << "Average number of embeddings per point: " << s << std::endl;
	}
	// if (comp_lb){
	//   auto lb = lowerbound<Range>(points, k, s);
	//   if (verbose)
	//     std::cout << "Naive Lower Bound: " << lb << std::endl;
	// }
	// Step 1: Initialization
	parlay::internal::timer st;
	st.start();
	cluster_ids.resize(n);
	if (seeding == "Random") {
		centers = UniformlyRandomMV(points, k);
	} else {
		std::cout << "Error: seeding algorithm not specified correctly"
							<< std::endl;
		abort();
	}
	compute_cluster_ids(points, cluster_ids);
	st.stop();

	std::vector<float> lloyds_times; // TODO: move this to a struct
	std::vector<float> costs;
	float seed_cost = sum_of_squared_cost(points, cluster_ids);
	costs.push_back(seed_cost);
	lloyds_times.push_back(st.total_time());
	if (verbose) {
		std::cout << "Seeding cost: " << seed_cost << std::endl;
		std::cout << "Seeding time: " << st.total_time() << " seconds" << std::endl;
	}

	// Step 2: Lloyd's Iteration
	float cost;
	parlay::internal::timer it_timer;
	for (long it = 0; it < iters; it++) {
		it_timer.start();
		// Step 2A: Compute new centers
		auto id_pt = parlay::delayed_seq<std::pair<size_t, size_t>>(
				n, [&](size_t i) { return std::make_pair(cluster_ids[i], i); });
		auto grouped = parlay::group_by_index(id_pt, k);
		parlay::sequence<parlay::sequence<parlay::sequence<float>>> new_centers(
				k, parlay::sequence<parlay::sequence<float>>(
							 s, parlay::sequence<float>(d)));
		// static size_t seed = 42;
		// size_t current_seed = seed;
		// parlay::parallel_for(0, k, [&](size_t i) {
		for (size_t i = 0; i < k; i++) {
			if (grouped[i].size() > 0) {
				auto data = points.filter_flattened(grouped[i]);
				if (s >= data.size()) {
					new_centers[i].resize(data.size());
					new_centers[i] = data;
				} else {
					// new_centers[i] = faiss_kmeans(data, d, s, metric, os_rate);
					auto data_range = Range(data, d);
					new_centers[i] =
							kmeans_subsample<float, Point>(data_range, s, os_rate);
				}
			} else { // Empty Cluster, sample from input
				if (verbose) {
					std::cout << "Cluster " << i << ": empty" << std::endl;
					std::cout << "Sampling from input" << std::endl;
				}
				// parlay::sequence<size_t> id = {parlay::hash32(current_seed+i) % n};
				parlay::sequence<size_t> id = {parlay::hash32(i) % n};
				auto data = points.filter_flattened(id);
				if (s >= data.size()) {
					new_centers[i].resize(data.size());
					new_centers[i] = data;
				} else {
					// new_centers[i] = faiss_kmeans(data, d, s, metric, os_rate);
					auto data_range = Range(data, d);
					new_centers[i] =
							kmeans_subsample<float, Point>(data_range, s, os_rate);
				}
			}
		} //);
		// seed += k;
		centers = PointCloudSet<ChPoint>(new_centers, d, {});
		// Step 2B: Reassign points
		compute_cluster_ids(points, cluster_ids);
		it_timer.stop();
		double round_time = it_timer.total_time();
		lloyds_times.push_back(round_time);
		it_timer.reset();
		if (verbose) {
			cost = sum_of_squared_cost(points, cluster_ids);
			costs.push_back(cost);
			std::cout << "Lloyd's iteration " << it << ": cost = " << cost
								<< ", time = " << round_time << " seconds" << std::endl;
		}
	}
	if (verbose) {
		for (auto c : costs) {
			std::cout << -c << std::endl;
		}
		for (auto t : lloyds_times) {
			std::cout << t << std::endl;
		}
	}
}

template struct MVClustering<true>;	 // Instantiates for L2 metric (metric =
																		 // true)
template struct MVClustering<false>; // Instantiates for MIPS      (metric =
																		 // false)

} // namespace mvivf