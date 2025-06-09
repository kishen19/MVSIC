#pragma once

#include "mvc/utils/point_cloud_set.h"
#include "parlay/primitives.h"

namespace mvivf {

template <typename ChPoint>
std::pair<parlay::sequence<std::pair<size_t, float>>, size_t>
get_knn(const ChPoint &q, const PointCloudSet<ChPoint> &points, size_t k) {
	size_t dist_cmps = 0;
	auto dists = parlay::tabulate(points.size(), [&](size_t i) {
		return std::pair(q.distance(points[i]), points.get_id(i));
	});
	dist_cmps += dists.size();
	parlay::sort_inplace(dists);
	auto knn = parlay::sequence<std::pair<size_t, float>>::from_function(
			std::min(k, dists.size()), [&](size_t i) {
				return std::make_pair(dists[i].second, dists[i].first);
			});
	return std::make_pair(knn, dist_cmps);
}

template <typename Point, typename Range>
std::pair<parlay::sequence<std::pair<size_t, float>>, size_t>
get_knn_ids(const Point &q, const Range &points,
						const parlay::sequence<std::pair<size_t, size_t>> &ids, size_t k) {
	size_t dist_cmps = 0;
	auto dists = parlay::tabulate(points.size(), [&](size_t i) {
		return std::pair(ids[i].first, q.distance(points[i]));
	});
	dist_cmps += dists.size();
	parlay::sort_inplace(dists);
	// Pick only first copy of same id elements
	auto cutoff_indices =
			parlay::delayed_seq<size_t>(dists.size(), [&](size_t i) {
				return i == 0 || dists[i].first != dists[i - 1].first;
			});
	auto indices = parlay::pack_index(cutoff_indices);
	auto new_dists = parlay::sequence<std::pair<float, size_t>>::from_function(
			indices.size(), [&](size_t i) {
				return std::make_pair(dists[indices[i]].second,
															dists[indices[i]].first);
			});
	parlay::sort_inplace(new_dists);
	auto knn = parlay::sequence<std::pair<size_t, float>>::from_function(
			std::min(k, new_dists.size()), [&](size_t i) {
				return std::make_pair(new_dists[i].second, new_dists[i].first);
			});
	return std::make_pair(knn, dist_cmps);
}

} // namespace mvivf