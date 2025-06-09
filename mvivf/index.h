#pragma once

#include "mvc/utils/chamfer_ip_point.h"
#include "mvc/utils/chamfer_l2_point.h"
#include "mvc/utils/point_cloud_set.h"
#include "parlay/primitives.h"
#include "utils/search_params.h"

namespace mvivf {

template <bool metric> struct Index {
	using ChPoint = std::conditional_t<metric, ChamferL2_Point, ChamferIP_Point>;

	// Embedding dimension
	size_t d = 0;

	// Builds the index given PointCloudSet object.
	virtual void build(const PointCloudSet<ChPoint> &points) {}

	// Builds the index given raw data.
	virtual void build(size_t n, const float *data, const size_t *offsets,
										 const size_t *ids) {
		PointCloudSet<ChPoint> points(n, d, data, offsets, ids);
		build(points);
	}

	// Returns the top-k point clouds for the query point cloud
	// Output format: < [<id, distance>, ...], # distance comparisons>
	virtual std::pair<parlay::sequence<std::pair<size_t, float>>, size_t>
	search(const ChPoint &query, const PointCloudSet<ChPoint> &points,
				 const SearchParams &params) {}

	// Write the index to a file in disk
	virtual void save(const std::string &filename) {}

	// Read the index from a file in disk
	virtual void load(const std::string &filename,
										const PointCloudSet<ChPoint> &points) {}
};

template struct Index<true>;	// Instantiates for L2 metric (metric = true)
template struct Index<false>; // Instantiates for MIPS      (metric = false)

} // namespace mvivf