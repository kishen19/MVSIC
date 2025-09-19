#pragma once

#include "parlay/primitives.h"

#include "mvsic/core/utils/chamfer_ip_point.h"
#include "mvsic/core/utils/chamfer_l2_point.h"
#include "mvsic/core/utils/point_cloud_set.h"
#include "search_params.h"
#include "index_params.h"

namespace mvsic {

template<bool metric>
class Index {
 public:
  using ChPoint = std::conditional_t<metric, ChamferL2_Point, ChamferIP_Point>;
  // Embedding dimension
  uint32_t d = 0;
  /* ------------------------------Build Functions------------------------------ */
  // Builds the index given PointCloudSet object.
  virtual void build(const PointCloudSet<ChPoint> &points) {}
  // Builds the index given raw data.
  virtual void build(uint32_t n, const float *data, const size_t *offsets, const uint32_t *ids) {
    PointCloudSet<ChPoint> points(n, d, data, offsets, ids);
    build(points);
  }
  /* ------------------------------Search Functions----------------------------- */
  // Returns the top-k point clouds for the query point cloud
  // Output format: < [<id, distance>, ...], # distance comparisons>
  virtual std::pair<parlay::sequence<std::pair<uint32_t, float>>, size_t> search(
      const ChPoint &query, const PointCloudSet<ChPoint> &points,
      const SearchParams &search_params) {}
  // Returns the top-k point clouds for each of the query point clouds
  // Default: runs search in parallel for each query
  virtual std::pair<parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>>, size_t>
  search_all(const PointCloudSet<ChPoint> &query_points, const PointCloudSet<ChPoint> &points,
             const SearchParams &search_params) {
    auto cmps = parlay::sequence<size_t>::uninitialized(query_points.size());
    auto pred = parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>>(query_points.size());
    parlay::parallel_for(0, query_points.size(), [&](size_t i) {
      auto [results, dist_cmps_i] = search(query_points[i], points, search_params);
      pred[i] = results;
      cmps[i] = dist_cmps_i;
    });
    return std::make_pair(pred, parlay::reduce(cmps));
  }
  // Returns some running time stats, specific to the index type
  virtual std::pair<parlay::sequence<std::pair<uint32_t, float>>, size_t> search_with_stats(
      const ChPoint &query, const PointCloudSet<ChPoint> &points, const SearchParams &params) {}
  /* -----------------------------Load/Save Functions-------------------------- */
  // Write the index to a file in disk
  virtual void save(const std::string &filename) {}
  // Read the index from a file in disk
  virtual void load(const std::string &filename, const PointCloudSet<ChPoint> &points) {}
  /* ------------------------------Helper Functions------------------------------ */
  // Only valid for MVIVF
  virtual size_t mean_cluster_size() const noexcept {
    std::cout << "mean_cluster_size() not implemented for this index type" << std::endl;
    return 0;
  }
  virtual size_t max_cluster_size() const noexcept {
    std::cout << "max_cluster_size() not implemented for this index type" << std::endl;
    return 0;
  }
};

// template struct Index<true>;   // Instantiates for L2 metric (metric = true)
// template struct Index<false>;  // Instantiates for MIPS      (metric = false)

}  // namespace mvsic