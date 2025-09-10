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
  // Builds the index given PointCloudSet object.
  virtual void build(const PointCloudSet<ChPoint> &points) {}
  // Builds the index given raw data.
  virtual void build(uint32_t n, const float *data, const size_t *offsets, const uint32_t *ids) {
    PointCloudSet<ChPoint> points(n, d, data, offsets, ids);
    build(points);
  }
  // Returns the top-k point clouds for the query point cloud
  // Output format: < [<id, distance>, ...], # distance comparisons>
  virtual std::pair<parlay::sequence<std::pair<uint32_t, float>>, size_t> search(
      const ChPoint &query, const PointCloudSet<ChPoint> &points,
      const SearchParams &search_params) {}
  // Write the index to a file in disk
  virtual void save(const std::string &filename) {}
  // Read the index from a file in disk
  virtual void load(const std::string &filename, const PointCloudSet<ChPoint> &points) {}
};

template struct Index<true>;   // Instantiates for L2 metric (metric = true)
template struct Index<false>;  // Instantiates for MIPS      (metric = false)

}  // namespace mvsic