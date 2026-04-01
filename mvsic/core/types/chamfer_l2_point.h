#pragma once

#include <Eigen/Core>
#include "parlay/primitives.h"
#include "mvsic/core/distance_measures/one_to_one.h"

#include "l2_point.h"

namespace mvsic {

// Computes the Chamfer L2 distance, given two point clouds
float chamfer_l2_distance(const float* a, uint32_t n_a, const float* b, uint32_t n_b, uint32_t dim);

/* Chamfer L2 Point Cloud Type */
struct ChamferL2_Point {
 private:
  uint32_t n = 0;
  uint32_t dims = 0;
  uint32_t aligned_dims = 0;
  float* values = nullptr;
  uint32_t id = std::numeric_limits<uint32_t>::max();

 public:
  using distanceType = float;
  ChamferL2_Point() noexcept {}
  // Non-owning version
  ChamferL2_Point(uint32_t n, uint32_t dims, float* values, uint32_t id) noexcept :
      n(n), dims(dims), values(values), id(id) {}

  // Returns number of embeddings in the point cloud
  inline uint32_t size() const noexcept { return n; }
  // Returns embedding dimension
  inline uint32_t get_dims() const noexcept { return dims; }
  // Returns id of the pointcloud
  inline uint32_t get_id() const noexcept { return id; }
  // Returns non-owning IP_Point view of the i-th embedding
  inline auto operator[](size_t i) const noexcept {
    return L2_Point<float>(values + i * dims, dims, aligned_dims, id);
  }
  // Returns pointer to start of all embeddings
  inline float* data() const noexcept { return values; }
  // Returns pointer to i-th embedding
  inline float* data(size_t i) const noexcept { return values + i * dims; }
  // Returns non-owning view of all coordinates
  inline auto get_slice() const noexcept { return parlay::make_slice(values, values + n * dims); }

  // Returns True since L2 is a metric
  // constexpr inline bool is_metric() const noexcept { return true; }
  static constexpr inline bool is_metric() noexcept { return true; }
  // Computes the (asymmetric) distance from the current point cloud
  // to the given point cloud
  float distance(const ChamferL2_Point& x) const {
    return chamfer_l2_distance(values, n, x.values, x.n, dims);
  }
  // Also computes the amount of data accessed by the distance function
  std::pair<float, size_t> distance_w_cmps(const ChamferL2_Point& x) const {
    return std::make_pair(chamfer_l2_distance(values, n, x.values, x.n, dims),
                          x.n * dims * sizeof(distanceType));
  }
};

}  // namespace mvsic