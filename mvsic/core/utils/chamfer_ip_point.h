#pragma once

#include <Eigen/Core>
#include "parlay/primitives.h"
#include "ip_point.h"

namespace mvsic {

// Computes the Chamfer IP distance, given two point clouds
float chamfer_ip_distance(const float *a, uint32_t n_a, const float *b, uint32_t n_b, uint32_t dim);

/* Chamfer IP Point Cloud Type */
struct ChamferIP_Point {
 private:
  uint32_t n = 0;
  uint32_t dims = 0;
  uint32_t aligned_dims = 0;
  float *values = nullptr;
  uint32_t id = std::numeric_limits<uint32_t>::max();

 public:
  using distanceType = float;
  ChamferIP_Point() noexcept {}
  // Non-owning version
  ChamferIP_Point(uint32_t n, uint32_t dims, float *values, uint32_t id) noexcept :
      n(n), dims(dims), values(values), id(id) {}

  // Returns number of embeddings in the point cloud
  inline uint32_t size() const noexcept { return n; }
  // Returns embedding dimension
  inline uint32_t get_dims() const noexcept { return dims; }
  // Returns id of the pointcloud
  inline uint32_t get_id() const noexcept { return id; }
  // Returns non-owning IP_Point view of the i-th embedding
  inline auto operator[](size_t i) const noexcept {
    return IP_Point<float>(values + i * dims, dims, aligned_dims, id);
  }
  // Returns pointer to start of all embeddings
  inline float *data() const noexcept { return values; }
  // Returns pointer to i-th embedding
  inline float *data(size_t i) const noexcept { return values + i * dims; }
  // Returns non-owning view of all coordinates
  inline auto get_slice() const noexcept { return parlay::make_slice(values, values + n * dims); }

  // Returns False: since IP is not a metric
  constexpr inline bool is_metric() const noexcept { return false; }
  // Computes the (asymmetric) distance from the current point cloud
  // to the given point cloud
  float distance(const ChamferIP_Point &x) const {
    return chamfer_ip_distance(values, n, x.values, x.n, dims);
  }
  // Also computes the amount of data accessed by the distance function
  std::pair<float, size_t> distance_w_cmps(const ChamferIP_Point &x) const {
    return std::make_pair(chamfer_ip_distance(values, n, x.values, x.n, dims), (n + x.n) * dims);
  }
};

/* -----------------------------------------Implementation-----------------------------------------*/

// Computes the Chamfer IP distance, given two point clouds
float chamfer_ip_distance(const float *a, uint32_t n_a, const float *b, uint32_t n_b,
                          uint32_t dim) {
  if (n_a == 0 || n_b == 0) {
    std::cout << "Invalid input to Chamfer IP distance, na = " << n_a << ", nb = " << n_b
              << std::endl;
    abort();
  }
  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mat_a(
      a, n_a, dim);
  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mat_b(
      b, n_b, dim);
  // (n_a x dim) * (dim x n_b) = (n_a x n_b)
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> inner_product_matrix =
      mat_a * mat_b.transpose();
  // Compute Max sim from each vector in A, and then take average.
  float sim = inner_product_matrix.rowwise().maxCoeff().mean();
  return -sim;
}

}  // namespace mvsic