#pragma once

#include <Eigen/Core>
#include "parlay/primitives.h"
#include "l2_point.h"

namespace mvivf {

// Computes the Chamfer L2 distance, given two point clouds
float chamfer_l2_distance(const float *a, size_t n_a, const float *b, size_t n_b, size_t dim);

/* Chamfer L2 Point Cloud Type */
struct ChamferL2_Point {
 private:
  size_t n = 0;
  size_t dims = 0;
  size_t aligned_dims = 0;
  float *values = nullptr;
  size_t id = std::numeric_limits<size_t>::max();

 public:
  using distanceType = float;
  ChamferL2_Point() noexcept {}
  // Non-owning version
  ChamferL2_Point(size_t n, size_t dims, float *values, size_t id) noexcept :
      n(n), dims(dims), values(values), id(id) {}

  // Returns number of embeddings in the point cloud
  inline size_t size() const noexcept { return n; }
  // Returns embedding dimension
  inline size_t get_dims() const noexcept { return dims; }
  // Returns id of the pointcloud
  inline size_t get_id() const noexcept { return id; }
  // Returns non-owning view of the i-th embedding
  inline auto operator[](size_t i) const noexcept {
    return L2_Point<float>(values + i * dims, dims, aligned_dims, id);
    // return parlay::make_slice(values + i * dims, values + (i + 1) * dims);
  }
  // Returns pointer to i-th embedding
  inline float *get_coords(size_t i) const noexcept { return values + i * dims; }
  // Returns True since L2 is a metric
  constexpr inline bool is_metric() const noexcept { return true; }
  // Computes the (asymmetric) distance from the current point cloud
  // to the given point cloud
  float distance(const ChamferL2_Point &x) const {
    return chamfer_l2_distance(values, n, x.values, x.n, dims);
  }
  std::pair<float, size_t> distance_w_cmps(const ChamferL2_Point &x) const {
    return std::make_pair(chamfer_l2_distance(values, n, x.values, x.n, dims), n + x.n);
  }
  // Returns non-owning view of all coordinates
  inline auto get_slice() const noexcept { return parlay::make_slice(values, values + n * dims); }
  inline float *data() const noexcept { return values; }
};

/* -----------------------------------------Implementation-----------------------------------------*/

// Computes the Chamfer L2 distance, given two point clouds
float chamfer_l2_distance(const float *a, size_t n_a, const float *b, size_t n_b, size_t dim) {
  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mat_a(
      a, n_a, dim);
  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mat_b(
      b, n_b, dim);
  // n_a x 1
  Eigen::Matrix<float, Eigen::Dynamic, 1> sq_norms_a =
      mat_a.rowwise().squaredNorm();  // Column vector,
  // 1 x n_b
  Eigen::Matrix<float, 1, Eigen::Dynamic> sq_norms_b = mat_b.rowwise().squaredNorm().transpose();
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> sq_dist_matrix =
      sq_norms_a.replicate(1, n_b)       // n_a x n_b
      - 2 * (mat_a * mat_b.transpose())  // n_a x n_b
      + sq_norms_b.replicate(n_a, 1);    // n_a x n_b

  sq_dist_matrix = sq_dist_matrix.cwiseMax(0.0);
  // n_a x 1
  Eigen::Matrix<float, Eigen::Dynamic, 1> min_dists_A_to_B = sq_dist_matrix.rowwise().minCoeff();
  float sum_min_dists = min_dists_A_to_B.sum();
  return sum_min_dists / static_cast<float>(n_a);
}

}  // namespace mvivf