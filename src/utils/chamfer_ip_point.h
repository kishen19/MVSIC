#pragma once

#include <Eigen/Core>
#include "parlay/primitives.h"

namespace mvivf {

// Computes the Chamfer IP distance, given two point clouds
float chamfer_ip_distance(const float *a, size_t n_a, const float *b, size_t n_b, size_t dim);

/* Chamfer IP Point Cloud Type */
struct ChamferIP_Point {
 private:
  size_t n = 0;
  size_t dims = 0;
  size_t aligned_dims = 0;
  float *values = nullptr;
  size_t id = std::numeric_limits<size_t>::max();
  bool owns = false;

 public:
  using distanceType = float;
  ChamferIP_Point() noexcept {}
  // Non-owning version
  ChamferIP_Point(size_t n, size_t dims, float *values, size_t id) noexcept :
      n(n), dims(dims), values(values), id(id), owns(false) {}
  // Owning version, creates a copy. Typically no id associated. TODO: deprecate
  ChamferIP_Point(size_t n, size_t dims, const float *values_) : n(n), dims(dims), owns(true) {
    values = static_cast<float *>(parlay::p_malloc(n * dims * sizeof(float)));
    std::memcpy(values, values_, n * dims * sizeof(float));
  }
  // Move Constructor: does exactly what the input does
  ChamferIP_Point(ChamferIP_Point &&other) noexcept;
  // Copy Constructor: does exactly what the input does
  ChamferIP_Point(const ChamferIP_Point &other);
  // Destructor: free if owns values
  ~ChamferIP_Point() noexcept {
    if (owns && (values != nullptr)) {
      parlay::p_free(values);
      values = nullptr;
      owns = false;
    }
  }
  // Move Assignment Operator: does exactly what the input does
  ChamferIP_Point &operator=(ChamferIP_Point &&other) noexcept;
  // *Copy Assignment Operator: creates owning copy of values regardless of
  // input TODO: deprecate
  ChamferIP_Point &operator=(const ChamferIP_Point &other);
  // Returns number of embeddings in the point cloud
  inline size_t size() const noexcept { return n; }
  // Returns embedding dimension
  inline size_t get_dims() const noexcept { return dims; }
  // Returns id of the pointcloud
  inline size_t get_id() const noexcept { return id; }
  // Returns non-owning view of the i-th embedding
  inline auto operator[](size_t i) const noexcept {
    return parlay::make_slice(values + i * dims, values + (i + 1) * dims);
  }
  // Returns pointer to i-th embedding
  inline float *get_coords(size_t i) const noexcept { return values + i * dims; }
  // Returns False since IP is not a metric
  constexpr inline bool is_metric() const noexcept { return false; }
  // Computes the (asymmetric) distance from the current point cloud
  // to the given point cloud
  float distance(const ChamferIP_Point &x) const {
    return chamfer_ip_distance(values, n, x.values, x.n, dims);
  }
  // Returns non-owning view of all coordinates
  inline auto get_slice() const noexcept { return parlay::make_slice(values, values + n * dims); }
  inline float *data() const noexcept { return values; }
};

/* -----------------------------------------Implementation-----------------------------------------*/

// Computes the Chamfer IP distance, given two point clouds
float chamfer_ip_distance(const float *a, size_t n_a, const float *b, size_t n_b, size_t dim) {
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
  // (n_a x 1)
  Eigen::Matrix<float, Eigen::Dynamic, 1> max_ips_A_to_B =
      inner_product_matrix.rowwise().maxCoeff();
  float sim = max_ips_A_to_B.mean();
  return -sim;
}

// Move Constructor
ChamferIP_Point::ChamferIP_Point(ChamferIP_Point &&other) noexcept :
    n(other.n),
    dims(other.dims),
    aligned_dims(other.aligned_dims),
    values(other.values),
    id(other.id),
    owns(other.owns) {
  other.n = 0;
  other.dims = 0;
  other.aligned_dims = 0;
  other.values = nullptr;
  other.id = std::numeric_limits<size_t>::max();
  other.owns = false;
}

// Copy Constructor: does exactly what the input does
ChamferIP_Point::ChamferIP_Point(const ChamferIP_Point &other) :
    n(other.n), dims(other.dims), aligned_dims(other.aligned_dims), id(other.id), owns(other.owns) {
  if (owns) {
    values = static_cast<float *>(parlay::p_malloc(n * dims * sizeof(float)));
    std::memcpy(values, other.values, n * dims * sizeof(float));
  } else {
    values = other.values;
  }
}

// Move Assignment Operator
ChamferIP_Point &ChamferIP_Point::operator=(ChamferIP_Point &&other) noexcept {
  if (this != &other) {
    if (owns && (values != nullptr)) {
      parlay::p_free(values);
      owns = false;
    }
    n = other.n;
    dims = other.dims;
    aligned_dims = other.aligned_dims;
    values = other.values;
    id = other.id;
    owns = other.owns;

    other.n = 0;
    other.dims = 0;
    other.aligned_dims = 0;
    other.values = nullptr;
    other.id = std::numeric_limits<size_t>::max();
    other.owns = false;
  }
  return *this;
}

// Copy Assignment Operator: creates owning copy of values regardless
// of input
ChamferIP_Point &ChamferIP_Point::operator=(const ChamferIP_Point &other) {
  if (this != &other) {
    if (owns && (values != nullptr)) {
      parlay::p_free(values);
      owns = false;
    }
    n = other.n;
    dims = other.dims;
    aligned_dims = other.aligned_dims;
    id = other.id;
    if (other.values == nullptr) {
      values = nullptr;
      owns = false;
    } else {
      values = static_cast<float *>(parlay::p_malloc(n * dims * sizeof(float)));
      std::memcpy(values, other.values, n * dims * sizeof(float));
      owns = true;
    }
  }
  return *this;
}

}  // namespace mvivf