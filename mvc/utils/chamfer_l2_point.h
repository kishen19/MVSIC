#pragma once

#include <Eigen/Core>
#include "parlay/parallel.h"
#include "parlay/primitives.h"

float chamfer_l2_distance(const float* a, size_t n_a,
  const float* b, size_t n_b, size_t dim) {
  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
    mat_a(a, n_a, dim);
  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
    mat_b(b, n_b, dim);
  // n_a x 1
  Eigen::Matrix<float, Eigen::Dynamic, 1> sq_norms_a = mat_a.rowwise().squaredNorm(); // Column vector, 
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

struct ChamferL2_Point {
private:
  size_t n = 0;
  size_t dims = 0;
  size_t aligned_dims = 0;
  float* values;
  uint32_t id = std::numeric_limits<uint32_t>::max();
  bool owns = false;

public:
  inline size_t size() const { return n; }
  inline size_t get_dims() const { return dims; }
  inline auto operator[](long i) const {
    return parlay::make_slice(values + i * dims, values + (i + 1) * dims);
  }
  // inline float operator[](long i, long j) const {
  //   return values[i * dims + j];
  // }
  inline float* get_values(long i) const {
    return values + i * dims;
  }
  uint32_t get_id() const { return id; }
  bool is_metric() const {return true; }
  float distance(const ChamferL2_Point& x) const {
    return chamfer_l2_distance(values, n, x.values, x.n, dims);
  }
  auto coords() const{
    return parlay::make_slice(values, values + n * dims);
  }

  ChamferL2_Point() {}
  ChamferL2_Point(long id, float* values_, size_t n, size_t dims)
    : n(n), dims(dims), values(values_), id(id), owns(false) {
  }
  ChamferL2_Point(float* values_, size_t n, size_t dims)
    : n(n), dims(dims), owns(true) {
    values = static_cast<float*>(parlay::p_malloc(n * dims * sizeof(float)));
    std::memcpy(values, values_, n * dims * sizeof(float));
  }
  ~ChamferL2_Point() {
    if (owns && (values != nullptr)) {
      parlay::p_free(values);
      values = nullptr;
      owns = false;
    }
  }
  ChamferL2_Point& operator=(const ChamferL2_Point& other) {
    if (this != &other) {
      if (owns && (values != nullptr)) {
        parlay::p_free(values);
        values = nullptr;
        owns = false;
      }
      id = other.id;
      n = other.n;
      dims = other.dims;
      aligned_dims = other.aligned_dims;
      values = static_cast<float*>(parlay::p_malloc(n * dims * sizeof(float)));
      std::memcpy(values, other.values, n * dims * sizeof(float));
      owns = true;
    }
    return *this;
  }
};