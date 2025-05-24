#pragma once

#include <Eigen/Core>
#include "parlay/parallel.h"
#include "parlay/primitives.h"

float chamfer_ip_distance(const float* a, size_t n_a,
  const float* b, size_t n_b, size_t dim) {
  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
    mat_a(a, n_a, dim);
  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
    mat_b(b, n_b, dim);
  // (n_a x dim) * (dim x n_b) = (n_a x n_b)
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> inner_product_matrix =
    mat_a * mat_b.transpose();
  // (n_a x 1)
  Eigen::Matrix<float, Eigen::Dynamic, 1> max_ips_A_to_B =
    inner_product_matrix.rowwise().maxCoeff();
  float sim = max_ips_A_to_B.mean();
  return -sim;
}

struct ChamferIP_Point {
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
  bool is_metric() const {return false; }
  float distance(const ChamferIP_Point& x) const {
    return chamfer_ip_distance(values, n, x.values, x.n, dims);
  }
  auto coords() const{
    return parlay::make_slice(values, values + n * dims);
  }

  ChamferIP_Point() {}
  ChamferIP_Point(long id, float* values_, size_t n, size_t dims)
    : n(n), dims(dims), values(values_), id(id), owns(false) {}
  ChamferIP_Point(float* values_, size_t n, size_t dims)
    : n(n), dims(dims), owns(false) {
    values = static_cast<float*>(parlay::p_malloc(n * dims * sizeof(float)));
    std::memcpy(values, values_, n * dims * sizeof(float));
  }
  ~ChamferIP_Point() {
    if (owns && (values != nullptr)) {
      parlay::p_free(values);
      values = nullptr;
      owns = false;
    }
  }
  ChamferIP_Point& operator=(const ChamferIP_Point& other) {
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