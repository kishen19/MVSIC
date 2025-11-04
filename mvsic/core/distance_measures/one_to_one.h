#pragma once

#include <sys/types.h>

#include <Eigen/Core>
#include "NSGDist.h"

namespace mvsic {

template<typename T>
float ip_distance_integer(const T* p, const T* q, unsigned d) {
  static_assert(std::is_integral<T>::value, "This function is only for integral types.");
  int result = 0;
  for (int i = 0; i < d; i++) {
    result += ((int32_t)q[i]) * ((int32_t)p[i]);
  }
  return -((float)result);
}
float ip_distance(const uint8_t* p, const uint8_t* q, unsigned d) {
  return ip_distance_integer(p, q, d);
}
float ip_distance(const int8_t* p, const int8_t* q, unsigned d) {
  return ip_distance_integer(p, q, d);
}

float ip_distance(const float* p, const float* q, unsigned d) {
  efanna2e::DistanceInnerProduct distfunc;
  return -distfunc.compare(p, q, d);
}
template<typename T>
float l2_distance_integer(const T* p, const T* q, unsigned d) {
  static_assert(std::is_integral<T>::value, "This function is only for integral types.");
  int result = 0;
  for (int i = 0; i < d; i++) {
    result +=
        ((int32_t)((int16_t)q[i] - (int16_t)p[i])) * ((int32_t)((int16_t)q[i] - (int16_t)p[i]));
  }
  return (float)result;
}
float l2_distance(const uint8_t* p, const uint8_t* q, unsigned d) {
  return l2_distance_integer(p, q, d);
}
float l2_distance(const int8_t* p, const int8_t* q, unsigned d) {
  return l2_distance_integer(p, q, d);
}

float l2_distance(const float* p, const float* q, unsigned d) {
  efanna2e::DistanceL2 distfunc;
  return distfunc.compare(p, q, d);
}

// Computes the Chamfer IP distance, given two point clouds
float chamfer_ip_distance(const float* a, uint32_t n_a, const float* b, uint32_t n_b,
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

// Computes the Chamfer L2 distance, given two point clouds
float chamfer_l2_distance(const float* a, uint32_t n_a, const float* b, uint32_t n_b,
                          uint32_t dim) {
  if (n_a == 0 || n_b == 0) {
    std::cout << "Invalid input to Chamfer L2 distance, na = " << n_a << ", nb = " << n_b
              << std::endl;
    abort();
  }
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
      -2 * (mat_a * mat_b.transpose());
  sq_dist_matrix.colwise() += sq_norms_a;
  sq_dist_matrix.rowwise() += sq_norms_b;

  sq_dist_matrix = sq_dist_matrix.cwiseMax(0.0);
  // The mean of the minimum squared distances for each point in A.
  return sq_dist_matrix.rowwise().minCoeff().mean();
}

}  // namespace mvsic