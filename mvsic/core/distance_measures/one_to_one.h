#pragma once

#include <sys/types.h>

#include <Eigen/Core>
#include "NSGDist.h"

namespace mvsic {

float ip_distance(const uint8_t *p, const uint8_t *q, unsigned d) {
  int result = 0;
  for (int i = 0; i < d; i++) {
    result += ((int32_t)q[i]) * ((int32_t)p[i]);
  }
  return -((float)result);
}

float ip_distance(const int8_t *p, const int8_t *q, unsigned d) {
  int result = 0;
  for (int i = 0; i < d; i++) {
    result += ((int32_t)q[i]) * ((int32_t)p[i]);
  }
  return -((float)result);
}

float ip_distance(const float *p, const float *q, unsigned d) {
  float result = 0;
  for (int i = 0; i < d; i++) {
    result += (q[i]) * (p[i]);
  }
  return -result;
}

float l2_distance(const uint8_t *p, const uint8_t *q, unsigned d) {
  int result = 0;
  for (int i = 0; i < d; i++) {
    result +=
        ((int32_t)((int16_t)q[i] - (int16_t)p[i])) * ((int32_t)((int16_t)q[i] - (int16_t)p[i]));
  }
  return (float)result;
}

float l2_distance(const int8_t *p, const int8_t *q, unsigned d) {
  int result = 0;
  for (int i = 0; i < d; i++) {
    result +=
        ((int32_t)((int16_t)q[i] - (int16_t)p[i])) * ((int32_t)((int16_t)q[i] - (int16_t)p[i]));
  }
  return (float)result;
}

float l2_distance(const float *p, const float *q, unsigned d) {
  efanna2e::DistanceL2 distfunc;
  return distfunc.compare(p, q, d);
}

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

// Computes the Chamfer L2 distance, given two point clouds
float chamfer_l2_distance(const float *a, uint32_t n_a, const float *b, uint32_t n_b,
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
      sq_norms_a.replicate(1, n_b)       // n_a x n_b
      - 2 * (mat_a * mat_b.transpose())  // n_a x n_b
      + sq_norms_b.replicate(n_a, 1);    // n_a x n_b

  sq_dist_matrix = sq_dist_matrix.cwiseMax(0.0);
  float dist = sq_dist_matrix.rowwise().minCoeff().mean();
  return dist;
}

}  // namespace mvsic