#pragma once

#include <sys/types.h>
#include <immintrin.h>
#include <iostream>
#include <vector>
#include <limits>

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

// Trying out some optimizations. None of the following are currently able to beat the above
// implementation in the microbenchmark: chamfer_ip_benchmark.cpp

namespace mvsic {
// Fast horizontal sum of a 256-bit AVX register
inline float hsum256_ps_avx(__m256 v) {
  __m128 vlow = _mm256_castps256_ps128(v);
  __m128 vhigh = _mm256_extractf128_ps(v, 1);
  vlow = _mm_add_ps(vlow, vhigh);
  __m128 shuf = _mm_movehl_ps(vlow, vlow);
  vlow = _mm_add_ps(vlow, shuf);
  shuf = _mm_shuffle_ps(vlow, vlow, 0x55);
  vlow = _mm_add_ss(vlow, shuf);
  return _mm_cvtss_f32(vlow);
}

// Computes the Chamfer IP distance, given two point clouds
float chamfer_ip_distance_opt(const float* a, uint32_t n_a, const float* b, uint32_t n_b,
                              uint32_t dim) {
  if (n_a == 0 || n_b == 0) {
    std::cout << "Invalid input to Chamfer IP distance, na = " << n_a << ", nb = " << n_b << "\n";
    abort();
  }

  // 1. Pure Stack Workspace: Zero branching, zero TLS guards, zero heap locks.
  // 128 floats = 512 bytes, executing entirely within the thread's L1 cache.
  constexpr uint32_t MAX_QUERY_SIZE = 128;
  alignas(32) float max_sims[MAX_QUERY_SIZE];

  if (n_a > MAX_QUERY_SIZE) {
    std::cout << "Query size n_a (" << n_a << ") exceeds stack buffer limit of " << MAX_QUERY_SIZE
              << "!\n";
    abort();
  }

  // Fast initialization
  for (uint32_t i = 0; i < n_a; ++i) {
    max_sims[i] = -std::numeric_limits<float>::infinity();
  }

  uint32_t j = 0;
  for (; j + 3 < n_b; j += 4) {
    const float* b0 = b + j * dim;
    const float* b1 = b + (j + 1) * dim;
    const float* b2 = b + (j + 2) * dim;
    const float* b3 = b + (j + 3) * dim;

    // 2. Software Prefetching: Tell the CPU to fetch the NEXT 4 rows of B into the L1 cache
    // _MM_HINT_T0 requests data be brought all the way to the lowest level cache.
    if (j + 7 < n_b) {
      _mm_prefetch(reinterpret_cast<const char*>(b + (j + 4) * dim), _MM_HINT_T0);
      _mm_prefetch(reinterpret_cast<const char*>(b + (j + 5) * dim), _MM_HINT_T0);
      _mm_prefetch(reinterpret_cast<const char*>(b + (j + 6) * dim), _MM_HINT_T0);
      _mm_prefetch(reinterpret_cast<const char*>(b + (j + 7) * dim), _MM_HINT_T0);
    }

    uint32_t i = 0;
    for (; i + 1 < n_a; i += 2) {
      const float* a0 = a + i * dim;
      const float* a1 = a + (i + 1) * dim;

      __m256 acc00 = _mm256_setzero_ps();
      __m256 acc01 = _mm256_setzero_ps();
      __m256 acc02 = _mm256_setzero_ps();
      __m256 acc03 = _mm256_setzero_ps();
      __m256 acc10 = _mm256_setzero_ps();
      __m256 acc11 = _mm256_setzero_ps();
      __m256 acc12 = _mm256_setzero_ps();
      __m256 acc13 = _mm256_setzero_ps();

      uint32_t d = 0;
      for (; d + 7 < dim; d += 8) {
        __m256 va0 = _mm256_loadu_ps(a0 + d);
        __m256 va1 = _mm256_loadu_ps(a1 + d);

        __m256 vb0 = _mm256_loadu_ps(b0 + d);
        __m256 vb1 = _mm256_loadu_ps(b1 + d);
        __m256 vb2 = _mm256_loadu_ps(b2 + d);
        __m256 vb3 = _mm256_loadu_ps(b3 + d);

        acc00 = _mm256_fmadd_ps(va0, vb0, acc00);
        acc01 = _mm256_fmadd_ps(va0, vb1, acc01);
        acc02 = _mm256_fmadd_ps(va0, vb2, acc02);
        acc03 = _mm256_fmadd_ps(va0, vb3, acc03);

        acc10 = _mm256_fmadd_ps(va1, vb0, acc10);
        acc11 = _mm256_fmadd_ps(va1, vb1, acc11);
        acc12 = _mm256_fmadd_ps(va1, vb2, acc12);
        acc13 = _mm256_fmadd_ps(va1, vb3, acc13);
      }

      float dp00 = hsum256_ps_avx(acc00);
      float dp01 = hsum256_ps_avx(acc01);
      float dp02 = hsum256_ps_avx(acc02);
      float dp03 = hsum256_ps_avx(acc03);
      float dp10 = hsum256_ps_avx(acc10);
      float dp11 = hsum256_ps_avx(acc11);
      float dp12 = hsum256_ps_avx(acc12);
      float dp13 = hsum256_ps_avx(acc13);

      for (; d < dim; ++d) {
        dp00 += a0[d] * b0[d];
        dp01 += a0[d] * b1[d];
        dp02 += a0[d] * b2[d];
        dp03 += a0[d] * b3[d];
        dp10 += a1[d] * b0[d];
        dp11 += a1[d] * b1[d];
        dp12 += a1[d] * b2[d];
        dp13 += a1[d] * b3[d];
      }

      max_sims[i] = std::max({max_sims[i], dp00, dp01, dp02, dp03});
      max_sims[i + 1] = std::max({max_sims[i + 1], dp10, dp11, dp12, dp13});
    }

    // Handle tail of A (if n_a is odd)
    for (; i < n_a; ++i) {
      const float* a0 = a + i * dim;
      __m256 acc00 = _mm256_setzero_ps();
      __m256 acc01 = _mm256_setzero_ps();
      __m256 acc02 = _mm256_setzero_ps();
      __m256 acc03 = _mm256_setzero_ps();

      uint32_t d = 0;
      for (; d + 7 < dim; d += 8) {
        __m256 va0 = _mm256_loadu_ps(a0 + d);
        acc00 = _mm256_fmadd_ps(va0, _mm256_loadu_ps(b0 + d), acc00);
        acc01 = _mm256_fmadd_ps(va0, _mm256_loadu_ps(b1 + d), acc01);
        acc02 = _mm256_fmadd_ps(va0, _mm256_loadu_ps(b2 + d), acc02);
        acc03 = _mm256_fmadd_ps(va0, _mm256_loadu_ps(b3 + d), acc03);
      }

      float dp00 = hsum256_ps_avx(acc00);
      float dp01 = hsum256_ps_avx(acc01);
      float dp02 = hsum256_ps_avx(acc02);
      float dp03 = hsum256_ps_avx(acc03);
      for (; d < dim; ++d) {
        dp00 += a0[d] * b0[d];
        dp01 += a0[d] * b1[d];
        dp02 += a0[d] * b2[d];
        dp03 += a0[d] * b3[d];
      }
      max_sims[i] = std::max({max_sims[i], dp00, dp01, dp02, dp03});
    }
  }

  // Handle tail of B
  for (; j < n_b; ++j) {
    const float* b0 = b + j * dim;
    for (uint32_t i = 0; i < n_a; ++i) {
      const float* a0 = a + i * dim;
      __m256 acc = _mm256_setzero_ps();
      uint32_t d = 0;
      for (; d + 7 < dim; d += 8) {
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a0 + d), _mm256_loadu_ps(b0 + d), acc);
      }
      float dp = hsum256_ps_avx(acc);
      for (; d < dim; ++d)
        dp += a0[d] * b0[d];
      max_sims[i] = std::max(max_sims[i], dp);
    }
  }

  float total_sim = 0.0f;
  for (uint32_t i = 0; i < n_a; ++i) {
    total_sim += max_sims[i];
  }

  return -(total_sim / n_a);
}

// Computes the Chamfer IP distance utilizing Eigen's AVX-512 GEMM on a stack buffer
inline float chamfer_ip_distance_eigen_opt(const float* a, uint32_t n_a, const float* b,
                                           uint32_t n_b, uint32_t dim) {
  if (n_a == 0 || n_b == 0) {
    std::cout << "Invalid input to Chamfer IP distance, na = " << n_a << ", nb = " << n_b << "\n";
    abort();
  }

  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mat_a(
      a, n_a, dim);
  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mat_b(
      b, n_b, dim);

  // 1. Pure Stack Workspace: Zero heap allocations.
  // We allocate enough space for n_a * n_b.
  // Assuming max n_a = 64 and max n_b = 256, 16384 floats is 65 KB.
  // 65 KB is perfectly safe for a deep nested thread stack.
  constexpr uint32_t MAX_ELEMENTS = 16384;
  alignas(64) float stack_buf[MAX_ELEMENTS];

  uint32_t required_size = n_a * n_b;

  if (required_size > MAX_ELEMENTS) {
    // Fallback to heap if we encounter an anomalously massive point cloud
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> ip_mat = mat_a * mat_b.transpose();
    return -ip_mat.rowwise().maxCoeff().mean();
  }

  // 2. Map the stack memory to an Eigen matrix
  Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> ip_mat(
      stack_buf, n_a, n_b);

  // 3. .noalias() is critical here!
  // It guarantees Eigen writes directly into our stack buffer without creating temporary matrices.
  ip_mat.noalias() = mat_a * mat_b.transpose();

  // 4. L1 Cache Reduction
  return -ip_mat.rowwise().maxCoeff().mean();
}

}  // namespace mvsic