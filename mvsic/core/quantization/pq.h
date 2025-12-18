#pragma once

#include <cmath>
#include <vector>
#include <iostream>
#include <algorithm>
#include <random>
#include <cstring>

#include "parlay/primitives.h"
#include <Eigen/Core>

#include "mvsic/core/utils/kmeans_util.h"

namespace mvsic {
namespace pq {

// ---------------------------------------------------------
// Distance Kernels (Unrolled & Specialized)
// ---------------------------------------------------------

// Generic fallback for arbitrary num_blocks
inline float distance_generic(const float* lut, const uint8_t* codes, uint32_t m) {
  float dist = 0.0f;
  for (uint32_t b = 0; b < m; ++b) {
    dist += lut[b * 256 + codes[b]];
  }
  return dist;
}

// Template for compile-time constants (Compiler will unroll these)
template<int M>
inline float distance_fixed(const float* lut, const uint8_t* codes, uint32_t /*unused*/) {
  float dist = 0.0f;
// GCC/Clang hint to unroll loops with known bounds
#pragma GCC unroll 16
  for (int b = 0; b < M; ++b) {
    dist += lut[b * 256 + codes[b]];
  }
  return dist;
}

// Specialization for M=8 (Highly common case)
// Manually unrolled to ensure independent instruction scheduling
template<>
inline float distance_fixed<8>(const float* lut, const uint8_t* codes, uint32_t) {
  return lut[0 * 256 + codes[0]] + lut[1 * 256 + codes[1]] + lut[2 * 256 + codes[2]] +
         lut[3 * 256 + codes[3]] + lut[4 * 256 + codes[4]] + lut[5 * 256 + codes[5]] +
         lut[6 * 256 + codes[6]] + lut[7 * 256 + codes[7]];
}

// ---------------------------------------------------------
// Quantized Query
// ---------------------------------------------------------

// Forward Declaration
template<bool Metric>
class Quantized_Point;

template<bool Metric>
class Quantized_Query {
 public:
  using distanceType = float;

  // Layout: Flat array of [num_blocks * 256] floats.
  std::vector<float> lut;
  uint32_t num_blocks;

  // Function pointer to the optimal kernel
  float (*dist_func)(const float*, const uint8_t*, uint32_t);

  Quantized_Query(uint32_t m) : num_blocks(m) {
    // lut fits in L1 cache (e.g., 8 * 256 * 4B = 8KB)
    lut.resize(m * 256);

    // [OPT 1] Select optimized kernel based on M
    switch (m) {
      case 4: dist_func = &distance_fixed<4>; break;
      case 8: dist_func = &distance_fixed<8>; break;
      case 12: dist_func = &distance_fixed<12>; break;
      case 16: dist_func = &distance_fixed<16>; break;
      case 24: dist_func = &distance_fixed<24>; break;
      case 32: dist_func = &distance_fixed<32>; break;
      case 64: dist_func = &distance_fixed<64>; break;
      default: dist_func = &distance_generic; break;
    }
  }

  inline float distance(const Quantized_Point<Metric>& p) const;
};

// ---------------------------------------------------------
// Quantized Point: Aux data type representing a compressed vector
// ---------------------------------------------------------
template<bool Metric>
class Quantized_Point {
 public:
  const uint8_t* code_ptr;
  uint32_t num_blocks;

  Quantized_Point(const uint8_t* ptr, uint32_t m) : code_ptr(ptr), num_blocks(m) {}

  inline float distance(const Quantized_Query<Metric>& qq) const { return qq.distance(*this); }

  // Loads the compressed code block into L1 cache before usage.
  // Useful in random-access searches (Vamana graph traversal).
  void prefetch() const { __builtin_prefetch(code_ptr, 0, 1); }

  // ParlayANN requirements (in beam_search)
  bool same_as(const Quantized_Point<Metric>& q) const { return false; }
  bool same_as(const Quantized_Query<Metric>& q) const { return false; }

  bool is_metric() const { return Metric; }
};

template<bool Metric>
inline float Quantized_Query<Metric>::distance(const Quantized_Point<Metric>& p) const {
  // Indirect call to optimized kernel
  return dist_func(lut.data(), p.code_ptr, num_blocks);
}

// ---------------------------------------------------------
// Quantized Point Range (Main Container)
// ---------------------------------------------------------
template<typename PointRange, bool Metric>
class Quantized_Point_Range {
 public:
  uint32_t num_blocks;
  uint32_t num_clusters_per_block;
  uint32_t num_points_per_cluster;

  size_t n_points;
  size_t dim;
  size_t dim_per_block;

  std::vector<Eigen::MatrixXf> codebooks;
  std::vector<Eigen::VectorXf> codebook_norms;

  parlay::sequence<uint8_t> codes;

  Quantized_Point_Range() {}

  Quantized_Point_Range(const PointRange& data, uint32_t m = 8, uint32_t k = 256,
                        uint32_t subsample_mult = 20) :
      num_blocks(m), num_clusters_per_block(k), num_points_per_cluster(subsample_mult) {

    n_points = data.size();
    dim = data.get_dims();

    if (dim % num_blocks != 0) {
      std::cerr << "Error: PQ Dimension " << dim << " not divisible by " << num_blocks << std::endl;
      abort();
    }
    dim_per_block = dim / num_blocks;

    parlay::internal::timer t;
    // 1. Train
    t.start();
    train(data);
    // 2. Encode
    encode_database(data);
  }

  Quantized_Point<Metric> operator[](size_t i) const {
    return Quantized_Point<Metric>(&codes[i * num_blocks], num_blocks);
  }

  template<typename PointTy>
  Quantized_Query<Metric> quantize_query(const PointTy& query) const {
    Quantized_Query<Metric> qq(num_blocks);

    for (uint32_t b = 0; b < num_blocks; ++b) {
      // Map query part to Eigen (safe copy)
      Eigen::VectorXf q_sub(dim_per_block);
      size_t offset = b * dim_per_block;
      for (size_t j = 0; j < dim_per_block; ++j) {
        q_sub[j] = query[offset + j];
      }

      // 1. Calculate Dot Products (GEMV)
      Eigen::VectorXf dot_products = codebooks[b] * q_sub;

      // 2. Combine results into LUT
      Eigen::Map<Eigen::VectorXf> lut_segment(&qq.lut[b * 256], num_clusters_per_block);

      if constexpr (Metric) {  // Euclidean
        float q_sq_norm = q_sub.squaredNorm();
        const auto& c_sq_norms = codebook_norms[b];
        // LUT[c] = ||c||^2 - 2<c,q> + ||q||^2
        lut_segment = c_sq_norms - (2.0f * dot_products);
        lut_segment.array() += q_sq_norm;
      } else {  // Inner Product (MIPS)
        // LUT[c] = -<c,q> (Minimize negative dot product to maximize dot product)
        lut_segment = -dot_products;
      }
    }
    return qq;
  }

  template<typename PointCloudTy>
  void quantize_query_batch(const PointCloudTy& query_cloud,
                            std::vector<Quantized_Query<Metric>>& out_luts) const {
    uint32_t num_q = query_cloud.size();
    out_luts.clear();
    out_luts.reserve(num_q);
    for (uint32_t i = 0; i < num_q; ++i)
      out_luts.emplace_back(num_blocks);

    for (uint32_t b = 0; b < num_blocks; ++b) {
      // 1. Map query cloud to a Matrix [num_q x dim_per_block]
      // We use Eigen::Map to avoid copying query data
      Eigen::MatrixXf Q_sub(num_q, dim_per_block);
      size_t offset = b * dim_per_block;
      for (size_t i = 0; i < num_q; ++i) {
        for (size_t j = 0; j < dim_per_block; ++j) {
          Q_sub(i, j) = query_cloud[i][offset + j];
        }
      }
      // 2. GEMM: [num_clusters x dim_per_block] * [dim_per_block x num_q]
      // This is the primary speedup: codebook is loaded once and reused for all query vectors.
      Eigen::MatrixXf dot_products = codebooks[b] * Q_sub.transpose();
      // 3. Populate all LUTs
      for (size_t i = 0; i < num_q; ++i) {
        Eigen::Map<Eigen::VectorXf> lut_segment(&out_luts[i].lut[b * 256], num_clusters_per_block);
        if constexpr (Metric) {
          float q_sq_norm = Q_sub.row(i).squaredNorm();
          lut_segment = codebook_norms[b] - (2.0f * dot_products.col(i));
          lut_segment.array() += q_sq_norm;
        } else {
          lut_segment = -dot_products.col(i);
        }
      }
    }
  }

  void save(std::ostream& out) const {
    out.write((char*)&num_blocks, sizeof(num_blocks));
    out.write((char*)&num_clusters_per_block, sizeof(num_clusters_per_block));
    out.write((char*)&n_points, sizeof(n_points));
    out.write((char*)&dim, sizeof(dim));
    out.write((char*)&dim_per_block, sizeof(dim_per_block));

    for (const auto& cb : codebooks) {
      out.write((char*)cb.data(), cb.size() * sizeof(float));
    }

    if (!codes.empty()) {
      out.write((char*)&codes[0], codes.size() * sizeof(uint8_t));
    }
  }

  void load(std::istream& in) {
    in.read((char*)&num_blocks, sizeof(num_blocks));
    in.read((char*)&num_clusters_per_block, sizeof(num_clusters_per_block));
    in.read((char*)&n_points, sizeof(n_points));
    in.read((char*)&dim, sizeof(dim));
    in.read((char*)&dim_per_block, sizeof(dim_per_block));

    codebooks.resize(num_blocks);
    codebook_norms.resize(num_blocks);
    codes.resize(n_points * num_blocks);

    for (uint32_t b = 0; b < num_blocks; ++b) {
      codebooks[b] = Eigen::MatrixXf(num_clusters_per_block, dim_per_block);
      in.read((char*)codebooks[b].data(), codebooks[b].size() * sizeof(float));
      codebook_norms[b] = codebooks[b].rowwise().squaredNorm();
    }

    if (!codes.empty()) {
      in.read((char*)&codes[0], codes.size() * sizeof(uint8_t));
    }
  }

  inline uint32_t size() const noexcept { return n_points; }
  inline uint32_t get_dims() const noexcept { return dim; }

 private:
  void train(const PointRange& data) {
    codebooks.resize(num_blocks);
    codebook_norms.resize(num_blocks);

    size_t sample_size = num_clusters_per_block * num_points_per_cluster;

    parlay::parallel_for(0, num_blocks, [&](size_t b) {
      size_t offset = b * dim_per_block;
      size_t actual_sample_size = std::min(sample_size, n_points);

      parlay::sequence<parlay::sequence<float>> subsample(actual_sample_size);
      std::mt19937 rng(b + 1);
      std::uniform_int_distribution<size_t> dist(0, n_points - 1);

      for (size_t i = 0; i < actual_sample_size; ++i) {
        size_t pid = dist(rng);
        const float* raw_data = reinterpret_cast<const float*>(data.location(pid));
        parlay::sequence<float> vec(dim_per_block);
        std::copy(raw_data + offset, raw_data + offset + dim_per_block, vec.begin());
        subsample[i] = std::move(vec);
      }

      auto [centers, _] = mvsic::kmeans_subsample_assign_only<Metric>(
          subsample, num_clusters_per_block, actual_sample_size, false);

      codebooks[b] = Eigen::MatrixXf(num_clusters_per_block, dim_per_block);
      for (size_t c = 0; c < centers.size(); ++c) {
        for (size_t d = 0; d < dim_per_block; ++d) {
          codebooks[b](c, d) = centers[c][d];
        }
      }
      for (size_t c = centers.size(); c < num_clusters_per_block; ++c) {
        codebooks[b].row(c).setZero();
      }
      codebook_norms[b] = codebooks[b].rowwise().squaredNorm();
    });
  }

  void encode_database(const PointRange& data) {
    codes.resize(n_points * num_blocks);

    parlay::parallel_for(0, n_points, [&](size_t i) {
      const float* raw_point_ptr = reinterpret_cast<const float*>(data.location(i));

      static thread_local Eigen::VectorXf dot_products;

      if (dot_products.size() != num_clusters_per_block) {
        dot_products.resize(num_clusters_per_block);
      }

      for (size_t b = 0; b < num_blocks; ++b) {
        size_t offset = b * dim_per_block;
        Eigen::Map<const Eigen::VectorXf> p_sub(raw_point_ptr + offset, dim_per_block);

        dot_products.noalias() = codebooks[b] * p_sub;

        float min_val = std::numeric_limits<float>::max();
        uint8_t best_code = 0;
        const auto& c_sq_norms = codebook_norms[b];

        for (int c = 0; c < (int)num_clusters_per_block; ++c) {
          float val;
          if constexpr (Metric) {  // Euclidean: ||c||^2 - 2<x,c>
            val = c_sq_norms[c] - 2 * dot_products[c];
          } else {  // IP: -<x,c> (minimize negative dot product)
            val = -dot_products[c];
          }

          if (val < min_val) {
            min_val = val;
            best_code = static_cast<uint8_t>(c);
          }
        }
        codes[i * num_blocks + b] = best_code;
      }
    });
  }
};

}  // namespace pq
}  // namespace mvsic