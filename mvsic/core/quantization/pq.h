#pragma once

#include <cmath>
#include <vector>
#include <iostream>
#include <algorithm>
#include <random>
#include <cstring>  // For std::memcpy if needed

#include "parlay/primitives.h"
#include <Eigen/Core>

#include "mvsic/core/utils/kmeans_util.h"

namespace mvsic {

// ---------------------------------------------------------
// Quantized Query: Holds the Lookup Table (LUT)
// ---------------------------------------------------------
class Quantized_Point;  // Forward decl

class Quantized_Query {
 public:
  // Layout: Flat array of [num_blocks * 256] floats.
  // Access: lut[block_idx * 256 + cluster_idx]
  std::vector<float> lut;
  uint32_t num_blocks;

  Quantized_Query(uint32_t m) : num_blocks(m) {
    // lut size is small (e.g., 8*256 floats = 8KB), fits in L1 cache.
    lut.resize(m * 256);
  }

  // The hot path function
  inline float distance(const Quantized_Point& p) const;
};

// ---------------------------------------------------------
// Quantized Point: Lightweight Handle
// ---------------------------------------------------------
class Quantized_Point {
 public:
  const uint8_t* code_ptr;
  uint32_t num_blocks;

  Quantized_Point(const uint8_t* ptr, uint32_t m) : code_ptr(ptr), num_blocks(m) {}

  // Symmetric API: Delegates back to Query
  inline float distance(const Quantized_Query& qq) const { return qq.distance(*this); }
};

// Inline definition for performance
inline float Quantized_Query::distance(const Quantized_Point& p) const {
  float dist = 0.0f;
  const uint8_t* codes = p.code_ptr;

  // Hint to compiler: Unroll this loop.
  // The 'lut' access pattern is contiguous within a block, but jumps 256 floats between blocks.
  // The 'codes' access is contiguous.
  for (uint32_t b = 0; b < num_blocks; ++b) {
    dist += lut[b * 256 + codes[b]];
  }
  return dist;
}

// ---------------------------------------------------------
// Manager: Quantized_Point_Range
// ---------------------------------------------------------
template<typename PointRange>
class Quantized_Point_Range {
 public:
  // PQ Params
  uint32_t num_blocks;
  uint32_t num_clusters_per_block;
  uint32_t num_points_per_cluster;

  size_t n_points;
  size_t dim;
  size_t dim_per_block;

  // Data
  std::vector<Eigen::MatrixXf> codebooks;
  // OPTIMIZATION: Precomputed squared norms of centroids
  // codebook_norms[block_idx](cluster_idx)
  std::vector<Eigen::VectorXf> codebook_norms;

  parlay::sequence<uint8_t> codes;  // Flat compressed data

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

    // 1. Train (Should be constant time approx)
    t.start();
    train(data);
    double train_time = t.next_time();

    // 2. Encode (Should be linear O(N))
    encode_database(data);
    double encode_time = t.next_time();

    // std::cout << "PQ Build Breakout:" << std::endl;
    // std::cout << "  - Train (K-Means on subsample): " << train_time << " s" << std::endl;
    // std::cout << "  - Encode (Assign all " << n_points << " points): " << encode_time << " s"
    //           << std::endl;
  }

  // Accessor returns a lightweight handle
  Quantized_Point operator[](size_t i) const {
    return Quantized_Point(&codes[i * num_blocks], num_blocks);
  }

  // Create the LUT for a query vector
  template<typename PointTy>
  Quantized_Query quantize_query(const PointTy& query) const {
    Quantized_Query qq(num_blocks);

    for (uint32_t b = 0; b < num_blocks; ++b) {
      // Map query part to Eigen (safe copy since query is small)
      Eigen::VectorXf q_sub(dim_per_block);
      size_t offset = b * dim_per_block;
      for (size_t j = 0; j < dim_per_block; ++j) {
        q_sub[j] = query[offset + j];
      }

      // OPTIMIZATION: GEMV for Distances
      // ||c - q||^2 = ||c||^2 + ||q||^2 - 2<c, q>

      // 1. Calculate Dot Products: (256 x D) * (D x 1) -> (256 x 1)
      // Eigen uses highly optimized AVX kernels for this matrix-vector mult
      Eigen::VectorXf dot_products = codebooks[b] * q_sub;

      float q_sq_norm = q_sub.squaredNorm();
      const auto& c_sq_norms = codebook_norms[b];

      // 2. Combine results directly into the LUT
      // Using Eigen::Map to write directly into std::vector memory
      Eigen::Map<Eigen::VectorXf> lut_segment(&qq.lut[b * 256], num_clusters_per_block);

      // Vectorized calculation: LUT = ||c||^2 - 2*dots + ||q||^2
      lut_segment = c_sq_norms - (2.0f * dot_products);
      lut_segment.array() += q_sq_norm;
    }
    return qq;
  }

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

      auto [centers, _] = mvsic::kmeans_subsample_assign_only<true>(
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

      // OPTIMIZATION: Precompute norms for fast distance calc later
      codebook_norms[b] = codebooks[b].rowwise().squaredNorm();
    });
  }

  void encode_database(const PointRange& data) {
    codes.resize(n_points * num_blocks);

    parlay::parallel_for(0, n_points, [&](size_t i) {
      const float* raw_point_ptr = reinterpret_cast<const float*>(data.location(i));

      for (size_t b = 0; b < num_blocks; ++b) {
        size_t offset = b * dim_per_block;
        Eigen::Map<const Eigen::VectorXf> p_sub(raw_point_ptr + offset, dim_per_block);

        // OPTIMIZATION: GEMV for Encoding
        // argmin ||p - c||^2 == argmin (||c||^2 - 2<p, c>)
        // (Note: ||p||^2 is constant for all c, so we ignore it)

        // 1. Dot products (AVX optimized)
        Eigen::VectorXf dot_products = codebooks[b] * p_sub;

        // 2. Find min index
        float min_val = std::numeric_limits<float>::max();
        uint8_t best_code = 0;

        const auto& c_sq_norms = codebook_norms[b];

        // This loop is now very simple scalar arithmetic, easy to unroll/pipeline
        // compared to full distance calcs
        for (int c = 0; c < (int)num_clusters_per_block; ++c) {
          float val = c_sq_norms[c] - 2 * dot_products[c];
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

}  // namespace mvsic