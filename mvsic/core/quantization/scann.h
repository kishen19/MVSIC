#pragma once

#include <cmath>
#include <vector>
#include <iostream>
#include <algorithm>
#include <random>

#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include <Eigen/Core>

// Reuse existing PQ structures
#include "pq.h"
#include "mvsic/core/utils/kmeans_util.h"

namespace mvsic {

// ---------------------------------------------------------
// ScaNN Point Range
// ---------------------------------------------------------
// Implements Anisotropic Vector Quantization (AVQ)
// Reference: "Accelerating Large-Scale Inference with Anisotropic Vector Quantization" (ICML 2020)
template<typename PointRange>
class ScaNN_Point_Range {
 public:
  // Layout parameters
  uint32_t num_blocks;
  uint32_t num_clusters_per_block;
  uint32_t num_points_per_cluster;
  float anisotropic_threshold;  // T in the paper (usually ~0.2)

  size_t n_points;
  size_t dim;
  size_t dim_per_block;

  // Codebooks [num_blocks] -> Matrix(256, dim_per_block)
  std::vector<Eigen::MatrixXf> codebooks;

  // Optimization: Precomputed norms ||c||^2
  std::vector<Eigen::VectorXf> codebook_norms;

  // Storage: Flat array of uint8 codes
  parlay::sequence<uint8_t> codes;

  ScaNN_Point_Range(const PointRange& data, uint32_t m = 8, uint32_t k = 256,
                    uint32_t subsample_mult = 20, float T = 0.2f) :
      num_blocks(m),
      num_clusters_per_block(k),
      num_points_per_cluster(subsample_mult),
      anisotropic_threshold(T) {
    n_points = data.size();
    dim = data.get_dims();

    if (dim % num_blocks != 0) {
      std::cerr << "ScaNN Error: Dimension " << dim << " not divisible by " << num_blocks
                << std::endl;
      abort();
    }
    dim_per_block = dim / num_blocks;

    parlay::internal::timer t;

    // 1. Train Codebooks (Standard L2 K-Means)
    // The paper explicitly states they use standard k-means for codebook learning.
    t.start();
    train_codebooks(data);
    double train_time = t.next_time();

    // 2. Anisotropic Encoding (Coordinate Descent)
    // This is the "Secret Sauce" of ScaNN.
    encode_anisotropic(data);
    double encode_time = t.next_time();

    std::cout << "ScaNN Init: Train=" << train_time << "s, Encode=" << encode_time << "s"
              << std::endl;
  }

  // Reuse the lightweight handle from PQ
  Quantized_Point operator[](size_t i) const {
    return Quantized_Point(&codes[i * num_blocks], num_blocks);
  }

  // Reuse standard PQ lookup table generation
  // (ScaNN compatible since the distance metric at search time is the same as PQ)
  template<typename PointTy>
  Quantized_Query quantize_query(const PointTy& query) const {
    // We reuse the PQ logic because ScaNN modifies the *codes* stored,
    // not necessarily the lookup table mechanism itself.
    Quantized_Query qq(num_blocks);

    for (uint32_t b = 0; b < num_blocks; ++b) {
      Eigen::VectorXf q_sub(dim_per_block);
      size_t offset = b * dim_per_block;
      for (size_t j = 0; j < dim_per_block; ++j) {
        q_sub[j] = query[offset + j];
      }

      // Precompute LUT: ||c||^2 - 2<c,q> + ||q||^2
      Eigen::VectorXf dot_products = codebooks[b] * q_sub;
      float q_sq_norm = q_sub.squaredNorm();

      Eigen::Map<Eigen::VectorXf> lut_segment(&qq.lut[b * 256], num_clusters_per_block);
      lut_segment = codebook_norms[b] - (2.0f * dot_products);
      lut_segment.array() += q_sq_norm;
    }
    return qq;
  }

 private:
  void train_codebooks(const PointRange& data) {
    codebooks.resize(num_blocks);
    codebook_norms.resize(num_blocks);
    size_t sample_size = num_clusters_per_block * num_points_per_cluster;

    // Train each subspace independently in parallel
    parlay::parallel_for(0, num_blocks, [&](size_t b) {
      size_t offset = b * dim_per_block;
      size_t actual_sample_size = std::min(sample_size, n_points);

      // Subsample data for this block
      parlay::sequence<parlay::sequence<float>> subsample(actual_sample_size);
      std::mt19937 rng(b + 1234);
      std::uniform_int_distribution<size_t> dist(0, n_points - 1);

      for (size_t i = 0; i < actual_sample_size; ++i) {
        size_t pid = dist(rng);
        const float* raw_data = reinterpret_cast<const float*>(data.location(pid));
        parlay::sequence<float> vec(dim_per_block);
        std::copy(raw_data + offset, raw_data + offset + dim_per_block, vec.begin());
        subsample[i] = std::move(vec);
      }

      // Run K-Means (Standard L2) using your utility
      auto [centers, _] = mvsic::kmeans_subsample_assign_only<true>(
          subsample, num_clusters_per_block, actual_sample_size, false);

      // Store in Eigen Matrix
      codebooks[b] = Eigen::MatrixXf(num_clusters_per_block, dim_per_block);
      for (size_t c = 0; c < centers.size(); ++c) {
        for (size_t d = 0; d < dim_per_block; ++d) {
          codebooks[b](c, d) = centers[c][d];
        }
      }
      // Fill empty clusters if any
      for (size_t c = centers.size(); c < num_clusters_per_block; ++c) {
        codebooks[b].row(c).setZero();
      }

      codebook_norms[b] = codebooks[b].rowwise().squaredNorm();
    });
  }

  void encode_anisotropic(const PointRange& data) {
    codes.resize(n_points * num_blocks);

    // We use Coordinate Descent (Iterative refinement).
    // Paper suggests 10 iterations, but 2-3 gets most of the benefit.
    const int num_iterations = 2;

    parlay::parallel_for(0, n_points, [&](size_t i) {
      const float* raw_ptr = reinterpret_cast<const float*>(data.location(i));

      // 1. Map entire vector to Eigen
      Eigen::Map<const Eigen::VectorXf> x_full(raw_ptr, dim);
      float x_sq_norm = x_full.squaredNorm();

      // Avoid division by zero for zero vectors
      float inv_norm_sq = (x_sq_norm > 1e-9f) ? (1.0f / x_sq_norm) : 0.0f;

      // 2. Initialize with standard L2 quantization (Greedy)
      // We store the current reconstruction `x_hat` implicitly via residuals or codes.
      // To save memory, we just store the codes and recompute x_hat components on fly.

      // current_codes: Stores the selected centroid index for each block
      std::vector<uint8_t> local_codes(num_blocks);

      // Initialize Standard PQ
      for (size_t b = 0; b < num_blocks; ++b) {
        size_t offset = b * dim_per_block;
        // Sub-vector x_b
        Eigen::VectorXf x_sub = x_full.segment(offset, dim_per_block);

        // ||c - x||^2 = ||c||^2 - 2<c,x> + ...
        Eigen::VectorXf dots = codebooks[b] * x_sub;
        Eigen::VectorXf dists = codebook_norms[b] - 2.0f * dots;

        Eigen::Index min_idx;
        dists.minCoeff(&min_idx);
        local_codes[b] = static_cast<uint8_t>(min_idx);
      }

      // 3. Coordinate Descent Optimization
      // Minimize: ||x - x_hat||^2 + T * (parallel_error)^2
      // This couples the blocks! Choosing c_i affects the parallel error for c_j.

      // Precompute parallel projection of all centroids onto x:  <c_ij, x_j>
      // This is static for the duration of the encoding of point x.
      std::vector<Eigen::VectorXf> parallel_projections(num_blocks);
      for (size_t b = 0; b < num_blocks; ++b) {
        size_t offset = b * dim_per_block;
        // Project codebook b onto the relevant part of x
        parallel_projections[b] = codebooks[b] * x_full.segment(offset, dim_per_block);
      }

      // Calculate initial Total Parallel Projection: sum( <c_selected, x_sub> )
      float total_parallel = 0.0f;
      for (size_t b = 0; b < num_blocks; ++b) {
        total_parallel += parallel_projections[b][local_codes[b]];
      }

      // Iterative Refinement
      for (int iter = 0; iter < num_iterations; ++iter) {
        for (size_t b = 0; b < num_blocks; ++b) {
          size_t offset = b * dim_per_block;

          // Remove current block's contribution to parallel projection
          float current_p_proj = parallel_projections[b][local_codes[b]];
          float other_parallel = total_parallel - current_p_proj;

          // We want to minimize Cost(c) for this block:
          // Cost(c) = ||x_sub - c||^2 + T * ( (other_parallel + <c, x_sub>) - ||x||^2 )^2 / ||x||^2
          //
          // Actually, ScaNN minimizes: ||x - x_hat||^2 + T * || (x - x_hat)_parallel ||^2
          // (assuming T is a weight, paper formulation slightly different but this is the code
          // equivalent)
          //
          // Let residual r = x - x_hat.
          // Loss = ||r_perp||^2 + (1+T) ||r_para||^2
          //      = ||r||^2 + T ||r_para||^2
          //      = ||x - x_hat||^2 + T * (<x - x_hat, x> / ||x||)^2
          //      = ||x - x_hat||^2 + (T / ||x||^2) * ( ||x||^2 - <x_hat, x> )^2

          // Let C be the candidate centroid in this block.
          // x_hat_new = x_hat_old - c_old + C
          // <x_hat_new, x> = other_parallel + <C, x_sub>

          // L2 Part: ||x_sub - C||^2 = ||C||^2 - 2<C, x_sub> + ||x_sub||^2
          // Anisotropic Part: Weight * ( ||x||^2 - (other_parallel + <C, x_sub>) )^2

          // Let proj = <C, x_sub> (precomputed in parallel_projections[b])
          // Let target_parallel = ||x||^2 - other_parallel
          // Aniso Term = (T / ||x||^2) * (target_parallel - proj)^2

          float weight = anisotropic_threshold * inv_norm_sq;
          float target = x_sq_norm - other_parallel;

          // Precomputed terms
          const auto& norms = codebook_norms[b];
          const auto& projs = parallel_projections[b];  // <C, x_sub>

          // Vectorized Cost Calculation
          // Cost = norms - 2*projs + weight * (target - projs).square()
          // (Note: we drop ||x_sub||^2 as it's constant)

          Eigen::VectorXf diff = projs.array() - target;          // (proj - target)
          Eigen::VectorXf aniso_penalty = diff.array().square();  // (proj - target)^2

          // Total Score
          Eigen::VectorXf scores = norms - (2.0f * projs) + (weight * aniso_penalty);

          Eigen::Index best_idx;
          scores.minCoeff(&best_idx);

          // Update state
          local_codes[b] = static_cast<uint8_t>(best_idx);
          total_parallel = other_parallel + projs[best_idx];
        }
      }

      // Store final codes
      for (size_t b = 0; b < num_blocks; ++b) {
        codes[i * num_blocks + b] = local_codes[b];
      }
    });
  }
};

}  // namespace mvsic