#pragma once

#include <cmath>
#include <vector>
#include <iostream>
#include <algorithm>
#include <random>
#include <cstring>

#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include <Eigen/Core>

#include "pq.h"
#include "mvsic/core/utils/kmeans_util.h"

namespace mvsic {
namespace pq {

// ---------------------------------------------------------
// ScaNN Point Range
// ---------------------------------------------------------
// Implements Anisotropic Vector Quantization (AVQ)
template<typename PointRange, bool Metric>
class ScaNN_Point_Range {
 public:
  // Layout parameters
  uint32_t num_blocks;
  uint32_t num_clusters_per_block;
  uint32_t num_points_per_cluster;
  float anisotropic_threshold;

  size_t n_points;
  size_t dim;
  size_t dim_per_block;

  // Codebooks [num_blocks] -> Matrix(K, dim_per_block)
  std::vector<Eigen::MatrixXf> codebooks;

  // Optimization: Precomputed norms ||c||^2
  std::vector<Eigen::VectorXf> codebook_norms;

  // Storage: Flat array of uint8 codes
  parlay::sequence<uint8_t> codes;

  ScaNN_Point_Range() {}

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

    // 1. Train Codebooks (Standard K-Means, Metric-aware)
    t.start();
    train_codebooks(data);
    double train_time = t.next_time();

    // 2. Anisotropic Encoding (Coordinate Descent)
    encode_anisotropic(data);
    double encode_time = t.next_time();

    std::cout << "ScaNN Init: Train=" << train_time << "s, Encode=" << encode_time << "s"
              << std::endl;
  }

  Quantized_Point<Metric> operator[](size_t i) const {
    return Quantized_Point<Metric>(&codes[i * num_blocks], num_blocks);
  }

  // ScaNN optimizes the codes, but the search metric (LUT generation) remains L2-based structure
  template<typename PointTy>
  Quantized_Query<Metric> quantize_query(const PointTy& query) const {
    Quantized_Query<Metric> qq(num_blocks, num_clusters_per_block);

    // Reuse buffers across blocks (avoids repeated mallocs)
    Eigen::VectorXf q_sub(dim_per_block);
    Eigen::VectorXf dot_products(num_clusters_per_block);

    for (uint32_t b = 0; b < num_blocks; ++b) {
      const size_t offset = static_cast<size_t>(b) * dim_per_block;
      for (size_t j = 0; j < dim_per_block; ++j) {
        q_sub[static_cast<Eigen::Index>(j)] = query[offset + j];
      }

      dot_products.noalias() = codebooks[b] * q_sub;

      Eigen::Map<Eigen::VectorXf> lut_segment(&qq.lut[b * 256], num_clusters_per_block);

      if constexpr (Metric) {  // Euclidean
        const float q_sq = q_sub.squaredNorm();
        lut_segment.noalias() = codebook_norms[b] - (2.0f * dot_products);
        lut_segment.array() += q_sq;
      } else {  // Inner Product (MIPS)
        lut_segment.noalias() = -dot_products;
      }
    }
    return qq;
  }

  Quantized_Query<Metric> quantize_query(const float* qptr) const {
    Quantized_Query<Metric> qq(num_blocks, num_clusters_per_block);

    // Reuse buffer across blocks
    Eigen::VectorXf dot_products(num_clusters_per_block);

    for (uint32_t b = 0; b < num_blocks; ++b) {
      const float* sub = qptr + static_cast<size_t>(b) * dim_per_block;
      Eigen::Map<const Eigen::VectorXf> q_map(sub, dim_per_block);

      dot_products.noalias() = codebooks[b] * q_map;

      Eigen::Map<Eigen::VectorXf> lut_segment(&qq.lut[b * 256], num_clusters_per_block);

      if constexpr (Metric) {  // Euclidean
        const float q_sq = q_map.squaredNorm();
        lut_segment.noalias() = codebook_norms[b] - (2.0f * dot_products);
        lut_segment.array() += q_sq;
      } else {  // Inner Product (MIPS)
        lut_segment.noalias() = -dot_products;
      }
    }
    return qq;
  }

  template<typename PointCloudTy>
  void quantize_query_batch(const PointCloudTy& query_cloud,
                            parlay::sequence<Quantized_Query<Metric>>& out_luts) const {
    const uint32_t num_q = query_cloud.size();
    const uint32_t dims = query_cloud.get_dims();
    const float* base = query_cloud.data();  // contiguous [num_q * dims]

    // Reuse out_luts storage if already correctly sized
    if (out_luts.size() != num_q) {
      out_luts.clear();
      out_luts.reserve(num_q);
      for (uint32_t i = 0; i < num_q; ++i)
        out_luts.emplace_back(num_blocks, num_clusters_per_block);
    } else {
      // If you want to be extra safe:
      // for (auto& q : out_luts) { assert(q.K == num_clusters_per_block && q.num_blocks ==
      // num_blocks); }
    }

    // Allocate once per call
    Eigen::MatrixXf dot_products(num_clusters_per_block, num_q);  // [K x num_q]
    Eigen::VectorXf q_sq;                                         // [num_q] (Metric only)

    using RowMajorMat = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using StrideT = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;

    for (uint32_t b = 0; b < num_blocks; ++b) {
      const size_t offset = static_cast<size_t>(b) * dim_per_block;

      // Strided view: Q_view(i,j) = base[offset + i*dims + j]
      Eigen::Map<const RowMajorMat, 0, StrideT> Q_view(
          base + offset, num_q, dim_per_block, StrideT(/*outerStride=*/dims, /*innerStride=*/1));

      if constexpr (Metric) {
        q_sq = Q_view.rowwise().squaredNorm();
      }

      // GEMM: [K x d] * [d x num_q] -> [K x num_q]
      dot_products.noalias() = codebooks[b] * Q_view.transpose();

      for (uint32_t i = 0; i < num_q; ++i) {
        Eigen::Map<Eigen::VectorXf> lut_segment(
            &out_luts[i].lut[static_cast<size_t>(b) * num_clusters_per_block],
            num_clusters_per_block);

        if constexpr (Metric) {
          lut_segment.noalias() = codebook_norms[b] - (2.0f * dot_products.col(i));
          lut_segment.array() += q_sq[i];
        } else {
          lut_segment.noalias() = -dot_products.col(i);
        }
      }
    }
  }

  void save(std::ostream& out) const {
    // 1. Metadata
    out.write((char*)&num_blocks, sizeof(num_blocks));
    out.write((char*)&num_clusters_per_block, sizeof(num_clusters_per_block));
    out.write((char*)&n_points, sizeof(n_points));
    out.write((char*)&dim, sizeof(dim));
    out.write((char*)&dim_per_block, sizeof(dim_per_block));
    out.write((char*)&anisotropic_threshold, sizeof(anisotropic_threshold));

    // 2. Codebooks
    for (const auto& cb : codebooks) {
      out.write((char*)cb.data(), cb.size() * sizeof(float));
    }

    // 3. Codes
    if (!codes.empty()) {
      out.write((char*)&codes[0], codes.size() * sizeof(uint8_t));
    }
  }

  void load(std::istream& in) {
    // 1. Metadata
    in.read((char*)&num_blocks, sizeof(num_blocks));
    in.read((char*)&num_clusters_per_block, sizeof(num_clusters_per_block));
    in.read((char*)&n_points, sizeof(n_points));
    in.read((char*)&dim, sizeof(dim));
    in.read((char*)&dim_per_block, sizeof(dim_per_block));
    in.read((char*)&anisotropic_threshold, sizeof(anisotropic_threshold));

    codebooks.resize(num_blocks);
    codebook_norms.resize(num_blocks);
    codes.resize(n_points * num_blocks);

    // 2. Codebooks
    for (uint32_t b = 0; b < num_blocks; ++b) {
      codebooks[b] = Eigen::MatrixXf(num_clusters_per_block, dim_per_block);
      in.read((char*)codebooks[b].data(), codebooks[b].size() * sizeof(float));
      codebook_norms[b] = codebooks[b].rowwise().squaredNorm();
    }

    // 3. Codes
    if (!codes.empty()) {
      in.read((char*)&codes[0], codes.size() * sizeof(uint8_t));
    }
  }

  inline uint32_t size() const noexcept { return n_points; }
  inline uint32_t get_dims() const noexcept { return dim; }

 private:
  void train_codebooks(const PointRange& data) {
    codebooks.resize(num_blocks);
    codebook_norms.resize(num_blocks);
    size_t sample_size = num_clusters_per_block * num_points_per_cluster;

    parlay::parallel_for(0, num_blocks, [&](size_t b) {
      size_t offset = b * dim_per_block;
      size_t actual_sample_size = std::min(sample_size, n_points);

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

  // [OPT] Workspace struct to be used with thread_local
  struct EncodeWorkspace {
    std::vector<uint8_t> local_codes;
    Eigen::MatrixXf parallel_projections;  // Size: K x M (Flattened for cache)
    Eigen::VectorXf scores;

    void resize(uint32_t m, uint32_t k) {
      if (local_codes.size() != m) local_codes.resize(m);
      if (parallel_projections.cols() != (int)m || parallel_projections.rows() != (int)k)
        parallel_projections.resize(k, m);
      if (scores.size() != (int)k) scores.resize(k);
    }
  };

  void encode_anisotropic(const PointRange& data) {
    codes.resize(n_points * num_blocks);
    const int num_iterations = 2;

    parlay::parallel_for(0, n_points, [&](size_t i) {
      // [OPT] Thread-Local Storage
      // Allocates workspace ONCE per thread, preventing Malloc traffic in loop
      static thread_local EncodeWorkspace ws;
      ws.resize(num_blocks, num_clusters_per_block);

      const float* raw_ptr = reinterpret_cast<const float*>(data.location(i));
      Eigen::Map<const Eigen::VectorXf> x_full(raw_ptr, dim);
      float x_sq_norm = x_full.squaredNorm();
      float inv_norm_sq = (x_sq_norm > 1e-9f) ? (1.0f / x_sq_norm) : 0.0f;

      // 1. Initialize & Precompute Projections
      for (size_t b = 0; b < num_blocks; ++b) {
        size_t offset = b * dim_per_block;
        // Project all K centroids in block b onto x
        // Store in column b of workspace matrix (Size K)
        ws.parallel_projections.col((int)b).noalias() =
            codebooks[b] * x_full.segment((int)offset, (int)dim_per_block);
      }

      // Initial Greedy Selection
      for (size_t b = 0; b < num_blocks; ++b) {
        if constexpr (Metric) {  // Euclidean
          // dist = ||c||^2 - 2<c, x>
          ws.scores = codebook_norms[b] - (2.0f * ws.parallel_projections.col((int)b));
        } else {  // Inner Product
          // dist = -<c, x>
          ws.scores = -ws.parallel_projections.col((int)b);
        }

        Eigen::Index min_idx;
        ws.scores.minCoeff(&min_idx);
        ws.local_codes[b] = static_cast<uint8_t>(min_idx);
      }

      // 2. Coordinate Descent Optimization
      // Calculate initial Total Parallel Projection
      float total_parallel = 0.0f;
      for (size_t b = 0; b < num_blocks; ++b) {
        total_parallel += ws.parallel_projections((int)ws.local_codes[b], (int)b);
      }

      float weight = anisotropic_threshold * inv_norm_sq;

      for (int iter = 0; iter < num_iterations; ++iter) {
        for (size_t b = 0; b < num_blocks; ++b) {
          // Remove current block contribution
          float current_p_proj = ws.parallel_projections((int)ws.local_codes[b], (int)b);
          float other_parallel = total_parallel - current_p_proj;
          float target = x_sq_norm - other_parallel;

          const auto& projs = ws.parallel_projections.col((int)b);

          if constexpr (Metric) {  // Euclidean Base
            ws.scores = codebook_norms[b] - (2.0f * projs);
          } else {  // IP Base
            ws.scores = -projs;
          }

          // Apply Anisotropic Penalty
          ws.scores.array() += weight * (projs.array() - target).square();

          Eigen::Index best_idx;
          ws.scores.minCoeff(&best_idx);

          // Update
          ws.local_codes[b] = static_cast<uint8_t>(best_idx);
          total_parallel = other_parallel + ws.parallel_projections(best_idx, (int)b);
        }
      }

      // Store final codes
      for (size_t b = 0; b < num_blocks; ++b) {
        codes[i * num_blocks + b] = ws.local_codes[b];
      }
    });
  }
};

}  // namespace pq
}  // namespace mvsic
