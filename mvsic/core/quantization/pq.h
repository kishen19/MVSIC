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
// Distance Kernel
// ---------------------------------------------------------
inline float distance_generic(const float* lut, const uint8_t* codes, uint32_t m, uint32_t K) {
  float dist = 0.0f;
  const float* lp = lut;
#pragma GCC unroll 16
  for (uint32_t b = 0; b < m; ++b) {
    dist += lp[codes[b]];
    lp += K;
  }
  return dist;
}

// ---------------------------------------------------------
// Quantized Query Vector Type
// ---------------------------------------------------------

// Forward Declaration
template<bool Metric>
class Quantized_Point;

template<bool Metric>
class Quantized_Query {
 public:
  using distanceType = float;

  std::vector<float> lut;  // Lookup Table
  uint32_t num_blocks;     // Number of blocks
  uint32_t K;              // Number of clusters per block

  Quantized_Query(uint32_t m, uint32_t k = 256) : num_blocks(m), K(k) {
    lut.resize(static_cast<size_t>(m) * static_cast<size_t>(K));
  }

  inline float distance(const Quantized_Point<Metric>& p) const {
    return distance_generic(lut.data(), p.code_ptr, num_blocks, K);
  }
};

// ---------------------------------------------------------
// Quantized Point: Aux data type representing a compressed vector
// ---------------------------------------------------------
template<bool Metric>
class Quantized_Point {
 public:
  const uint8_t* code_ptr;  // codebook of this vector
  uint32_t num_blocks;      // Number of blocks

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

// ---------------------------------------------------------
// Quantized Point Range (Main Container)
// ---------------------------------------------------------
template<typename PointRange, bool Metric>
class Quantized_Point_Range {
 public:
  uint32_t num_blocks;
  uint32_t num_clusters_per_block;
  uint32_t num_points_per_cluster;  // Default 20

  size_t n_points;
  size_t dim;
  size_t dim_per_block;  // = dim/num_blocks

  std::vector<Eigen::MatrixXf> codebooks;
  std::vector<Eigen::VectorXf> codebook_norms;
  parlay::sequence<uint8_t> codes;

  Quantized_Point_Range() {}

  Quantized_Point_Range(const PointRange& data, uint32_t m = 8, uint32_t k = 256,
                        uint32_t subsample_mult = 20) :
      num_blocks(m), num_clusters_per_block(k), num_points_per_cluster(subsample_mult) {

    n_points = data.size();
    dim = data.get_dims();

    // With uint8_t codes, we cannot represent >256 clusters per block.
    if (num_clusters_per_block > 256) {
      std::cerr << "Error: PQ num_clusters_per_block=" << num_clusters_per_block
                << " > 256 not supported with uint8 codes.\n";
      abort();
    }

    if (dim % num_blocks != 0) {
      std::cerr << "Error: PQ Dimension " << dim << " not divisible by " << num_blocks << std::endl;
      abort();
    }
    dim_per_block = dim / num_blocks;

    // Train the PQ
    train(data);
    // Encode and compute the codebooks
    encode_database(data);
  }

  Quantized_Point<Metric> operator[](size_t i) const {
    return Quantized_Point<Metric>(&codes[i * num_blocks], num_blocks);
  }

  // Computes the LUT
  template<typename PointTy>
  Quantized_Query<Metric> quantize_query(const PointTy& query) const {
    Quantized_Query<Metric> qq(num_blocks, num_clusters_per_block);

    Eigen::VectorXf q_sub(dim_per_block);
    Eigen::VectorXf dot_products(num_clusters_per_block);

    for (uint32_t b = 0; b < num_blocks; ++b) {
      const size_t offset = static_cast<size_t>(b) * dim_per_block;
      for (size_t j = 0; j < dim_per_block; ++j) {
        q_sub[static_cast<Eigen::Index>(j)] = query[offset + j];
      }
      dot_products.noalias() = codebooks[b] * q_sub;

      Eigen::Map<Eigen::VectorXf> lut_segment(
          &qq.lut[static_cast<size_t>(b) * num_clusters_per_block], num_clusters_per_block);

      if constexpr (Metric) {
        const float q_sq = q_sub.squaredNorm();
        lut_segment.noalias() = codebook_norms[b] - (2.0f * dot_products);
        lut_segment.array() += q_sq;
      } else {
        lut_segment.noalias() = -dot_products;
      }
    }

    return qq;
  }

  Quantized_Query<Metric> quantize_query(const float* qptr) const {
    Quantized_Query<Metric> qq(num_blocks, num_clusters_per_block);

    Eigen::VectorXf dot_products(num_clusters_per_block);

    for (uint32_t b = 0; b < num_blocks; ++b) {
      const float* sub = qptr + static_cast<size_t>(b) * dim_per_block;
      Eigen::Map<const Eigen::VectorXf> q_map(sub, dim_per_block);
      dot_products.noalias() = codebooks[b] * q_map;

      Eigen::Map<Eigen::VectorXf> lut_segment(
          &qq.lut[static_cast<size_t>(b) * num_clusters_per_block], num_clusters_per_block);

      if constexpr (Metric) {
        const float q_sq = q_map.squaredNorm();
        lut_segment.noalias() = codebook_norms[b] - (2.0f * dot_products);
        lut_segment.array() += q_sq;
      } else {
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

    out_luts.clear();
    out_luts.reserve(num_q);
    for (uint32_t i = 0; i < num_q; ++i) {
      out_luts.emplace_back(num_blocks, num_clusters_per_block);
    }

    // RowMajor so each row is contiguous; memcpy fills are fast.
    using RowMajorMat = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    RowMajorMat Q_sub(num_q, dim_per_block);
    // [K x num_q]
    Eigen::MatrixXf dot_products(num_clusters_per_block, num_q);
    // [num_q] (only used when Metric==true)
    Eigen::VectorXf q_sq;
    for (uint32_t b = 0; b < num_blocks; ++b) {
      const uint32_t offset = b * dim_per_block;
      // Fill Q_sub from contiguous cloud memory
      for (uint32_t i = 0; i < num_q; ++i) {
        const float* row_ptr = base + static_cast<size_t>(i) * dims + offset;
        std::memcpy(&Q_sub(i, 0), row_ptr, sizeof(float) * dim_per_block);
      }
      if constexpr (Metric) {
        // Compute once per query vector for this block
        q_sq = Q_sub.rowwise().squaredNorm();
      }
      // GEMM: [K x d] * [d x num_q] -> [K x num_q]
      dot_products.noalias() = codebooks[b] * Q_sub.transpose();
      // Write LUT segments for all query vectors
      const size_t lut_block_base = static_cast<size_t>(b) * num_clusters_per_block;
      for (uint32_t i = 0; i < num_q; ++i) {
        float* lut_ptr = out_luts[i].lut.data() + lut_block_base;
        Eigen::Map<Eigen::VectorXf> lut_segment(lut_ptr, num_clusters_per_block);
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

    if (num_clusters_per_block > 256) {
      std::cerr << "Error: PQ num_clusters_per_block=" << num_clusters_per_block
                << " > 256 not supported with uint8 codes.\n";
      abort();
    }

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

    size_t sample_size =
        std::min(static_cast<size_t>(num_clusters_per_block) * num_points_per_cluster, n_points);

    parlay::parallel_for(0, num_blocks, [&](size_t b) {
      // Subsampling here sequentially: Faster
      size_t offset = b * dim_per_block;
      parlay::sequence<parlay::sequence<float>> subsample(sample_size);
      std::mt19937 rng(b + 1);
      std::uniform_int_distribution<size_t> dist(0, n_points - 1);
      for (size_t i = 0; i < sample_size; ++i) {
        size_t pid = dist(rng);
        const float* raw_data = reinterpret_cast<const float*>(data.location(pid));
        parlay::sequence<float> vec(dim_per_block);
        std::copy(raw_data + offset, raw_data + offset + dim_per_block, vec.begin());
        subsample[i] = std::move(vec);
      }

      auto [centers, _] = mvsic::kmeans_subsample_assign_only<Metric>(
          subsample, num_clusters_per_block, sample_size, false);

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
