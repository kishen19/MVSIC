// mvsic/core/quantization/pq.h
#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include <Eigen/Core>

#include "parlay/parallel.h"
#include "parlay/sequence.h"

#include "mvsic/core/utils/kmeans_util.h"

namespace mvsic {
namespace pq {

// =========================================================================
// Distance Kernels
// =========================================================================
namespace internal {
inline float distance_generic_float(const float* lut, const uint8_t* codes, uint32_t m,
                                    uint32_t K) {
  float dist = 0.0f;
  const float* lp = lut;
#pragma GCC unroll 16
  for (uint32_t b = 0; b < m; ++b) {
    dist += lp[codes[b]];
    lp += K;
  }
  return dist;
}

inline float distance_generic_int(const uint8_t* int_lut, const uint8_t* codes, uint32_t m,
                                  uint32_t K, float min_val, float scale) {
  uint32_t acc = 0;
  const uint8_t* lp = int_lut;
#pragma GCC unroll 16
  for (uint32_t b = 0; b < m; ++b) {
    acc += static_cast<uint32_t>(lp[codes[b]]);
    lp += K;
  }
  return min_val * static_cast<float>(m) + scale * static_cast<float>(acc);
}
}  // namespace internal

// =========================================================================
// Quantized Query Vector Type (owns LUT)
// =========================================================================

// Forward Declaration
template<bool Metric>
class Quantized_Point;

template<bool Metric>
class Quantized_Query {
 public:
  using distanceType = float;

  // Scalar-quantized LUT: 8-bit per entry, decode = min_val*m + scale*sum.
  alignas(64) std::vector<uint8_t> int_lut;  // [m * K]
  float min_val = 0.0f;
  float scale = 1.0f;
  uint32_t num_blocks = 0;
  uint32_t K = 0;

  Quantized_Query() = default;
  Quantized_Query(uint32_t m, uint32_t k) : num_blocks(m), K(k) {
    int_lut.resize(static_cast<size_t>(m) * static_cast<size_t>(K));
  }

  // Build int_lut from float LUT (min/max over all entries, scale = range/255).
  void set_lut_from_float(const float* float_lut, size_t size) {
    if (size == 0) return;
    float g_min = float_lut[0];
    float g_max = float_lut[0];
    for (size_t i = 1; i < size; ++i) {
      g_min = std::min(g_min, float_lut[i]);
      g_max = std::max(g_max, float_lut[i]);
    }
    min_val = g_min;
    const float range = g_max - g_min;
    scale = (range > 1e-9f) ? (range / 255.0f) : 1e-9f;
    for (size_t i = 0; i < size; ++i) {
      float v = (float_lut[i] - g_min) / scale;
      int_lut[i] = static_cast<uint8_t>(std::clamp(static_cast<int>(v + 0.5f), 0, 255));
    }
  }

  inline float distance(const Quantized_Point<Metric>& p) const {
    return internal::distance_generic_int(int_lut.data(), p.code_ptr, num_blocks, K, min_val,
                                          scale);
  }

  // Compute distances to every encoded vector in an encoded range.
  template<typename EncRange>
  void distances_all(const EncRange& db, float* out) const {
    const size_t N = static_cast<size_t>(db.size());
    const uint8_t* base = db.codes.data();
    const size_t stride = static_cast<size_t>(num_blocks);

    parlay::parallel_for(0, N, [&](size_t i) {
      out[i] = internal::distance_generic_int(int_lut.data(), base + i * stride, num_blocks, K,
                                              min_val, scale);
    });
  }
};

// ---------------------------------------------------------
// Quantized Point: (aux) handle into encoded codes
// ---------------------------------------------------------
template<bool Metric>
class Quantized_Point {
 public:
  const uint8_t* code_ptr = nullptr;  // points at codes[i*m]
  uint32_t num_blocks = 0;

  Quantized_Point() = default;
  Quantized_Point(const uint8_t* ptr, uint32_t m) : code_ptr(ptr), num_blocks(m) {}

  inline float distance(const Quantized_Query<Metric>& qq) const { return qq.distance(*this); }

  // Useful in random-access searches
  void prefetch() const { __builtin_prefetch(code_ptr, 0, 1); }

  // ParlayANN requirements
  bool same_as(const Quantized_Point<Metric>&) const { return false; }
  bool same_as(const Quantized_Query<Metric>&) const { return false; }
  bool is_metric() const { return Metric; }
};

// ---------------------------------------------------------
// Encoded Point Range
//  - owns codes + metadata
//  - Does NOT own centroids/codebooks.
//  - It is the output of PQ::encode(...).
// ---------------------------------------------------------
template<typename PointRange, bool Metric>
class Quantized_Point_Range {
 public:
  static constexpr bool is_fastscan = false;

  uint32_t num_blocks = 0;              // m
  uint32_t num_clusters_per_block = 0;  // K (<=256)
  uint32_t dim_per_block = 0;           // block_size

  size_t n_points = 0;
  size_t dim = 0;

  parlay::sequence<uint8_t> codes;  // [n_points * m]

  Quantized_Point_Range() = default;

  Quantized_Point<Metric> operator[](size_t i) const {
    return Quantized_Point<Metric>(&codes[i * static_cast<size_t>(num_blocks)], num_blocks);
  }

  inline uint32_t size() const noexcept { return static_cast<uint32_t>(n_points); }
  inline uint32_t get_dims() const noexcept { return static_cast<uint32_t>(dim); }
  inline size_t num_bytes_per_point() const noexcept { return static_cast<size_t>(num_blocks); }

  // Save/load encoded data only (codebooks live in PQ model).
  void save(std::ostream& out) const {
    out.write(reinterpret_cast<const char*>(&num_blocks), sizeof(num_blocks));
    out.write(reinterpret_cast<const char*>(&num_clusters_per_block),
              sizeof(num_clusters_per_block));
    out.write(reinterpret_cast<const char*>(&n_points), sizeof(n_points));
    out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    out.write(reinterpret_cast<const char*>(&dim_per_block), sizeof(dim_per_block));

    size_t codes_sz = codes.size();
    out.write(reinterpret_cast<const char*>(&codes_sz), sizeof(codes_sz));
    if (codes_sz) {
      out.write(reinterpret_cast<const char*>(codes.data()), codes_sz * sizeof(uint8_t));
    }
  }

  void load(std::istream& in) {
    in.read(reinterpret_cast<char*>(&num_blocks), sizeof(num_blocks));
    in.read(reinterpret_cast<char*>(&num_clusters_per_block), sizeof(num_clusters_per_block));
    in.read(reinterpret_cast<char*>(&n_points), sizeof(n_points));
    in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    in.read(reinterpret_cast<char*>(&dim_per_block), sizeof(dim_per_block));

    if (num_clusters_per_block > 256) {
      std::cerr << "Error: PQ K=" << num_clusters_per_block
                << " > 256 not supported with uint8 codes.\n";
      abort();
    }

    size_t codes_sz = 0;
    in.read(reinterpret_cast<char*>(&codes_sz), sizeof(codes_sz));
    codes.resize(codes_sz);
    if (codes_sz) {
      in.read(reinterpret_cast<char*>(codes.data()), codes_sz * sizeof(uint8_t));
    }
  }
};

// ---------------------------------------------------------
// PQ model: owns codebooks + norms, supports train/encode/query LUT.
// ---------------------------------------------------------
template<bool Metric>
class Model {
 public:
  static constexpr bool is_fastscan = false;
  // Model params / metadata
  uint32_t num_blocks = 0;              // m
  uint32_t num_clusters_per_block = 0;  // K
  uint32_t num_points_per_cluster = 0;  // subsample_mult (default 20)

  size_t dim = 0;
  size_t dim_per_block = 0;

  // Codebooks: [b] is (K x dim_per_block)
  std::vector<Eigen::MatrixXf> codebooks;
  std::vector<Eigen::VectorXf> codebook_norms;

  Model() = default;

  template<typename PointRange>
  Model(const PointRange& train_data, uint32_t block_size = 8, uint32_t k = 256,
        uint32_t subsample_mult = 20) {
    train(train_data, block_size, k, subsample_mult);
  }

  template<typename PointRange>
  void train(const PointRange& train_data, uint32_t block_size = 8, uint32_t k = 256,
             uint32_t subsample_mult = 20) {
    num_clusters_per_block = k;
    num_points_per_cluster = subsample_mult;

    dim = train_data.get_dims();
    dim_per_block = block_size;

    if (num_clusters_per_block > 256) {
      std::cerr << "Error: PQ K=" << num_clusters_per_block
                << " > 256 not supported with uint8 codes.\n";
      abort();
    }
    if (dim_per_block == 0 || (dim % dim_per_block) != 0) {
      std::cerr << "Error: PQ Dimension " << dim << " not divisible by block_size=" << dim_per_block
                << "\n";
      abort();
    }

    num_blocks = static_cast<uint32_t>(dim / dim_per_block);

    codebooks.resize(num_blocks);
    codebook_norms.resize(num_blocks);

    const size_t n_points = train_data.size();
    const size_t sample_size =
        std::min(static_cast<size_t>(num_clusters_per_block) * num_points_per_cluster, n_points);

    // Train each block independently
    parlay::parallel_for(0, num_blocks, [&](size_t b) {
      const size_t offset = b * dim_per_block;

      // Subsample data for faster training
      parlay::sequence<parlay::sequence<float>> subsample(sample_size);
      std::mt19937 rng(static_cast<unsigned>(b + 1));
      std::uniform_int_distribution<size_t> distu(0, n_points - 1);

      for (size_t i = 0; i < sample_size; ++i) {
        const size_t pid = distu(rng);
        const float* raw = reinterpret_cast<const float*>(train_data.location(pid));
        parlay::sequence<float> vec(dim_per_block);
        std::copy(raw + offset, raw + offset + dim_per_block, vec.begin());
        subsample[i] = std::move(vec);
      }

      // Always use L2-based kmeans, even for Inner Product/Cosine Similarity
      // as subvectors are not L2-normalized typically.
      auto [centers, _] = mvsic::kmeans_subsample_assign_only<true>(
          subsample, num_clusters_per_block, sample_size, false);

      codebooks[b] = Eigen::MatrixXf(static_cast<Eigen::Index>(num_clusters_per_block),
                                     static_cast<Eigen::Index>(dim_per_block));
      for (size_t c = 0; c < centers.size(); ++c) {
        for (size_t d = 0; d < dim_per_block; ++d) {
          codebooks[b](static_cast<Eigen::Index>(c), static_cast<Eigen::Index>(d)) = centers[c][d];
        }
      }
      // pad any unused centers with 0s
      for (size_t c = centers.size(); c < num_clusters_per_block; ++c) {
        codebooks[b].row(static_cast<Eigen::Index>(c)).setZero();
      }
      codebook_norms[b] = codebooks[b].rowwise().squaredNorm();
    });
  }

  // Encode a dataset using current model into an encoded range object
  template<typename PointRange>
  Quantized_Point_Range<PointRange, Metric> encode(const PointRange& data) const {
    Quantized_Point_Range<PointRange, Metric> enc;
    enc.num_blocks = num_blocks;
    enc.num_clusters_per_block = num_clusters_per_block;
    enc.dim_per_block = static_cast<uint32_t>(dim_per_block);
    enc.n_points = data.size();
    enc.dim = data.get_dims();

    if (enc.dim != dim) {
      std::cerr << "Error: PQ::encode data dim=" << enc.dim << " differs from trained dim=" << dim
                << "\n";
      abort();
    }

    enc.codes.resize(enc.n_points * static_cast<size_t>(num_blocks));

    // Per-thread scratch (avoids reallocation churn)
    parlay::parallel_for(0, enc.n_points, [&](size_t i) {
      const float* raw_point_ptr = reinterpret_cast<const float*>(data.location(i));

      static thread_local Eigen::VectorXf dot_products;
      if (dot_products.size() != static_cast<Eigen::Index>(num_clusters_per_block)) {
        dot_products.resize(static_cast<Eigen::Index>(num_clusters_per_block));
      }

      uint8_t* out_codes = enc.codes.data() + i * static_cast<size_t>(num_blocks);

      for (uint32_t b = 0; b < num_blocks; ++b) {
        const size_t offset = static_cast<size_t>(b) * dim_per_block;
        Eigen::Map<const Eigen::VectorXf> p_sub(raw_point_ptr + offset,
                                                static_cast<Eigen::Index>(dim_per_block));

        dot_products.noalias() = codebooks[b] * p_sub;

        float best_val = std::numeric_limits<float>::max();
        uint8_t best_code = 0;
        const auto& c_sq_norms = codebook_norms[b];

        for (uint32_t c = 0; c < num_clusters_per_block; ++c) {
          float val;
          if constexpr (Metric) {  // L2: ||c||^2 - 2<x,c> (q_sq handled in LUT)
            val = c_sq_norms[static_cast<Eigen::Index>(c)] -
                  2.0f * dot_products[static_cast<Eigen::Index>(c)];
          } else {  // IP: -<x,c>
            val = -dot_products[static_cast<Eigen::Index>(c)];
          }
          if (val < best_val) {
            best_val = val;
            best_code = static_cast<uint8_t>(c);
          }
        }
        out_codes[b] = best_code;
      }
    });

    return enc;
  }

  // LUT for a single query vector (build float LUT, then scalar-quantize to 8-bit).
  template<typename PointTy>
  Quantized_Query<Metric> quantize_query(const PointTy& query) const {
    Quantized_Query<Metric> qq(num_blocks, num_clusters_per_block);
    std::vector<float> float_lut(static_cast<size_t>(num_blocks) * num_clusters_per_block);

    Eigen::VectorXf q_sub(static_cast<Eigen::Index>(dim_per_block));
    Eigen::VectorXf dot_products(static_cast<Eigen::Index>(num_clusters_per_block));

    for (uint32_t b = 0; b < num_blocks; ++b) {
      const size_t offset = static_cast<size_t>(b) * dim_per_block;
      for (size_t j = 0; j < dim_per_block; ++j) {
        q_sub[static_cast<Eigen::Index>(j)] = query[offset + j];
      }

      dot_products.noalias() = codebooks[b] * q_sub;

      float* lut_segment = float_lut.data() + static_cast<size_t>(b) * num_clusters_per_block;

      if constexpr (Metric) {
        const float q_sq = q_sub.squaredNorm();
        for (uint32_t c = 0; c < num_clusters_per_block; ++c) {
          lut_segment[c] = codebook_norms[b][static_cast<Eigen::Index>(c)] -
                           2.0f * dot_products[static_cast<Eigen::Index>(c)] + q_sq;
        }
      } else {
        for (uint32_t c = 0; c < num_clusters_per_block; ++c) {
          lut_segment[c] = -dot_products[static_cast<Eigen::Index>(c)];
        }
      }
    }

    qq.set_lut_from_float(float_lut.data(), float_lut.size());
    return qq;
  }

  Quantized_Query<Metric> quantize_query(const float* qptr) const {
    Quantized_Query<Metric> qq(num_blocks, num_clusters_per_block);
    std::vector<float> float_lut(static_cast<size_t>(num_blocks) * num_clusters_per_block);

    Eigen::VectorXf dot_products(static_cast<Eigen::Index>(num_clusters_per_block));

    for (uint32_t b = 0; b < num_blocks; ++b) {
      const float* sub = qptr + static_cast<size_t>(b) * dim_per_block;
      Eigen::Map<const Eigen::VectorXf> q_map(sub, static_cast<Eigen::Index>(dim_per_block));
      dot_products.noalias() = codebooks[b] * q_map;

      float* lut_segment = float_lut.data() + static_cast<size_t>(b) * num_clusters_per_block;

      if constexpr (Metric) {
        const float q_sq = q_map.squaredNorm();
        for (uint32_t c = 0; c < num_clusters_per_block; ++c) {
          lut_segment[c] = codebook_norms[b][static_cast<Eigen::Index>(c)] -
                           2.0f * dot_products[static_cast<Eigen::Index>(c)] + q_sq;
        }
      } else {
        for (uint32_t c = 0; c < num_clusters_per_block; ++c) {
          lut_segment[c] = -dot_products[static_cast<Eigen::Index>(c)];
        }
      }
    }

    qq.set_lut_from_float(float_lut.data(), float_lut.size());
    return qq;
  }

  // Batch LUT building for multiple queries (float LUT then scalar-quantize each).
  template<typename PointCloudTy>
  void quantize_query_batch(const PointCloudTy& query_cloud,
                            parlay::sequence<Quantized_Query<Metric>>& out_luts) const {
    const uint32_t num_q = query_cloud.size();
    const uint32_t dims = query_cloud.get_dims();
    const float* base = query_cloud.data();

    out_luts.clear();
    out_luts.reserve(num_q);
    for (uint32_t i = 0; i < num_q; ++i) {
      out_luts.emplace_back(num_blocks, num_clusters_per_block);
    }

    const size_t lut_size = static_cast<size_t>(num_blocks) * num_clusters_per_block;
    std::vector<float> float_luts(num_q * lut_size);

    using RowMajorMat = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    RowMajorMat Q_sub(static_cast<Eigen::Index>(num_q), static_cast<Eigen::Index>(dim_per_block));
    Eigen::MatrixXf dot_products(static_cast<Eigen::Index>(num_clusters_per_block),
                                 static_cast<Eigen::Index>(num_q));
    Eigen::VectorXf q_sq;

    for (uint32_t b = 0; b < num_blocks; ++b) {
      const uint32_t offset = b * static_cast<uint32_t>(dim_per_block);
      for (uint32_t i = 0; i < num_q; ++i) {
        const float* row_ptr = base + static_cast<size_t>(i) * dims + offset;
        std::memcpy(&Q_sub(static_cast<Eigen::Index>(i), 0), row_ptr,
                    sizeof(float) * dim_per_block);
      }

      if constexpr (Metric) {
        q_sq = Q_sub.rowwise().squaredNorm();
      }

      dot_products.noalias() = codebooks[b] * Q_sub.transpose();

      for (uint32_t i = 0; i < num_q; ++i) {
        float* lut_ptr =
            float_luts.data() + i * lut_size + static_cast<size_t>(b) * num_clusters_per_block;
        const Eigen::Index col = static_cast<Eigen::Index>(i);
        if constexpr (Metric) {
          for (uint32_t c = 0; c < num_clusters_per_block; ++c) {
            lut_ptr[c] = codebook_norms[b][static_cast<Eigen::Index>(c)] -
                         2.0f * dot_products(c, col) + q_sq[col];
          }
        } else {
          for (uint32_t c = 0; c < num_clusters_per_block; ++c) {
            lut_ptr[c] = -dot_products(c, col);
          }
        }
      }
    }

    for (uint32_t i = 0; i < num_q; ++i) {
      out_luts[i].set_lut_from_float(float_luts.data() + i * lut_size, lut_size);
    }
  }

  // Save/load model only (codebooks + metadata). Encoded ranges are saved separately.
  void save(std::ostream& out) const {
    out.write(reinterpret_cast<const char*>(&num_blocks), sizeof(num_blocks));
    out.write(reinterpret_cast<const char*>(&num_clusters_per_block),
              sizeof(num_clusters_per_block));
    out.write(reinterpret_cast<const char*>(&num_points_per_cluster),
              sizeof(num_points_per_cluster));
    out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    out.write(reinterpret_cast<const char*>(&dim_per_block), sizeof(dim_per_block));

    for (uint32_t b = 0; b < num_blocks; ++b) {
      // store rows/cols for robustness
      size_t rs = static_cast<size_t>(codebooks[b].rows());
      size_t cs = static_cast<size_t>(codebooks[b].cols());
      out.write(reinterpret_cast<const char*>(&rs), sizeof(rs));
      out.write(reinterpret_cast<const char*>(&cs), sizeof(cs));
      out.write(reinterpret_cast<const char*>(codebooks[b].data()), rs * cs * sizeof(float));
    }
  }

  void load(std::istream& in) {
    in.read(reinterpret_cast<char*>(&num_blocks), sizeof(num_blocks));
    in.read(reinterpret_cast<char*>(&num_clusters_per_block), sizeof(num_clusters_per_block));
    in.read(reinterpret_cast<char*>(&num_points_per_cluster), sizeof(num_points_per_cluster));
    in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    in.read(reinterpret_cast<char*>(&dim_per_block), sizeof(dim_per_block));

    if (num_clusters_per_block > 256) {
      std::cerr << "Error: PQ K=" << num_clusters_per_block
                << " > 256 not supported with uint8 codes.\n";
      abort();
    }

    codebooks.resize(num_blocks);
    codebook_norms.resize(num_blocks);

    for (uint32_t b = 0; b < num_blocks; ++b) {
      size_t rs = 0, cs = 0;
      in.read(reinterpret_cast<char*>(&rs), sizeof(rs));
      in.read(reinterpret_cast<char*>(&cs), sizeof(cs));
      codebooks[b] = Eigen::MatrixXf(static_cast<Eigen::Index>(rs), static_cast<Eigen::Index>(cs));
      in.read(reinterpret_cast<char*>(codebooks[b].data()), rs * cs * sizeof(float));
      codebook_norms[b] = codebooks[b].rowwise().squaredNorm();
    }
  }
};

}  // namespace pq
}  // namespace mvsic
