#pragma once

#include <immintrin.h>
#include <Eigen/Core>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <cstring>
#include <random>
#include <fstream>

#include "parlay/parallel.h"
#include "parlay/sequence.h"
#include "mvsic/core/utils/kmeans_util.h"

namespace mvsic {
namespace fastscan {

template<bool Metric>
class Quantized_Point;

// ---------------------------------------------------------
// FastScan Quantized Query (8-bit Fixed Point)
// ---------------------------------------------------------
template<bool Metric>
class Quantized_Query {
 public:
  using distanceType = float;

  // 8-bit LUTs packed for SIMD: [num_blocks][K=16]
  alignas(64) std::vector<uint8_t> int_lut;

  float min_dist;
  float scale;
  uint32_t num_blocks;
  static constexpr uint32_t K = 16;

  Quantized_Query(uint32_t m) : num_blocks(m) { int_lut.resize(static_cast<size_t>(m) * K); }

  /**
   * @brief Converts the integer accumulated distance back to floating point.
   */
  inline float decode(uint16_t int_dist) const {
    return (min_dist * static_cast<float>(num_blocks)) + (static_cast<float>(int_dist) * scale);
  }

  /**
   * @brief Scalar distance used by the generic wrapper.h distance() loop.
   */
  inline float distance(const Quantized_Point<Metric>& p) const {
    uint16_t acc = 0;
    // p.code_ptr points to the start of the block for the 64-vector strip.
    // The stride between blocks in a strip is 32 bytes (64 vectors * 4 bits).
    for (uint32_t b = 0; b < num_blocks; ++b) {
      uint8_t packed = p.code_ptr[b * 32];
      uint8_t code = (p.lane_idx % 2 == 0) ? (packed & 0x0F) : (packed >> 4);
      acc += static_cast<uint16_t>(int_lut[b * K + code]);
    }
    return decode(acc);
  }
};

// ---------------------------------------------------------
// FastScan Point Handle
// ---------------------------------------------------------
template<bool Metric>
class Quantized_Point {
 public:
  const uint8_t* code_ptr;
  uint32_t lane_idx;

  Quantized_Point(const uint8_t* ptr, uint32_t lane) : code_ptr(ptr), lane_idx(lane) {}

  inline float distance(const Quantized_Query<Metric>& qq) const { return qq.distance(*this); }

  void prefetch() const { __builtin_prefetch(code_ptr, 0, 3); }

  bool same_as(const Quantized_Point<Metric>& q) const { return false; }
  bool same_as(const Quantized_Query<Metric>& q) const { return false; }
  bool is_metric() const { return Metric; }
};

// ---------------------------------------------------------
// FastScan Point Range
// ---------------------------------------------------------
template<typename PointRange, bool Metric>
class Quantized_Point_Range {
 public:
  static constexpr bool is_fastscan = true;
  uint32_t num_blocks;
  static constexpr uint32_t K = 16;
  size_t n_points, dim, dim_per_block;

  std::vector<Eigen::MatrixXf> codebooks;
  std::vector<Eigen::VectorXf> codebook_norms;

  // Storage: Interleaved strips of 64 vectors.
  // Each strip byte contains two 4-bit codes.
  parlay::sequence<uint8_t> packed_codes;

  Quantized_Point_Range() {}

  Quantized_Point_Range(const PointRange& data, uint32_t block_size = 32) {
    n_points = data.size();
    dim = data.get_dims();
    dim_per_block = block_size;
    num_blocks = dim / block_size;

    train(data);
    encode_database_4bit(data);
  }

  /**
   * @brief Accessor for individual vectors.
   */
  Quantized_Point<Metric> operator[](size_t i) const {
    size_t strip_idx = i / 64;
    size_t lane_idx = i % 64;
    const uint8_t* ptr = &packed_codes[strip_idx * num_blocks * 32 + (lane_idx / 2)];
    return Quantized_Point<Metric>(ptr, static_cast<uint32_t>(lane_idx));
  }

  template<typename PointTy>
  Quantized_Query<Metric> quantize_query(const PointTy& query) const {
    Quantized_Query<Metric> qq(num_blocks);
    std::vector<float> float_lut(num_blocks * K);
    float g_min = std::numeric_limits<float>::max(), g_max = std::numeric_limits<float>::lowest();

    for (uint32_t b = 0; b < num_blocks; ++b) {
      // Pull data into Eigen vector using indexing
      Eigen::VectorXf q_sub(dim_per_block);
      const size_t offset = static_cast<size_t>(b) * dim_per_block;
      for (size_t j = 0; j < dim_per_block; ++j) {
        q_sub[static_cast<Eigen::Index>(j)] = query[offset + j];
      }

      Eigen::VectorXf dots = codebooks[b] * q_sub;
      float q_sq = Metric ? q_sub.squaredNorm() : 0.0f;
      for (uint32_t i = 0; i < K; ++i) {
        float val = Metric ? (codebook_norms[b][i] - 2.0f * dots[i] + q_sq) : -dots[i];
        float_lut[b * K + i] = val;
        g_min = std::min(g_min, val);
        g_max = std::max(g_max, val);
      }
    }
    return finish_quantize_query(qq, float_lut, g_min, g_max);
  }

  /**
   * @brief Fixed-Point Query Quantization.
   */
  Quantized_Query<Metric> quantize_query(const float* qptr) const {
    Quantized_Query<Metric> qq(num_blocks);
    std::vector<float> float_lut(num_blocks * K);

    float global_min = std::numeric_limits<float>::max();
    float global_max = std::numeric_limits<float>::lowest();

    for (uint32_t b = 0; b < num_blocks; ++b) {
      Eigen::Map<const Eigen::VectorXf> q_sub(qptr + b * dim_per_block, dim_per_block);
      Eigen::VectorXf dots = codebooks[b] * q_sub;
      float q_sq = Metric ? q_sub.squaredNorm() : 0.0f;

      for (uint32_t i = 0; i < K; ++i) {
        float val = Metric ? (codebook_norms[b][i] - 2.0f * dots[i] + q_sq) : -dots[i];
        float_lut[b * K + i] = val;
        global_min = std::min(global_min, val);
        global_max = std::max(global_max, val);
      }
    }

    qq.min_dist = global_min;
    float range = (global_max - global_min);
    qq.scale = std::max(1e-6f, range / 255.0f);

    for (size_t i = 0; i < float_lut.size(); ++i) {
      qq.int_lut[i] = static_cast<uint8_t>((float_lut[i] - global_min) / qq.scale);
    }
    return qq;
  }

  template<typename PointCloudTy>
  void quantize_query_batch(const PointCloudTy& query_cloud,
                            parlay::sequence<Quantized_Query<Metric>>& out_luts) const {
    const uint32_t num_q = query_cloud.size();
    const uint32_t dims = query_cloud.get_dims();
    const float* base = query_cloud.data();

    out_luts.clear();
    for (uint32_t i = 0; i < num_q; ++i)
      out_luts.emplace_back(num_blocks);

    using RowMajorMat = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    RowMajorMat Q_sub(num_q, dim_per_block);
    Eigen::MatrixXf dot_products(K, num_q);

    // We need to store intermediate floats to compute global min/max for scaling per query
    std::vector<std::vector<float>> batch_float_luts(num_q, std::vector<float>(num_blocks * K));
    std::vector<float> q_mins(num_q, std::numeric_limits<float>::max());
    std::vector<float> q_maxs(num_q, std::numeric_limits<float>::lowest());

    for (uint32_t b = 0; b < num_blocks; ++b) {
      const uint32_t offset = b * dim_per_block;
      for (uint32_t i = 0; i < num_q; ++i) {
        std::memcpy(&Q_sub(i, 0), base + static_cast<size_t>(i) * dims + offset,
                    sizeof(float) * dim_per_block);
      }

      Eigen::VectorXf q_sq;
      if constexpr (Metric) q_sq = Q_sub.rowwise().squaredNorm();

      dot_products.noalias() = codebooks[b] * Q_sub.transpose();

      for (uint32_t i = 0; i < num_q; ++i) {
        for (uint32_t k_idx = 0; k_idx < K; ++k_idx) {
          float val = Metric ? (codebook_norms[b][k_idx] - 2.0f * dot_products(k_idx, i) + q_sq[i])
                             : -dot_products(k_idx, i);
          batch_float_luts[i][b * K + k_idx] = val;
          q_mins[i] = std::min(q_mins[i], val);
          q_maxs[i] = std::max(q_maxs[i], val);
        }
      }
    }

    // Finalize all queries in the batch with their specific scaling factors
    for (uint32_t i = 0; i < num_q; ++i) {
      finish_quantize_query(out_luts[i], batch_float_luts[i], q_mins[i], q_maxs[i]);
    }
  }

  /**
   * @brief AVX-512 FASTSCAN KERNEL.
   */
  void scan_64_chunk(const Quantized_Query<Metric>& q, const uint8_t* codes_ptr,
                     float* results) const {
    __m512i acc_even = _mm512_setzero_si512();
    __m512i acc_odd = _mm512_setzero_si512();
    __m512i low_mask = _mm512_set1_epi8(0x0F);

    for (uint32_t b = 0; b < num_blocks; ++b) {
      __m256i packed_raw = _mm256_loadu_si256((const __m256i*)codes_ptr);
      __m512i packed = _mm512_castsi256_si512(packed_raw);
      packed = _mm512_inserti64x4(packed, packed_raw, 1);
      codes_ptr += 32;

      __m512i lut = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i*)&q.int_lut[b * K]));

      // Even Lanes
      __m512i codes_even = _mm512_and_si512(packed, low_mask);
      __m512i scores_even = _mm512_shuffle_epi8(lut, codes_even);
      acc_even = _mm512_add_epi16(acc_even,
                                  _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(scores_even, 0)));

      // Odd Lanes
      __m512i codes_odd = _mm512_and_si512(_mm512_srli_epi16(packed, 4), low_mask);
      __m512i scores_odd = _mm512_shuffle_epi8(lut, codes_odd);
      acc_odd =
          _mm512_add_epi16(acc_odd, _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(scores_odd, 0)));
    }

    alignas(64) uint16_t raw_even[32], raw_odd[32];
    _mm512_store_si512((__m512i*)raw_even, acc_even);
    _mm512_store_si512((__m512i*)raw_odd, acc_odd);

    for (int i = 0; i < 32; ++i) {
      results[i * 2] = q.decode(raw_even[i]);
      results[i * 2 + 1] = q.decode(raw_odd[i]);
    }
  }

  /**
   * @brief Manual horizontal minimum using AVX-512 shuffles.
   * Folds the 512-bit register into the lowest 16-bit lane via a tournament bracket.
   */
  inline uint16_t hmin_512_epu16(__m512i v) const {
    // 1. Fold 512 bits to 256 bits
    // Compare lanes [3,2,1,0] with [1,0,3,2] to cross-pollinate
    v = _mm512_min_epu16(v, _mm512_shuffle_i32x4(v, v, _MM_SHUFFLE(1, 0, 3, 2)));

    // 2. Fold 256 bits to 128 bits
    // Compare lanes with [0,0,1,1]
    v = _mm512_min_epu16(v, _mm512_shuffle_i32x4(v, v, _MM_SHUFFLE(0, 0, 1, 1)));

    // 3. Extract the lowest 128-bit lane for the final stage
    __m128i v128 = _mm512_castsi512_si128(v);

    // 4. Final 128-bit reduction using standard SSE/AVX shuffles
    v128 = _mm_min_epu16(v128, _mm_srli_si128(v128, 8));
    v128 = _mm_min_epu16(v128, _mm_srli_si128(v128, 4));
    v128 = _mm_min_epu16(v128, _mm_srli_si128(v128, 2));

    return static_cast<uint16_t>(_mm_extract_epi16(v128, 0));
  }

  float scan_64_chunk_min(const Quantized_Query<Metric>& q, const uint8_t* codes_ptr) const {
    // Initialize accumulators with 0 to start fresh
    __m512i acc_even = _mm512_setzero_si512();
    __m512i acc_odd = _mm512_setzero_si512();
    __m512i low_mask = _mm512_set1_epi8(0x0F);

    for (uint32_t b = 0; b < num_blocks; ++b) {
      __m256i packed_raw = _mm256_loadu_si256((const __m256i*)codes_ptr);
      __m512i packed = _mm512_castsi256_si512(packed_raw);
      packed = _mm512_inserti64x4(packed, packed_raw, 1);
      codes_ptr += 32;

      __m512i lut = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i*)&q.int_lut[b * K]));

      // Extract and lookup even lanes
      __m512i codes_even = _mm512_and_si512(packed, low_mask);
      __m512i scores_even = _mm512_shuffle_epi8(lut, codes_even);
      acc_even = _mm512_add_epi16(acc_even,
                                  _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(scores_even, 0)));

      // Extract and lookup odd lanes
      __m512i codes_odd = _mm512_and_si512(_mm512_srli_epi16(packed, 4), low_mask);
      __m512i scores_odd = _mm512_shuffle_epi8(lut, codes_odd);
      acc_odd =
          _mm512_add_epi16(acc_odd, _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(scores_odd, 0)));
    }

    // Combine even and odd lane results to find the minimum across all 64 vectors
    __m512i global_min_v = _mm512_min_epu16(acc_even, acc_odd);

    // Reduce the combined register to a single scalar integer
    uint16_t final_min_int = hmin_512_epu16(global_min_v);

    return q.decode(final_min_int);
  }

  /**
   * @brief Accumulates distances for 64 vectors and updates a running minimum
   * register entirely within the SIMD unit.
   */
  inline __m512i scan_64_running_min(const Quantized_Query<Metric>& q, const uint8_t* codes_ptr,
                                     __m512i current_min_v) const {
    __m512i acc_even = _mm512_setzero_si512();
    __m512i acc_odd = _mm512_setzero_si512();
    __m512i low_mask = _mm512_set1_epi8(0x0F);

    for (uint32_t b = 0; b < num_blocks; ++b) {
      // Load codes (1 strip = 32 bytes = 64 vectors)
      __m256i packed_raw = _mm256_loadu_si256((const __m256i*)codes_ptr);
      __m512i packed = _mm512_inserti64x4(_mm512_castsi256_si512(packed_raw), packed_raw, 1);
      codes_ptr += 32;

      // Broadcast LUT for this block
      __m512i lut = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i*)&q.int_lut[b * K]));

      // Gather scores and add to 16-bit accumulators
      acc_even = _mm512_add_epi16(
          acc_even, _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(
                        _mm512_shuffle_epi8(lut, _mm512_and_si512(packed, low_mask)), 0)));
      acc_odd = _mm512_add_epi16(
          acc_odd,
          _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(
              _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_srli_epi16(packed, 4), low_mask)),
              0)));
    }

    // Update the running minimum register (unsigned 16-bit)
    current_min_v = _mm512_min_epu16(current_min_v, acc_even);
    current_min_v = _mm512_min_epu16(current_min_v, acc_odd);
    return current_min_v;
  }

  /**
   * @brief Final reduction: Horizontal min + float decode.
   */
  inline float reduce_running_min(const Quantized_Query<Metric>& q, __m512i running_min_v) const {
    uint16_t final_min_int = hmin_512_epu16(running_min_v);
    return q.decode(final_min_int);
  }

  inline void scan_64_dual_query(const Quantized_Query<Metric>& q1,
                                 const Quantized_Query<Metric>& q2, const uint8_t* codes_ptr,
                                 __m512i& min_v1, __m512i& min_v2) const {
    __m512i acc1_even = _mm512_setzero_si512();
    __m512i acc1_odd = _mm512_setzero_si512();
    __m512i acc2_even = _mm512_setzero_si512();
    __m512i acc2_odd = _mm512_setzero_si512();
    __m512i low_mask = _mm512_set1_epi8(0x0F);

    for (uint32_t b = 0; b < num_blocks; ++b) {
      __m256i raw = _mm256_loadu_si256((const __m256i*)codes_ptr);
      __m512i packed = _mm512_inserti64x4(_mm512_castsi256_si512(raw), raw, 1);
      codes_ptr += 32;

      __m512i lut1 = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i*)&q1.int_lut[b * K]));
      __m512i lut2 = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i*)&q2.int_lut[b * K]));

      __m512i c_even = _mm512_and_si512(packed, low_mask);
      __m512i c_odd = _mm512_and_si512(_mm512_srli_epi16(packed, 4), low_mask);

      // Compute scores for Query 1
      acc1_even = _mm512_add_epi16(acc1_even, _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(
                                                  _mm512_shuffle_epi8(lut1, c_even), 0)));
      acc1_odd = _mm512_add_epi16(acc1_odd, _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(
                                                _mm512_shuffle_epi8(lut1, c_odd), 0)));

      // Compute scores for Query 2 (Reuses c_even and c_odd!)
      acc2_even = _mm512_add_epi16(acc2_even, _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(
                                                  _mm512_shuffle_epi8(lut2, c_even), 0)));
      acc2_odd = _mm512_add_epi16(acc2_odd, _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(
                                                _mm512_shuffle_epi8(lut2, c_odd), 0)));
    }

    min_v1 = _mm512_min_epu16(min_v1, _mm512_min_epu16(acc1_even, acc1_odd));
    min_v2 = _mm512_min_epu16(min_v2, _mm512_min_epu16(acc2_even, acc2_odd));
  }

  void save(std::ofstream& out) const {
    out.write((char*)&num_blocks, sizeof(num_blocks));
    out.write((char*)&n_points, sizeof(n_points));
    out.write((char*)&dim, sizeof(dim));
    for (const auto& cb : codebooks) {
      size_t rows = cb.rows(), cols = cb.cols();
      out.write((char*)&rows, sizeof(size_t));
      out.write((char*)&cols, sizeof(size_t));
      out.write((char*)cb.data(), cb.size() * sizeof(float));
    }
    size_t sz = packed_codes.size();
    out.write((char*)&sz, sizeof(sz));
    out.write((char*)packed_codes.data(), sz);
  }

  void load(std::ifstream& in) {
    in.read((char*)&num_blocks, sizeof(num_blocks));
    in.read((char*)&n_points, sizeof(n_points));
    in.read((char*)&dim, sizeof(dim));
    dim_per_block = dim / num_blocks;
    codebooks.resize(num_blocks);
    codebook_norms.resize(num_blocks);
    for (uint32_t b = 0; b < num_blocks; ++b) {
      size_t rows, cols;
      in.read((char*)&rows, sizeof(size_t));
      in.read((char*)&cols, sizeof(size_t));
      codebooks[b] = Eigen::MatrixXf(rows, cols);
      in.read((char*)codebooks[b].data(), rows * cols * sizeof(float));
      codebook_norms[b] = codebooks[b].rowwise().squaredNorm();
    }
    size_t sz;
    in.read((char*)&sz, sizeof(sz));
    packed_codes.resize(sz);
    in.read((char*)packed_codes.data(), sz);
  }

  inline uint32_t size() const noexcept { return static_cast<uint32_t>(n_points); }
  inline uint32_t get_dims() const noexcept { return static_cast<uint32_t>(dim); }

 private:
  Quantized_Query<Metric>& finish_quantize_query(Quantized_Query<Metric>& qq,
                                                 const std::vector<float>& float_lut, float g_min,
                                                 float g_max) const {
    qq.min_dist = g_min;
    float range = (g_max - g_min);
    qq.scale = std::max(1e-6f, range / 255.0f);
    for (size_t i = 0; i < float_lut.size(); ++i) {
      qq.int_lut[i] = static_cast<uint8_t>((float_lut[i] - g_min) / qq.scale);
    }
    return qq;
  }

  void train(const PointRange& data) {
    codebooks.resize(num_blocks);
    codebook_norms.resize(num_blocks);

    size_t sample_size = std::min(static_cast<size_t>(K * 50), n_points);
    parlay::parallel_for(0, num_blocks, [&](size_t b) {
      size_t offset = b * dim_per_block;
      parlay::sequence<parlay::sequence<float>> sub(sample_size);
      std::mt19937 rng(static_cast<unsigned int>(b + 1));
      std::uniform_int_distribution<size_t> dist(0, n_points - 1);
      for (size_t i = 0; i < sample_size; ++i) {
        const float* raw = reinterpret_cast<const float*>(data.location(dist(rng)));
        sub[i] = parlay::sequence<float>(raw + offset, raw + offset + dim_per_block);
      }

      auto [centers, _] = mvsic::kmeans_subsample_assign_only<Metric>(sub, K, sample_size, false);
      codebooks[b] = Eigen::MatrixXf(K, dim_per_block);
      for (size_t c = 0; c < centers.size(); ++c) {
        for (size_t d_idx = 0; d_idx < dim_per_block; ++d_idx)
          codebooks[b](c, d_idx) = centers[c][d_idx];
      }
      codebook_norms[b] = codebooks[b].rowwise().squaredNorm();
    });
  }

  void encode_database_4bit(const PointRange& data) {
    size_t num_strips = (n_points + 63) / 64;
    packed_codes.resize(num_strips * num_blocks * 32, 0);

    parlay::parallel_for(0, num_strips, [&](size_t s) {
      for (size_t lane_pair = 0; lane_pair < 32; ++lane_pair) {
        size_t v_even = s * 64 + lane_pair * 2;
        size_t v_odd = v_even + 1;
        if (v_even >= n_points) break;

        for (uint32_t b = 0; b < num_blocks; ++b) {
          uint8_t c_e = find_best(data.location(v_even), b);
          uint8_t c_o = (v_odd < n_points) ? find_best(data.location(v_odd), b) : 0;
          packed_codes[s * num_blocks * 32 + b * 32 + lane_pair] =
              (c_e & 0x0F) | ((c_o & 0x0F) << 4);
        }
      }
    });
  }

  uint8_t find_best(const uint8_t* vec_ptr, uint32_t b) const {
    const float* sub = reinterpret_cast<const float*>(vec_ptr) + b * dim_per_block;
    Eigen::Map<const Eigen::VectorXf> q_map(sub, dim_per_block);
    Eigen::VectorXf dots = codebooks[b] * q_map;
    float min_val = std::numeric_limits<float>::max();
    uint8_t best = 0;
    for (uint32_t i = 0; i < K; ++i) {
      float val = Metric ? (codebook_norms[b][i] - 2.0f * dots[i]) : -dots[i];
      if (val < min_val) {
        min_val = val;
        best = static_cast<uint8_t>(i);
      }
    }
    return best;
  }
};

}  // namespace fastscan
}  // namespace mvsic