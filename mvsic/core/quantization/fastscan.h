#pragma once

// fastscan.h
//
// FastScan with AVX512 and AVX2 fallback.
// Pure Single-Vector Implementation.
// - When __AVX512F__ is defined: all kernels use AVX-512 (shuffle + int16 accumulate).
// - When only __AVX2__ is defined: all kernels use AVX2 (256-bit shuffle + int16 accumulate).

#include <immintrin.h>
#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

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

  alignas(64) std::vector<uint8_t> int_lut;  // [num_blocks * K]
  float min_dist = 0.0f;
  float scale = 1.0f;
  uint32_t num_blocks = 0;
  static constexpr uint32_t K = 16;

  Quantized_Query() = default;
  explicit Quantized_Query(uint32_t m) : num_blocks(m) {
    int_lut.resize(static_cast<size_t>(m) * K);
  }

  inline float decode(uint16_t int_dist) const {
    return (min_dist * static_cast<float>(num_blocks)) + (static_cast<float>(int_dist) * scale);
  }

  inline float distance(const Quantized_Point<Metric>& p) const {
    uint16_t acc = 0;
    for (uint32_t b = 0; b < num_blocks; ++b) {
      uint8_t packed = p.code_ptr[b * 32];
      uint8_t code = (p.lane_idx % 2 == 0) ? (packed & 0x0F) : (packed >> 4);
      acc += static_cast<uint16_t>(int_lut[static_cast<size_t>(b) * K + code]);
    }
    return decode(acc);
  }

  template<typename EncRange>
  void distances_all(const EncRange& db, float* out) const {
    const size_t N = db.size();
    if (N == 0) return;

    const size_t strip_stride = static_cast<size_t>(db.num_blocks) * 32;
    const size_t n_full_strips = N / 64;
    const size_t full = n_full_strips * 64;

    parlay::parallel_for(
        0, n_full_strips,
        [&](size_t s) {
          const uint8_t* codes_ptr = db.packed_codes.data() + s * strip_stride;
          db.scan_64_chunk(*this, codes_ptr, out + s * 64);
        },
        /*granularity=*/64);  // Tuned via microbenchmark

    if (full < N) {
      alignas(64) float tmp[64];
      const size_t tail_strip = n_full_strips;
      const uint8_t* codes_ptr = db.packed_codes.data() + tail_strip * strip_stride;
      db.scan_64_chunk(*this, codes_ptr, tmp);
      const size_t tail = N - full;
      std::memcpy(out + full, tmp, tail * sizeof(float));
    }
  }
};

// ---------------------------------------------------------
// FastScan Point Handle
// ---------------------------------------------------------
template<bool Metric>
class Quantized_Point {
 public:
  const uint8_t* code_ptr = nullptr;
  uint32_t lane_idx = 0;

  Quantized_Point() = default;
  Quantized_Point(const uint8_t* ptr, uint32_t lane) : code_ptr(ptr), lane_idx(lane) {}

  inline float distance(const Quantized_Query<Metric>& qq) const { return qq.distance(*this); }
  void prefetch() const { __builtin_prefetch(code_ptr, 0, 3); }

  bool same_as(const Quantized_Point<Metric>&) const { return false; }
  bool same_as(const Quantized_Query<Metric>&) const { return false; }
  bool is_metric() const { return Metric; }
};

// ---------------------------------------------------------
// Quantized Point Range with SIMD kernels
// ---------------------------------------------------------
template<typename PointRange, bool Metric>
class Quantized_Point_Range {
 public:
  static constexpr bool is_fastscan = true;
  static constexpr uint32_t K = 16;

  uint32_t num_blocks = 0;
  size_t n_points = 0;
  size_t dim = 0;
  size_t dim_per_block = 0;

  parlay::sequence<uint8_t> packed_codes;

  Quantized_Point_Range() = default;

  Quantized_Point<Metric> operator[](size_t i) const {
    const size_t strip_idx = i / 64;
    const size_t lane_idx = i % 64;
    const size_t strip_stride = static_cast<size_t>(num_blocks) * 32;
    const uint8_t* ptr = &packed_codes[strip_idx * strip_stride + (lane_idx / 2)];
    return Quantized_Point<Metric>(ptr, static_cast<uint32_t>(lane_idx));
  }

#ifdef __AVX512F__
  void scan_64_chunk(const Quantized_Query<Metric>& q, const uint8_t* codes_ptr,
                     float* results) const {
    __m512i acc_even = _mm512_setzero_si512();
    __m512i acc_odd = _mm512_setzero_si512();
    const __m256i low_mask = _mm256_set1_epi8(0x0F);

    for (uint32_t b = 0; b < num_blocks; ++b) {
      const __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(codes_ptr));
      codes_ptr += 32;

      const __m256i codes_even = _mm256_and_si256(packed, low_mask);
      const __m256i codes_odd = _mm256_and_si256(_mm256_srli_epi16(packed, 4), low_mask);

      const __m128i lut128 =
          _mm_loadu_si128(reinterpret_cast<const __m128i*>(&q.int_lut[static_cast<size_t>(b) * K]));
      const __m256i lut256 = _mm256_broadcastsi128_si256(lut128);

      const __m256i scores_even_u8 = _mm256_shuffle_epi8(lut256, codes_even);
      const __m256i scores_odd_u8 = _mm256_shuffle_epi8(lut256, codes_odd);

      acc_even = _mm512_add_epi16(acc_even, _mm512_cvtepu8_epi16(scores_even_u8));
      acc_odd = _mm512_add_epi16(acc_odd, _mm512_cvtepu8_epi16(scores_odd_u8));
    }

    alignas(64) uint16_t raw_even[32];
    alignas(64) uint16_t raw_odd[32];
    _mm512_store_si512(reinterpret_cast<__m512i*>(raw_even), acc_even);
    _mm512_store_si512(reinterpret_cast<__m512i*>(raw_odd), acc_odd);

    for (int i = 0; i < 32; ++i) {
      results[i * 2] = q.decode(raw_even[i]);
      results[i * 2 + 1] = q.decode(raw_odd[i]);
    }
  }
#elif defined(__AVX2__)
  void scan_64_chunk(const Quantized_Query<Metric>& q, const uint8_t* codes_ptr,
                     float* results) const {
    __m256i acc_even_lo = _mm256_setzero_si256();
    __m256i acc_even_hi = _mm256_setzero_si256();
    __m256i acc_odd_lo = _mm256_setzero_si256();
    __m256i acc_odd_hi = _mm256_setzero_si256();
    const __m256i low_mask = _mm256_set1_epi8(0x0F);

    for (uint32_t b = 0; b < num_blocks; ++b) {
      const __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(codes_ptr));
      codes_ptr += 32;
      const __m256i codes_even = _mm256_and_si256(packed, low_mask);
      const __m256i codes_odd = _mm256_and_si256(_mm256_srli_epi16(packed, 4), low_mask);
      const __m128i lut128 =
          _mm_loadu_si128(reinterpret_cast<const __m128i*>(&q.int_lut[static_cast<size_t>(b) * K]));
      const __m256i lut256 = _mm256_broadcastsi128_si256(lut128);
      const __m256i scores_even_u8 = _mm256_shuffle_epi8(lut256, codes_even);
      const __m256i scores_odd_u8 = _mm256_shuffle_epi8(lut256, codes_odd);

      acc_even_lo = _mm256_add_epi16(acc_even_lo,
                                     _mm256_cvtepu8_epi16(_mm256_castsi256_si128(scores_even_u8)));
      acc_even_hi = _mm256_add_epi16(
          acc_even_hi, _mm256_cvtepu8_epi16(_mm256_extracti128_si256(scores_even_u8, 1)));
      acc_odd_lo =
          _mm256_add_epi16(acc_odd_lo, _mm256_cvtepu8_epi16(_mm256_castsi256_si128(scores_odd_u8)));
      acc_odd_hi = _mm256_add_epi16(
          acc_odd_hi, _mm256_cvtepu8_epi16(_mm256_extracti128_si256(scores_odd_u8, 1)));
    }

    alignas(32) uint16_t raw_even[32];
    alignas(32) uint16_t raw_odd[32];
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(raw_even), acc_even_lo);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(raw_even + 16), acc_even_hi);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(raw_odd), acc_odd_lo);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(raw_odd + 16), acc_odd_hi);

    for (int i = 0; i < 32; ++i) {
      results[i * 2] = q.decode(raw_even[i]);
      results[i * 2 + 1] = q.decode(raw_odd[i]);
    }
  }
#else
  void scan_64_chunk(const Quantized_Query<Metric>& q, const uint8_t* codes_ptr,
                     float* results) const {
    for (int lane = 0; lane < 64; ++lane) {
      uint16_t acc = 0;
      for (uint32_t b = 0; b < num_blocks; ++b) {
        uint8_t packed = codes_ptr[b * 32 + (lane / 2)];
        uint8_t code = (lane % 2 == 0) ? (packed & 0x0F) : (packed >> 4);
        acc += static_cast<uint16_t>(q.int_lut[b * K + code]);
      }
      results[lane] = q.decode(acc);
    }
  }
#endif

  void save(std::ostream& out) const {
    out.write(reinterpret_cast<const char*>(&num_blocks), sizeof(num_blocks));
    out.write(reinterpret_cast<const char*>(&n_points), sizeof(n_points));
    out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    out.write(reinterpret_cast<const char*>(&dim_per_block), sizeof(dim_per_block));
    size_t sz = packed_codes.size();
    out.write(reinterpret_cast<const char*>(&sz), sizeof(sz));
    if (sz) out.write(reinterpret_cast<const char*>(packed_codes.data()), sz * sizeof(uint8_t));
  }

  void load(std::istream& in) {
    in.read(reinterpret_cast<char*>(&num_blocks), sizeof(num_blocks));
    in.read(reinterpret_cast<char*>(&n_points), sizeof(n_points));
    in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    in.read(reinterpret_cast<char*>(&dim_per_block), sizeof(dim_per_block));
    size_t sz = 0;
    in.read(reinterpret_cast<char*>(&sz), sizeof(sz));
    packed_codes.resize(sz);
    if (sz) in.read(reinterpret_cast<char*>(packed_codes.data()), sz * sizeof(uint8_t));
  }

  inline uint32_t size() const noexcept { return static_cast<uint32_t>(n_points); }
  inline uint32_t get_dims() const noexcept { return static_cast<uint32_t>(dim); }
};

// ---------------------------------------------------------
// Model
// ---------------------------------------------------------
template<bool Metric>
class Model {
 public:
  static constexpr bool is_fastscan = true;
  static constexpr uint32_t K = 16;

  uint32_t num_blocks = 0;
  size_t dim = 0;
  size_t dim_per_block = 0;

  std::vector<Eigen::MatrixXf> codebooks;
  std::vector<Eigen::VectorXf> codebook_norms;

  Model() = default;

  template<typename PointRange>
  Model(const PointRange& train_data, uint32_t block_size = 32) {
    train(train_data, block_size);
  }

  template<typename PointRange>
  void train(const PointRange& data, uint32_t block_size = 32) {
    dim = data.get_dims();
    dim_per_block = block_size;
    if (dim_per_block == 0 || (dim % dim_per_block) != 0) {
      std::cerr << "Error: FastScan dim=" << dim << " not divisible by block_size=" << dim_per_block
                << "\n";
      abort();
    }
    num_blocks = static_cast<uint32_t>(dim / dim_per_block);
    codebooks.resize(num_blocks);
    codebook_norms.resize(num_blocks);

    const size_t n_points = data.size();
    const size_t sample_size = std::min(static_cast<size_t>(4096), n_points);

    parlay::parallel_for(0, num_blocks, [&](size_t b) {
      const size_t offset = b * dim_per_block;
      parlay::sequence<parlay::sequence<float>> sub(sample_size);
      std::mt19937 rng(static_cast<unsigned>(b + 1));
      std::uniform_int_distribution<size_t> distu(0, n_points - 1);

      for (size_t i = 0; i < sample_size; ++i) {
        const float* raw = reinterpret_cast<const float*>(data.location(distu(rng)));
        sub[i] = parlay::sequence<float>(raw + offset, raw + offset + dim_per_block);
      }

      auto [centers, _] = mvsic::kmeans_subsample_assign_only<true>(sub, K, sample_size, false);
      codebooks[b] =
          Eigen::MatrixXf(static_cast<Eigen::Index>(K), static_cast<Eigen::Index>(dim_per_block));
      for (size_t c = 0; c < centers.size(); ++c) {
        for (size_t d = 0; d < dim_per_block; ++d)
          codebooks[b](static_cast<Eigen::Index>(c), static_cast<Eigen::Index>(d)) = centers[c][d];
      }
      codebook_norms[b] = codebooks[b].rowwise().squaredNorm();
    });
  }

  template<typename VecLocPtr>
  uint8_t find_best(const VecLocPtr vec_ptr, uint32_t b) const {
    const float* raw = reinterpret_cast<const float*>(vec_ptr);
    const float* sub_ptr = raw + b * dim_per_block;
    Eigen::Map<const Eigen::VectorXf> q_sub(sub_ptr, static_cast<Eigen::Index>(dim_per_block));
    Eigen::VectorXf dots = codebooks[b] * q_sub;
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

  template<typename PointRange>
  Quantized_Point_Range<PointRange, Metric> encode(const PointRange& data) const {
    Quantized_Point_Range<PointRange, Metric> enc;
    enc.num_blocks = num_blocks;
    enc.dim = data.get_dims();
    enc.dim_per_block = dim_per_block;
    enc.n_points = data.size();

    const size_t n_padded = ((enc.n_points + 63) / 64) * 64;
    const size_t num_strips = n_padded / 64;
    const size_t strip_stride = static_cast<size_t>(enc.num_blocks) * 32;

    enc.packed_codes.resize(num_strips * strip_stride);
    std::fill(enc.packed_codes.begin(), enc.packed_codes.end(), uint8_t{0});

    parlay::parallel_for(0, num_strips, [&](size_t s) {
      uint8_t* strip_base = enc.packed_codes.data() + s * strip_stride;
      for (size_t lane_pair = 0; lane_pair < 32; ++lane_pair) {
        const size_t v_even = s * 64 + lane_pair * 2;
        const size_t v_odd = v_even + 1;
        const bool has_even = (v_even < enc.n_points);
        const bool has_odd = (v_odd < enc.n_points);
        for (uint32_t b = 0; b < enc.num_blocks; ++b) {
          const uint8_t c_e = has_even ? find_best(data.location(v_even), b) : 0;
          const uint8_t c_o = has_odd ? find_best(data.location(v_odd), b) : 0;
          strip_base[b * 32 + lane_pair] = (c_e & 0x0F) | ((c_o & 0x0F) << 4);
        }
      }
    });

    return enc;
  }

  template<typename PointTy>
  Quantized_Query<Metric> quantize_query(const PointTy& query) const {
    Quantized_Query<Metric> qq(num_blocks);
    std::vector<float> float_lut(static_cast<size_t>(num_blocks) * K);
    float g_min = std::numeric_limits<float>::max();
    float g_max = std::numeric_limits<float>::lowest();
    for (uint32_t b = 0; b < num_blocks; ++b) {
      Eigen::VectorXf q_sub(static_cast<Eigen::Index>(dim_per_block));
      for (size_t j = 0; j < dim_per_block; ++j)
        q_sub(static_cast<Eigen::Index>(j)) = query[b * dim_per_block + j];
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

  Quantized_Query<Metric> quantize_query(const float* qptr) const {
    Quantized_Query<Metric> qq(num_blocks);
    std::vector<float> float_lut(static_cast<size_t>(num_blocks) * K);
    float g_min = std::numeric_limits<float>::max();
    float g_max = std::numeric_limits<float>::lowest();
    for (uint32_t b = 0; b < num_blocks; ++b) {
      Eigen::Map<const Eigen::VectorXf> q_sub(qptr + b * dim_per_block,
                                              static_cast<Eigen::Index>(dim_per_block));
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

  template<typename PointCloudTy>
  void quantize_query_batch(const PointCloudTy& query_cloud,
                            parlay::sequence<Quantized_Query<Metric>>& out_luts) const {
    const uint32_t num_q = query_cloud.size();
    const uint32_t dims = query_cloud.get_dims();
    const float* base = query_cloud.data();
    out_luts.clear();
    out_luts.reserve(num_q);
    for (uint32_t i = 0; i < num_q; ++i)
      out_luts.emplace_back(num_blocks);
    using RowMajorMat = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    RowMajorMat Q_sub(static_cast<Eigen::Index>(num_q), static_cast<Eigen::Index>(dim_per_block));
    Eigen::MatrixXf dot_products(static_cast<Eigen::Index>(K), static_cast<Eigen::Index>(num_q));
    std::vector<std::vector<float>> batch_float_luts(num_q, std::vector<float>(num_blocks * K));
    std::vector<float> q_mins(num_q, std::numeric_limits<float>::max());
    std::vector<float> q_maxs(num_q, std::numeric_limits<float>::lowest());
    for (uint32_t b = 0; b < num_blocks; ++b) {
      for (uint32_t i = 0; i < num_q; ++i)
        std::memcpy(&Q_sub(i, 0), base + i * dims + b * dim_per_block,
                    sizeof(float) * dim_per_block);
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
    for (uint32_t i = 0; i < num_q; ++i)
      finish_quantize_query(out_luts[i], batch_float_luts[i], q_mins[i], q_maxs[i]);
  }

  void save(std::ostream& out) const {
    out.write(reinterpret_cast<const char*>(&num_blocks), sizeof(num_blocks));
    out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    out.write(reinterpret_cast<const char*>(&dim_per_block), sizeof(dim_per_block));
    for (uint32_t b = 0; b < num_blocks; ++b) {
      size_t rs = static_cast<size_t>(codebooks[b].rows());
      size_t cs = static_cast<size_t>(codebooks[b].cols());
      out.write(reinterpret_cast<const char*>(&rs), sizeof(rs));
      out.write(reinterpret_cast<const char*>(&cs), sizeof(cs));
      out.write(reinterpret_cast<const char*>(codebooks[b].data()), rs * cs * sizeof(float));
    }
  }

  void load(std::istream& in) {
    in.read(reinterpret_cast<char*>(&num_blocks), sizeof(num_blocks));
    in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    in.read(reinterpret_cast<char*>(&dim_per_block), sizeof(dim_per_block));
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

 private:
  Quantized_Query<Metric>& finish_quantize_query(Quantized_Query<Metric>& qq,
                                                 const std::vector<float>& float_lut, float g_min,
                                                 float g_max) const {
    qq.min_dist = g_min;
    qq.scale = std::max(1e-6f, (g_max - g_min) / 255.0f);
    for (size_t i = 0; i < float_lut.size(); ++i)
      qq.int_lut[i] = static_cast<uint8_t>((float_lut[i] - g_min) / qq.scale);
    return qq;
  }
};

}  // namespace fastscan
}  // namespace mvsic