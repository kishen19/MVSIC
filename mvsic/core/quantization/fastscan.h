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
#include <cstdint>

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

  // 8-bit LUTs: [num_blocks][K=16]
  // (alignment helps but note: std::vector's data is not guaranteed 64B-aligned,
  //  however unaligned loads are fine for LUT loads; you can switch to an aligned
  //  allocator later if you want.)
  alignas(64) std::vector<uint8_t> int_lut;

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

  // Scalar distance used by generic wrapper paths (slow).
  inline float distance(const Quantized_Point<Metric>& p) const {
    uint16_t acc = 0;
    for (uint32_t b = 0; b < num_blocks; ++b) {
      uint8_t packed = p.code_ptr[b * 32];
      uint8_t code = (p.lane_idx % 2 == 0) ? (packed & 0x0F) : (packed >> 4);
      acc += static_cast<uint16_t>(int_lut[static_cast<size_t>(b) * K + code]);
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
  const uint8_t* code_ptr = nullptr;  // points at lane_pair byte for block0
  uint32_t lane_idx = 0;              // 0..63 within strip

  Quantized_Point() = default;
  Quantized_Point(const uint8_t* ptr, uint32_t lane) : code_ptr(ptr), lane_idx(lane) {}

  inline float distance(const Quantized_Query<Metric>& qq) const { return qq.distance(*this); }

  void prefetch() const { __builtin_prefetch(code_ptr, 0, 3); }

  bool same_as(const Quantized_Point<Metric>&) const { return false; }
  bool same_as(const Quantized_Query<Metric>&) const { return false; }
  bool is_metric() const { return Metric; }
};

// ---------------------------------------------------------
// FastScan Point Range
// ---------------------------------------------------------
template<typename PointRange, bool Metric>
class Quantized_Point_Range {
 public:
  static constexpr bool is_fastscan = true;
  uint32_t num_blocks = 0;
  static constexpr uint32_t K = 16;

  size_t n_points_raw = 0;  // number of vectors in flattened database
  size_t dim = 0;
  size_t dim_per_block = 0;

  std::vector<Eigen::MatrixXf> codebooks;       // [b] is (K x dim_per_block)
  std::vector<Eigen::VectorXf> codebook_norms;  // [b] is (K)

  // Storage: strip-major for 64 vectors.
  // For each strip s and block b: 32 bytes (each byte packs 2x4-bit codes).
  parlay::sequence<uint8_t> packed_codes;

  // Optional: store per-cloud offsets in VECTOR indices (for wrapper convenience).
  // These are derived from CSR-like float offsets / dim.
  parlay::sequence<size_t> cloud_vec_offsets;

  Quantized_Point_Range() = default;

  template<typename Seq>
  Quantized_Point_Range(const PointRange& data, const Seq& cloud_offsets_float,
                        uint32_t block_size = 32) {
    n_points_raw = data.size();
    dim = data.get_dims();
    dim_per_block = block_size;
    num_blocks = static_cast<uint32_t>(dim / dim_per_block);

    train(data);
    encode_database_4bit_no_padding(data, cloud_offsets_float);
  }

  // Access a single vector handle (by *global vector index* in packed layout).
  Quantized_Point<Metric> operator[](size_t i) const {
    const size_t strip_idx = i / 64;
    const size_t lane_idx = i % 64;
    const size_t strip_stride = static_cast<size_t>(num_blocks) * 32;
    const uint8_t* ptr = &packed_codes[strip_idx * strip_stride + (lane_idx / 2)];
    return Quantized_Point<Metric>(ptr, static_cast<uint32_t>(lane_idx));
  }

  // ---------------------------------------------------------
  // LUT construction (float -> uint8)
  // ---------------------------------------------------------
  template<typename PointTy>
  Quantized_Query<Metric> quantize_query(const PointTy& query) const {
    Quantized_Query<Metric> qq(num_blocks);
    std::vector<float> float_lut(static_cast<size_t>(num_blocks) * K);

    float g_min = std::numeric_limits<float>::max();
    float g_max = std::numeric_limits<float>::lowest();

    for (uint32_t b = 0; b < num_blocks; ++b) {
      Eigen::VectorXf q_sub(static_cast<Eigen::Index>(dim_per_block));
      const size_t offset = static_cast<size_t>(b) * dim_per_block;
      for (size_t j = 0; j < dim_per_block; ++j) {
        q_sub[static_cast<Eigen::Index>(j)] = query[offset + j];
      }

      Eigen::VectorXf dots = codebooks[b] * q_sub;
      float q_sq = Metric ? q_sub.squaredNorm() : 0.0f;

      for (uint32_t i = 0; i < K; ++i) {
        float val = Metric ? (codebook_norms[b][i] - 2.0f * dots[i] + q_sq) : -dots[i];
        float_lut[static_cast<size_t>(b) * K + i] = val;
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
      Eigen::Map<const Eigen::VectorXf> q_sub(qptr + static_cast<size_t>(b) * dim_per_block,
                                              static_cast<Eigen::Index>(dim_per_block));
      Eigen::VectorXf dots = codebooks[b] * q_sub;
      float q_sq = Metric ? q_sub.squaredNorm() : 0.0f;

      for (uint32_t i = 0; i < K; ++i) {
        float val = Metric ? (codebook_norms[b][i] - 2.0f * dots[i] + q_sq) : -dots[i];
        float_lut[static_cast<size_t>(b) * K + i] = val;
        g_min = std::min(g_min, val);
        g_max = std::max(g_max, val);
      }
    }

    return finish_quantize_query(qq, float_lut, g_min, g_max);
  }

  // You said batch LUT compute isn't the bottleneck; leaving it as-is (but still correct).
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

    std::vector<std::vector<float>> batch_float_luts(
        num_q, std::vector<float>(static_cast<size_t>(num_blocks) * K));
    std::vector<float> q_mins(num_q, std::numeric_limits<float>::max());
    std::vector<float> q_maxs(num_q, std::numeric_limits<float>::lowest());

    for (uint32_t b = 0; b < num_blocks; ++b) {
      const uint32_t offset = b * static_cast<uint32_t>(dim_per_block);
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
          batch_float_luts[i][static_cast<size_t>(b) * K + k_idx] = val;
          q_mins[i] = std::min(q_mins[i], val);
          q_maxs[i] = std::max(q_maxs[i], val);
        }
      }
    }

    for (uint32_t i = 0; i < num_q; ++i) {
      finish_quantize_query(out_luts[i], batch_float_luts[i], q_mins[i], q_maxs[i]);
    }
  }

  // ---------------------------------------------------------
  // SIMD kernels (FIXED: no 256->512 duplication)
  // Use AVX2 for shuffle and AVX-512 for widening + u16 accumulation.
  // ---------------------------------------------------------

  void scan_64_chunk(const Quantized_Query<Metric>& q, const uint8_t* codes_ptr,
                     float* results) const {
    __m512i acc_even = _mm512_setzero_si512();  // 32x u16 (even lanes 0,2,...,62)
    __m512i acc_odd = _mm512_setzero_si512();   // 32x u16 (odd lanes 1,3,...,63)
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

  inline uint16_t hmin_512_epu16(__m512i v) const {
    v = _mm512_min_epu16(v, _mm512_shuffle_i32x4(v, v, _MM_SHUFFLE(1, 0, 3, 2)));
    v = _mm512_min_epu16(v, _mm512_shuffle_i32x4(v, v, _MM_SHUFFLE(0, 0, 1, 1)));

    __m128i v128 = _mm512_castsi512_si128(v);
    v128 = _mm_min_epu16(v128, _mm_srli_si128(v128, 8));
    v128 = _mm_min_epu16(v128, _mm_srli_si128(v128, 4));
    v128 = _mm_min_epu16(v128, _mm_srli_si128(v128, 2));
    return static_cast<uint16_t>(_mm_extract_epi16(v128, 0));
  }

  float scan_64_chunk_min(const Quantized_Query<Metric>& q, const uint8_t* codes_ptr) const {
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

    const __m512i vmin = _mm512_min_epu16(acc_even, acc_odd);
    const uint16_t best = hmin_512_epu16(vmin);
    return q.decode(best);
  }

  // Update running min (32 lanes), where lane i tracks min over the pair {2i,2i+1} across strips.
  inline __m512i scan_64_running_min(const Quantized_Query<Metric>& q, const uint8_t* codes_ptr,
                                     __m512i current_min_v) const {
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

    current_min_v = _mm512_min_epu16(current_min_v, acc_even);
    current_min_v = _mm512_min_epu16(current_min_v, acc_odd);
    return current_min_v;
  }

  inline float reduce_running_min(const Quantized_Query<Metric>& q, __m512i running_min_v) const {
    const uint16_t best = hmin_512_epu16(running_min_v);
    return q.decode(best);
  }

  // Dual-query scan for one strip (64 vectors): update two running-min vectors.
  inline void scan_64_dual_query(const Quantized_Query<Metric>& q1,
                                 const Quantized_Query<Metric>& q2, const uint8_t* codes_ptr,
                                 __m512i& min_v1, __m512i& min_v2) const {
    __m512i acc1_even = _mm512_setzero_si512();
    __m512i acc1_odd = _mm512_setzero_si512();
    __m512i acc2_even = _mm512_setzero_si512();
    __m512i acc2_odd = _mm512_setzero_si512();
    const __m256i low_mask = _mm256_set1_epi8(0x0F);

    for (uint32_t b = 0; b < num_blocks; ++b) {
      const __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(codes_ptr));
      codes_ptr += 32;

      const __m256i codes_even = _mm256_and_si256(packed, low_mask);
      const __m256i codes_odd = _mm256_and_si256(_mm256_srli_epi16(packed, 4), low_mask);

      const __m128i lut1_128 = _mm_loadu_si128(
          reinterpret_cast<const __m128i*>(&q1.int_lut[static_cast<size_t>(b) * K]));
      const __m128i lut2_128 = _mm_loadu_si128(
          reinterpret_cast<const __m128i*>(&q2.int_lut[static_cast<size_t>(b) * K]));
      const __m256i lut1_256 = _mm256_broadcastsi128_si256(lut1_128);
      const __m256i lut2_256 = _mm256_broadcastsi128_si256(lut2_128);

      const __m256i s1_even_u8 = _mm256_shuffle_epi8(lut1_256, codes_even);
      const __m256i s1_odd_u8 = _mm256_shuffle_epi8(lut1_256, codes_odd);
      const __m256i s2_even_u8 = _mm256_shuffle_epi8(lut2_256, codes_even);
      const __m256i s2_odd_u8 = _mm256_shuffle_epi8(lut2_256, codes_odd);

      acc1_even = _mm512_add_epi16(acc1_even, _mm512_cvtepu8_epi16(s1_even_u8));
      acc1_odd = _mm512_add_epi16(acc1_odd, _mm512_cvtepu8_epi16(s1_odd_u8));
      acc2_even = _mm512_add_epi16(acc2_even, _mm512_cvtepu8_epi16(s2_even_u8));
      acc2_odd = _mm512_add_epi16(acc2_odd, _mm512_cvtepu8_epi16(s2_odd_u8));
    }

    min_v1 = _mm512_min_epu16(min_v1, _mm512_min_epu16(acc1_even, acc1_odd));
    min_v2 = _mm512_min_epu16(min_v2, _mm512_min_epu16(acc2_even, acc2_odd));
  }

  // ---------------------------------------------------------
  // NO-PADDING support: masked strip min for partial strips
  // lo/hi are lane indices in [0,64], and we take min over lanes [lo,hi).
  // ---------------------------------------------------------
  static inline __mmask32 mask_even_lanes(int lo, int hi) {
    __mmask32 m = 0;
    for (int i = 0; i < 32; ++i) {
      const int lane = 2 * i;
      if (lo <= lane && lane < hi) m |= (__mmask32(1) << i);
    }
    return m;
  }

  static inline __mmask32 mask_odd_lanes(int lo, int hi) {
    __mmask32 m = 0;
    for (int i = 0; i < 32; ++i) {
      const int lane = 2 * i + 1;
      if (lo <= lane && lane < hi) m |= (__mmask32(1) << i);
    }
    return m;
  }

  float scan_64_chunk_min_masked(const Quantized_Query<Metric>& q, const uint8_t* codes_ptr, int lo,
                                 int hi) const {
    // Clamp defensively.
    lo = std::max(lo, 0);
    hi = std::min(hi, 64);
    if (hi <= lo) return std::numeric_limits<float>::infinity();

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

    const __m512i INF = _mm512_set1_epi16(0xFFFF);
    const __mmask32 me = mask_even_lanes(lo, hi);
    const __mmask32 mo = mask_odd_lanes(lo, hi);

    // Keep valid lanes, set invalid to INF.
    acc_even = _mm512_mask_mov_epi16(INF, me, acc_even);
    acc_odd = _mm512_mask_mov_epi16(INF, mo, acc_odd);

    const __m512i vmin = _mm512_min_epu16(acc_even, acc_odd);
    const uint16_t best = hmin_512_epu16(vmin);
    return q.decode(best);
  }

  // ---------------------------------------------------------
  // IO / misc
  // ---------------------------------------------------------
  void save(std::ofstream& out) const {
    out.write(reinterpret_cast<const char*>(&num_blocks), sizeof(num_blocks));
    out.write(reinterpret_cast<const char*>(&n_points_raw), sizeof(n_points_raw));
    out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    out.write(reinterpret_cast<const char*>(&dim_per_block), sizeof(dim_per_block));

    // Save cloud_vec_offsets (vector indices)
    size_t n_off = cloud_vec_offsets.size();
    out.write(reinterpret_cast<const char*>(&n_off), sizeof(n_off));
    out.write(reinterpret_cast<const char*>(cloud_vec_offsets.data()), n_off * sizeof(size_t));

    for (const auto& cb : codebooks) {
      size_t rs = static_cast<size_t>(cb.rows());
      size_t cs = static_cast<size_t>(cb.cols());
      out.write(reinterpret_cast<const char*>(&rs), sizeof(size_t));
      out.write(reinterpret_cast<const char*>(&cs), sizeof(size_t));
      out.write(reinterpret_cast<const char*>(cb.data()), cb.size() * sizeof(float));
    }

    size_t sz = packed_codes.size();
    out.write(reinterpret_cast<const char*>(&sz), sizeof(sz));
    out.write(reinterpret_cast<const char*>(packed_codes.data()), sz);
  }

  void load(std::ifstream& in) {
    in.read(reinterpret_cast<char*>(&num_blocks), sizeof(num_blocks));
    in.read(reinterpret_cast<char*>(&n_points_raw), sizeof(n_points_raw));
    in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    in.read(reinterpret_cast<char*>(&dim_per_block), sizeof(dim_per_block));

    // Load cloud_vec_offsets
    size_t n_off = 0;
    in.read(reinterpret_cast<char*>(&n_off), sizeof(n_off));
    cloud_vec_offsets.resize(n_off);
    in.read(reinterpret_cast<char*>(cloud_vec_offsets.data()), n_off * sizeof(size_t));

    codebooks.resize(num_blocks);
    codebook_norms.resize(num_blocks);

    for (uint32_t b = 0; b < num_blocks; ++b) {
      size_t rs = 0, cs = 0;
      in.read(reinterpret_cast<char*>(&rs), sizeof(size_t));
      in.read(reinterpret_cast<char*>(&cs), sizeof(size_t));
      codebooks[b] = Eigen::MatrixXf(static_cast<Eigen::Index>(rs), static_cast<Eigen::Index>(cs));
      in.read(reinterpret_cast<char*>(codebooks[b].data()), rs * cs * sizeof(float));
      codebook_norms[b] = codebooks[b].rowwise().squaredNorm();
    }

    size_t sz = 0;
    in.read(reinterpret_cast<char*>(&sz), sizeof(sz));
    packed_codes.resize(sz);
    in.read(reinterpret_cast<char*>(packed_codes.data()), sz);
  }

  inline uint32_t size() const noexcept { return static_cast<uint32_t>(n_points_raw); }
  inline uint32_t get_dims() const noexcept { return static_cast<uint32_t>(dim); }

 private:
  Quantized_Query<Metric>& finish_quantize_query(Quantized_Query<Metric>& qq,
                                                 const std::vector<float>& float_lut, float g_min,
                                                 float g_max) const {
    qq.min_dist = g_min;
    const float range = (g_max - g_min);
    qq.scale = std::max(1e-6f, range / 255.0f);

    for (size_t i = 0; i < float_lut.size(); ++i) {
      qq.int_lut[i] = static_cast<uint8_t>((float_lut[i] - g_min) / qq.scale);
    }
    return qq;
  }

  void train(const PointRange& data) {
    codebooks.resize(num_blocks);
    codebook_norms.resize(num_blocks);

    const size_t sample_size = std::min(static_cast<size_t>(K * 50), n_points_raw);

    parlay::parallel_for(0, num_blocks, [&](size_t b) {
      const size_t offset = b * dim_per_block;
      parlay::sequence<parlay::sequence<float>> sub(sample_size);

      std::mt19937 rng(static_cast<unsigned int>(b + 1));
      std::uniform_int_distribution<size_t> dist(0, n_points_raw - 1);

      for (size_t i = 0; i < sample_size; ++i) {
        const float* raw = reinterpret_cast<const float*>(data.location(dist(rng)));
        sub[i] = parlay::sequence<float>(raw + offset, raw + offset + dim_per_block);
      }

      auto [centers, _] = mvsic::kmeans_subsample_assign_only<Metric>(sub, K, sample_size, false);

      codebooks[b] =
          Eigen::MatrixXf(static_cast<Eigen::Index>(K), static_cast<Eigen::Index>(dim_per_block));
      for (size_t c = 0; c < centers.size(); ++c) {
        for (size_t d_idx = 0; d_idx < dim_per_block; ++d_idx) {
          codebooks[b](static_cast<Eigen::Index>(c), static_cast<Eigen::Index>(d_idx)) =
              centers[c][d_idx];
        }
      }
      codebook_norms[b] = codebooks[b].rowwise().squaredNorm();
    });
  }

  template<typename Seq>
  void encode_database_4bit_no_padding(const PointRange& data, const Seq& cloud_offsets_float) {
    // Store cloud offsets in VECTOR indices (optional; used by wrapper to get start/end).
    // cloud_offsets_float are offsets in FLOATS.
    const size_t n_clouds = cloud_offsets_float.size() > 0 ? cloud_offsets_float.size() - 1 : 0;
    cloud_vec_offsets.resize(n_clouds + 1);
    for (size_t i = 0; i <= n_clouds; ++i) {
      cloud_vec_offsets[i] = static_cast<size_t>(cloud_offsets_float[i] / dim);
    }

    // Pack codes for all vectors contiguously in natural order: strip s = v/64.
    const size_t strip_stride = static_cast<size_t>(num_blocks) * 32;
    const size_t num_strips = (n_points_raw + 63) / 64;

    packed_codes.resize(num_strips * strip_stride);
    std::fill(packed_codes.begin(), packed_codes.end(), uint8_t{0});

    parlay::parallel_for(0, num_strips, [&](size_t s) {
      uint8_t* strip_base = packed_codes.data() + s * strip_stride;

      for (size_t lane_pair = 0; lane_pair < 32; ++lane_pair) {
        const size_t v_even = s * 64 + lane_pair * 2;
        if (v_even >= n_points_raw) break;

        const size_t v_odd = v_even + 1;

        for (uint32_t b = 0; b < num_blocks; ++b) {
          const uint8_t c_e = find_best(data.location(v_even), b);
          const uint8_t c_o = (v_odd < n_points_raw) ? find_best(data.location(v_odd), b) : 0;

          strip_base[static_cast<size_t>(b) * 32 + lane_pair] = (c_e & 0x0F) | ((c_o & 0x0F) << 4);
        }
      }
    });
  }

  uint8_t find_best(const uint8_t* vec_ptr, uint32_t b) const {
    const float* raw = reinterpret_cast<const float*>(vec_ptr);
    const float* sub_ptr = raw + static_cast<size_t>(b) * dim_per_block;

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
};

}  // namespace fastscan
}  // namespace mvsic
