#pragma once

// byte_turboquant.h
//
// "Byte TurboQuant": stores 1 byte per dimension (int8/uint8) instead of
// 4-bit nibble-packed centroids.  Eliminates all codebook lookup / decode
// overhead — the GEMM kernel does: load → maddubs → accumulate directly.
//
// 2× space vs 4-bit TQ, but the kernel is pure compute with no decode step.
// Uses the same preprocessing as regular TQ:
//   rotate (Hadamard) → normalize → random sign flip → quantize to int8.
//
// Interface matches one_to_many_turboquant:
//   Model<Metric>, Quantized_Query<Metric>, Quantized_Point<Metric>,
//   Quantized_Point_Range<PR, Metric>
// Compatible with mvsic::MultiVecQuantizer (wrapper.h).

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>

#include <immintrin.h>

#include "parlay/parallel.h"
#include "parlay/sequence.h"

#include "rabitqlib/utils/rotator.hpp"

namespace mvsic {
namespace byte_turboquant {

// =========================================================================
// Constants
// =========================================================================
namespace internal {

// Cap for normalized values before int8 quantization.
static constexpr float kValueCap = 3.91724f;
static constexpr size_t kStripSize = 64;

}  // namespace internal

// =========================================================================
// Forward declarations
// =========================================================================
template<bool Metric>
class Quantized_Point;

template<bool Metric>
class Quantized_Query;

// =========================================================================
// Quantized_Point: lightweight view (for per-point fallback)
// =========================================================================
template<bool Metric>
class Quantized_Point {
 public:
  const uint8_t* code_ptr = nullptr;  // uint8 codes (biased +128)
  size_t num_bytes = 0;               // = padded_dim (1 byte per dim)
  float norm_scaling_factor = 0.0f;
  float unquantized_squared_norm = 0.0f;

  // Owned buffer for gathered bytes.
  std::vector<uint8_t> owned_codes;

  Quantized_Point() = default;
  Quantized_Point(const uint8_t* ptr, size_t nb, float nsf, float usn)
      : code_ptr(ptr), num_bytes(nb), norm_scaling_factor(nsf),
        unquantized_squared_norm(usn) {}
  Quantized_Point(std::vector<uint8_t>&& codes, size_t nb, float nsf, float usn)
      : num_bytes(nb), norm_scaling_factor(nsf),
        unquantized_squared_norm(usn), owned_codes(std::move(codes)) {
    code_ptr = owned_codes.data();
  }

  inline float distance(const Quantized_Query<Metric>& qq) const;

  void prefetch() const { if (code_ptr) __builtin_prefetch(code_ptr, 0, 3); }
  bool same_as(const Quantized_Point<Metric>&) const { return false; }
  bool same_as(const Quantized_Query<Metric>&) const { return false; }
  bool is_metric() const { return Metric; }
};

// =========================================================================
// Quantized_Query: int8-encoded query for byte TQ
// =========================================================================
template<bool Metric>
class Quantized_Query {
 public:
  using distanceType = float;

  size_t dim = 0;
  size_t num_bytes_per_datapoint = 0;  // = padded_dim

  // Row-major int8 query data (signed, 1 byte per dim).
  parlay::sequence<int8_t> query_data;

  float norm_scaling_factor = 0.0f;
  float unquantized_squared_norm = 0.0f;

  Quantized_Query() = default;

  inline float distance(const Quantized_Point<Metric>& p) const {
    return p.distance(*this);
  }

  // Tag for SFINAE detection in wrapper.
  static constexpr bool has_batch_distances = true;
};

// =========================================================================
// Scalar fallback: Quantized_Point::distance
// =========================================================================
template<bool Metric>
inline float Quantized_Point<Metric>::distance(
    const Quantized_Query<Metric>& qq) const {
  // DB codes are uint8 (biased by +128). Query is int8.
  // Raw dot = sum_i (code_uint8[i] * query_int8[i])
  // True dot = raw_dot - 128 * sum_i(query_int8[i])
  int32_t raw_dot = 0;
  int32_t q_byte_sum = 0;
  const size_t nb = num_bytes;
  for (size_t d = 0; d < nb; ++d) {
    raw_dot += static_cast<int32_t>(code_ptr[d]) *
               static_cast<int32_t>(qq.query_data[d]);
    q_byte_sum += static_cast<int32_t>(qq.query_data[d]);
  }

  const int32_t corrected_dot = raw_dot - 128 * q_byte_sum;

  float neg_dot = -static_cast<float>(corrected_dot) * norm_scaling_factor *
                  qq.norm_scaling_factor;
  if constexpr (Metric) {
    return unquantized_squared_norm + 2.0f * neg_dot +
           qq.unquantized_squared_norm;
  } else {
    return neg_dot;
  }
}

// =========================================================================
// AVX-512 GEMM Chamfer kernel (dpbusd: 16 points per __m512i)
// =========================================================================
// Loads two consecutive 8-point groups into 512-bit registers and uses
// vpdpbusd (VNNI) or emulated maddubs for maximum throughput.

#ifdef __AVX512F__

static constexpr size_t kByteTq512Points = 16;  // int32 lanes in __m512i
static constexpr size_t kByteTq512Mq = 4;

// AVX-512 dpbusd: native VNNI or emulated.
inline __m512i byte_tq_dpbusd_512(__m512i acc, __m512i a_unsigned, __m512i b_signed) {
#ifdef __AVX512VNNI__
  return _mm512_dpbusd_epi32(acc, a_unsigned, b_signed);
#else
  const __m512i prod16 = _mm512_maddubs_epi16(a_unsigned, b_signed);
  const __m512i prod32 = _mm512_madd_epi16(prod16, _mm512_set1_epi16(1));
  return _mm512_add_epi32(acc, prod32);
#endif
}

// Epilogue: bias correct, float post-transform, update running min (16 points).
template<bool Metric>
inline void byte_tq_epilogue_512(
    __m512i acc, int32_t q_byte_sum, float q_nsf, float q_sqn,
    const float* norms16, const float* sqn16,
    __m512& running_min) {
  const __m512i bias = _mm512_set1_epi32(128 * q_byte_sum);
  const __m512i corrected = _mm512_sub_epi32(acc, bias);

  __m512 fdot = _mm512_cvtepi32_ps(corrected);
  const __m512 norm = _mm512_loadu_ps(norms16);
  const __m512 nsf = _mm512_set1_ps(q_nsf);

  __m512 neg_dot = _mm512_mul_ps(fdot, norm);
  neg_dot = _mm512_mul_ps(neg_dot, nsf);
  neg_dot = _mm512_sub_ps(_mm512_setzero_ps(), neg_dot);

  __m512 dist;
  if constexpr (Metric) {
    const __m512 sqn_v = _mm512_loadu_ps(sqn16);
    const __m512 sqn_q = _mm512_set1_ps(q_sqn);
    dist = _mm512_add_ps(sqn_v, _mm512_add_ps(
        _mm512_add_ps(neg_dot, neg_dot), sqn_q));
  } else {
    dist = neg_dot;
  }

  running_min = _mm512_min_ps(running_min, dist);
}

template<bool Metric>
inline float chamfer_byte_tq_gemm_512(
    const Quantized_Query<Metric>* const* query_ptrs,
    size_t num_queries,
    const uint8_t* strip_data,
    const float* norms,
    const float* squared_norms,
    size_t strip_stride,
    size_t n_strips,
    size_t num_bytes_per_point,
    size_t cloud_size) {

  const size_t padded_dim = num_bytes_per_point;
  const size_t total_tiles = (padded_dim + 3) / 4;
  const size_t group8_bytes = total_tiles * 32;         // bytes per 8-pt group
  const size_t n_groups16 = (cloud_size + kByteTq512Points - 1) / kByteTq512Points;

  // Step 1: query byte sums.
  thread_local std::vector<int32_t> all_q_byte_sums;
  all_q_byte_sums.resize(num_queries);
  for (size_t qi = 0; qi < num_queries; ++qi) {
    int32_t bsum = 0;
    for (size_t d = 0; d < padded_dim; ++d)
      bsum += static_cast<int32_t>(query_ptrs[qi]->query_data[d]);
    all_q_byte_sums[qi] = bsum;
  }

  // Step 2: pad norms.
  const size_t padded_pts = n_groups16 * kByteTq512Points;
  thread_local std::vector<float> padded_norms;
  thread_local std::vector<float> padded_sqn;
  padded_norms.resize(padded_pts);
  std::memcpy(padded_norms.data(), norms, cloud_size * sizeof(float));
  std::memset(padded_norms.data() + cloud_size, 0,
              (padded_pts - cloud_size) * sizeof(float));
  if constexpr (Metric) {
    padded_sqn.resize(padded_pts);
    std::memcpy(padded_sqn.data(), squared_norms, cloud_size * sizeof(float));
    std::memset(padded_sqn.data() + cloud_size, 0,
                (padded_pts - cloud_size) * sizeof(float));
  }

  // Step 3: Score.
  float total_chamfer = 0.0f;
  size_t qi = 0;

  for (; qi + kByteTq512Mq <= num_queries; qi += kByteTq512Mq) {
    __m512 mins[kByteTq512Mq];
    for (size_t q = 0; q < kByteTq512Mq; ++q)
      mins[q] = _mm512_set1_ps(std::numeric_limits<float>::max());

    for (size_t g16 = 0; g16 < n_groups16; ++g16) {
      const size_t point_start = g16 * kByteTq512Points;
      const size_t strip = point_start / 64;
      const size_t group8_lo = (point_start % 64) / 8;  // first 8-pt group in strip
      const uint8_t* base_lo = strip_data + strip * strip_stride +
                                group8_lo * group8_bytes;
      const uint8_t* base_hi = base_lo + group8_bytes;   // next 8-pt group

      __m512i acc[kByteTq512Mq];
      for (size_t q = 0; q < kByteTq512Mq; ++q)
        acc[q] = _mm512_setzero_si512();

      for (size_t t = 0; t < total_tiles; ++t) {
        // Load two 32-byte tiles → one 64-byte __m512i (16 points).
        const __m256i lo = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(base_lo + t * 32));
        const __m256i hi = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(base_hi + t * 32));
        const __m512i tile = _mm512_inserti64x4(
            _mm512_castsi256_si512(lo), hi, 1);

        for (size_t q = 0; q < kByteTq512Mq; ++q) {
          const __m512i qv = _mm512_set1_epi32(
              reinterpret_cast<const int32_t*>(
                  query_ptrs[qi + q]->query_data.data())[t]);
          acc[q] = byte_tq_dpbusd_512(acc[q], tile, qv);
        }
      }

      for (size_t q = 0; q < kByteTq512Mq; ++q) {
        byte_tq_epilogue_512<Metric>(
            acc[q], all_q_byte_sums[qi + q],
            query_ptrs[qi + q]->norm_scaling_factor,
            query_ptrs[qi + q]->unquantized_squared_norm,
            padded_norms.data() + g16 * kByteTq512Points,
            padded_sqn.data() + g16 * kByteTq512Points,
            mins[q]);
      }
    }

    for (size_t q = 0; q < kByteTq512Mq; ++q)
      total_chamfer += _mm512_reduce_min_ps(mins[q]);
  }

  // Tail: 1 query at a time.
  for (; qi < num_queries; ++qi) {
    __m512 running_min = _mm512_set1_ps(std::numeric_limits<float>::max());
    const int32_t bsum = all_q_byte_sums[qi];
    const int8_t* qdata = query_ptrs[qi]->query_data.data();

    for (size_t g16 = 0; g16 < n_groups16; ++g16) {
      const size_t point_start = g16 * kByteTq512Points;
      const size_t strip = point_start / 64;
      const size_t group8_lo = (point_start % 64) / 8;
      const uint8_t* base_lo = strip_data + strip * strip_stride +
                                group8_lo * group8_bytes;
      const uint8_t* base_hi = base_lo + group8_bytes;

      __m512i acc = _mm512_setzero_si512();
      for (size_t t = 0; t < total_tiles; ++t) {
        const __m256i lo = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(base_lo + t * 32));
        const __m256i hi = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(base_hi + t * 32));
        const __m512i tile = _mm512_inserti64x4(
            _mm512_castsi256_si512(lo), hi, 1);
        const __m512i qv = _mm512_set1_epi32(
            reinterpret_cast<const int32_t*>(qdata)[t]);
        acc = byte_tq_dpbusd_512(acc, tile, qv);
      }

      byte_tq_epilogue_512<Metric>(
          acc, bsum,
          query_ptrs[qi]->norm_scaling_factor,
          query_ptrs[qi]->unquantized_squared_norm,
          padded_norms.data() + g16 * kByteTq512Points,
          padded_sqn.data() + g16 * kByteTq512Points,
          running_min);
    }

    total_chamfer += _mm512_reduce_min_ps(running_min);
  }

  return total_chamfer;
}

#endif  // __AVX512F__

// =========================================================================
// AVX2 GEMM Chamfer kernel (maddubs: 8 points per __m256i, fallback)
// =========================================================================
// DB is stored as uint8 (biased +128) in pre-interleaved tile layout.
// Each tile = 4 dims × 8 points = 32 bytes, already in maddubs order.
// Within a 64-point strip, 8 groups of 8 points, each with total_tiles tiles.
// Layout: strip_data[strip * stride + group * group_bytes + tile * 32 + ...].
// The kernel does: _mm256_load_si256 → maddubs → accumulate. Zero interleave.

#ifdef __AVX2__

static constexpr size_t kByteTqPoints = 8;   // int32 lanes in __m256i
static constexpr size_t kByteTqMq = 4;       // queries per batch

// AVX2 unsigned×signed int8 dot product accumulate.
inline __m256i byte_tq_dpbusd(__m256i acc, __m256i a_unsigned, __m256i b_signed) {
  const __m256i prod16 = _mm256_maddubs_epi16(a_unsigned, b_signed);
  const __m256i prod32 = _mm256_madd_epi16(prod16, _mm256_set1_epi16(1));
  return _mm256_add_epi32(acc, prod32);
}

// Reduce __m256 (8 floats) to scalar min.
inline float byte_tq_reduce_min_ps(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v);
  __m128 hi = _mm256_extractf128_ps(v, 1);
  __m128 m = _mm_min_ps(lo, hi);
  __m128 m2 = _mm_shuffle_ps(m, m, _MM_SHUFFLE(1,0,3,2));
  m = _mm_min_ps(m, m2);
  __m128 m3 = _mm_shuffle_ps(m, m, _MM_SHUFFLE(0,1,0,1));
  m = _mm_min_ps(m, m3);
  return _mm_cvtss_f32(m);
}

// Epilogue: bias correct, float post-transform, update running min (8 points).
template<bool Metric>
inline void byte_tq_epilogue(
    __m256i acc, int32_t q_byte_sum, float q_nsf, float q_sqn,
    const float* norms8, const float* sqn8,
    __m256& running_min) {
  const __m256i bias = _mm256_set1_epi32(128 * q_byte_sum);
  const __m256i corrected = _mm256_sub_epi32(acc, bias);

  __m256 fdot = _mm256_cvtepi32_ps(corrected);
  const __m256 norm = _mm256_loadu_ps(norms8);
  const __m256 nsf = _mm256_set1_ps(q_nsf);

  __m256 neg_dot = _mm256_mul_ps(fdot, norm);
  neg_dot = _mm256_mul_ps(neg_dot, nsf);
  neg_dot = _mm256_sub_ps(_mm256_setzero_ps(), neg_dot);

  __m256 dist;
  if constexpr (Metric) {
    const __m256 sqn_v = _mm256_loadu_ps(sqn8);
    const __m256 sqn_q = _mm256_set1_ps(q_sqn);
    dist = _mm256_add_ps(sqn_v, _mm256_add_ps(
        _mm256_add_ps(neg_dot, neg_dot), sqn_q));
  } else {
    dist = neg_dot;
  }

  running_min = _mm256_min_ps(running_min, dist);
}

// Fused GEMM kernel: loads directly from strip layout (no panel buffer).
// For each group of 8 points, iterates dims 4 at a time, loading 32 bytes
// (4 dims × 8 points) from the strip and immediately multiplying.
template<bool Metric>
inline float chamfer_byte_tq_gemm(
    const Quantized_Query<Metric>* const* query_ptrs,
    size_t num_queries,
    const uint8_t* strip_data,
    const float* norms,
    const float* squared_norms,
    size_t strip_stride,
    size_t n_strips,
    size_t num_bytes_per_point,   // = padded_dim (1 byte per dim)
    size_t cloud_size) {

  const size_t padded_dim = num_bytes_per_point;
  const size_t total_tiles = (padded_dim + 3) / 4;  // 4 dims per tile
  const size_t group_bytes = total_tiles * 32;  // 32 bytes per tile
  const size_t n_groups = (cloud_size + kByteTqPoints - 1) / kByteTqPoints;

  // Step 1: Prepare query data + byte sums.
  thread_local std::vector<int32_t> all_q_byte_sums;
  all_q_byte_sums.resize(num_queries);

  for (size_t qi = 0; qi < num_queries; ++qi) {
    int32_t bsum = 0;
    for (size_t d = 0; d < padded_dim; ++d)
      bsum += static_cast<int32_t>(query_ptrs[qi]->query_data[d]);
    all_q_byte_sums[qi] = bsum;
  }

  // Step 2: Pad norms arrays.
  const size_t padded_pts = n_groups * kByteTqPoints;
  thread_local std::vector<float> padded_norms;
  thread_local std::vector<float> padded_sqn;
  padded_norms.resize(padded_pts);
  std::memcpy(padded_norms.data(), norms, cloud_size * sizeof(float));
  std::memset(padded_norms.data() + cloud_size, 0,
              (padded_pts - cloud_size) * sizeof(float));
  if constexpr (Metric) {
    padded_sqn.resize(padded_pts);
    std::memcpy(padded_sqn.data(), squared_norms, cloud_size * sizeof(float));
    std::memset(padded_sqn.data() + cloud_size, 0,
                (padded_pts - cloud_size) * sizeof(float));
  }

  // Step 3: Score — fused load+compute, no panel buffer needed.
  float total_chamfer = 0.0f;
  size_t qi = 0;

  // Batch of kByteTqMq queries.
  for (; qi + kByteTqMq <= num_queries; qi += kByteTqMq) {
    __m256 mins[kByteTqMq];
    for (size_t q = 0; q < kByteTqMq; ++q)
      mins[q] = _mm256_set1_ps(std::numeric_limits<float>::max());

    for (size_t g = 0; g < n_groups; ++g) {
      const size_t point_start = g * kByteTqPoints;
      const size_t strip = point_start / 64;
      const size_t group_in_strip = (point_start % 64) / kByteTqPoints;
      const uint8_t* group_base = strip_data + strip * strip_stride +
                                   group_in_strip * group_bytes;

      // Accumulate dot products: direct aligned loads, no interleave.
      __m256i acc[kByteTqMq];
      for (size_t q = 0; q < kByteTqMq; ++q)
        acc[q] = _mm256_setzero_si256();

      for (size_t t = 0; t < total_tiles; ++t) {
        // Direct load: tile is already [d0_p0,d1_p0,d2_p0,d3_p0,...] for 8 pts.
        const __m256i tile = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(group_base + t * 32));

        for (size_t q = 0; q < kByteTqMq; ++q) {
          const __m256i qv = _mm256_set1_epi32(
              reinterpret_cast<const int32_t*>(
                  query_ptrs[qi + q]->query_data.data())[t]);
          acc[q] = byte_tq_dpbusd(acc[q], tile, qv);
        }
      }

      // Epilogue for this group of 8 points.
      for (size_t q = 0; q < kByteTqMq; ++q) {
        byte_tq_epilogue<Metric>(
            acc[q], all_q_byte_sums[qi + q],
            query_ptrs[qi + q]->norm_scaling_factor,
            query_ptrs[qi + q]->unquantized_squared_norm,
            padded_norms.data() + g * kByteTqPoints,
            padded_sqn.data() + g * kByteTqPoints,
            mins[q]);
      }
    }

    for (size_t q = 0; q < kByteTqMq; ++q)
      total_chamfer += byte_tq_reduce_min_ps(mins[q]);
  }

  // Tail: remaining queries (1 at a time).
  for (; qi < num_queries; ++qi) {
    __m256 running_min = _mm256_set1_ps(std::numeric_limits<float>::max());
    const int32_t bsum = all_q_byte_sums[qi];
    const int8_t* qdata = query_ptrs[qi]->query_data.data();

    for (size_t g = 0; g < n_groups; ++g) {
      const size_t point_start = g * kByteTqPoints;
      const size_t strip = point_start / 64;
      const size_t group_in_strip = (point_start % 64) / kByteTqPoints;
      const uint8_t* group_base = strip_data + strip * strip_stride +
                                   group_in_strip * group_bytes;

      __m256i acc = _mm256_setzero_si256();
      for (size_t t = 0; t < total_tiles; ++t) {
        const __m256i tile = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(group_base + t * 32));
        const __m256i qv = _mm256_set1_epi32(
            reinterpret_cast<const int32_t*>(qdata)[t]);
        acc = byte_tq_dpbusd(acc, tile, qv);
      }

      byte_tq_epilogue<Metric>(
          acc, bsum,
          query_ptrs[qi]->norm_scaling_factor,
          query_ptrs[qi]->unquantized_squared_norm,
          padded_norms.data() + g * kByteTqPoints,
          padded_sqn.data() + g * kByteTqPoints,
          running_min);
    }

    total_chamfer += byte_tq_reduce_min_ps(running_min);
  }

  return total_chamfer;
}

#endif  // __AVX2__

// =========================================================================
// Quantized_Point_Range: strip-interleaved encoded base points (byte layout)
// =========================================================================
template<typename PointRange, bool Metric>
class Quantized_Point_Range {
 public:
  static constexpr bool is_fastscan = false;

  size_t n_points_raw = 0;
  size_t n_points_raw_unpadded = 0;
  size_t dim = 0;
  size_t num_bytes_per_datapoint = 0;  // = padded_dim (1 byte per dim)
  size_t stride = 0;                   // strip stride = padded_dim * 64

  parlay::sequence<uint8_t> packed_codes;   // uint8 codes (biased +128)
  parlay::sequence<float> norm_scaling_factors;
  parlay::sequence<float> unquantized_squared_norms;
  parlay::sequence<size_t> cloud_vec_offsets;

  Quantized_Point_Range() = default;

  // Per-point access: gather bytes from pre-interleaved tile layout.
  Quantized_Point<Metric> operator[](size_t i) const {
    const size_t strip = i / 64;
    const size_t group = (i % 64) / 8;
    const size_t pt_in_group = i % 8;
    const size_t nb = num_bytes_per_datapoint;
    const size_t total_tiles = (nb + 3) / 4;
    const size_t group_bytes = total_tiles * 32;

    std::vector<uint8_t> codes(nb);
    for (size_t d = 0; d < nb; ++d) {
      const size_t tile = d / 4;
      const size_t dim_in_tile = d % 4;
      codes[d] = packed_codes[strip * stride + group * group_bytes +
                               tile * 32 + pt_in_group * 4 + dim_in_tile];
    }

    return Quantized_Point<Metric>(
        std::move(codes), nb,
        norm_scaling_factors[i],
        Metric ? unquantized_squared_norms[i] : 0.0f);
  }

  inline uint32_t size() const noexcept {
    return static_cast<uint32_t>(n_points_raw_unpadded);
  }
  inline uint32_t get_dims() const noexcept {
    return static_cast<uint32_t>(dim);
  }

  void save(std::ostream& out) const {
    out.write(reinterpret_cast<const char*>(&n_points_raw), sizeof(n_points_raw));
    out.write(reinterpret_cast<const char*>(&n_points_raw_unpadded), sizeof(n_points_raw_unpadded));
    out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    out.write(reinterpret_cast<const char*>(&num_bytes_per_datapoint), sizeof(num_bytes_per_datapoint));
    out.write(reinterpret_cast<const char*>(&stride), sizeof(stride));

    size_t code_size = packed_codes.size();
    out.write(reinterpret_cast<const char*>(&code_size), sizeof(code_size));
    if (code_size)
      out.write(reinterpret_cast<const char*>(packed_codes.data()), code_size);

    size_t nf_size = norm_scaling_factors.size();
    out.write(reinterpret_cast<const char*>(&nf_size), sizeof(nf_size));
    if (nf_size)
      out.write(reinterpret_cast<const char*>(norm_scaling_factors.data()), nf_size * sizeof(float));

    size_t sqn_size = unquantized_squared_norms.size();
    out.write(reinterpret_cast<const char*>(&sqn_size), sizeof(sqn_size));
    if (sqn_size)
      out.write(reinterpret_cast<const char*>(unquantized_squared_norms.data()), sqn_size * sizeof(float));

    size_t off_size = cloud_vec_offsets.size();
    out.write(reinterpret_cast<const char*>(&off_size), sizeof(off_size));
    if (off_size)
      out.write(reinterpret_cast<const char*>(cloud_vec_offsets.data()), off_size * sizeof(size_t));
  }

  void load(std::istream& in) {
    in.read(reinterpret_cast<char*>(&n_points_raw), sizeof(n_points_raw));
    in.read(reinterpret_cast<char*>(&n_points_raw_unpadded), sizeof(n_points_raw_unpadded));
    in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    in.read(reinterpret_cast<char*>(&num_bytes_per_datapoint), sizeof(num_bytes_per_datapoint));
    in.read(reinterpret_cast<char*>(&stride), sizeof(stride));

    size_t code_size = 0;
    in.read(reinterpret_cast<char*>(&code_size), sizeof(code_size));
    packed_codes.resize(code_size);
    if (code_size)
      in.read(reinterpret_cast<char*>(packed_codes.data()), code_size);

    size_t nf_size = 0;
    in.read(reinterpret_cast<char*>(&nf_size), sizeof(nf_size));
    norm_scaling_factors.resize(nf_size);
    if (nf_size)
      in.read(reinterpret_cast<char*>(norm_scaling_factors.data()), nf_size * sizeof(float));

    size_t sqn_size = 0;
    in.read(reinterpret_cast<char*>(&sqn_size), sizeof(sqn_size));
    unquantized_squared_norms.resize(sqn_size);
    if (sqn_size)
      in.read(reinterpret_cast<char*>(unquantized_squared_norms.data()), sqn_size * sizeof(float));

    size_t off_size = 0;
    in.read(reinterpret_cast<char*>(&off_size), sizeof(off_size));
    cloud_vec_offsets.resize(off_size);
    if (off_size)
      in.read(reinterpret_cast<char*>(cloud_vec_offsets.data()), off_size * sizeof(size_t));
  }
};

// =========================================================================
// Model: train / encode (strip layout) / quantize_query
// =========================================================================
template<bool Metric>
class Model {
 public:
  static constexpr bool is_fastscan = false;

  size_t dim = 0;
  size_t padded_dim = 0;
  size_t num_bytes_per_datapoint = 0;  // = padded_dim (1 byte per dim)
  size_t seed_ = 42;

  std::vector<float> signs;
  rabitqlib::RotatorType rotator_type = rabitqlib::RotatorType::FhtKacRotator;
  std::unique_ptr<rabitqlib::Rotator<float>> rotator;

  Model() = default;

  template<typename PointRange>
  void train(const PointRange& data) {
    dim = data.get_dims();
    if (dim == 0) return;
    rotator.reset(rabitqlib::choose_rotator<float>(dim, rotator_type));
    if (!rotator) return;
    padded_dim = rotator->size();
    // Key difference: 1 byte per dim instead of 0.5.
    num_bytes_per_datapoint = padded_dim;

    signs.resize(padded_dim);
    std::mt19937 gen(seed_);
    std::uniform_int_distribution<> dist(0, 1);
    for (size_t i = 0; i < padded_dim; ++i) {
      signs[i] = (2.0f * dist(gen) - 1.0f);
    }
  }

 private:
  // Encode a single point to uint8 (biased +128).
  // Returns {squared_norm, norm_scaling_factor}.
  std::pair<float, float> encode_single(
      const float* p, uint8_t* output,
      std::vector<float>& ws) const {
    rotator->rotate(p, ws.data());

    float sqr_norm = 0.0f;
    for (size_t i = 0; i < padded_dim; ++i)
      sqr_norm += ws[i] * ws[i];

    if (sqr_norm == 0.0f) {
      // Zero vector → code = 128 (neutral for maddubs).
      std::memset(output, 128, num_bytes_per_datapoint);
      return {0.0f, 0.0f};
    }

    const float norm = std::sqrt(sqr_norm);
    const float inv_norm = 1.0f / norm;
    for (size_t i = 0; i < padded_dim; ++i)
      ws[i] = std::clamp(ws[i] * signs[i] * inv_norm,
                          -internal::kValueCap, internal::kValueCap);

    // Quantize to int8 range, then bias to uint8 for maddubs.
    const float scale = 127.0f / internal::kValueCap;
    int32_t quant_norm_sq = 0;
    for (size_t i = 0; i < padded_dim; ++i) {
      float scaled = std::clamp(std::round(ws[i] * scale), -127.0f, 127.0f);
      int8_t s = static_cast<int8_t>(scaled);
      output[i] = static_cast<uint8_t>(static_cast<int16_t>(s) + 128);
      quant_norm_sq += static_cast<int32_t>(s) * static_cast<int32_t>(s);
    }

    float nsf = quant_norm_sq > 0
        ? norm / std::sqrt(static_cast<float>(quant_norm_sq))
        : 0.0f;
    return {sqr_norm, nsf};
  }

 public:
  // ---- Encode: strip-interleaved layout (1 byte per dim) ----
  template<typename PointRangeTy>
  Quantized_Point_Range<PointRangeTy, Metric> encode(
      const PointRangeTy& data) const {
    if (!rotator) {
      std::cerr << "byte_turboquant::encode: rotator is null.\n";
      return Quantized_Point_Range<PointRangeTy, Metric>();
    }

    const size_t N = data.size();
    const size_t N_padded = ((N + 63) / 64) * 64;
    const size_t n_strips = N_padded / 64;
    const size_t strip_stride = num_bytes_per_datapoint * 64;

    Quantized_Point_Range<PointRangeTy, Metric> enc;
    enc.dim = padded_dim;
    enc.num_bytes_per_datapoint = num_bytes_per_datapoint;
    enc.stride = strip_stride;
    enc.n_points_raw = N_padded;
    enc.n_points_raw_unpadded = N;
    enc.cloud_vec_offsets.resize(2);
    enc.cloud_vec_offsets[0] = 0;
    enc.cloud_vec_offsets[1] = N_padded;

    // Initialize codes to 128 (neutral for maddubs: 128 * signed = 0 after bias).
    enc.packed_codes.resize(n_strips * strip_stride, 128);
    enc.norm_scaling_factors.resize(N_padded, 0.0f);
    enc.unquantized_squared_norms.resize(N_padded, 0.0f);

    struct Workspace {
      std::vector<float> rot;
      std::vector<uint8_t> point_codes;
      void ensure(size_t pdim) {
        if (rot.size() != pdim) rot.resize(pdim);
        if (point_codes.size() != pdim) point_codes.resize(pdim);
      }
    };

    parlay::parallel_for(0, N, [&](size_t vi) {
      static thread_local Workspace ws;
      ws.ensure(padded_dim);

      const float* p = reinterpret_cast<const float*>(data.location(vi));
      auto [sqn, nsf] = encode_single(p, ws.point_codes.data(), ws.rot);
      enc.norm_scaling_factors[vi] = nsf;
      enc.unquantized_squared_norms[vi] = sqn;

      // Scatter into pre-interleaved tile layout.
      const size_t strip = vi / 64;
      const size_t group = (vi % 64) / 8;
      const size_t pt_in_group = vi % 8;
      const size_t total_tiles = (num_bytes_per_datapoint + 3) / 4;
      const size_t group_bytes = total_tiles * 32;
      const size_t strip_base = strip * strip_stride + group * group_bytes;
      for (size_t d = 0; d < num_bytes_per_datapoint; ++d) {
        const size_t tile = d / 4;
        const size_t dim_in_tile = d % 4;
        enc.packed_codes[strip_base + tile * 32 + pt_in_group * 4 + dim_in_tile] =
            ws.point_codes[d];
      }
    });

    return enc;
  }

  // ---- Encode with cloud offsets (multi-cloud) ----
  template<typename PointRangeTy, typename SeqOffsetsFloat>
  Quantized_Point_Range<PointRangeTy, Metric> encode(
      const PointRangeTy& data,
      const SeqOffsetsFloat& cloud_offsets_float) const {
    if (!rotator) return Quantized_Point_Range<PointRangeTy, Metric>();

    const size_t D = padded_dim;
    const size_t n_clouds =
        cloud_offsets_float.size() > 0 ? cloud_offsets_float.size() - 1 : 0;

    Quantized_Point_Range<PointRangeTy, Metric> enc;
    enc.dim = padded_dim;
    enc.num_bytes_per_datapoint = num_bytes_per_datapoint;
    const size_t strip_stride = num_bytes_per_datapoint * 64;
    enc.stride = strip_stride;
    enc.n_points_raw_unpadded = data.size();

    enc.cloud_vec_offsets.resize(n_clouds + 1);
    size_t cur = 0;
    for (size_t c = 0; c < n_clouds; ++c) {
      enc.cloud_vec_offsets[c] = cur;
      const size_t start_f = static_cast<size_t>(cloud_offsets_float[c]);
      const size_t end_f = static_cast<size_t>(cloud_offsets_float[c + 1]);
      const size_t n_vecs = (end_f - start_f) / D;
      cur += ((n_vecs + 63) / 64) * 64;
    }
    enc.cloud_vec_offsets[n_clouds] = cur;
    enc.n_points_raw = cur;

    const size_t n_strips = cur / 64;
    enc.packed_codes.resize(n_strips * strip_stride, 128);
    enc.norm_scaling_factors.resize(cur, 0.0f);
    enc.unquantized_squared_norms.resize(cur, 0.0f);

    struct Workspace {
      std::vector<float> rot;
      std::vector<uint8_t> point_codes;
      void ensure(size_t pdim) {
        if (rot.size() != pdim) rot.resize(pdim);
        if (point_codes.size() != pdim) point_codes.resize(pdim);
      }
    };

    for (size_t c = 0; c < n_clouds; ++c) {
      const size_t start_f = static_cast<size_t>(cloud_offsets_float[c]);
      const size_t end_f = static_cast<size_t>(cloud_offsets_float[c + 1]);
      const size_t src_start = start_f / D;
      const size_t n_vecs = (end_f - start_f) / D;
      const size_t dst_start = enc.cloud_vec_offsets[c];

      parlay::parallel_for(0, n_vecs, [&](size_t i) {
        static thread_local Workspace ws;
        ws.ensure(padded_dim);

        const float* p = reinterpret_cast<const float*>(
            data.location(src_start + i));
        auto [sqn, nsf] = encode_single(p, ws.point_codes.data(), ws.rot);

        const size_t dst_idx = dst_start + i;
        enc.norm_scaling_factors[dst_idx] = nsf;
        enc.unquantized_squared_norms[dst_idx] = sqn;

        const size_t strip = dst_idx / 64;
        const size_t group = (dst_idx % 64) / 8;
        const size_t pt_in_group = dst_idx % 8;
        const size_t total_tiles = (num_bytes_per_datapoint + 3) / 4;
        const size_t grp_bytes = total_tiles * 32;
        const size_t strip_base = strip * strip_stride + group * grp_bytes;
        for (size_t d = 0; d < num_bytes_per_datapoint; ++d) {
          const size_t tile = d / 4;
          const size_t dim_in_tile = d % 4;
          enc.packed_codes[strip_base + tile * 32 + pt_in_group * 4 + dim_in_tile] =
              ws.point_codes[d];
        }
      });
    }

    return enc;
  }

  // ---- Quantize query ----
  Quantized_Query<Metric> quantize_query(const float* qptr) const {
    if (!rotator) return Quantized_Query<Metric>();

    Quantized_Query<Metric> qq;
    qq.dim = padded_dim;
    qq.num_bytes_per_datapoint = num_bytes_per_datapoint;
    qq.query_data.resize(padded_dim, 0);

    std::vector<float> q_rot(padded_dim);
    rotator->rotate(qptr, q_rot.data());

    float sqr_norm = 0.0f;
    for (size_t i = 0; i < padded_dim; ++i)
      sqr_norm += q_rot[i] * q_rot[i];

    if (sqr_norm == 0.0f || !std::isfinite(sqr_norm)) {
      qq.norm_scaling_factor = 0.0f;
      qq.unquantized_squared_norm = 0.0f;
      return qq;
    }

    const float norm = std::sqrt(sqr_norm);
    const float inv_norm = 1.0f / norm;
    for (size_t i = 0; i < padded_dim; ++i)
      q_rot[i] = std::clamp(q_rot[i] * signs[i] * inv_norm,
                             -internal::kValueCap, internal::kValueCap);

    // Scale to int8 range using same kValueCap as DB encoding.
    const float scale = 127.0f / internal::kValueCap;

    int32_t quant_norm_sq = 0;
    for (size_t i = 0; i < padded_dim; ++i) {
      float scaled = std::clamp(std::round(q_rot[i] * scale), -127.0f, 127.0f);
      int8_t snapped = static_cast<int8_t>(scaled);
      qq.query_data[i] = snapped;
      quant_norm_sq += static_cast<int32_t>(snapped) * static_cast<int32_t>(snapped);
    }

    qq.norm_scaling_factor = quant_norm_sq > 0
        ? norm / std::sqrt(static_cast<float>(quant_norm_sq))
        : 0.0f;
    qq.unquantized_squared_norm = sqr_norm;
    return qq;
  }

  template<typename PointTy>
  typename std::enable_if<!std::is_pointer<PointTy>::value,
                          Quantized_Query<Metric>>::type
  quantize_query(const PointTy& query) const {
    const float* ptr = reinterpret_cast<const float*>(&query);
    return quantize_query(ptr);
  }

  template<typename PointCloudTy>
  void quantize_query_batch(
      const PointCloudTy& qc,
      parlay::sequence<Quantized_Query<Metric>>& out) const {
    const uint32_t nq = qc.size();
    const uint32_t d = qc.get_dims();
    const float* base = qc.data();
    out.clear();
    out.reserve(nq);
    for (uint32_t i = 0; i < nq; ++i)
      out.emplace_back(quantize_query(base + static_cast<size_t>(i) * d));
  }

  void save(std::ofstream& out) const {
    out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    out.write(reinterpret_cast<const char*>(&padded_dim), sizeof(padded_dim));
    out.write(reinterpret_cast<const char*>(&num_bytes_per_datapoint), sizeof(num_bytes_per_datapoint));
    out.write(reinterpret_cast<const char*>(&seed_), sizeof(seed_));
    int rt = static_cast<int>(rotator_type);
    out.write(reinterpret_cast<const char*>(&rt), sizeof(rt));
    out.write(reinterpret_cast<const char*>(signs.data()), signs.size() * sizeof(float));
  }

  void load(std::ifstream& in) {
    in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    in.read(reinterpret_cast<char*>(&padded_dim), sizeof(padded_dim));
    in.read(reinterpret_cast<char*>(&num_bytes_per_datapoint), sizeof(num_bytes_per_datapoint));
    in.read(reinterpret_cast<char*>(&seed_), sizeof(seed_));
    int rt = 0;
    in.read(reinterpret_cast<char*>(&rt), sizeof(rt));
    rotator_type = static_cast<rabitqlib::RotatorType>(rt);
    rotator.reset(rabitqlib::choose_rotator<float>(dim, rotator_type));
    if (rotator) padded_dim = rotator->size();
    signs.resize(padded_dim);
    in.read(reinterpret_cast<char*>(signs.data()), padded_dim * sizeof(float));
  }
};

}  // namespace byte_turboquant
}  // namespace mvsic
