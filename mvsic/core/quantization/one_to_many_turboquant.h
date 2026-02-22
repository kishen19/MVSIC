#pragma once

// one_to_many_turboquant.h
//
// Fast TurboQuant implementation with FastScan-style strip-interleaved layout.
// Uses a 4-bit codebook (16 centroids per dimension) with random-sign flip +
// Hadamard rotation preprocessing, int8 query encoding, and AVX-512 SIMD
// scoring with 64-point strips for amortized throughput.
//
// Interface: Model<Metric>, Quantized_Query<Metric>, Quantized_Point<Metric>,
//            Quantized_Point_Range<PointRange, Metric>
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
namespace one_to_many_turboquant {

// =========================================================================
// Constants
// =========================================================================
namespace internal {

static constexpr std::array<int8_t, 16> kTurboQuantCentroidsInt8 = {
    6, 18, 31, 44, 58, 75, 96, 127, -6, -18, -31, -44, -58, -75, -96, -127};

static constexpr std::array<float, 16> kTurboQuantCentroidsFloat = {
    0.06429295f,  0.19320825f, 0.32435477f, 0.45861828f,
    0.59504265f,  0.73458207f, 0.87637275f, 1.0211549f,
    -0.06429295f, -0.19320825f, -0.32435477f, -0.45861828f,
    -0.59504265f, -0.73458207f, -0.87637275f, -1.0211549f};

static constexpr std::array<float, 8> kSquaredCentroidsMag = {
    0.00413358f, 0.03732943f, 0.10520603f, 0.21032069f,
    0.35407575f, 0.53961036f, 0.76803890f, 1.04275720f};

static constexpr std::array<float, 7> kBoundaries = {
    0.12875060f, 0.39148653f, 0.66481236f, 0.94876383f,
    1.25821375f, 1.62382795f, 2.09153930f};

static constexpr float kValueCap = 3.91724f;
static constexpr size_t kStripSize = 64;
static constexpr size_t kQueryPadding = 64;

inline uint8_t FindBucket(float x) {
  uint8_t idx = static_cast<uint8_t>(kBoundaries.size());
  while (idx > 0 && x < kBoundaries[idx - 1]) --idx;
  return idx;
}

inline uint8_t FourBitEncoding(float x) {
  return FindBucket(std::abs(x)) | (x > 0 ? 0 : 8);
}

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
  const uint8_t* code_ptr = nullptr;
  size_t num_bytes = 0;
  float norm_scaling_factor = 0.0f;
  float unquantized_squared_norm = 0.0f;

  // Owned buffer for when point bytes are gathered from strip layout.
  std::vector<uint8_t> owned_codes;

  Quantized_Point() = default;
  Quantized_Point(const uint8_t* ptr, size_t nb, float nsf, float usn)
      : code_ptr(ptr), num_bytes(nb), norm_scaling_factor(nsf),
        unquantized_squared_norm(usn) {}

  // Constructor that takes ownership of gathered bytes.
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
// Quantized_Query: int8-encoded query with strip-based SIMD scoring
// =========================================================================
template<bool Metric>
class Quantized_Query {
 public:
  using distanceType = float;

  // NON-deinterleaved int8 query: [dim0, dim1, dim2, ..., dim_{padded_dim-1}]
  // For strip scoring, we need query[dim_2j] and query[dim_2j+1] as scalars.
  std::vector<int8_t> query_data;
  size_t dim = 0;
  size_t num_bytes_per_datapoint = 0;

  float norm_scaling_factor = 0.0f;
  float unquantized_squared_norm = 0.0f;

  Quantized_Query() = default;

  inline float distance(const Quantized_Point<Metric>& p) const {
    return p.distance(*this);
  }

#ifdef __AVX512F__
  // ---- Strip-based scoring: 64 points at once ----
  // Scores 64 points from a strip. strip_ptr points to the strip start
  // (byte_pos=0, point_0). Layout: for byte_pos j, 64 contiguous bytes
  // at strip_ptr + j * 64.
  // Results written to dots[0..63] as raw int32 dot products.
  inline void scan_strip_64_dots(
      const uint8_t* strip_ptr,
      int32_t* dots) const {
    const __m512i codebook = _mm512_broadcast_i32x4(
        _mm_load_si128(reinterpret_cast<const __m128i*>(
            internal::kTurboQuantCentroidsInt8.data())));
    const __m512i mask_0f = _mm512_set1_epi8(0x0F);

    // 4 __m512i int32 accumulators for 64 points (16 int32 each).
    __m512i acc0 = _mm512_setzero_si512();
    __m512i acc1 = _mm512_setzero_si512();
    __m512i acc2 = _mm512_setzero_si512();
    __m512i acc3 = _mm512_setzero_si512();

    const size_t nb = num_bytes_per_datapoint;

    for (size_t j = 0; j < nb; ++j) {
      // Load 64 packed bytes (64 points' byte j).
      const __m512i codes = _mm512_loadu_si512(strip_ptr + j * 64);

      // Split nibbles → centroid lookup.
      const __m512i even_idxs = _mm512_and_si512(codes, mask_0f);
      const __m512i odd_idxs = _mm512_and_si512(
          _mm512_srli_epi16(codes, 4), mask_0f);
      const __m512i even_c = _mm512_shuffle_epi8(codebook, even_idxs);
      const __m512i odd_c = _mm512_shuffle_epi8(codebook, odd_idxs);

      // Broadcast query values for dims 2j and 2j+1.
      const int16_t q_even = static_cast<int16_t>(query_data[2 * j]);
      const int16_t q_odd = static_cast<int16_t>(query_data[2 * j + 1]);
      const __m512i q_even_v = _mm512_set1_epi16(q_even);
      const __m512i q_odd_v = _mm512_set1_epi16(q_odd);

      // Extend int8 centroids → int16, multiply by query, accumulate int32.
      // Process points 0-31 (low 256 bits of even_c/odd_c).
      const __m256i even_lo = _mm512_castsi512_si256(even_c);
      const __m256i even_hi = _mm512_extracti64x4_epi64(even_c, 1);
      const __m256i odd_lo = _mm512_castsi512_si256(odd_c);
      const __m256i odd_hi = _mm512_extracti64x4_epi64(odd_c, 1);

      // Points 0-31: even dim.
      const __m512i even16_lo = _mm512_cvtepi8_epi16(even_lo);  // 32 int16
      __m512i prod_lo = _mm512_mullo_epi16(even16_lo, q_even_v);
      // Points 0-31: odd dim.
      const __m512i odd16_lo = _mm512_cvtepi8_epi16(odd_lo);
      prod_lo = _mm512_add_epi16(prod_lo,
                                  _mm512_mullo_epi16(odd16_lo, q_odd_v));

      // Points 32-63: even dim.
      const __m512i even16_hi = _mm512_cvtepi8_epi16(even_hi);
      __m512i prod_hi = _mm512_mullo_epi16(even16_hi, q_even_v);
      // Points 32-63: odd dim.
      const __m512i odd16_hi = _mm512_cvtepi8_epi16(odd_hi);
      prod_hi = _mm512_add_epi16(prod_hi,
                                  _mm512_mullo_epi16(odd16_hi, q_odd_v));

      // Widen int16 → int32 and accumulate.
      // prod_lo has 32 int16 (points 0-31). Split to 2 × 16 int32.
      const __m256i prod_lo_lo = _mm512_castsi512_si256(prod_lo);
      const __m256i prod_lo_hi = _mm512_extracti64x4_epi64(prod_lo, 1);
      acc0 = _mm512_add_epi32(acc0, _mm512_cvtepi16_epi32(prod_lo_lo));
      acc1 = _mm512_add_epi32(acc1, _mm512_cvtepi16_epi32(prod_lo_hi));

      // prod_hi has 32 int16 (points 32-63). Split to 2 × 16 int32.
      const __m256i prod_hi_lo = _mm512_castsi512_si256(prod_hi);
      const __m256i prod_hi_hi = _mm512_extracti64x4_epi64(prod_hi, 1);
      acc2 = _mm512_add_epi32(acc2, _mm512_cvtepi16_epi32(prod_hi_lo));
      acc3 = _mm512_add_epi32(acc3, _mm512_cvtepi16_epi32(prod_hi_hi));
    }

    _mm512_storeu_si512(dots + 0, acc0);
    _mm512_storeu_si512(dots + 16, acc1);
    _mm512_storeu_si512(dots + 32, acc2);
    _mm512_storeu_si512(dots + 48, acc3);
  }

  // ---- Chamfer-optimized: scan strip, post-transform, running min ----
  // Scores 64 points and updates per-point running minimum distances.
  inline void scan_strip_64_min(
      const uint8_t* strip_ptr,
      const float* norms,          // norm_scaling_factor[64]
      const float* squared_norms,  // unquantized_squared_norm[64] (L2 only)
      __m512& min0, __m512& min1, __m512& min2, __m512& min3) const {
    alignas(64) int32_t dots[64];
    scan_strip_64_dots(strip_ptr, dots);

    // Post-transform: convert to float distances, update running min.
    const __m512 nsf = _mm512_set1_ps(norm_scaling_factor);

    for (int chunk = 0; chunk < 4; ++chunk) {
      const __m512i raw = _mm512_load_si512(dots + chunk * 16);
      __m512 fdot = _mm512_cvtepi32_ps(raw);
      const __m512 norm = _mm512_loadu_ps(norms + chunk * 16);

      // neg_dot = -fdot * norm[i] * norm_query
      __m512 neg_dot = _mm512_mul_ps(fdot, norm);
      neg_dot = _mm512_mul_ps(neg_dot, nsf);
      neg_dot = _mm512_sub_ps(_mm512_setzero_ps(), neg_dot);

      __m512 dist;
      if constexpr (Metric) {
        const __m512 sqn = _mm512_loadu_ps(squared_norms + chunk * 16);
        const __m512 sqn_q = _mm512_set1_ps(unquantized_squared_norm);
        // L2: sqn[i] + 2*neg_dot + sqn_q
        dist = _mm512_add_ps(sqn, _mm512_add_ps(
            _mm512_add_ps(neg_dot, neg_dot), sqn_q));
      } else {
        dist = neg_dot;
      }

      // Update running min.
      __m512& min_ref = (chunk == 0) ? min0 :
                        (chunk == 1) ? min1 :
                        (chunk == 2) ? min2 : min3;
      min_ref = _mm512_min_ps(min_ref, dist);
    }
  }

  // ---- Batch scoring: contiguous raw-pointer interface ----
  // strip_data + norms + squared_norms are in strip layout.
  // N = number of points (strip-padded to multiple of 64).
  // N_real = actual number of points.
  void distances_contiguous(
      const uint8_t* strip_data,
      const float* norms,
      const float* squared_norms,
      size_t /* stride_unused */,
      size_t N_real,
      float* out) const {
    if (N_real == 0) return;

    const size_t n_strips = (N_real + 63) / 64;
    const size_t strip_stride = num_bytes_per_datapoint * 64;

    alignas(64) int32_t dots[64];

    for (size_t s = 0; s < n_strips; ++s) {
      const uint8_t* sp = strip_data + s * strip_stride;
      const float* ns = norms + s * 64;
      const float* sq = squared_norms + s * 64;
      const size_t base = s * 64;
      const size_t count = std::min<size_t>(64, N_real - base);

      scan_strip_64_dots(sp, dots);

      for (size_t i = 0; i < count; ++i) {
        float neg_dot = -static_cast<float>(dots[i]) *
                        ns[i] * norm_scaling_factor;
        if constexpr (Metric) {
          out[base + i] = sq[i] + 2.0f * neg_dot + unquantized_squared_norm;
        } else {
          out[base + i] = neg_dot;
        }
      }
    }
  }

  // ---- Chamfer: one query vector vs entire cloud, return min ----
  float chamfer_min_strip(
      const uint8_t* strip_data,
      const float* norms,
      const float* squared_norms,
      size_t strip_stride,
      size_t n_strips) const {
    __m512 min0 = _mm512_set1_ps(std::numeric_limits<float>::max());
    __m512 min1 = min0, min2 = min0, min3 = min0;

    for (size_t s = 0; s < n_strips; ++s) {
      scan_strip_64_min(
          strip_data + s * strip_stride,
          norms + s * 64,
          squared_norms + s * 64,
          min0, min1, min2, min3);
    }

    // Reduce: min across all 64 lanes.
    __m512 m01 = _mm512_min_ps(min0, min1);
    __m512 m23 = _mm512_min_ps(min2, min3);
    __m512 m = _mm512_min_ps(m01, m23);
    return _mm512_reduce_min_ps(m);
  }

  // Tag for SFINAE detection in wrapper.
  static constexpr bool has_batch_distances = true;
#endif  // __AVX512F__

#if !defined(__AVX512F__) && defined(__AVX2__)
  // AVX2 path also supports batch distances.
  static constexpr bool has_batch_distances = true;
#endif  // !__AVX512F__ && __AVX2__
};

#if defined(__AVX512F__) || defined(__AVX2__)
// =========================================================================
// VNNI GEMM Chamfer: vpdpbusd-based micro-kernel for maximum throughput
// =========================================================================
//
// Adapts the many-to-many TurboQuant GEMM pattern for one-to-many Chamfer:
// - Queries decoded to row-major int8 (signed centroids)
// - DB decoded to block-transposed uint8 panels (unsigned centroids = signed+128)
// - Micro-kernel: vpdpbusd (uint8 × int8, 4 MADs/lane/cycle)
// - Epilogue: bias correction (-128 * byte_sum), float post-transform, running min

// Centroids shifted by +128 for native vpdpbusd (unsigned × signed).
static constexpr std::array<uint8_t, 16> kCentroidsUint8 = {
    134, 146, 159, 172, 186, 203, 224, 255,
    122, 110,  97,  84,  70,  53,  32,   1};

static constexpr size_t kVnniPoints = 16;  // int32 lanes in __m512i
static constexpr size_t kVnniMq = 8;       // queries per batch (2-panel path)
static constexpr size_t kVnniMq4 = 4;      // queries per batch (4-panel path)

// Decode one query from nibble-packed to row-major int8.
// Returns byte_sum (sum of all decoded int8 values) for bias correction.
inline int32_t decode_query_vnni(
    const int8_t* query_data,  // int8 per dim (already decoded by quantize_query)
    size_t decoded_dim,
    int8_t* out) {
  // query_data is already in row-major int8 format from quantize_query.
  // Just copy and compute byte sum.
  int32_t byte_sum = 0;
  for (size_t d = 0; d < decoded_dim; ++d) {
    out[d] = query_data[d];
    byte_sum += static_cast<int32_t>(query_data[d]);
  }
  // Pad to multiple of 4 for vpbroadcastd.
  const size_t padded = (decoded_dim + 3) & ~3;
  for (size_t d = decoded_dim; d < padded; ++d)
    out[d] = 0;
  return byte_sum;
}

// SIMD bulk decode: 16 points from a strip into one block-transposed panel.
// Processes all byte-positions using SIMD instead of per-point scalar decode.
// For byte-positions j and j+1, loads 16 packed bytes each, decodes nibbles,
// and interleaves to form 4-byte tiles for vpdpbusd.
inline void decode_strip_to_panel_simd(
    const uint8_t* strip_ptr,   // pointer to strip start
    size_t base_lane,           // starting lane within strip (must be < 64)
    size_t num_bytes,           // bytes per datapoint
    size_t total_tiles,         // padded_decoded_dim / 4
    uint8_t* panel) {           // output panel, aligned 64
  const __m128i codebook_u8 = _mm_loadu_si128(
      reinterpret_cast<const __m128i*>(kCentroidsUint8.data()));
  const __m128i mask_0f = _mm_set1_epi8(0x0F);

  constexpr size_t N = kVnniPoints * 4;  // 64 bytes per tile
  const size_t decoded_dim = 2 * num_bytes;

  // Process pairs of byte-positions: (j, j+1) → 4 decoded dims = 1 tile group.
  // Each pair produces two tiles (tile_2g for dims from j, tile_2g+1 for j+1).
  // Wait — actually the vpdpbusd layout from the reference is:
  // 4 consecutive decoded dims per tile slot per point.
  // Decoded dims 2j are even, 2j+1 are odd.
  // From byte j: even{d=2j} and odd{d=2j+1}.
  // From byte j+1: even{d=2j+2} and odd{d=2j+3}.
  // So dims {2j, 2j+1, 2j+2, 2j+3} = 4 consecutive dims.
  // This maps to loading byte j and j+1, decoding, and interleaving into
  // [even_j, odd_j, even_j+1, odd_j+1] per point.
  //
  // For 16 points, each tile = 64 bytes = 16 × 4 bytes.

  size_t tile = 0;
  size_t j = 0;
  for (; j + 1 < num_bytes; j += 2) {
    // Load 16 packed bytes for byte-positions j and j+1.
    const __m128i packed_j = _mm_loadu_si128(
        reinterpret_cast<const __m128i*>(strip_ptr + j * 64 + base_lane));
    const __m128i packed_j1 = _mm_loadu_si128(
        reinterpret_cast<const __m128i*>(strip_ptr + (j + 1) * 64 + base_lane));

    // Decode nibbles for byte j: 16 even centroids + 16 odd centroids.
    const __m128i even_idx_j = _mm_and_si128(packed_j, mask_0f);
    const __m128i odd_idx_j = _mm_and_si128(_mm_srli_epi16(packed_j, 4), mask_0f);
    const __m128i even_j = _mm_shuffle_epi8(codebook_u8, even_idx_j);
    const __m128i odd_j = _mm_shuffle_epi8(codebook_u8, odd_idx_j);

    // Decode nibbles for byte j+1.
    const __m128i even_idx_j1 = _mm_and_si128(packed_j1, mask_0f);
    const __m128i odd_idx_j1 = _mm_and_si128(_mm_srli_epi16(packed_j1, 4), mask_0f);
    const __m128i even_j1 = _mm_shuffle_epi8(codebook_u8, even_idx_j1);
    const __m128i odd_j1 = _mm_shuffle_epi8(codebook_u8, odd_idx_j1);

    // Interleave to form 4-byte groups per point:
    // [even_j[p], odd_j[p], even_j1[p], odd_j1[p]] = 4 decoded dims per point.
    // Use two interleave steps:
    //   step 1: pair(even_j, odd_j) → [ej0,oj0, ej1,oj1, ...] (2 bytes per point)
    //   step 2: pair(even_j1, odd_j1) → [ej10,oj10, ej11,oj11, ...]
    //   step 3: interleave the int16 pairs → 4 bytes per point.
    const __m128i pair_j = _mm_unpacklo_epi8(even_j, odd_j);    // low 8 points: 16 bytes
    const __m128i pair_j_hi = _mm_unpackhi_epi8(even_j, odd_j); // high 8 points
    const __m128i pair_j1 = _mm_unpacklo_epi8(even_j1, odd_j1);
    const __m128i pair_j1_hi = _mm_unpackhi_epi8(even_j1, odd_j1);

    // Now interleave int16 pairs: [ej0,oj0,ej10,oj10, ej1,oj1,ej11,oj11, ...]
    const __m128i tile_lo = _mm_unpacklo_epi16(pair_j, pair_j1);   // points 0-3: 16B
    const __m128i tile_mid_lo = _mm_unpackhi_epi16(pair_j, pair_j1); // points 4-7
    const __m128i tile_mid_hi = _mm_unpacklo_epi16(pair_j_hi, pair_j1_hi); // points 8-11
    const __m128i tile_hi = _mm_unpackhi_epi16(pair_j_hi, pair_j1_hi); // points 12-15

    // Store as one tile: 16 points × 4 bytes = 64 bytes.
    _mm_storeu_si128(reinterpret_cast<__m128i*>(panel + tile * N + 0), tile_lo);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(panel + tile * N + 16), tile_mid_lo);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(panel + tile * N + 32), tile_mid_hi);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(panel + tile * N + 48), tile_hi);
    ++tile;
  }

  // Handle odd last byte-position (if num_bytes is odd).
  if (j < num_bytes) {
    const __m128i packed_j = _mm_loadu_si128(
        reinterpret_cast<const __m128i*>(strip_ptr + j * 64 + base_lane));
    const __m128i even_idx_j = _mm_and_si128(packed_j, mask_0f);
    const __m128i odd_idx_j = _mm_and_si128(_mm_srli_epi16(packed_j, 4), mask_0f);
    const __m128i even_j = _mm_shuffle_epi8(codebook_u8, even_idx_j);
    const __m128i odd_j = _mm_shuffle_epi8(codebook_u8, odd_idx_j);
    const __m128i zeros = _mm_setzero_si128();

    const __m128i pair_j = _mm_unpacklo_epi8(even_j, odd_j);
    const __m128i pair_j_hi = _mm_unpackhi_epi8(even_j, odd_j);
    const __m128i tile_lo = _mm_unpacklo_epi16(pair_j, zeros);
    const __m128i tile_mid_lo = _mm_unpackhi_epi16(pair_j, zeros);
    const __m128i tile_mid_hi = _mm_unpacklo_epi16(pair_j_hi, zeros);
    const __m128i tile_hi = _mm_unpackhi_epi16(pair_j_hi, zeros);

    _mm_storeu_si128(reinterpret_cast<__m128i*>(panel + tile * N + 0), tile_lo);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(panel + tile * N + 16), tile_mid_lo);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(panel + tile * N + 32), tile_mid_hi);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(panel + tile * N + 48), tile_hi);
    ++tile;
  }

  // Zero-pad remaining tiles.
  const __m128i neutral = _mm_set1_epi8(static_cast<char>(0x80));
  for (; tile < total_tiles; ++tile) {
    _mm_storeu_si128(reinterpret_cast<__m128i*>(panel + tile * N + 0), neutral);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(panel + tile * N + 16), neutral);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(panel + tile * N + 32), neutral);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(panel + tile * N + 48), neutral);
  }
}


#ifdef __AVX512F__
// Portable unsigned×signed int8 dot product accumulate.
// When VNNI is available, uses the native vpdpbusd instruction (1 cycle).
// Otherwise, emulates with pmaddubsw + pmaddwd + paddd (3 instructions).
inline __m512i tq_dpbusd(__m512i acc, __m512i a_unsigned, __m512i b_signed) {
#ifdef __AVX512VNNI__
  return _mm512_dpbusd_epi32(acc, a_unsigned, b_signed);
#else
  // pmaddubsw: unsigned×signed byte pairs → int16 (adjacent pairs summed)
  const __m512i prod16 = _mm512_maddubs_epi16(a_unsigned, b_signed);
  // pmaddwd: adjacent int16 pairs → int32
  const __m512i prod32 = _mm512_madd_epi16(prod16, _mm512_set1_epi16(1));
  // Accumulate into int32
  return _mm512_add_epi32(acc, prod32);
#endif
}

// VNNI micro-kernel: accumulate kVnniMq queries × 1 panel.
// acc[q] accumulates kVnniPoints int32 dot products (one per DB point in panel).
template<size_t Mq>
inline void vnni_micro_kernel_1panel(
    const int8_t* const* query_ptrs,  // Mq decoded query pointers
    const uint8_t* panel,
    size_t total_tiles,
    __m512i* acc) {  // Mq accumulators
  constexpr size_t N = kVnniPoints * 4;  // 64 bytes per tile

  for (size_t q = 0; q < Mq; ++q)
    acc[q] = _mm512_setzero_si512();

  for (size_t t = 0; t < total_tiles; ++t) {
    // Load panel tile: 16 int32 slots = 64 bytes (kVnniPoints points × 4 bytes).
    const __m512i b = _mm512_load_si512(
        reinterpret_cast<const __m512i*>(panel + t * N));

    for (size_t q = 0; q < Mq; ++q) {
      // Broadcast 4 query bytes as int32.
      const __m512i qv = _mm512_set1_epi32(
          reinterpret_cast<const int32_t*>(query_ptrs[q])[t]);
      acc[q] = tq_dpbusd(acc[q], b, qv);
    }
  }
}

// VNNI micro-kernel: 2-panel variant for better ILP.
template<size_t Mq>
inline void vnni_micro_kernel_2panel(
    const int8_t* const* query_ptrs,
    const uint8_t* panel0,
    const uint8_t* panel1,
    size_t total_tiles,
    __m512i* acc0,
    __m512i* acc1) {
  constexpr size_t N = kVnniPoints * 4;

  for (size_t q = 0; q < Mq; ++q) {
    acc0[q] = _mm512_setzero_si512();
    acc1[q] = _mm512_setzero_si512();
  }

  for (size_t t = 0; t < total_tiles; ++t) {
    const __m512i b0 = _mm512_load_si512(
        reinterpret_cast<const __m512i*>(panel0 + t * N));
    const __m512i b1 = _mm512_load_si512(
        reinterpret_cast<const __m512i*>(panel1 + t * N));
    for (size_t q = 0; q < Mq; ++q) {
      const __m512i qv = _mm512_set1_epi32(
          reinterpret_cast<const int32_t*>(query_ptrs[q])[t]);
      acc0[q] = tq_dpbusd(acc0[q], b0, qv);
      acc1[q] = tq_dpbusd(acc1[q], b1, qv);
    }
  }
}

// VNNI micro-kernel: 4-panel variant for maximum ILP.
// With 4 independent accumulator chains per query, the 5-cycle dpbusd latency
// is fully hidden (4 dpbusd per query per tile iteration = 4 cycles of work).
template<size_t Mq>
inline void vnni_micro_kernel_4panel(
    const int8_t* const* query_ptrs,
    const uint8_t* panel0,
    const uint8_t* panel1,
    const uint8_t* panel2,
    const uint8_t* panel3,
    size_t total_tiles,
    __m512i* acc0,
    __m512i* acc1,
    __m512i* acc2,
    __m512i* acc3) {
  constexpr size_t N = kVnniPoints * 4;

  for (size_t q = 0; q < Mq; ++q) {
    acc0[q] = _mm512_setzero_si512();
    acc1[q] = _mm512_setzero_si512();
    acc2[q] = _mm512_setzero_si512();
    acc3[q] = _mm512_setzero_si512();
  }

  for (size_t t = 0; t < total_tiles; ++t) {
    const __m512i b0 = _mm512_load_si512(
        reinterpret_cast<const __m512i*>(panel0 + t * N));
    const __m512i b1 = _mm512_load_si512(
        reinterpret_cast<const __m512i*>(panel1 + t * N));
    const __m512i b2 = _mm512_load_si512(
        reinterpret_cast<const __m512i*>(panel2 + t * N));
    const __m512i b3 = _mm512_load_si512(
        reinterpret_cast<const __m512i*>(panel3 + t * N));
    for (size_t q = 0; q < Mq; ++q) {
      const __m512i qv = _mm512_set1_epi32(
          reinterpret_cast<const int32_t*>(query_ptrs[q])[t]);
      acc0[q] = tq_dpbusd(acc0[q], b0, qv);
      acc1[q] = tq_dpbusd(acc1[q], b1, qv);
      acc2[q] = tq_dpbusd(acc2[q], b2, qv);
      acc3[q] = tq_dpbusd(acc3[q], b3, qv);
    }
  }
}

// VNNI Chamfer epilogue: bias correct, float post-transform, update min.
template<bool Metric>
inline void vnni_chamfer_epilogue(
    __m512i acc,
    int32_t byte_sum,
    float q_nsf,
    float q_sqn,
    const float* db_norms,      // kVnniPoints floats
    const float* db_sqn,        // kVnniPoints floats (L2 only)
    __m512& running_min) {
  // Bias correction: acc -= 128 * byte_sum.
  const __m512i correction = _mm512_set1_epi32(128 * byte_sum);
  acc = _mm512_sub_epi32(acc, correction);

  // Convert to float and apply norms.
  const __m512 fdot = _mm512_cvtepi32_ps(acc);
  const __m512 db_norm_v = _mm512_loadu_ps(db_norms);
  const __m512 q_nsf_v = _mm512_set1_ps(q_nsf);

  // neg_dot = -(fdot * db_norm * q_norm)
  __m512 neg_dot = _mm512_mul_ps(fdot, db_norm_v);
  neg_dot = _mm512_mul_ps(neg_dot, q_nsf_v);
  neg_dot = _mm512_sub_ps(_mm512_setzero_ps(), neg_dot);

  __m512 dist;
  if constexpr (Metric) {
    const __m512 sqn_v = _mm512_loadu_ps(db_sqn);
    const __m512 sqn_qv = _mm512_set1_ps(q_sqn);
    dist = _mm512_add_ps(sqn_v, _mm512_add_ps(
        _mm512_add_ps(neg_dot, neg_dot), sqn_qv));
  } else {
    dist = neg_dot;
  }

  running_min = _mm512_min_ps(running_min, dist);
}

// Full VNNI GEMM Chamfer distance.
template<bool Metric>
inline float chamfer_vnni_gemm(
    const Quantized_Query<Metric>* const* query_ptrs,
    size_t num_queries,
    const uint8_t* strip_data,
    const float* norms,
    const float* squared_norms,
    size_t strip_stride,
    size_t n_strips,
    size_t num_bytes_per_point,
    size_t cloud_size) {

  const size_t decoded_dim = 2 * num_bytes_per_point;
  const size_t padded_dim = (decoded_dim + 3) & ~3;
  const size_t total_tiles = padded_dim / 4;
  constexpr size_t N = kVnniPoints * 4;  // 64 bytes per panel tile
  const size_t panel_bytes = total_tiles * N;
  const size_t n_panels = (cloud_size + kVnniPoints - 1) / kVnniPoints;

  // Step 1: Decode all queries to row-major int8.  (thread_local to avoid
  // repeated heap allocation on every call.)
  const size_t q_stride = padded_dim;
  thread_local std::vector<int8_t> all_q_decoded;
  thread_local std::vector<int32_t> all_q_byte_sums;
  all_q_decoded.resize(num_queries * q_stride);
  std::memset(all_q_decoded.data(), 0, num_queries * q_stride);
  all_q_byte_sums.resize(num_queries);

  for (size_t qi = 0; qi < num_queries; ++qi) {
    all_q_byte_sums[qi] = decode_query_vnni(
        query_ptrs[qi]->query_data.data(),
        decoded_dim,
        all_q_decoded.data() + qi * q_stride);
  }

  // Step 2: Decode DB cloud to block-transposed panels (all at once).
  thread_local std::vector<uint8_t> panels;
  panels.resize(n_panels * panel_bytes + 64);
  std::memset(panels.data(), 0x80, panels.size());
  uint8_t* panels_aligned = reinterpret_cast<uint8_t*>(
      (reinterpret_cast<uintptr_t>(panels.data()) + 63) & ~63);

  for (size_t p = 0; p < n_panels; ++p) {
    const size_t point_start = p * kVnniPoints;
    const size_t strip = point_start / 64;
    const size_t base_lane = point_start % 64;
    decode_strip_to_panel_simd(
        strip_data + strip * strip_stride,
        base_lane,
        num_bytes_per_point,
        total_tiles,
        panels_aligned + p * panel_bytes);
  }

  // Step 3: Pad norm arrays (thread_local, dynamic — no 1024 limit).
  const size_t padded_pts = n_panels * kVnniPoints;
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

  // Step 4: Score all queries against all panels.
  float total_chamfer = 0.0f;
  size_t qi = 0;

  // Primary path: 4-panel ILP with kVnniMq4 queries per batch.
  for (; qi + kVnniMq4 <= num_queries; qi += kVnniMq4) {
    const int8_t* q_batch[kVnniMq4];
    float q_norms_arr[kVnniMq4];
    float q_sqn_arr[kVnniMq4];
    int32_t q_bsums[kVnniMq4];

    for (size_t q = 0; q < kVnniMq4; ++q) {
      q_batch[q] = all_q_decoded.data() + (qi + q) * q_stride;
      q_norms_arr[q] = query_ptrs[qi + q]->norm_scaling_factor;
      q_sqn_arr[q] = query_ptrs[qi + q]->unquantized_squared_norm;
      q_bsums[q] = all_q_byte_sums[qi + q];
    }

    __m512 mins[kVnniMq4];
    for (size_t q = 0; q < kVnniMq4; ++q)
      mins[q] = _mm512_set1_ps(std::numeric_limits<float>::max());

    // 4-panel scoring loop.
    size_t p = 0;
    for (; p + 4 <= n_panels; p += 4) {
      __m512i a0[kVnniMq4], a1[kVnniMq4], a2[kVnniMq4], a3[kVnniMq4];
      vnni_micro_kernel_4panel<kVnniMq4>(
          q_batch,
          panels_aligned + p * panel_bytes,
          panels_aligned + (p + 1) * panel_bytes,
          panels_aligned + (p + 2) * panel_bytes,
          panels_aligned + (p + 3) * panel_bytes,
          total_tiles, a0, a1, a2, a3);

      for (size_t q = 0; q < kVnniMq4; ++q) {
        vnni_chamfer_epilogue<Metric>(
            a0[q], q_bsums[q], q_norms_arr[q], q_sqn_arr[q],
            padded_norms.data() + p * kVnniPoints,
            padded_sqn.data() + p * kVnniPoints,
            mins[q]);
        vnni_chamfer_epilogue<Metric>(
            a1[q], q_bsums[q], q_norms_arr[q], q_sqn_arr[q],
            padded_norms.data() + (p + 1) * kVnniPoints,
            padded_sqn.data() + (p + 1) * kVnniPoints,
            mins[q]);
        vnni_chamfer_epilogue<Metric>(
            a2[q], q_bsums[q], q_norms_arr[q], q_sqn_arr[q],
            padded_norms.data() + (p + 2) * kVnniPoints,
            padded_sqn.data() + (p + 2) * kVnniPoints,
            mins[q]);
        vnni_chamfer_epilogue<Metric>(
            a3[q], q_bsums[q], q_norms_arr[q], q_sqn_arr[q],
            padded_norms.data() + (p + 3) * kVnniPoints,
            padded_sqn.data() + (p + 3) * kVnniPoints,
            mins[q]);
      }
    }
    // Remainder: 2-panel.
    for (; p + 2 <= n_panels; p += 2) {
      __m512i ac0[kVnniMq4], ac1[kVnniMq4];
      vnni_micro_kernel_2panel<kVnniMq4>(
          q_batch,
          panels_aligned + p * panel_bytes,
          panels_aligned + (p + 1) * panel_bytes,
          total_tiles, ac0, ac1);

      for (size_t q = 0; q < kVnniMq4; ++q) {
        vnni_chamfer_epilogue<Metric>(
            ac0[q], q_bsums[q], q_norms_arr[q], q_sqn_arr[q],
            padded_norms.data() + p * kVnniPoints,
            padded_sqn.data() + p * kVnniPoints,
            mins[q]);
        vnni_chamfer_epilogue<Metric>(
            ac1[q], q_bsums[q], q_norms_arr[q], q_sqn_arr[q],
            padded_norms.data() + (p + 1) * kVnniPoints,
            padded_sqn.data() + (p + 1) * kVnniPoints,
            mins[q]);
      }
    }
    // Remainder: 1-panel.
    for (; p < n_panels; ++p) {
      __m512i acc[kVnniMq4];
      vnni_micro_kernel_1panel<kVnniMq4>(
          q_batch,
          panels_aligned + p * panel_bytes,
          total_tiles, acc);

      for (size_t q = 0; q < kVnniMq4; ++q) {
        vnni_chamfer_epilogue<Metric>(
            acc[q], q_bsums[q], q_norms_arr[q], q_sqn_arr[q],
            padded_norms.data() + p * kVnniPoints,
            padded_sqn.data() + p * kVnniPoints,
            mins[q]);
      }
    }

    for (size_t q = 0; q < kVnniMq4; ++q)
      total_chamfer += _mm512_reduce_min_ps(mins[q]);
  }

  // Tail: remaining queries.
  for (; qi < num_queries; ++qi) {
    const int8_t* qp = all_q_decoded.data() + qi * q_stride;
    const float q_nsf = query_ptrs[qi]->norm_scaling_factor;
    const float q_sqn_val = query_ptrs[qi]->unquantized_squared_norm;
    const int32_t bsum = all_q_byte_sums[qi];

    __m512 running_min = _mm512_set1_ps(std::numeric_limits<float>::max());

    for (size_t p = 0; p < n_panels; ++p) {
      __m512i acc;
      vnni_micro_kernel_1panel<1>(
          &qp,
          panels_aligned + p * panel_bytes,
          total_tiles, &acc);
      vnni_chamfer_epilogue<Metric>(
          acc, bsum, q_nsf, q_sqn_val,
          padded_norms.data() + p * kVnniPoints,
          padded_sqn.data() + p * kVnniPoints,
          running_min);
    }

    total_chamfer += _mm512_reduce_min_ps(running_min);
  }

  return total_chamfer;
}
#endif  // __AVX512F__

// =========================================================================
// AVX2 GEMM Chamfer: maddubs-based kernel for non-AVX512 machines
// =========================================================================
// Reuses the SSE-based decode (decode_strip_to_panel_simd) and 16-point panels.
// Each 64-byte tile is processed as two __m256i halves (low/high 8 points).
// Uses _mm256_maddubs_epi16 + _mm256_madd_epi16 for unsigned×signed dot product.
#if !defined(__AVX512F__) && defined(__AVX2__)

static constexpr size_t kAvx2Points = 8;  // int32 lanes in __m256i
static constexpr size_t kAvx2Mq4 = 4;    // queries per batch (4-panel ILP)

// AVX2 unsigned×signed int8 dot product accumulate.
// Equivalent to dpbusd: for each 4-byte group, multiply unsigned×signed
// byte pairs and accumulate the 4 products into one int32.
inline __m256i avx2_dpbusd(__m256i acc, __m256i a_unsigned, __m256i b_signed) {
  const __m256i prod16 = _mm256_maddubs_epi16(a_unsigned, b_signed);
  const __m256i prod32 = _mm256_madd_epi16(prod16, _mm256_set1_epi16(1));
  return _mm256_add_epi32(acc, prod32);
}

// AVX2 micro-kernel: 1-panel, processes low and high 8-point halves.
template<size_t Mq>
inline void avx2_micro_kernel_1panel(
    const int8_t* const* query_ptrs,
    const uint8_t* panel,  // 16-point panel (64 bytes per tile)
    size_t total_tiles,
    __m256i* acc_lo,       // Mq accumulators for low 8 points
    __m256i* acc_hi) {     // Mq accumulators for high 8 points
  constexpr size_t N = kVnniPoints * 4;  // 64 bytes per tile

  for (size_t q = 0; q < Mq; ++q) {
    acc_lo[q] = _mm256_setzero_si256();
    acc_hi[q] = _mm256_setzero_si256();
  }

  for (size_t t = 0; t < total_tiles; ++t) {
    const __m256i b_lo = _mm256_load_si256(
        reinterpret_cast<const __m256i*>(panel + t * N));
    const __m256i b_hi = _mm256_load_si256(
        reinterpret_cast<const __m256i*>(panel + t * N + 32));

    for (size_t q = 0; q < Mq; ++q) {
      const __m256i qv = _mm256_set1_epi32(
          reinterpret_cast<const int32_t*>(query_ptrs[q])[t]);
      acc_lo[q] = avx2_dpbusd(acc_lo[q], b_lo, qv);
      acc_hi[q] = avx2_dpbusd(acc_hi[q], b_hi, qv);
    }
  }
}

// AVX2 micro-kernel: 4-panel ILP.
template<size_t Mq>
inline void avx2_micro_kernel_4panel(
    const int8_t* const* query_ptrs,
    const uint8_t* panel0, const uint8_t* panel1,
    const uint8_t* panel2, const uint8_t* panel3,
    size_t total_tiles,
    __m256i* a0_lo, __m256i* a0_hi,
    __m256i* a1_lo, __m256i* a1_hi,
    __m256i* a2_lo, __m256i* a2_hi,
    __m256i* a3_lo, __m256i* a3_hi) {
  constexpr size_t N = kVnniPoints * 4;

  for (size_t q = 0; q < Mq; ++q) {
    a0_lo[q] = a0_hi[q] = _mm256_setzero_si256();
    a1_lo[q] = a1_hi[q] = _mm256_setzero_si256();
    a2_lo[q] = a2_hi[q] = _mm256_setzero_si256();
    a3_lo[q] = a3_hi[q] = _mm256_setzero_si256();
  }

  for (size_t t = 0; t < total_tiles; ++t) {
    const __m256i b0_lo = _mm256_load_si256(reinterpret_cast<const __m256i*>(panel0 + t * N));
    const __m256i b0_hi = _mm256_load_si256(reinterpret_cast<const __m256i*>(panel0 + t * N + 32));
    const __m256i b1_lo = _mm256_load_si256(reinterpret_cast<const __m256i*>(panel1 + t * N));
    const __m256i b1_hi = _mm256_load_si256(reinterpret_cast<const __m256i*>(panel1 + t * N + 32));
    const __m256i b2_lo = _mm256_load_si256(reinterpret_cast<const __m256i*>(panel2 + t * N));
    const __m256i b2_hi = _mm256_load_si256(reinterpret_cast<const __m256i*>(panel2 + t * N + 32));
    const __m256i b3_lo = _mm256_load_si256(reinterpret_cast<const __m256i*>(panel3 + t * N));
    const __m256i b3_hi = _mm256_load_si256(reinterpret_cast<const __m256i*>(panel3 + t * N + 32));

    for (size_t q = 0; q < Mq; ++q) {
      const __m256i qv = _mm256_set1_epi32(
          reinterpret_cast<const int32_t*>(query_ptrs[q])[t]);
      a0_lo[q] = avx2_dpbusd(a0_lo[q], b0_lo, qv);
      a0_hi[q] = avx2_dpbusd(a0_hi[q], b0_hi, qv);
      a1_lo[q] = avx2_dpbusd(a1_lo[q], b1_lo, qv);
      a1_hi[q] = avx2_dpbusd(a1_hi[q], b1_hi, qv);
      a2_lo[q] = avx2_dpbusd(a2_lo[q], b2_lo, qv);
      a2_hi[q] = avx2_dpbusd(a2_hi[q], b2_hi, qv);
      a3_lo[q] = avx2_dpbusd(a3_lo[q], b3_lo, qv);
      a3_hi[q] = avx2_dpbusd(a3_hi[q], b3_hi, qv);
    }
  }
}

// AVX2 Chamfer epilogue: bias correct, float post-transform, update min.
// Processes 8 points (one __m256i half of a 16-point panel).
template<bool Metric>
inline void avx2_chamfer_epilogue(
    __m256i acc, int32_t byte_sum, float q_nsf, float q_sqn,
    const float* norms8, const float* sqn8,
    __m256& running_min) {
  // Bias correction: subtract 128 * byte_sum from each dot product.
  const __m256i bias = _mm256_set1_epi32(128 * byte_sum);
  const __m256i corrected = _mm256_sub_epi32(acc, bias);

  // Convert to float and apply norm scaling.
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

// AVX2 reduce __m256 to scalar min.
inline float avx2_reduce_min_ps(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v);
  __m128 hi = _mm256_extractf128_ps(v, 1);
  __m128 m = _mm_min_ps(lo, hi);      // 4 floats
  __m128 m2 = _mm_shuffle_ps(m, m, _MM_SHUFFLE(1,0,3,2));
  m = _mm_min_ps(m, m2);              // 2 floats
  __m128 m3 = _mm_shuffle_ps(m, m, _MM_SHUFFLE(0,1,0,1));
  m = _mm_min_ps(m, m3);              // 1 float
  return _mm_cvtss_f32(m);
}

// Full AVX2 GEMM Chamfer distance (reuses decode_strip_to_panel_simd).
template<bool Metric>
inline float chamfer_avx2_gemm(
    const Quantized_Query<Metric>* const* query_ptrs,
    size_t num_queries,
    const uint8_t* strip_data,
    const float* norms,
    const float* squared_norms,
    size_t strip_stride,
    size_t n_strips,
    size_t num_bytes_per_point,
    size_t cloud_size) {

  const size_t decoded_dim = 2 * num_bytes_per_point;
  const size_t padded_dim = (decoded_dim + 3) & ~3;
  const size_t total_tiles = padded_dim / 4;
  constexpr size_t N = kVnniPoints * 4;  // 64 bytes per tile (16-point panels)
  const size_t panel_bytes = total_tiles * N;
  const size_t n_panels = (cloud_size + kVnniPoints - 1) / kVnniPoints;

  // Step 1: Decode all queries to row-major int8.
  const size_t q_stride = padded_dim;
  thread_local std::vector<int8_t> all_q_decoded;
  thread_local std::vector<int32_t> all_q_byte_sums;
  all_q_decoded.resize(num_queries * q_stride);
  std::memset(all_q_decoded.data(), 0, num_queries * q_stride);
  all_q_byte_sums.resize(num_queries);

  for (size_t qi = 0; qi < num_queries; ++qi) {
    all_q_byte_sums[qi] = decode_query_vnni(
        query_ptrs[qi]->query_data.data(),
        decoded_dim,
        all_q_decoded.data() + qi * q_stride);
  }

  // Step 2: Decode DB to 16-point panels (reuses SSE decode function).
  thread_local std::vector<uint8_t> panels;
  panels.resize(n_panels * panel_bytes + 64);
  std::memset(panels.data(), 0x80, panels.size());
  uint8_t* panels_aligned = reinterpret_cast<uint8_t*>(
      (reinterpret_cast<uintptr_t>(panels.data()) + 63) & ~63);

  for (size_t p = 0; p < n_panels; ++p) {
    const size_t point_start = p * kVnniPoints;
    const size_t strip = point_start / 64;
    const size_t base_lane = point_start % 64;
    decode_strip_to_panel_simd(
        strip_data + strip * strip_stride,
        base_lane, num_bytes_per_point, total_tiles,
        panels_aligned + p * panel_bytes);
  }

  // Step 3: Pad norm arrays.
  const size_t padded_pts = n_panels * kVnniPoints;
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

  // Step 4: Score using AVX2 micro-kernels.
  float total_chamfer = 0.0f;
  size_t qi = 0;

  for (; qi + kAvx2Mq4 <= num_queries; qi += kAvx2Mq4) {
    const int8_t* q_batch[kAvx2Mq4];
    float q_norms_arr[kAvx2Mq4];
    float q_sqn_arr[kAvx2Mq4];
    int32_t q_bsums[kAvx2Mq4];

    for (size_t q = 0; q < kAvx2Mq4; ++q) {
      q_batch[q] = all_q_decoded.data() + (qi + q) * q_stride;
      q_norms_arr[q] = query_ptrs[qi + q]->norm_scaling_factor;
      q_sqn_arr[q] = query_ptrs[qi + q]->unquantized_squared_norm;
      q_bsums[q] = all_q_byte_sums[qi + q];
    }

    __m256 mins[kAvx2Mq4];
    for (size_t q = 0; q < kAvx2Mq4; ++q)
      mins[q] = _mm256_set1_ps(std::numeric_limits<float>::max());

    // 4-panel scoring loop.
    size_t p = 0;
    for (; p + 4 <= n_panels; p += 4) {
      __m256i a0_lo[kAvx2Mq4], a0_hi[kAvx2Mq4];
      __m256i a1_lo[kAvx2Mq4], a1_hi[kAvx2Mq4];
      __m256i a2_lo[kAvx2Mq4], a2_hi[kAvx2Mq4];
      __m256i a3_lo[kAvx2Mq4], a3_hi[kAvx2Mq4];
      avx2_micro_kernel_4panel<kAvx2Mq4>(
          q_batch,
          panels_aligned + p * panel_bytes,
          panels_aligned + (p + 1) * panel_bytes,
          panels_aligned + (p + 2) * panel_bytes,
          panels_aligned + (p + 3) * panel_bytes,
          total_tiles,
          a0_lo, a0_hi, a1_lo, a1_hi,
          a2_lo, a2_hi, a3_lo, a3_hi);

      for (size_t q = 0; q < kAvx2Mq4; ++q) {
        // Each panel has 16 points: low 8 + high 8.
        avx2_chamfer_epilogue<Metric>(
            a0_lo[q], q_bsums[q], q_norms_arr[q], q_sqn_arr[q],
            padded_norms.data() + p * kVnniPoints,
            padded_sqn.data() + p * kVnniPoints, mins[q]);
        avx2_chamfer_epilogue<Metric>(
            a0_hi[q], q_bsums[q], q_norms_arr[q], q_sqn_arr[q],
            padded_norms.data() + p * kVnniPoints + kAvx2Points,
            padded_sqn.data() + p * kVnniPoints + kAvx2Points, mins[q]);

        avx2_chamfer_epilogue<Metric>(
            a1_lo[q], q_bsums[q], q_norms_arr[q], q_sqn_arr[q],
            padded_norms.data() + (p+1) * kVnniPoints,
            padded_sqn.data() + (p+1) * kVnniPoints, mins[q]);
        avx2_chamfer_epilogue<Metric>(
            a1_hi[q], q_bsums[q], q_norms_arr[q], q_sqn_arr[q],
            padded_norms.data() + (p+1) * kVnniPoints + kAvx2Points,
            padded_sqn.data() + (p+1) * kVnniPoints + kAvx2Points, mins[q]);

        avx2_chamfer_epilogue<Metric>(
            a2_lo[q], q_bsums[q], q_norms_arr[q], q_sqn_arr[q],
            padded_norms.data() + (p+2) * kVnniPoints,
            padded_sqn.data() + (p+2) * kVnniPoints, mins[q]);
        avx2_chamfer_epilogue<Metric>(
            a2_hi[q], q_bsums[q], q_norms_arr[q], q_sqn_arr[q],
            padded_norms.data() + (p+2) * kVnniPoints + kAvx2Points,
            padded_sqn.data() + (p+2) * kVnniPoints + kAvx2Points, mins[q]);

        avx2_chamfer_epilogue<Metric>(
            a3_lo[q], q_bsums[q], q_norms_arr[q], q_sqn_arr[q],
            padded_norms.data() + (p+3) * kVnniPoints,
            padded_sqn.data() + (p+3) * kVnniPoints, mins[q]);
        avx2_chamfer_epilogue<Metric>(
            a3_hi[q], q_bsums[q], q_norms_arr[q], q_sqn_arr[q],
            padded_norms.data() + (p+3) * kVnniPoints + kAvx2Points,
            padded_sqn.data() + (p+3) * kVnniPoints + kAvx2Points, mins[q]);
      }
    }
    // Remainder: 1-panel.
    for (; p < n_panels; ++p) {
      __m256i acc_lo[kAvx2Mq4], acc_hi[kAvx2Mq4];
      avx2_micro_kernel_1panel<kAvx2Mq4>(
          q_batch, panels_aligned + p * panel_bytes,
          total_tiles, acc_lo, acc_hi);

      for (size_t q = 0; q < kAvx2Mq4; ++q) {
        avx2_chamfer_epilogue<Metric>(
            acc_lo[q], q_bsums[q], q_norms_arr[q], q_sqn_arr[q],
            padded_norms.data() + p * kVnniPoints,
            padded_sqn.data() + p * kVnniPoints, mins[q]);
        avx2_chamfer_epilogue<Metric>(
            acc_hi[q], q_bsums[q], q_norms_arr[q], q_sqn_arr[q],
            padded_norms.data() + p * kVnniPoints + kAvx2Points,
            padded_sqn.data() + p * kVnniPoints + kAvx2Points, mins[q]);
      }
    }

    for (size_t q = 0; q < kAvx2Mq4; ++q)
      total_chamfer += avx2_reduce_min_ps(mins[q]);
  }

  // Tail: remaining queries.
  for (; qi < num_queries; ++qi) {
    const int8_t* qp = all_q_decoded.data() + qi * q_stride;
    const float q_nsf = query_ptrs[qi]->norm_scaling_factor;
    const float q_sqn_val = query_ptrs[qi]->unquantized_squared_norm;
    const int32_t bsum = all_q_byte_sums[qi];

    __m256 running_min = _mm256_set1_ps(std::numeric_limits<float>::max());

    for (size_t p = 0; p < n_panels; ++p) {
      __m256i acc_lo, acc_hi;
      avx2_micro_kernel_1panel<1>(
          &qp, panels_aligned + p * panel_bytes,
          total_tiles, &acc_lo, &acc_hi);
      avx2_chamfer_epilogue<Metric>(
          acc_lo, bsum, q_nsf, q_sqn_val,
          padded_norms.data() + p * kVnniPoints,
          padded_sqn.data() + p * kVnniPoints, running_min);
      avx2_chamfer_epilogue<Metric>(
          acc_hi, bsum, q_nsf, q_sqn_val,
          padded_norms.data() + p * kVnniPoints + kAvx2Points,
          padded_sqn.data() + p * kVnniPoints + kAvx2Points, running_min);
    }

    total_chamfer += avx2_reduce_min_ps(running_min);
  }

  return total_chamfer;
}
#endif  // !__AVX512F__ && __AVX2__

#endif  // defined(__AVX512F__) || defined(__AVX2__)

// ---- Quantized_Point::distance (per-point fallback, uses strip byte order) ----
template<bool Metric>
inline float Quantized_Point<Metric>::distance(
    const Quantized_Query<Metric>& qq) const {
  int32_t dot = 0;
  const size_t nb = num_bytes;
  for (size_t j = 0; j < nb; ++j) {
    const uint8_t byte = code_ptr[j];
    const uint8_t b_even = byte & 0xF;
    const uint8_t b_odd = byte >> 4;
    dot += static_cast<int32_t>(internal::kTurboQuantCentroidsInt8[b_even]) *
           qq.query_data[2 * j];
    dot += static_cast<int32_t>(internal::kTurboQuantCentroidsInt8[b_odd]) *
           qq.query_data[2 * j + 1];
  }

  float neg_dot = -static_cast<float>(dot) * norm_scaling_factor *
                  qq.norm_scaling_factor;
  if constexpr (Metric) {
    return unquantized_squared_norm + 2.0f * neg_dot +
           qq.unquantized_squared_norm;
  } else {
    return neg_dot;
  }
}

// =========================================================================
// Quantized_Point_Range: strip-interleaved encoded base points
// =========================================================================
template<typename PointRange, bool Metric>
class Quantized_Point_Range {
 public:
  static constexpr bool is_fastscan = false;

  size_t n_points_raw = 0;           // total padded points
  size_t n_points_raw_unpadded = 0;  // actual points
  size_t dim = 0;                    // padded dimensionality
  size_t num_bytes_per_datapoint = 0;  // = dim / 2
  size_t stride = 0;                 // strip stride = num_bytes * 64

  // Strip-interleaved codes: for strip s, byte-pos j, point p within strip:
  //   packed_codes[s * stride + j * 64 + p]
  parlay::sequence<uint8_t> packed_codes;
  parlay::sequence<float> norm_scaling_factors;
  parlay::sequence<float> unquantized_squared_norms;
  parlay::sequence<size_t> cloud_vec_offsets;

  Quantized_Point_Range() = default;

  // Per-point access (for fallback path). Gathers bytes from strip layout
  // into a contiguous buffer.
  Quantized_Point<Metric> operator[](size_t i) const {
    const size_t strip = i / 64;
    const size_t lane = i % 64;
    const size_t nb = num_bytes_per_datapoint;

    std::vector<uint8_t> codes(nb);
    for (size_t j = 0; j < nb; ++j) {
      codes[j] = packed_codes[strip * stride + j * 64 + lane];
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
    size_t sz = packed_codes.size();
    out.write(reinterpret_cast<const char*>(&sz), sizeof(sz));
    if (sz) out.write(reinterpret_cast<const char*>(packed_codes.data()), sz);
    size_t nsz = norm_scaling_factors.size();
    out.write(reinterpret_cast<const char*>(&nsz), sizeof(nsz));
    if (nsz) out.write(reinterpret_cast<const char*>(norm_scaling_factors.data()), nsz * sizeof(float));
    size_t ssz = unquantized_squared_norms.size();
    out.write(reinterpret_cast<const char*>(&ssz), sizeof(ssz));
    if (ssz) out.write(reinterpret_cast<const char*>(unquantized_squared_norms.data()), ssz * sizeof(float));
    size_t osz = cloud_vec_offsets.size();
    out.write(reinterpret_cast<const char*>(&osz), sizeof(osz));
    if (osz) out.write(reinterpret_cast<const char*>(cloud_vec_offsets.data()), osz * sizeof(size_t));
  }

  void load(std::istream& in) {
    in.read(reinterpret_cast<char*>(&n_points_raw), sizeof(n_points_raw));
    in.read(reinterpret_cast<char*>(&n_points_raw_unpadded), sizeof(n_points_raw_unpadded));
    in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    in.read(reinterpret_cast<char*>(&num_bytes_per_datapoint), sizeof(num_bytes_per_datapoint));
    in.read(reinterpret_cast<char*>(&stride), sizeof(stride));
    size_t sz = 0;
    in.read(reinterpret_cast<char*>(&sz), sizeof(sz));
    packed_codes.resize(sz);
    if (sz) in.read(reinterpret_cast<char*>(packed_codes.data()), sz);
    size_t nsz = 0;
    in.read(reinterpret_cast<char*>(&nsz), sizeof(nsz));
    norm_scaling_factors.resize(nsz);
    if (nsz) in.read(reinterpret_cast<char*>(norm_scaling_factors.data()), nsz * sizeof(float));
    size_t ssz = 0;
    in.read(reinterpret_cast<char*>(&ssz), sizeof(ssz));
    unquantized_squared_norms.resize(ssz);
    if (ssz) in.read(reinterpret_cast<char*>(unquantized_squared_norms.data()), ssz * sizeof(float));
    size_t osz = 0;
    in.read(reinterpret_cast<char*>(&osz), sizeof(osz));
    cloud_vec_offsets.resize(osz);
    if (osz) in.read(reinterpret_cast<char*>(cloud_vec_offsets.data()), osz * sizeof(size_t));
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
  size_t num_bytes_per_datapoint = 0;
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
    num_bytes_per_datapoint = (padded_dim + 1) / 2;

    signs.resize(padded_dim);
    std::mt19937 gen(seed_);
    std::uniform_int_distribution<> dist(0, 1);
    for (size_t i = 0; i < padded_dim; ++i) {
      signs[i] = (2.0f * dist(gen) - 1.0f);
    }
  }

 private:
  // Encode a single point into per-point contiguous bytes.
  // Returns {squared_norm, norm_scaling_factor}.
  std::pair<float, float> encode_single(
      const float* p, uint8_t* output,
      std::vector<float>& ws) const {
    rotator->rotate(p, ws.data());

    float sqr_norm = 0.0f;
    for (size_t i = 0; i < padded_dim; ++i)
      sqr_norm += ws[i] * ws[i];

    if (sqr_norm == 0.0f) {
      std::memset(output, 0, num_bytes_per_datapoint);
      return {0.0f, 0.0f};
    }

    const float norm = std::sqrt(sqr_norm);
    const float inv_norm = 1.0f / norm;
    for (size_t i = 0; i < padded_dim; ++i)
      ws[i] = ws[i] * signs[i] * inv_norm;

    float q_sqr_norm = 0.0f;
    for (size_t i = 0; i < padded_dim; ++i) {
      uint8_t code = internal::FourBitEncoding(ws[i]);
      q_sqr_norm += internal::kSquaredCentroidsMag[code & 7];
      if (i % 2 == 0)
        output[i / 2] = code;
      else
        output[i / 2] |= (code << 4);
    }

    return {sqr_norm, norm / std::sqrt(q_sqr_norm)};
  }

 public:
  // ---- Encode: strip-interleaved layout ----
  template<typename PointRangeTy>
  Quantized_Point_Range<PointRangeTy, Metric> encode(
      const PointRangeTy& data) const {
    if (!rotator) {
      std::cerr << "one_to_many_turboquant::encode: rotator is null.\n";
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

    enc.packed_codes.resize(n_strips * strip_stride, 0);
    enc.norm_scaling_factors.resize(N_padded, 0.0f);
    enc.unquantized_squared_norms.resize(N_padded, 0.0f);

    struct Workspace {
      std::vector<float> rot;
      std::vector<uint8_t> point_codes;
      void ensure(size_t pdim, size_t nb) {
        if (rot.size() != pdim) rot.resize(pdim);
        if (point_codes.size() != nb) point_codes.resize(nb);
      }
    };

    parlay::parallel_for(0, N, [&](size_t vi) {
      static thread_local Workspace ws;
      ws.ensure(padded_dim, num_bytes_per_datapoint);

      const float* p = reinterpret_cast<const float*>(data.location(vi));
      auto [sqn, nsf] = encode_single(p, ws.point_codes.data(), ws.rot);
      enc.norm_scaling_factors[vi] = nsf;
      enc.unquantized_squared_norms[vi] = sqn;

      // Scatter contiguous bytes into strip layout.
      const size_t strip = vi / 64;
      const size_t lane = vi % 64;
      for (size_t j = 0; j < num_bytes_per_datapoint; ++j) {
        enc.packed_codes[strip * strip_stride + j * 64 + lane] =
            ws.point_codes[j];
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

    // Compute padded offsets (each cloud padded to 64).
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
    enc.packed_codes.resize(n_strips * strip_stride, 0);
    enc.norm_scaling_factors.resize(cur, 0.0f);
    enc.unquantized_squared_norms.resize(cur, 0.0f);

    struct Workspace {
      std::vector<float> rot;
      std::vector<uint8_t> point_codes;
      void ensure(size_t pdim, size_t nb) {
        if (rot.size() != pdim) rot.resize(pdim);
        if (point_codes.size() != nb) point_codes.resize(nb);
      }
    };

    // Encode each cloud.
    for (size_t c = 0; c < n_clouds; ++c) {
      const size_t start_f = static_cast<size_t>(cloud_offsets_float[c]);
      const size_t end_f = static_cast<size_t>(cloud_offsets_float[c + 1]);
      const size_t src_start = start_f / D;
      const size_t n_vecs = (end_f - start_f) / D;
      const size_t dst_start = enc.cloud_vec_offsets[c];

      parlay::parallel_for(0, n_vecs, [&](size_t i) {
        static thread_local Workspace ws;
        ws.ensure(padded_dim, num_bytes_per_datapoint);

        const float* p = reinterpret_cast<const float*>(
            data.location(src_start + i));
        auto [sqn, nsf] = encode_single(p, ws.point_codes.data(), ws.rot);

        const size_t dst_idx = dst_start + i;
        enc.norm_scaling_factors[dst_idx] = nsf;
        enc.unquantized_squared_norms[dst_idx] = sqn;

        const size_t strip = dst_idx / 64;
        const size_t lane = dst_idx % 64;
        for (size_t j = 0; j < num_bytes_per_datapoint; ++j) {
          enc.packed_codes[strip * strip_stride + j * 64 + lane] =
              ws.point_codes[j];
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

    // NON-deinterleaved: query_data[i] = int8 for dimension i.
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

    float max_value = 0.0f;
    for (size_t i = 0; i < padded_dim; ++i)
      max_value = std::max(max_value, std::abs(q_rot[i]));

    const float sf = max_value > 0.0f ? 127.0f / max_value : 0.0f;

    int quant_norm = 0;
    for (size_t i = 0; i < padded_dim; ++i) {
      float scaled = std::clamp(std::round(q_rot[i] * sf), -127.0f, 127.0f);
      int8_t snapped = static_cast<int8_t>(scaled);
      qq.query_data[i] = snapped;
      quant_norm += snapped * snapped;
    }

    qq.norm_scaling_factor = norm / std::sqrt(static_cast<float>(quant_norm));
    qq.unquantized_squared_norm = sqr_norm;
    return qq;
  }

  template<typename PointTy>
  typename std::enable_if<!std::is_pointer<PointTy>::value,
                          Quantized_Query<Metric>>::type
  quantize_query(const PointTy& query) const {
    std::vector<float> tmp(dim);
    for (size_t i = 0; i < dim; ++i) tmp[i] = query[i];
    return quantize_query(tmp.data());
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
    out.write(reinterpret_cast<const char*>(&seed_), sizeof(seed_));
    int rt = static_cast<int>(rotator_type);
    out.write(reinterpret_cast<const char*>(&rt), sizeof(rt));
    if (rotator) rotator->save(out);
    size_t ss = signs.size();
    out.write(reinterpret_cast<const char*>(&ss), sizeof(ss));
    out.write(reinterpret_cast<const char*>(signs.data()), ss * sizeof(float));
  }

  void load(std::ifstream& in) {
    in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    in.read(reinterpret_cast<char*>(&padded_dim), sizeof(padded_dim));
    in.read(reinterpret_cast<char*>(&seed_), sizeof(seed_));
    num_bytes_per_datapoint = (padded_dim + 1) / 2;
    int rt = 0;
    in.read(reinterpret_cast<char*>(&rt), sizeof(rt));
    rotator_type = static_cast<rabitqlib::RotatorType>(rt);
    rotator.reset(rabitqlib::choose_rotator<float>(dim, rotator_type));
    if (rotator) rotator->load(in);
    size_t ss = 0;
    in.read(reinterpret_cast<char*>(&ss), sizeof(ss));
    signs.resize(ss);
    in.read(reinterpret_cast<char*>(signs.data()), ss * sizeof(float));
  }
};

}  // namespace one_to_many_turboquant
}  // namespace mvsic
