#pragma once

// Multi-vector 1-bit (sign-bit) TurboQuant Chamfer with Hamming-distance scoring.
//
// This is the multi-vector counterpart to the symmetric 1-bit kernels in
// `1-bit-code/hamming_1bit.h` / `1-bit-code/one_to_many_1bit.cc`. The encoding
// is "TurboQuant with 1 bit": rotate via FhtKacRotator, normalize, then store
// only the sign of each padded dimension (1 bit / dim, packed 8 dims / byte).
//
// Storage layout (mirrors turboquant_mv.h panel layout but for bit-packed
// codes):
//   Each cloud is stored as a sequence of "panels". A panel holds
//   kPanelPoints = 16 consecutive points laid out so that one tile's worth of
//   bytes (4 bytes per point = 64 bytes total) maps to a single AVX-512
//   uint8 vector. Per panel:
//     - num_hamming_tiles = ceil(num_bytes_per_datapoint / 4) tiles
//     - Tile t is at offset (t * kPanelBytes), where
//       kPanelBytes = num_hamming_tiles * (kPanelPoints * 4) = num_hamming_tiles * 64
//     - Within tile t, point p occupies bytes [p*4 .. p*4+3], holding the
//       packed-bit bytes (c*4 .. c*4+3) of point p (zero-padded if past end).
//
//   This is the same format used by `BuildHammingPanel1Bit` in the reference
//   kernels, but built once at index time and stored permanently (instead of
//   rebuilt per query batch).
//
// Per-Chamfer kernel (one query cloud x one DB cloud):
//   for each query embedding q:
//     // Broadcast each query tile (4 bytes) to a 64-byte vector.
//     // For each panel, accumulate popcount(panel_tile XOR q_broadcast)
//     // across tiles. Periodically dump uint8 sums into int32 via vpdpbusd
//     // with ones to avoid u8 overflow (safe for up to 31 tiles per chunk).
//     hamming[p] = sum_t popcount(panel[t][p] XOR q_broadcast[t])  for p in panel
//     // Convert hamming -> dot via the analytic formula:
//     //   dot = 16129 * (padded_dim - 2 * hamming)        (16129 = 127^2)
//     // (Each centroid is +/-127 because ws is rescaled by sqrt(padded_dim),
//     //  so post-encode sqr norm = padded_dim * 127^2.)
//     // Distance:
//     //   neg_dot = -(dot * db_nsf * q_nsf)
//     //   metric:   sqn_db + 2*neg_dot + sqn_q
//     //   non-metric: neg_dot
//     min_q = min over panel-points of distance
//   chamfer = sum_q(min_q) / num_queries
//
// Two micro-kernel widths are supported (1 panel and 4 panels), batching
// across `kMq1bit` queries per outer pass for ILP (mirrors the
// HammingMicroKernelAccumulate{,4} pattern from hamming_1bit.h).

#include <vector>
#include <cstdint>
#include <immintrin.h>
#include <limits>
#include <algorithm>
#include <cstring>
#include <queue>
#include <array>

#include "parlay/parallel.h"
#include "parlay/sequence.h"
#include "mvsic/core/quantization/turboquant_utils.h"

namespace mvsic {
namespace turboquant_1bit_mv {

namespace internal {

// Each centroid has magnitude 127, so squared dimension contribution is 127^2.
static constexpr float k127Sq = 16129.0f;  // 127 * 127
static constexpr float kNeg2x127Sq = -2.0f * k127Sq;

// AVX-512 lane counts for the panel layout.
static constexpr size_t kPanelPoints = 16;     // points per panel (matches int32 lanes)
static constexpr size_t kPanelLaneBytes = 4;   // bytes per point per tile
static constexpr size_t kTileBytes = kPanelPoints * kPanelLaneBytes;  // 64 bytes per tile

// uint8 sums of popcount-bytes can hold at most 255. Each popcount byte is in
// [0, 8], so chunking 31 tiles at a time keeps us safely under 256 (31*8=248).
static constexpr size_t kMaxSafeAccumTiles = 31;

// Number of queries to batch together in the inner loop. 4 keeps register
// pressure manageable (4 panels * 4 queries * 1 acc each = 16 zmm) and matches
// the SPR sweet spot used elsewhere.
static constexpr size_t kMq1bit = 4;

// Compute number of tiles required to cover `nbytes` bytes per point. Each
// tile holds 4 bytes per point.
inline size_t num_hamming_tiles_for(size_t nbytes) { return (nbytes + 3) / 4; }

inline size_t panel_bytes_for(size_t num_hamming_tiles) {
  return num_hamming_tiles * kTileBytes;
}

#ifdef __AVX512F__

// Wide popcount: prefers VPOPCNTB (BITALG) when available, otherwise falls
// back to a nibble-LUT shuffle.
inline __m512i popcnt_u8(__m512i x) {
#if defined(__AVX512BITALG__)
  return _mm512_popcnt_epi8(x);
#else
  static const __m512i lut = _mm512_set_epi8(
      4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0,
      4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0,
      4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0,
      4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0);
  const __m512i mask = _mm512_set1_epi8(0x0F);
  const __m512i lo = _mm512_and_si512(x, mask);
  const __m512i hi = _mm512_and_si512(_mm512_srli_epi16(x, 4), mask);
  return _mm512_add_epi8(_mm512_shuffle_epi8(lut, lo), _mm512_shuffle_epi8(lut, hi));
#endif
}

// Broadcast each 4-byte tile word of `q_packed` (length nbytes) into the
// per-query buffer `qbuf`, which is `num_hamming_tiles * 64` bytes long. After
// this, qbuf[t * 64 + j] holds q_packed[t*4 + (j%4)] for all j.
inline void pre_broadcast_query(const uint8_t* q_packed, size_t nbytes,
                                size_t num_hamming_tiles, uint8_t* qbuf) {
  const size_t full_tiles = nbytes / 4;
  size_t t = 0;
  for (; t < full_tiles; ++t) {
    int32_t word;
    std::memcpy(&word, q_packed + t * 4, 4);
    const __m512i v = _mm512_set1_epi32(word);
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(qbuf + t * kTileBytes), v);
  }
  if (t < num_hamming_tiles) {
    int32_t word = 0;
    const size_t rem = nbytes - full_tiles * 4;
    std::memcpy(&word, q_packed + t * 4, rem);
    const __m512i v = _mm512_set1_epi32(word);
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(qbuf + t * kTileBytes), v);
  }
}

// VPDPBUSD with all-ones to convert per-byte popcount sums into per-point
// int32 lanes (sums 4 adjacent u8 values into one int32 lane).
inline __m512i sum_quad_accum(__m512i acc, __m512i u8_sum) {
#ifdef __AVX512VNNI__
  const __m512i ones = _mm512_set1_epi8(1);
  return _mm512_dpbusd_epi32(acc, u8_sum, ones);
#else
  // Fallback: widen via maddubs + madd.
  const __m512i ones16 = _mm512_set1_epi16(1);
  const __m512i ones8 = _mm512_set1_epi8(1);
  __m512i prod16 = _mm512_maddubs_epi16(u8_sum, ones8);
  __m512i prod32 = _mm512_madd_epi16(prod16, ones16);
  return _mm512_add_epi32(acc, prod32);
#endif
}

// Single-panel Hamming accumulation. Writes per-query int32 hamming totals
// into `acc[0..Mq)`. Each lane of acc[q] is the hamming distance of one of
// the (up to) 16 panel points against query q.
//
// Fast path (num_hamming_tiles <= kMaxSafeAccumTiles, i.e. padded_dim <=
// 992): single u8 popcount-byte accumulator per query, reduced to int32
// lanes ONCE at the end. The split between u8 chunk-sums and i32 final
// accumulators (and the kMaxSafeAccumTiles outer loop) collapses, dropping
// ~16 zmm of live state on the 4-panel kernel so the compiler stops
// spilling. Slow path retained for very-large padded_dim configurations.
template<size_t Mq>
inline void hamming_micro_kernel_1panel(const uint8_t* qbuf, size_t qbuf_tile_stride,
                                        const uint8_t* panel, size_t num_hamming_tiles,
                                        __m512i* acc) {
  if (num_hamming_tiles <= kMaxSafeAccumTiles) {
    __m512i sum[Mq];
    for (size_t q = 0; q < Mq; ++q) sum[q] = _mm512_setzero_si512();

    for (size_t t = 0; t < num_hamming_tiles; ++t) {
      const __m512i pv = _mm512_loadu_si512(
          reinterpret_cast<const __m512i*>(panel + t * kTileBytes));
      for (size_t q = 0; q < Mq; ++q) {
        const __m512i qb = _mm512_loadu_si512(
            reinterpret_cast<const __m512i*>(qbuf + q * qbuf_tile_stride + t * kTileBytes));
        sum[q] = _mm512_add_epi8(sum[q], popcnt_u8(_mm512_xor_si512(pv, qb)));
      }
    }

    const __m512i zero = _mm512_setzero_si512();
    for (size_t q = 0; q < Mq; ++q) acc[q] = sum_quad_accum(zero, sum[q]);
    return;
  }

  // Chunked fallback for very large padded_dim.
  for (size_t q = 0; q < Mq; ++q) acc[q] = _mm512_setzero_si512();
  for (size_t t0 = 0; t0 < num_hamming_tiles; t0 += kMaxSafeAccumTiles) {
    const size_t t_end = std::min(t0 + kMaxSafeAccumTiles, num_hamming_tiles);
    __m512i sum[Mq];
    for (size_t q = 0; q < Mq; ++q) sum[q] = _mm512_setzero_si512();
    for (size_t t = t0; t < t_end; ++t) {
      const __m512i pv = _mm512_loadu_si512(
          reinterpret_cast<const __m512i*>(panel + t * kTileBytes));
      for (size_t q = 0; q < Mq; ++q) {
        const __m512i qb = _mm512_loadu_si512(
            reinterpret_cast<const __m512i*>(qbuf + q * qbuf_tile_stride + t * kTileBytes));
        sum[q] = _mm512_add_epi8(sum[q], popcnt_u8(_mm512_xor_si512(pv, qb)));
      }
    }
    for (size_t q = 0; q < Mq; ++q) acc[q] = sum_quad_accum(acc[q], sum[q]);
  }
}

// Four-panel Hamming accumulation (best ILP). Same single-acc strategy as
// the 1-panel kernel — see that comment for the safety argument. With Mq=4
// this brings live state from ~38 zmm (16 u8 sums + 16 i32 accs + panel +
// query temps, previously spilling) down to ~22 zmm: 16 u8 accumulators +
// 4 panel loads + 1 query load + popcount intermediate. Within 32 zmm.
template<size_t Mq>
inline void hamming_micro_kernel_4panel(const uint8_t* qbuf, size_t qbuf_tile_stride,
                                        const uint8_t* panel0, const uint8_t* panel1,
                                        const uint8_t* panel2, const uint8_t* panel3,
                                        size_t num_hamming_tiles, __m512i* acc0,
                                        __m512i* acc1, __m512i* acc2, __m512i* acc3) {
  if (num_hamming_tiles <= kMaxSafeAccumTiles) {
    __m512i s0[Mq], s1[Mq], s2[Mq], s3[Mq];
    for (size_t q = 0; q < Mq; ++q) {
      s0[q] = _mm512_setzero_si512();
      s1[q] = _mm512_setzero_si512();
      s2[q] = _mm512_setzero_si512();
      s3[q] = _mm512_setzero_si512();
    }

    for (size_t t = 0; t < num_hamming_tiles; ++t) {
      const __m512i pa = _mm512_loadu_si512(
          reinterpret_cast<const __m512i*>(panel0 + t * kTileBytes));
      const __m512i pb = _mm512_loadu_si512(
          reinterpret_cast<const __m512i*>(panel1 + t * kTileBytes));
      const __m512i pc = _mm512_loadu_si512(
          reinterpret_cast<const __m512i*>(panel2 + t * kTileBytes));
      const __m512i pd = _mm512_loadu_si512(
          reinterpret_cast<const __m512i*>(panel3 + t * kTileBytes));
      for (size_t q = 0; q < Mq; ++q) {
        const __m512i qb = _mm512_loadu_si512(
            reinterpret_cast<const __m512i*>(qbuf + q * qbuf_tile_stride + t * kTileBytes));
        s0[q] = _mm512_add_epi8(s0[q], popcnt_u8(_mm512_xor_si512(pa, qb)));
        s1[q] = _mm512_add_epi8(s1[q], popcnt_u8(_mm512_xor_si512(pb, qb)));
        s2[q] = _mm512_add_epi8(s2[q], popcnt_u8(_mm512_xor_si512(pc, qb)));
        s3[q] = _mm512_add_epi8(s3[q], popcnt_u8(_mm512_xor_si512(pd, qb)));
      }
    }

    const __m512i zero = _mm512_setzero_si512();
    for (size_t q = 0; q < Mq; ++q) {
      acc0[q] = sum_quad_accum(zero, s0[q]);
      acc1[q] = sum_quad_accum(zero, s1[q]);
      acc2[q] = sum_quad_accum(zero, s2[q]);
      acc3[q] = sum_quad_accum(zero, s3[q]);
    }
    return;
  }

  // Chunked fallback for very large padded_dim.
  for (size_t q = 0; q < Mq; ++q) {
    acc0[q] = _mm512_setzero_si512();
    acc1[q] = _mm512_setzero_si512();
    acc2[q] = _mm512_setzero_si512();
    acc3[q] = _mm512_setzero_si512();
  }
  for (size_t t0 = 0; t0 < num_hamming_tiles; t0 += kMaxSafeAccumTiles) {
    const size_t t_end = std::min(t0 + kMaxSafeAccumTiles, num_hamming_tiles);
    __m512i s0[Mq], s1[Mq], s2[Mq], s3[Mq];
    for (size_t q = 0; q < Mq; ++q) {
      s0[q] = _mm512_setzero_si512();
      s1[q] = _mm512_setzero_si512();
      s2[q] = _mm512_setzero_si512();
      s3[q] = _mm512_setzero_si512();
    }
    for (size_t t = t0; t < t_end; ++t) {
      const __m512i pa = _mm512_loadu_si512(
          reinterpret_cast<const __m512i*>(panel0 + t * kTileBytes));
      const __m512i pb = _mm512_loadu_si512(
          reinterpret_cast<const __m512i*>(panel1 + t * kTileBytes));
      const __m512i pc = _mm512_loadu_si512(
          reinterpret_cast<const __m512i*>(panel2 + t * kTileBytes));
      const __m512i pd = _mm512_loadu_si512(
          reinterpret_cast<const __m512i*>(panel3 + t * kTileBytes));
      for (size_t q = 0; q < Mq; ++q) {
        const __m512i qb = _mm512_loadu_si512(
            reinterpret_cast<const __m512i*>(qbuf + q * qbuf_tile_stride + t * kTileBytes));
        s0[q] = _mm512_add_epi8(s0[q], popcnt_u8(_mm512_xor_si512(pa, qb)));
        s1[q] = _mm512_add_epi8(s1[q], popcnt_u8(_mm512_xor_si512(pb, qb)));
        s2[q] = _mm512_add_epi8(s2[q], popcnt_u8(_mm512_xor_si512(pc, qb)));
        s3[q] = _mm512_add_epi8(s3[q], popcnt_u8(_mm512_xor_si512(pd, qb)));
      }
    }
    for (size_t q = 0; q < Mq; ++q) {
      acc0[q] = sum_quad_accum(acc0[q], s0[q]);
      acc1[q] = sum_quad_accum(acc1[q], s1[q]);
      acc2[q] = sum_quad_accum(acc2[q], s2[q]);
      acc3[q] = sum_quad_accum(acc3[q], s3[q]);
    }
  }
}

// Convert int32 hamming acc (one lane per panel point) into a float distance
// vector and merge into the per-query running min. The dot is computed as
// base_dot + (-2*16129)*hamming where base_dot = 16129 * padded_dim.
template<bool Metric>
inline void hamming_dist_epilogue(__m512i hamming, float q_nsf, float q_sqn, float base_dot,
                                  const float* db_norms, const float* db_sqn,
                                  __m512& running_min) {
  const __m512 hamming_f = _mm512_cvtepi32_ps(hamming);
  const __m512 dot_f =
      _mm512_fmadd_ps(_mm512_set1_ps(kNeg2x127Sq), hamming_f, _mm512_set1_ps(base_dot));
  const __m512 db_norm_v = _mm512_loadu_ps(db_norms);
  const __m512 q_nsf_v = _mm512_set1_ps(q_nsf);
  __m512 neg_dot = _mm512_mul_ps(dot_f, db_norm_v);
  neg_dot = _mm512_mul_ps(neg_dot, q_nsf_v);
  neg_dot = _mm512_sub_ps(_mm512_setzero_ps(), neg_dot);

  __m512 dist;
  if constexpr (Metric) {
    const __m512 sqn_v = _mm512_loadu_ps(db_sqn);
    const __m512 sqn_qv = _mm512_set1_ps(q_sqn);
    dist = _mm512_add_ps(sqn_v, _mm512_add_ps(_mm512_add_ps(neg_dot, neg_dot), sqn_qv));
  } else {
    dist = neg_dot;
  }
  running_min = _mm512_min_ps(running_min, dist);
}

// Core Chamfer kernel over PRE-PANELED 1-bit codes.
//
// Inputs:
//   q_packed_data: contiguous packed-bit query codes, num_q * num_bytes
//   q_norms / q_sqns: per-query norm scaling factor and unquantized sqr norm
//   num_q: number of query embeddings
//   num_bytes: bytes per encoded point (== (padded_dim + 7) / 8)
//   panel_data: pre-built panels for one DB cloud
//   db_norms / db_sqn: per-point arrays (padded so each panel sees 16 lanes)
//   panel_bytes / num_hamming_tiles / n_panels: panel geometry for the cloud
//   padded_dim: padded dimensionality (== num_bytes * 8 for full dim)
//
// Returns sum_{q} min_{point} dist(q, point); caller divides by num_q.
template<bool Metric>
inline float chamfer_panels(const uint8_t* q_packed_data, const float* q_norms,
                            const float* q_sqns, size_t q_byte_stride, size_t num_q,
                            size_t num_bytes, const uint8_t* panel_data, const float* db_norms,
                            const float* db_sqn, size_t panel_bytes, size_t num_hamming_tiles,
                            size_t n_panels, size_t padded_dim) {
  if (num_q == 0 || n_panels == 0) return 0.0f;

  const float base_dot = k127Sq * static_cast<float>(padded_dim);
  const size_t qbuf_tile_stride = num_hamming_tiles * kTileBytes;

  // Per-query broadcast buffer for up to kMq1bit queries at a time.
  // Inline scratch covers kMaxInlineTiles tiles per query (== 32 -> 1024-dim
  // packed at full density). Anything larger falls back to a heap allocation.
  static constexpr size_t kMaxInlineTiles = 32;
  alignas(64) uint8_t qbuf_storage[kMq1bit * kMaxInlineTiles * kTileBytes];
  std::vector<uint8_t> qbuf_heap;
  uint8_t* qbuf;
  if (num_hamming_tiles <= kMaxInlineTiles) {
    qbuf = qbuf_storage;
  } else {
    qbuf_heap.resize(num_hamming_tiles * kMq1bit * kTileBytes);
    qbuf = qbuf_heap.data();
  }

  float total_chamfer = 0.0f;
  size_t qi = 0;

  for (; qi + kMq1bit <= num_q; qi += kMq1bit) {
    // Build broadcast buffers for kMq1bit consecutive queries.
    for (size_t q = 0; q < kMq1bit; ++q) {
      pre_broadcast_query(q_packed_data + (qi + q) * q_byte_stride, num_bytes,
                          num_hamming_tiles, qbuf + q * qbuf_tile_stride);
    }

    __m512 mins[kMq1bit];
    for (size_t q = 0; q < kMq1bit; ++q)
      mins[q] = _mm512_set1_ps(std::numeric_limits<float>::max());

    size_t p = 0;
    for (; p + 4 <= n_panels; p += 4) {
      __m512i a0[kMq1bit], a1[kMq1bit], a2[kMq1bit], a3[kMq1bit];
      hamming_micro_kernel_4panel<kMq1bit>(
          qbuf, qbuf_tile_stride, panel_data + p * panel_bytes,
          panel_data + (p + 1) * panel_bytes, panel_data + (p + 2) * panel_bytes,
          panel_data + (p + 3) * panel_bytes, num_hamming_tiles, a0, a1, a2, a3);
      for (size_t q = 0; q < kMq1bit; ++q) {
        const float qn = q_norms[qi + q];
        const float qs = q_sqns[qi + q];
        hamming_dist_epilogue<Metric>(a0[q], qn, qs, base_dot,
                                       db_norms + p * kPanelPoints,
                                       db_sqn + p * kPanelPoints, mins[q]);
        hamming_dist_epilogue<Metric>(a1[q], qn, qs, base_dot,
                                       db_norms + (p + 1) * kPanelPoints,
                                       db_sqn + (p + 1) * kPanelPoints, mins[q]);
        hamming_dist_epilogue<Metric>(a2[q], qn, qs, base_dot,
                                       db_norms + (p + 2) * kPanelPoints,
                                       db_sqn + (p + 2) * kPanelPoints, mins[q]);
        hamming_dist_epilogue<Metric>(a3[q], qn, qs, base_dot,
                                       db_norms + (p + 3) * kPanelPoints,
                                       db_sqn + (p + 3) * kPanelPoints, mins[q]);
      }
    }
    for (; p < n_panels; ++p) {
      __m512i acc[kMq1bit];
      hamming_micro_kernel_1panel<kMq1bit>(qbuf, qbuf_tile_stride,
                                            panel_data + p * panel_bytes, num_hamming_tiles, acc);
      for (size_t q = 0; q < kMq1bit; ++q) {
        hamming_dist_epilogue<Metric>(acc[q], q_norms[qi + q], q_sqns[qi + q], base_dot,
                                       db_norms + p * kPanelPoints,
                                       db_sqn + p * kPanelPoints, mins[q]);
      }
    }

    for (size_t q = 0; q < kMq1bit; ++q) total_chamfer += _mm512_reduce_min_ps(mins[q]);
  }

  // Tail (< kMq1bit queries).
  for (; qi < num_q; ++qi) {
    pre_broadcast_query(q_packed_data + qi * q_byte_stride, num_bytes, num_hamming_tiles,
                        qbuf);
    __m512 mv = _mm512_set1_ps(std::numeric_limits<float>::max());
    for (size_t p = 0; p < n_panels; ++p) {
      __m512i acc;
      hamming_micro_kernel_1panel<1>(qbuf, qbuf_tile_stride, panel_data + p * panel_bytes,
                                      num_hamming_tiles, &acc);
      hamming_dist_epilogue<Metric>(acc, q_norms[qi], q_sqns[qi], base_dot,
                                     db_norms + p * kPanelPoints,
                                     db_sqn + p * kPanelPoints, mv);
    }
    total_chamfer += _mm512_reduce_min_ps(mv);
  }

  return total_chamfer;
}

#endif  // __AVX512F__

// =========================================================================
// Scalar fallback (correct, slow). Used when AVX-512 is unavailable.
// =========================================================================
template<bool Metric>
inline float chamfer_panels_scalar(const uint8_t* q_packed_data, const float* q_norms,
                                   const float* q_sqns, size_t q_byte_stride, size_t num_q,
                                   size_t num_bytes, const uint8_t* panel_data,
                                   const float* db_norms, const float* db_sqn, size_t panel_bytes,
                                   size_t num_hamming_tiles, size_t n_panels, size_t padded_dim,
                                   size_t cloud_size) {
  const float base_dot = k127Sq * static_cast<float>(padded_dim);
  float total = 0.0f;
  for (size_t qi = 0; qi < num_q; ++qi) {
    const uint8_t* qp = q_packed_data + qi * q_byte_stride;
    float min_d = std::numeric_limits<float>::max();
    for (size_t i = 0; i < cloud_size; ++i) {
      const size_t panel = i / kPanelPoints;
      const size_t lane = i % kPanelPoints;
      const uint8_t* base = panel_data + panel * panel_bytes;
      // Hamming distance: pop count of XOR over all packed bytes.
      int hamming = 0;
      for (size_t t = 0; t < num_hamming_tiles; ++t) {
        const uint8_t* tile = base + t * kTileBytes + lane * kPanelLaneBytes;
        for (size_t b = 0; b < kPanelLaneBytes; ++b) {
          const size_t byte_idx = t * 4 + b;
          const uint8_t qb = (byte_idx < num_bytes) ? qp[byte_idx] : 0;
          hamming += __builtin_popcount(static_cast<unsigned>(tile[b] ^ qb));
        }
      }
      const float dot = base_dot + kNeg2x127Sq * static_cast<float>(hamming);
      const float neg_dot = -(dot * db_norms[panel * kPanelPoints + lane] * q_norms[qi]);
      float d;
      if constexpr (Metric) {
        d = db_sqn[panel * kPanelPoints + lane] + 2.0f * neg_dot + q_sqns[qi];
      } else {
        d = neg_dot;
      }
      if (d < min_d) min_d = d;
    }
    total += min_d;
  }
  return total;
}

}  // namespace internal

// =========================================================================
// Forward declarations
// =========================================================================
template<bool Metric>
class Quantized_Point_Cloud_Set;
template<bool Metric>
class Quantized_Query_Point_Cloud;

template<bool Metric>
float turboquant_1bit_mv_chamfer_distance(const Quantized_Query_Point_Cloud<Metric>& q,
                                          const Quantized_Point_Cloud_Set<Metric>& db,
                                          size_t cloud_idx);

// =========================================================================
// Cloud handle
// =========================================================================
template<bool Metric>
class Quantized_Point_Cloud {
 public:
  const Quantized_Point_Cloud_Set<Metric>* db = nullptr;
  size_t cloud_idx = 0;

  Quantized_Point_Cloud() = default;
  Quantized_Point_Cloud(const Quantized_Point_Cloud_Set<Metric>* d, size_t idx) :
      db(d), cloud_idx(idx) {}

  size_t size() const;

  static constexpr bool is_metric() { return Metric; }

  template<typename Query>
  bool same_as(const Query&) const { return false; }
};

// =========================================================================
// Multi-Vector Query
// =========================================================================
template<bool Metric>
class Quantized_Query_Point_Cloud {
 public:
  using distanceType = float;

  size_t num_queries = 0;
  size_t num_bytes_per_datapoint = 0;
  size_t padded_dim = 0;

  // Flat per-query packed bit codes: num_queries * num_bytes_per_datapoint
  std::vector<uint8_t> flat_query_codes;
  std::vector<float> norm_scaling_factors;       // size: num_queries
  std::vector<float> unquantized_squared_norms;  // size: num_queries

  Quantized_Query_Point_Cloud() = default;

  inline const uint8_t* get_q_codes(size_t i) const {
    return flat_query_codes.data() + i * num_bytes_per_datapoint;
  }

  template<typename CloudView>
  float distance(const CloudView& cloud) const {
    return turboquant_1bit_mv_chamfer_distance(*this, *cloud.db, cloud.cloud_idx);
  }

  template<typename CloudHandle>
  std::pair<float, size_t> distance_w_cmps(const CloudHandle& cloud) const {
    const size_t bytes_per_vec =
        cloud.db->num_bytes_per_datapoint + sizeof(float) + (Metric ? sizeof(float) : 0);
    return {this->distance(cloud), cloud.size() * bytes_per_vec};
  }

  static constexpr bool is_metric() { return Metric; }
};

// =========================================================================
// Multi-Vector Set (panel layout)
// =========================================================================
template<bool Metric>
class Quantized_Point_Cloud_Set {
 public:
  // Encoder geometry (same for every cloud)
  size_t num_bytes_per_datapoint = 0;  // bytes per packed-bit point
  size_t padded_dim = 0;               // dimensions per point (rotator size)
  size_t num_hamming_tiles = 0;        // ceil(num_bytes / 4)
  size_t panel_bytes = 0;              // num_hamming_tiles * 64

  // Concatenated panel data, one cloud after another. Cloud c lives at
  // [panel_offsets[c], panel_offsets[c+1]) bytes.
  parlay::sequence<uint8_t> panel_data;
  parlay::sequence<size_t> panel_offsets;  // size: n_clouds + 1, in BYTES

  // Per-point arrays, padded so each cloud is a multiple of kPanelPoints.
  parlay::sequence<size_t> point_offsets;  // size: n_clouds + 1, in points
  parlay::sequence<float> norm_scaling_factors;
  parlay::sequence<float> unquantized_squared_norms;

  // Unpadded sizes per cloud.
  parlay::sequence<size_t> cloud_sizes;
  parlay::sequence<uint32_t> ids;

  inline size_t num_clouds() const noexcept {
    return panel_offsets.size() > 0 ? panel_offsets.size() - 1 : 0;
  }
  inline size_t n_panels(size_t c) const noexcept {
    return (panel_offsets[c + 1] - panel_offsets[c]) / panel_bytes;
  }
  inline size_t cloud_size(size_t c) const noexcept { return cloud_sizes[c]; }

  Quantized_Point_Cloud_Set() = default;
  static constexpr bool is_metric() noexcept { return Metric; }

  Quantized_Point_Cloud<Metric> operator[](size_t i) const {
    return Quantized_Point_Cloud<Metric>(this, i);
  }

  inline uint32_t get_id(size_t i) const noexcept {
    return (ids.size() > 0) ? ids[i] : static_cast<uint32_t>(i);
  }
  inline size_t num_bytes() const noexcept {
    return panel_data.size() * sizeof(uint8_t) +
           norm_scaling_factors.size() * sizeof(float) +
           unquantized_squared_norms.size() * sizeof(float);
  }

  void distances_all(const Quantized_Query_Point_Cloud<Metric>& q,
                     std::pair<uint32_t, float>* results) const {
    const size_t num_q = q.num_queries;
    const size_t nc = num_clouds();
    if (num_q == 0 || nc == 0) return;

    parlay::parallel_for(0, nc, [&](size_t c) {
      if (cloud_sizes[c] == 0) {
        results[c] = {get_id(c), std::numeric_limits<float>::max()};
        return;
      }
      const float d = turboquant_1bit_mv_chamfer_distance(q, *this, c);
      results[c] = {get_id(c), d};
    });
  }

  void save(std::ofstream& out) const {
    out.write(reinterpret_cast<const char*>(&num_bytes_per_datapoint),
              sizeof(num_bytes_per_datapoint));
    out.write(reinterpret_cast<const char*>(&padded_dim), sizeof(padded_dim));
    out.write(reinterpret_cast<const char*>(&num_hamming_tiles), sizeof(num_hamming_tiles));
    out.write(reinterpret_cast<const char*>(&panel_bytes), sizeof(panel_bytes));

    auto write_seq = [&](const auto& seq) {
      using T = typename std::remove_reference_t<decltype(seq)>::value_type;
      size_t sz = seq.size();
      out.write(reinterpret_cast<const char*>(&sz), sizeof(sz));
      if (sz) out.write(reinterpret_cast<const char*>(seq.data()), sz * sizeof(T));
    };
    write_seq(panel_data);
    write_seq(panel_offsets);
    write_seq(point_offsets);
    write_seq(norm_scaling_factors);
    write_seq(unquantized_squared_norms);
    write_seq(cloud_sizes);
    write_seq(ids);
  }

  void load(std::ifstream& in) {
    in.read(reinterpret_cast<char*>(&num_bytes_per_datapoint), sizeof(num_bytes_per_datapoint));
    in.read(reinterpret_cast<char*>(&padded_dim), sizeof(padded_dim));
    in.read(reinterpret_cast<char*>(&num_hamming_tiles), sizeof(num_hamming_tiles));
    in.read(reinterpret_cast<char*>(&panel_bytes), sizeof(panel_bytes));

    auto read_seq = [&](auto& seq) {
      using T = typename std::remove_reference_t<decltype(seq)>::value_type;
      size_t sz = 0;
      in.read(reinterpret_cast<char*>(&sz), sizeof(sz));
      seq.resize(sz);
      if (sz) in.read(reinterpret_cast<char*>(seq.data()), sz * sizeof(T));
    };
    read_seq(panel_data);
    read_seq(panel_offsets);
    read_seq(point_offsets);
    read_seq(norm_scaling_factors);
    read_seq(unquantized_squared_norms);
    read_seq(cloud_sizes);
    read_seq(ids);
  }
};

template<bool Metric>
inline size_t Quantized_Point_Cloud<Metric>::size() const {
  return db ? db->cloud_sizes[cloud_idx] : 0;
}

// =========================================================================
// Chamfer kernel: one query cloud x one DB cloud
// =========================================================================
template<bool Metric>
float turboquant_1bit_mv_chamfer_distance(const Quantized_Query_Point_Cloud<Metric>& q,
                                          const Quantized_Point_Cloud_Set<Metric>& db,
                                          size_t c) {
  const size_t num_q = q.num_queries;
  if (num_q == 0) return 0.0f;
  const size_t cs = db.cloud_sizes[c];
  if (cs == 0) return std::numeric_limits<float>::max();

  const size_t panel_byte_off = db.panel_offsets[c];
  const size_t pt_off = db.point_offsets[c];
  const size_t np = db.n_panels(c);
  const uint8_t* panel_ptr = db.panel_data.data() + panel_byte_off;
  const float* ns = db.norm_scaling_factors.data() + pt_off;
  const float* sqn = db.unquantized_squared_norms.data() + pt_off;

  float dist_sum = 0.0f;
#ifdef __AVX512F__
  dist_sum = internal::chamfer_panels<Metric>(
      q.flat_query_codes.data(), q.norm_scaling_factors.data(),
      q.unquantized_squared_norms.data(), q.num_bytes_per_datapoint, num_q,
      q.num_bytes_per_datapoint, panel_ptr, ns, sqn, db.panel_bytes, db.num_hamming_tiles, np,
      db.padded_dim);
#else
  dist_sum = internal::chamfer_panels_scalar<Metric>(
      q.flat_query_codes.data(), q.norm_scaling_factors.data(),
      q.unquantized_squared_norms.data(), q.num_bytes_per_datapoint, num_q,
      q.num_bytes_per_datapoint, panel_ptr, ns, sqn, db.panel_bytes, db.num_hamming_tiles, np,
      db.padded_dim, cs);
#endif
  return dist_sum / static_cast<float>(num_q);
}

// =========================================================================
// Multi-Vector Model
// =========================================================================
template<bool Metric>
class Model {
 public:
  // Reuses the TQ rotator (FhtKacRotator). We don't use BaseEncoder's 4-bit
  // encode_single — only its rotator setup.
  turboquant::BaseEncoder encoder;

  Model() = default;

  template<typename PCSet>
  void train(const PCSet& pcs) {
    encoder.train(pcs.get_dims());
  }

  // Encode a single rotated, normalized vector into 1-bit codes (sign bits).
  // Sets `*out_sqn` to the unquantized squared norm and returns the
  // norm-scaling-factor (norm / (127 * sqrt(padded_dim))).
  inline float encode_single_bits(const float* p, std::vector<float>& ws, uint8_t* out_codes,
                                  float* out_sqn) const {
    const size_t pdim = encoder.padded_dim;
    const size_t nbytes = (pdim + 7) / 8;
    std::memset(out_codes, 0, nbytes);
    encoder.rotator->rotate(p, ws.data());

    float sqr_norm = 0.0f;
    for (size_t i = 0; i < pdim; ++i) sqr_norm += ws[i] * ws[i];
    *out_sqn = sqr_norm;
    if (sqr_norm == 0.0f) return 0.0f;

    // Sign bit only. Convention (matches one_to_many_1bit.cc / hamming_1bit.h):
    //   bit set  => coordinate is *negative*  (so XOR Hamming counts mismatches)
    //   bit clear => coordinate is non-negative
    for (size_t i = 0; i < pdim; ++i) {
      if (ws[i] < 0.0f) out_codes[i / 8] |= static_cast<uint8_t>(1u << (i % 8));
    }
    // Encoded sqr norm = padded_dim * 127^2.
    const float q_sqr = static_cast<float>(pdim) * internal::k127Sq;
    return std::sqrt(sqr_norm / q_sqr);
  }

  template<typename PCSet>
  Quantized_Point_Cloud_Set<Metric> encode(const PCSet& pcs) const {
    Quantized_Point_Cloud_Set<Metric> enc;
    enc.padded_dim = encoder.padded_dim;
    enc.num_bytes_per_datapoint = (encoder.padded_dim + 7) / 8;
    enc.num_hamming_tiles = internal::num_hamming_tiles_for(enc.num_bytes_per_datapoint);
    enc.panel_bytes = internal::panel_bytes_for(enc.num_hamming_tiles);

    auto float_offsets = pcs.get_offsets();
    const size_t n_clouds = float_offsets.size() > 0 ? float_offsets.size() - 1 : 0;

    enc.panel_offsets.resize(n_clouds + 1);
    enc.point_offsets.resize(n_clouds + 1);
    enc.cloud_sizes.resize(n_clouds);

    size_t cur_panel_bytes = 0;
    size_t cur_padded_pts = 0;
    for (size_t c = 0; c < n_clouds; ++c) {
      const size_t n_vecs = (float_offsets[c + 1] - float_offsets[c]) / encoder.dim;
      const size_t np = (n_vecs + internal::kPanelPoints - 1) / internal::kPanelPoints;
      enc.panel_offsets[c] = cur_panel_bytes;
      enc.point_offsets[c] = cur_padded_pts;
      enc.cloud_sizes[c] = n_vecs;
      cur_panel_bytes += np * enc.panel_bytes;
      cur_padded_pts += np * internal::kPanelPoints;
    }
    enc.panel_offsets[n_clouds] = cur_panel_bytes;
    enc.point_offsets[n_clouds] = cur_padded_pts;

    enc.panel_data.resize(cur_panel_bytes, 0);
    enc.norm_scaling_factors.resize(cur_padded_pts, 0.0f);
    if constexpr (Metric) {
      // Padding lanes get +inf so they can never win the min.
      enc.unquantized_squared_norms.assign(cur_padded_pts, std::numeric_limits<float>::infinity());
    } else {
      enc.unquantized_squared_norms.resize(cur_padded_pts, 0.0f);
    }

    auto pcs_ids = pcs.get_ids();
    enc.ids = parlay::sequence<uint32_t>(pcs_ids.begin(), pcs_ids.end());

    parlay::parallel_for(0, n_clouds, [&](size_t c) {
      const size_t n_vecs = enc.cloud_sizes[c];
      if (n_vecs == 0) return;
      const size_t np = (n_vecs + internal::kPanelPoints - 1) / internal::kPanelPoints;

      // Per-cloud temporary: flat encoded bytes for each point in the cloud.
      std::vector<uint8_t> flat_codes(n_vecs * enc.num_bytes_per_datapoint, 0);
      std::vector<float> ws(encoder.padded_dim);

      const size_t src_start = float_offsets[c] / encoder.dim;
      const size_t pt_off = enc.point_offsets[c];

      for (size_t i = 0; i < n_vecs; ++i) {
        const float* p =
            reinterpret_cast<const float*>(pcs.data() + (src_start + i) * encoder.dim);
        float sqn = 0.0f;
        const float nsf = encode_single_bits(p, ws,
                                             flat_codes.data() + i * enc.num_bytes_per_datapoint,
                                             &sqn);
        enc.norm_scaling_factors[pt_off + i] = nsf;
        enc.unquantized_squared_norms[pt_off + i] = sqn;
      }

      // Repack into the panel layout: for each panel of kPanelPoints points,
      // for each tile of 4 bytes, store the 4 bytes per point at offset
      // p*4 within the tile.
      uint8_t* cloud_panels = enc.panel_data.data() + enc.panel_offsets[c];
      for (size_t pi = 0; pi < np; ++pi) {
        uint8_t* panel = cloud_panels + pi * enc.panel_bytes;
        const size_t base = pi * internal::kPanelPoints;
        for (size_t lane = 0; lane < internal::kPanelPoints; ++lane) {
          const size_t pt = base + lane;
          if (pt >= n_vecs) {
            // Pad with zeros (panel was already memset). Padding norm is 0
            // and sqn is +inf (metric) so this point can never win the min.
            continue;
          }
          const uint8_t* src = flat_codes.data() + pt * enc.num_bytes_per_datapoint;
          for (size_t t = 0; t < enc.num_hamming_tiles; ++t) {
            uint8_t* dst = panel + t * internal::kTileBytes + lane * internal::kPanelLaneBytes;
            const size_t avail =
                (t * 4 + 4 <= enc.num_bytes_per_datapoint) ? 4 : (enc.num_bytes_per_datapoint - t * 4);
            std::memcpy(dst, src + t * 4, avail);
            // Remainder bytes in this tile lane stay 0 (already memset).
          }
        }
      }
    });

    return enc;
  }

  template<typename ChPoint>
  Quantized_Query_Point_Cloud<Metric> quantize_query(const ChPoint& query_cloud) const {
    Quantized_Query_Point_Cloud<Metric> res;
    res.num_queries = query_cloud.size();
    res.padded_dim = encoder.padded_dim;
    res.num_bytes_per_datapoint = (encoder.padded_dim + 7) / 8;
    if (res.num_queries == 0) return res;

    res.flat_query_codes.assign(res.num_queries * res.num_bytes_per_datapoint, 0);
    res.norm_scaling_factors.assign(res.num_queries, 0.0f);
    res.unquantized_squared_norms.assign(res.num_queries, 0.0f);

    const float* base_ptr = query_cloud.data();
    std::vector<float> ws(encoder.padded_dim);
    for (size_t qi = 0; qi < res.num_queries; ++qi) {
      float sqn = 0.0f;
      const float nsf = encode_single_bits(base_ptr + qi * encoder.dim, ws,
                                           res.flat_query_codes.data() +
                                               qi * res.num_bytes_per_datapoint,
                                           &sqn);
      res.norm_scaling_factors[qi] = nsf;
      res.unquantized_squared_norms[qi] = sqn;
    }
    return res;
  }

  void save(std::ofstream& out) const { encoder.save(out); }
  void load(std::ifstream& in) { encoder.load(in); }
};

// =========================================================================
// Pre-Fused Query Batch (mirrors pqtq_mv::FusedQueryBatch). Holds the
// pre-broadcasted query tile buffers for ALL embeddings across a group of
// query clouds, so that one M2M sweep can stream each db cloud's panels
// exactly once across the entire query group.
// =========================================================================
template<bool Metric>
struct FusedQueryBatch {
  size_t num_source_clouds = 0;
  size_t total_embeddings = 0;
  size_t num_bytes_per_datapoint = 0;
  size_t padded_dim = 0;
  size_t num_hamming_tiles = 0;
  size_t qbuf_tile_stride = 0;  // num_hamming_tiles * kTileBytes

  // Pre-broadcasted query data, total_embeddings * qbuf_tile_stride bytes.
  // Layout: [embedding][tile * 64 bytes (broadcasted)].
  std::vector<uint8_t> flat_qbuf;
  std::vector<float> norm_scaling_factors;       // size: total_embeddings
  std::vector<float> unquantized_squared_norms;  // size: total_embeddings
  std::vector<size_t> emb_offsets;               // size: num_source_clouds + 1

  void Build(const std::vector<const Quantized_Query_Point_Cloud<Metric>*>& A) {
    num_source_clouds = A.size();
    emb_offsets.assign(num_source_clouds + 1, 0);
    if (num_source_clouds == 0) {
      total_embeddings = 0;
      return;
    }
    num_bytes_per_datapoint = A[0]->num_bytes_per_datapoint;
    padded_dim = A[0]->padded_dim;
    num_hamming_tiles = internal::num_hamming_tiles_for(num_bytes_per_datapoint);
    qbuf_tile_stride = num_hamming_tiles * internal::kTileBytes;

    for (size_t i = 0; i < num_source_clouds; ++i) {
      emb_offsets[i + 1] = emb_offsets[i] + A[i]->num_queries;
    }
    total_embeddings = emb_offsets[num_source_clouds];
    if (total_embeddings == 0) return;

    flat_qbuf.assign(total_embeddings * qbuf_tile_stride, 0);
    norm_scaling_factors.resize(total_embeddings);
    unquantized_squared_norms.resize(total_embeddings);

    parlay::parallel_for(0, num_source_clouds, [&](size_t i) {
      const auto* qc = A[i];
      const size_t off = emb_offsets[i];
      const size_t cnt = qc->num_queries;
      if (cnt == 0) return;
      for (size_t e = 0; e < cnt; ++e) {
        const uint8_t* src =
            qc->flat_query_codes.data() + e * num_bytes_per_datapoint;
        uint8_t* dst = flat_qbuf.data() + (off + e) * qbuf_tile_stride;
#ifdef __AVX512F__
        internal::pre_broadcast_query(src, num_bytes_per_datapoint, num_hamming_tiles, dst);
#else
        for (size_t t = 0; t < num_hamming_tiles; ++t) {
          uint8_t word[4] = {0, 0, 0, 0};
          const size_t base = t * 4;
          const size_t avail =
              (base + 4 <= num_bytes_per_datapoint) ? 4 : (num_bytes_per_datapoint - base);
          for (size_t b = 0; b < avail; ++b) word[b] = src[base + b];
          for (size_t lane = 0; lane < internal::kPanelPoints; ++lane) {
            for (size_t b = 0; b < 4; ++b) {
              dst[t * internal::kTileBytes + lane * 4 + b] = word[b];
            }
          }
        }
#endif
      }
      std::memcpy(norm_scaling_factors.data() + off, qc->norm_scaling_factors.data(),
                  cnt * sizeof(float));
      std::memcpy(unquantized_squared_norms.data() + off,
                  qc->unquantized_squared_norms.data(), cnt * sizeof(float));
    });
  }
};

// =========================================================================
// Score one db cloud against ALL embeddings in a FusedQueryBatch. Writes
// per-embedding min distance into `emb_min` (one float per embedding).
// =========================================================================
template<bool Metric>
inline void score_one_db_cloud(const FusedQueryBatch<Metric>& fq,
                                const Quantized_Point_Cloud_Set<Metric>& db, size_t c,
                                float* emb_min) {
  const size_t num_emb = fq.total_embeddings;
  const size_t cs = db.cloud_sizes[c];
  if (cs == 0) {
    for (size_t i = 0; i < num_emb; ++i) emb_min[i] = std::numeric_limits<float>::max();
    return;
  }

  const size_t panel_byte_off = db.panel_offsets[c];
  const size_t pt_off = db.point_offsets[c];
  const size_t np = db.n_panels(c);
  const uint8_t* panel_ptr = db.panel_data.data() + panel_byte_off;
  const float* db_norms = db.norm_scaling_factors.data() + pt_off;
  const float* db_sqn = db.unquantized_squared_norms.data() + pt_off;
  const size_t panel_bytes = db.panel_bytes;
  const size_t num_hamming_tiles = fq.num_hamming_tiles;
  const size_t qbuf_tile_stride = fq.qbuf_tile_stride;
  const float base_dot = internal::k127Sq * static_cast<float>(fq.padded_dim);

#ifdef __AVX512F__
  using internal::kMq1bit;
  using internal::kPanelPoints;

  size_t qi = 0;
  for (; qi + kMq1bit <= num_emb; qi += kMq1bit) {
    const uint8_t* qbuf = fq.flat_qbuf.data() + qi * qbuf_tile_stride;

    __m512 mins[kMq1bit];
    for (size_t q = 0; q < kMq1bit; ++q)
      mins[q] = _mm512_set1_ps(std::numeric_limits<float>::max());

    size_t p = 0;
    for (; p + 4 <= np; p += 4) {
      __m512i a0[kMq1bit], a1[kMq1bit], a2[kMq1bit], a3[kMq1bit];
      internal::hamming_micro_kernel_4panel<kMq1bit>(
          qbuf, qbuf_tile_stride,
          panel_ptr + p * panel_bytes, panel_ptr + (p + 1) * panel_bytes,
          panel_ptr + (p + 2) * panel_bytes, panel_ptr + (p + 3) * panel_bytes,
          num_hamming_tiles, a0, a1, a2, a3);
      for (size_t q = 0; q < kMq1bit; ++q) {
        const float qn = fq.norm_scaling_factors[qi + q];
        const float qs = fq.unquantized_squared_norms[qi + q];
        internal::hamming_dist_epilogue<Metric>(a0[q], qn, qs, base_dot,
                                                 db_norms + p * kPanelPoints,
                                                 db_sqn + p * kPanelPoints, mins[q]);
        internal::hamming_dist_epilogue<Metric>(a1[q], qn, qs, base_dot,
                                                 db_norms + (p + 1) * kPanelPoints,
                                                 db_sqn + (p + 1) * kPanelPoints, mins[q]);
        internal::hamming_dist_epilogue<Metric>(a2[q], qn, qs, base_dot,
                                                 db_norms + (p + 2) * kPanelPoints,
                                                 db_sqn + (p + 2) * kPanelPoints, mins[q]);
        internal::hamming_dist_epilogue<Metric>(a3[q], qn, qs, base_dot,
                                                 db_norms + (p + 3) * kPanelPoints,
                                                 db_sqn + (p + 3) * kPanelPoints, mins[q]);
      }
    }
    for (; p < np; ++p) {
      __m512i acc[kMq1bit];
      internal::hamming_micro_kernel_1panel<kMq1bit>(qbuf, qbuf_tile_stride,
                                                       panel_ptr + p * panel_bytes,
                                                       num_hamming_tiles, acc);
      for (size_t q = 0; q < kMq1bit; ++q) {
        internal::hamming_dist_epilogue<Metric>(
            acc[q], fq.norm_scaling_factors[qi + q],
            fq.unquantized_squared_norms[qi + q], base_dot,
            db_norms + p * kPanelPoints, db_sqn + p * kPanelPoints, mins[q]);
      }
    }

    for (size_t q = 0; q < kMq1bit; ++q)
      emb_min[qi + q] = _mm512_reduce_min_ps(mins[q]);
  }

  // Tail: < kMq1bit embeddings remaining.
  for (; qi < num_emb; ++qi) {
    const uint8_t* qbuf = fq.flat_qbuf.data() + qi * qbuf_tile_stride;
    __m512 mv = _mm512_set1_ps(std::numeric_limits<float>::max());
    for (size_t p = 0; p < np; ++p) {
      __m512i acc;
      internal::hamming_micro_kernel_1panel<1>(qbuf, qbuf_tile_stride,
                                                panel_ptr + p * panel_bytes,
                                                num_hamming_tiles, &acc);
      internal::hamming_dist_epilogue<Metric>(
          acc, fq.norm_scaling_factors[qi], fq.unquantized_squared_norms[qi], base_dot,
          db_norms + p * kPanelPoints, db_sqn + p * kPanelPoints, mv);
    }
    emb_min[qi] = _mm512_reduce_min_ps(mv);
  }
#else
  // Scalar fallback: leave +inf so callers see worst-case distances.
  (void)panel_ptr;
  (void)db_norms;
  (void)db_sqn;
  (void)panel_bytes;
  (void)num_hamming_tiles;
  (void)qbuf_tile_stride;
  (void)base_dot;
  for (size_t i = 0; i < num_emb; ++i) emb_min[i] = std::numeric_limits<float>::max();
#endif
}

// =========================================================================
// ManyToMany Batch Operator (mirrors pqtq_mv::ManyToMany).
// =========================================================================
template<typename PCS>
class ManyToMany {
 public:
  static void TopKIntoUninitialized(
      const std::vector<const Quantized_Query_Point_Cloud<PCS::is_metric()>*>& A, const PCS& B,
      uint32_t k, std::pair<uint32_t, float>* results, size_t q_block = 16,
      bool parallel_query_blocks = true) {
    constexpr bool Metric = PCS::is_metric();
    const size_t num_q_clouds = A.size();
    const size_t num_db_clouds = B.num_clouds();
    if (num_q_clouds == 0 || num_db_clouds == 0 || k == 0) return;
    if (q_block == 0) q_block = 1;

    auto process_query_range = [&](size_t q_start, size_t q_end) {
      const size_t q_count = q_end - q_start;
      if (q_count == 0) return;

      std::vector<std::priority_queue<std::pair<float, uint32_t>>> heaps(q_count);

      std::vector<const Quantized_Query_Point_Cloud<Metric>*> slice(q_count);
      for (size_t i = 0; i < q_count; ++i) slice[i] = A[q_start + i];
      FusedQueryBatch<Metric> fq;
      fq.Build(slice);

      std::vector<float> emb_min_dists(fq.total_embeddings);

      for (size_t c = 0; c < num_db_clouds; ++c) {
        const size_t cs = B.cloud_sizes[c];
        if (cs == 0) continue;

        score_one_db_cloud<Metric>(fq, B, c, emb_min_dists.data());

        for (size_t i = 0; i < q_count; ++i) {
          const size_t e_start = fq.emb_offsets[i];
          const size_t e_count = fq.emb_offsets[i + 1] - e_start;
          if (e_count == 0) continue;
          float dist_sum = 0.0f;
          for (size_t e = 0; e < e_count; ++e) dist_sum += emb_min_dists[e_start + e];
          const float chamfer_dist = dist_sum / static_cast<float>(e_count);

          if (heaps[i].size() < k) {
            heaps[i].push({chamfer_dist, B.get_id(c)});
          } else if (chamfer_dist < heaps[i].top().first) {
            heaps[i].pop();
            heaps[i].push({chamfer_dist, B.get_id(c)});
          }
        }
      }

      for (size_t i = 0; i < q_count; ++i) {
        const size_t cnt = heaps[i].size();
        const size_t global_idx = q_start + i;
        for (size_t ki = 0; ki < cnt; ++ki) {
          results[global_idx * k + (cnt - 1 - ki)] = {heaps[i].top().second,
                                                        heaps[i].top().first};
          heaps[i].pop();
        }
        for (size_t ki = cnt; ki < k; ++ki) {
          results[global_idx * k + ki] = {UINT32_MAX, std::numeric_limits<float>::max()};
        }
      }
    };

    if (parallel_query_blocks) {
      parlay::blocked_for(0, num_q_clouds, q_block,
                          [&](size_t /*block_idx*/, size_t q_start, size_t q_end) {
                            process_query_range(q_start, q_end);
                          });
    } else {
      for (size_t q_start = 0; q_start < num_q_clouds; q_start += q_block) {
        const size_t q_end = std::min(q_start + q_block, num_q_clouds);
        process_query_range(q_start, q_end);
      }
    }
  }
};

}  // namespace turboquant_1bit_mv
}  // namespace mvsic
