#pragma once

// Multi-vector 4-bit TurboQuant Chamfer with PRE-TRANSPOSED panel storage.
//
// Each cloud is stored as a contiguous run of "panels". A panel holds
// kVnniPoints = 16 consecutive points laid out so that one VPDPBUSD instruction
// produces one int32 lane per point. This is the same panel format used by the
// reference many-to-many TurboQuant kernels in `many_to_many_code/`, but kept
// here as raw AVX-512 intrinsics so we can reuse the existing micro-kernels.
//
// Pre-transposing the panels at *index build* time (instead of decoding strips
// on every Chamfer call) is the main optimization: a query that visits the
// same cluster multiple times no longer redoes the per-call decode.
//
// Encoding pipeline:
//   floats --(BaseEncoder::encode_single)--> packed nibbles per point
//   nibbles --(per-cloud temporary strip buffer)--> decode_strip_to_panel_simd
//                                                  --> panel_data[cloud_offset..]
//
// Storage layout:
//   panel_data        : concatenated panels for all clouds (aligned)
//   panel_offsets[c]  : byte offset of cloud c's first panel within panel_data
//   point_offsets[c]  : start of cloud c in the per-point arrays (padded to 16)
//   norms / sqns      : per-point arrays, padded so each cloud is a multiple of 16
//   cloud_sizes[c]    : unpadded number of points in cloud c
//
// Per-Chamfer:
//   for each query batch of kVnniMq queries:
//     decode is skipped, kernel runs straight on panel_data
//     min over panels via vnni_micro_kernel_{1,2,4}panel<Mq> + epilogue
//     final reduce_min per query, sum into chamfer
//
// kVnniMq is set to 6 to match the SPR sweet spot identified in the reference
// (24 acc + 4 panel B + 1 broadcast Q + spill room = ~30 ZMM live registers).

#include <vector>
#include <cstdint>
#include <immintrin.h>
#include <limits>
#include <algorithm>
#include <cstring>
#include <queue>

#include "parlay/parallel.h"
#include "parlay/sequence.h"
#include "mvsic/core/quantization/turboquant_utils.h"

namespace mvsic {
namespace turboquant_mv {

namespace internal {

// Centroids shifted by +128 for native vpdpbusd (unsigned * signed). Same as
// kTurboQuantCentroidsInt8 ^ 0x80.
static constexpr std::array<uint8_t, 16> kCentroidsUint8 = {134, 146, 159, 172, 186, 203, 224, 255,
                                                            122, 110, 97,  84,  70,  53,  32,  1};

static constexpr size_t kVnniPoints = 16;  // int32 lanes in __m512i (= panel point count)
static constexpr size_t kVnniMq = 6;       // queries per batch (SPR sweet spot)

// =========================================================================
// Panel build-time helpers
// =========================================================================

// Builds one block-transposed panel from a strip-laid-out source buffer
// `strip_ptr` (byte j of point lane => strip_ptr[j*64 + lane]). Reads
// kVnniPoints (= 16) consecutive lanes starting at `base_lane`. Pads any
// trailing tile slots with the neutral byte (0x80) so they don't move the min.
inline void decode_strip_to_panel_simd(const uint8_t* strip_ptr, size_t base_lane, size_t num_bytes,
                                       size_t total_tiles, uint8_t* panel) {
  const __m128i codebook_u8 =
      _mm_loadu_si128(reinterpret_cast<const __m128i*>(kCentroidsUint8.data()));
  const __m128i mask_0f = _mm_set1_epi8(0x0F);
  constexpr size_t N = kVnniPoints * 4;

  size_t tile = 0;
  size_t j = 0;
  for (; j + 1 < num_bytes; j += 2) {
    const __m128i packed_j =
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(strip_ptr + j * 64 + base_lane));
    const __m128i packed_j1 =
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(strip_ptr + (j + 1) * 64 + base_lane));

    const __m128i even_idx_j = _mm_and_si128(packed_j, mask_0f);
    const __m128i odd_idx_j = _mm_and_si128(_mm_srli_epi16(packed_j, 4), mask_0f);
    const __m128i even_j = _mm_shuffle_epi8(codebook_u8, even_idx_j);
    const __m128i odd_j = _mm_shuffle_epi8(codebook_u8, odd_idx_j);

    const __m128i even_idx_j1 = _mm_and_si128(packed_j1, mask_0f);
    const __m128i odd_idx_j1 = _mm_and_si128(_mm_srli_epi16(packed_j1, 4), mask_0f);
    const __m128i even_j1 = _mm_shuffle_epi8(codebook_u8, even_idx_j1);
    const __m128i odd_j1 = _mm_shuffle_epi8(codebook_u8, odd_idx_j1);

    const __m128i pair_j = _mm_unpacklo_epi8(even_j, odd_j);
    const __m128i pair_j_hi = _mm_unpackhi_epi8(even_j, odd_j);
    const __m128i pair_j1 = _mm_unpacklo_epi8(even_j1, odd_j1);
    const __m128i pair_j1_hi = _mm_unpackhi_epi8(even_j1, odd_j1);

    const __m128i tile_lo = _mm_unpacklo_epi16(pair_j, pair_j1);
    const __m128i tile_mid_lo = _mm_unpackhi_epi16(pair_j, pair_j1);
    const __m128i tile_mid_hi = _mm_unpacklo_epi16(pair_j_hi, pair_j1_hi);
    const __m128i tile_hi = _mm_unpackhi_epi16(pair_j_hi, pair_j1_hi);

    _mm_storeu_si128(reinterpret_cast<__m128i*>(panel + tile * N + 0), tile_lo);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(panel + tile * N + 16), tile_mid_lo);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(panel + tile * N + 32), tile_mid_hi);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(panel + tile * N + 48), tile_hi);
    ++tile;
  }

  if (j < num_bytes) {
    const __m128i packed_j =
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(strip_ptr + j * 64 + base_lane));
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

  // Neutral pad: 0x80 = unsigned form of int8 0, zero contribution to dot product.
  const __m128i neutral = _mm_set1_epi8(static_cast<char>(0x80));
  for (; tile < total_tiles; ++tile) {
    _mm_storeu_si128(reinterpret_cast<__m128i*>(panel + tile * N + 0), neutral);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(panel + tile * N + 16), neutral);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(panel + tile * N + 32), neutral);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(panel + tile * N + 48), neutral);
  }
}

// =========================================================================
// AVX-512 GEMM micro-kernels
// =========================================================================
#ifdef __AVX512F__
inline __m512i tq_dpbusd(__m512i acc, __m512i a_unsigned, __m512i b_signed) {
#ifdef __AVX512VNNI__
  return _mm512_dpbusd_epi32(acc, a_unsigned, b_signed);
#else
  const __m512i prod16 = _mm512_maddubs_epi16(a_unsigned, b_signed);
  const __m512i prod32 = _mm512_madd_epi16(prod16, _mm512_set1_epi16(1));
  return _mm512_add_epi32(acc, prod32);
#endif
}

template<size_t Mq>
inline void vnni_micro_kernel_1panel(const int8_t* const* query_ptrs, const uint8_t* panel,
                                     size_t total_tiles, __m512i* acc) {
  constexpr size_t N = kVnniPoints * 4;
  for (size_t q = 0; q < Mq; ++q)
    acc[q] = _mm512_setzero_si512();

  for (size_t t = 0; t < total_tiles; ++t) {
    const __m512i b = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(panel + t * N));
    for (size_t q = 0; q < Mq; ++q) {
      const __m512i qv = _mm512_set1_epi32(reinterpret_cast<const int32_t*>(query_ptrs[q])[t]);
      acc[q] = tq_dpbusd(acc[q], b, qv);
    }
  }
}

template<size_t Mq>
inline void vnni_micro_kernel_2panel(const int8_t* const* query_ptrs, const uint8_t* panel0,
                                     const uint8_t* panel1, size_t total_tiles, __m512i* acc0,
                                     __m512i* acc1) {
  constexpr size_t N = kVnniPoints * 4;
  for (size_t q = 0; q < Mq; ++q) {
    acc0[q] = _mm512_setzero_si512();
    acc1[q] = _mm512_setzero_si512();
  }

  for (size_t t = 0; t < total_tiles; ++t) {
    const __m512i b0 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(panel0 + t * N));
    const __m512i b1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(panel1 + t * N));
    for (size_t q = 0; q < Mq; ++q) {
      const __m512i qv = _mm512_set1_epi32(reinterpret_cast<const int32_t*>(query_ptrs[q])[t]);
      acc0[q] = tq_dpbusd(acc0[q], b0, qv);
      acc1[q] = tq_dpbusd(acc1[q], b1, qv);
    }
  }
}

template<size_t Mq>
inline void vnni_micro_kernel_4panel(const int8_t* const* query_ptrs, const uint8_t* panel0,
                                     const uint8_t* panel1, const uint8_t* panel2,
                                     const uint8_t* panel3, size_t total_tiles, __m512i* acc0,
                                     __m512i* acc1, __m512i* acc2, __m512i* acc3) {
  constexpr size_t N = kVnniPoints * 4;
  for (size_t q = 0; q < Mq; ++q) {
    acc0[q] = _mm512_setzero_si512();
    acc1[q] = _mm512_setzero_si512();
    acc2[q] = _mm512_setzero_si512();
    acc3[q] = _mm512_setzero_si512();
  }

  for (size_t t = 0; t < total_tiles; ++t) {
    const __m512i b0 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(panel0 + t * N));
    const __m512i b1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(panel1 + t * N));
    const __m512i b2 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(panel2 + t * N));
    const __m512i b3 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(panel3 + t * N));
    for (size_t q = 0; q < Mq; ++q) {
      const __m512i qv = _mm512_set1_epi32(reinterpret_cast<const int32_t*>(query_ptrs[q])[t]);
      acc0[q] = tq_dpbusd(acc0[q], b0, qv);
      acc1[q] = tq_dpbusd(acc1[q], b1, qv);
      acc2[q] = tq_dpbusd(acc2[q], b2, qv);
      acc3[q] = tq_dpbusd(acc3[q], b3, qv);
    }
  }
}

template<bool Metric>
inline void vnni_chamfer_epilogue(__m512i acc, int32_t byte_sum, float q_nsf, float q_sqn,
                                  const float* db_norms, const float* db_sqn, __m512& running_min) {
  const __m512i correction = _mm512_set1_epi32(128 * byte_sum);
  acc = _mm512_sub_epi32(acc, correction);

  const __m512 fdot = _mm512_cvtepi32_ps(acc);
  const __m512 db_norm_v = _mm512_loadu_ps(db_norms);
  const __m512 q_nsf_v = _mm512_set1_ps(q_nsf);

  __m512 neg_dot = _mm512_mul_ps(fdot, db_norm_v);
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

// Core Chamfer over a single (queries, cloud) pair on PRE-TRANSPOSED panels.
// Returns sum_{q in queries} min_{b in cloud} dist(q, b). Caller divides by
// num_queries if a true mean is needed.
template<bool Metric>
inline float chamfer_panels(const int8_t* q_flat_data, const float* q_norms, const float* q_sqns,
                            const int32_t* q_bsums, size_t q_stride, size_t num_queries,
                            const uint8_t* panel_data, const float* padded_norms,
                            const float* padded_sqn, size_t total_tiles, size_t panel_bytes,
                            size_t n_panels) {
  float total_chamfer = 0.0f;
  size_t qi = 0;

  for (; qi + kVnniMq <= num_queries; qi += kVnniMq) {
    const int8_t* q_batch[kVnniMq];
    for (size_t q = 0; q < kVnniMq; ++q)
      q_batch[q] = q_flat_data + (qi + q) * q_stride;

    __m512 mins[kVnniMq];
    for (size_t q = 0; q < kVnniMq; ++q)
      mins[q] = _mm512_set1_ps(std::numeric_limits<float>::max());

    size_t p = 0;
    for (; p + 4 <= n_panels; p += 4) {
      __m512i a0[kVnniMq], a1[kVnniMq], a2[kVnniMq], a3[kVnniMq];
      vnni_micro_kernel_4panel<kVnniMq>(
          q_batch, panel_data + p * panel_bytes, panel_data + (p + 1) * panel_bytes,
          panel_data + (p + 2) * panel_bytes, panel_data + (p + 3) * panel_bytes,
          total_tiles, a0, a1, a2, a3);

      for (size_t q = 0; q < kVnniMq; ++q) {
        const size_t qg = qi + q;
        vnni_chamfer_epilogue<Metric>(a0[q], q_bsums[qg], q_norms[qg], q_sqns[qg],
                                      padded_norms + p * kVnniPoints,
                                      padded_sqn + p * kVnniPoints, mins[q]);
        vnni_chamfer_epilogue<Metric>(a1[q], q_bsums[qg], q_norms[qg], q_sqns[qg],
                                      padded_norms + (p + 1) * kVnniPoints,
                                      padded_sqn + (p + 1) * kVnniPoints, mins[q]);
        vnni_chamfer_epilogue<Metric>(a2[q], q_bsums[qg], q_norms[qg], q_sqns[qg],
                                      padded_norms + (p + 2) * kVnniPoints,
                                      padded_sqn + (p + 2) * kVnniPoints, mins[q]);
        vnni_chamfer_epilogue<Metric>(a3[q], q_bsums[qg], q_norms[qg], q_sqns[qg],
                                      padded_norms + (p + 3) * kVnniPoints,
                                      padded_sqn + (p + 3) * kVnniPoints, mins[q]);
      }
    }
    for (; p + 2 <= n_panels; p += 2) {
      __m512i ac0[kVnniMq], ac1[kVnniMq];
      vnni_micro_kernel_2panel<kVnniMq>(q_batch, panel_data + p * panel_bytes,
                                        panel_data + (p + 1) * panel_bytes, total_tiles, ac0, ac1);
      for (size_t q = 0; q < kVnniMq; ++q) {
        const size_t qg = qi + q;
        vnni_chamfer_epilogue<Metric>(ac0[q], q_bsums[qg], q_norms[qg], q_sqns[qg],
                                      padded_norms + p * kVnniPoints,
                                      padded_sqn + p * kVnniPoints, mins[q]);
        vnni_chamfer_epilogue<Metric>(ac1[q], q_bsums[qg], q_norms[qg], q_sqns[qg],
                                      padded_norms + (p + 1) * kVnniPoints,
                                      padded_sqn + (p + 1) * kVnniPoints, mins[q]);
      }
    }
    for (; p < n_panels; ++p) {
      __m512i acc[kVnniMq];
      vnni_micro_kernel_1panel<kVnniMq>(q_batch, panel_data + p * panel_bytes, total_tiles, acc);
      for (size_t q = 0; q < kVnniMq; ++q) {
        const size_t qg = qi + q;
        vnni_chamfer_epilogue<Metric>(acc[q], q_bsums[qg], q_norms[qg], q_sqns[qg],
                                      padded_norms + p * kVnniPoints,
                                      padded_sqn + p * kVnniPoints, mins[q]);
      }
    }

    for (size_t q = 0; q < kVnniMq; ++q)
      total_chamfer += _mm512_reduce_min_ps(mins[q]);
  }

  // Tail: remaining queries (< kVnniMq).
  for (; qi < num_queries; ++qi) {
    const int8_t* qp = q_flat_data + qi * q_stride;
    __m512 running_min = _mm512_set1_ps(std::numeric_limits<float>::max());

    for (size_t p = 0; p < n_panels; ++p) {
      __m512i acc;
      vnni_micro_kernel_1panel<1>(&qp, panel_data + p * panel_bytes, total_tiles, &acc);
      vnni_chamfer_epilogue<Metric>(acc, q_bsums[qi], q_norms[qi], q_sqns[qi],
                                    padded_norms + p * kVnniPoints, padded_sqn + p * kVnniPoints,
                                    running_min);
    }
    total_chamfer += _mm512_reduce_min_ps(running_min);
  }

  return total_chamfer;
}

// Multi-query variant: writes per-query min distance into out_dists. Used by
// the many-to-many fused path so the caller can aggregate min distances per
// source cloud.
template<bool Metric>
inline void chamfer_panels_multi(const int8_t* q_flat_data, const float* q_norms,
                                 const float* q_sqns, const int32_t* q_bsums, size_t q_stride,
                                 size_t num_queries, const uint8_t* panel_data,
                                 const float* padded_norms, const float* padded_sqn,
                                 size_t total_tiles, size_t panel_bytes, size_t n_panels,
                                 float* out_dists) {
  size_t qi = 0;
  for (; qi + kVnniMq <= num_queries; qi += kVnniMq) {
    const int8_t* q_batch[kVnniMq];
    for (size_t q = 0; q < kVnniMq; ++q)
      q_batch[q] = q_flat_data + (qi + q) * q_stride;

    __m512 mins[kVnniMq];
    for (size_t q = 0; q < kVnniMq; ++q)
      mins[q] = _mm512_set1_ps(std::numeric_limits<float>::max());

    size_t p = 0;
    for (; p + 4 <= n_panels; p += 4) {
      __m512i a0[kVnniMq], a1[kVnniMq], a2[kVnniMq], a3[kVnniMq];
      vnni_micro_kernel_4panel<kVnniMq>(
          q_batch, panel_data + p * panel_bytes, panel_data + (p + 1) * panel_bytes,
          panel_data + (p + 2) * panel_bytes, panel_data + (p + 3) * panel_bytes,
          total_tiles, a0, a1, a2, a3);
      for (size_t q = 0; q < kVnniMq; ++q) {
        const size_t qg = qi + q;
        vnni_chamfer_epilogue<Metric>(a0[q], q_bsums[qg], q_norms[qg], q_sqns[qg],
                                      padded_norms + p * kVnniPoints,
                                      padded_sqn + p * kVnniPoints, mins[q]);
        vnni_chamfer_epilogue<Metric>(a1[q], q_bsums[qg], q_norms[qg], q_sqns[qg],
                                      padded_norms + (p + 1) * kVnniPoints,
                                      padded_sqn + (p + 1) * kVnniPoints, mins[q]);
        vnni_chamfer_epilogue<Metric>(a2[q], q_bsums[qg], q_norms[qg], q_sqns[qg],
                                      padded_norms + (p + 2) * kVnniPoints,
                                      padded_sqn + (p + 2) * kVnniPoints, mins[q]);
        vnni_chamfer_epilogue<Metric>(a3[q], q_bsums[qg], q_norms[qg], q_sqns[qg],
                                      padded_norms + (p + 3) * kVnniPoints,
                                      padded_sqn + (p + 3) * kVnniPoints, mins[q]);
      }
    }
    for (; p + 2 <= n_panels; p += 2) {
      __m512i ac0[kVnniMq], ac1[kVnniMq];
      vnni_micro_kernel_2panel<kVnniMq>(q_batch, panel_data + p * panel_bytes,
                                        panel_data + (p + 1) * panel_bytes, total_tiles, ac0, ac1);
      for (size_t q = 0; q < kVnniMq; ++q) {
        const size_t qg = qi + q;
        vnni_chamfer_epilogue<Metric>(ac0[q], q_bsums[qg], q_norms[qg], q_sqns[qg],
                                      padded_norms + p * kVnniPoints,
                                      padded_sqn + p * kVnniPoints, mins[q]);
        vnni_chamfer_epilogue<Metric>(ac1[q], q_bsums[qg], q_norms[qg], q_sqns[qg],
                                      padded_norms + (p + 1) * kVnniPoints,
                                      padded_sqn + (p + 1) * kVnniPoints, mins[q]);
      }
    }
    for (; p < n_panels; ++p) {
      __m512i acc[kVnniMq];
      vnni_micro_kernel_1panel<kVnniMq>(q_batch, panel_data + p * panel_bytes, total_tiles, acc);
      for (size_t q = 0; q < kVnniMq; ++q) {
        const size_t qg = qi + q;
        vnni_chamfer_epilogue<Metric>(acc[q], q_bsums[qg], q_norms[qg], q_sqns[qg],
                                      padded_norms + p * kVnniPoints,
                                      padded_sqn + p * kVnniPoints, mins[q]);
      }
    }

    for (size_t q = 0; q < kVnniMq; ++q)
      out_dists[qi + q] = _mm512_reduce_min_ps(mins[q]);
  }

  // Tail
  for (; qi < num_queries; ++qi) {
    const int8_t* qp = q_flat_data + qi * q_stride;
    __m512 running_min = _mm512_set1_ps(std::numeric_limits<float>::max());
    for (size_t p = 0; p < n_panels; ++p) {
      __m512i acc;
      vnni_micro_kernel_1panel<1>(&qp, panel_data + p * panel_bytes, total_tiles, &acc);
      vnni_chamfer_epilogue<Metric>(acc, q_bsums[qi], q_norms[qi], q_sqns[qi],
                                    padded_norms + p * kVnniPoints, padded_sqn + p * kVnniPoints,
                                    running_min);
    }
    out_dists[qi] = _mm512_reduce_min_ps(running_min);
  }
}
#endif  // __AVX512F__

// =========================================================================
// Scalar fallback (decodes panels on-the-fly; only used if AVX-512 is absent)
// =========================================================================
template<bool Metric>
inline float chamfer_panels_scalar(const int8_t* q_flat_data, const float* q_norms,
                                   const float* q_sqns, const int32_t* q_bsums, size_t q_stride,
                                   size_t num_queries, const uint8_t* panel_data,
                                   const float* padded_norms, const float* padded_sqn,
                                   size_t total_tiles, size_t panel_bytes, size_t n_panels,
                                   size_t cloud_size) {
  constexpr size_t N = kVnniPoints * 4;
  float total = 0.0f;
  for (size_t qi = 0; qi < num_queries; ++qi) {
    const int8_t* qp = q_flat_data + qi * q_stride;
    float min_d = std::numeric_limits<float>::max();
    for (size_t i = 0; i < cloud_size; ++i) {
      const size_t panel = i / kVnniPoints;
      const size_t lane = i % kVnniPoints;
      const uint8_t* base = panel_data + panel * panel_bytes;
      int32_t dot = 0;
      for (size_t t = 0; t < total_tiles; ++t) {
        const uint8_t* tile = base + t * N + lane * 4;
        for (size_t k = 0; k < 4; ++k) {
          dot += static_cast<int32_t>(tile[k]) * static_cast<int32_t>(qp[t * 4 + k]);
        }
      }
      dot -= 128 * q_bsums[qi];
      const float neg_dot =
          -static_cast<float>(dot) * padded_norms[panel * kVnniPoints + lane] * q_norms[qi];
      float d;
      if constexpr (Metric) {
        d = padded_sqn[panel * kVnniPoints + lane] + 2.0f * neg_dot + q_sqns[qi];
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
float turboquant_mv_chamfer_distance(const Quantized_Query_Point_Cloud<Metric>& q,
                                     const Quantized_Point_Cloud_Set<Metric>& db, size_t cloud_idx);

// =========================================================================
// Cloud handle: refers to one cloud (by index) within a set
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
  bool same_as(const Query&) const {
    return false;
  }
};

// =========================================================================
// Multi-Vector Query
// =========================================================================
template<bool Metric>
class Quantized_Query_Point_Cloud {
 public:
  using distanceType = float;

  size_t num_queries = 0;
  size_t q_stride = 0;  // Padded dimension required by GEMM (multiple of 4)

  // Flat, contiguous memory arrays for maximum cache locality during GEMM
  std::vector<int8_t> flat_query_data;           // Size: num_queries * q_stride
  std::vector<float> norm_scaling_factors;       // Size: num_queries
  std::vector<float> unquantized_squared_norms;  // Size: num_queries
  std::vector<int32_t> byte_sums;                // Size: num_queries

  Quantized_Query_Point_Cloud() = default;

  inline const int8_t* get_q_ptr(size_t i) const { return flat_query_data.data() + i * q_stride; }

  template<typename CloudView>
  float distance(const CloudView& cloud) const {
    return turboquant_mv_chamfer_distance(*this, *cloud.db, cloud.cloud_idx);
  }

  template<typename CloudHandle>
  std::pair<float, size_t> distance_w_cmps(const CloudHandle& cloud) const {
    const size_t bytes_per_vec = cloud.db->num_bytes_per_datapoint + sizeof(float) +
                                 (Metric ? sizeof(float) : 0);
    return {this->distance(cloud), cloud.size() * bytes_per_vec};
  }

  static constexpr bool is_metric() { return Metric; }
};

// =========================================================================
// Multi-Vector Set (Pre-Transposed Panel Layout)
// =========================================================================
template<bool Metric>
class Quantized_Point_Cloud_Set {
 public:
  // Encoder geometry (same for every cloud)
  size_t num_bytes_per_datapoint = 0;
  size_t total_tiles = 0;  // = ceil(2*num_bytes / 4) -- one VPDPBUSD's worth per tile
  size_t panel_bytes = 0;  // = total_tiles * (kVnniPoints * 4)

  // Concatenated, pre-transposed panel data. Each cloud occupies
  // n_panels(c) * panel_bytes bytes starting at panel_offsets[c].
  parlay::sequence<uint8_t> panel_data;
  parlay::sequence<size_t> panel_offsets;  // size: n_clouds + 1, in BYTES

  // Per-point arrays, padded so each cloud is a multiple of kVnniPoints.
  // point_offsets is in unpadded? -> NO, point_offsets is the padded offset
  // such that norms[point_offsets[c] + lane] gives the lane'th point of
  // cloud c. Padding lanes carry sentinel values that disqualify them from
  // the min (norms = 0, sqns = +inf).
  parlay::sequence<size_t> point_offsets;  // size: n_clouds + 1, in points
  parlay::sequence<float> norm_scaling_factors;
  parlay::sequence<float> unquantized_squared_norms;

  // Unpadded sizes per cloud (used to bound the on-the-fly scalar fallback
  // and reported via Quantized_Point_Cloud::size()).
  parlay::sequence<size_t> cloud_sizes;
  parlay::sequence<uint32_t> ids;

  // Backwards-compat alias used by ManyToMany scaffolding.
  // offsets[c] == point_offsets[c]; kept so external code that probes the
  // padded layout still works.
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

    auto score_one = [&](size_t c) {
      if (cloud_sizes[c] == 0) {
        results[c] = {get_id(c), std::numeric_limits<float>::max()};
        return;
      }
      const float d = turboquant_mv_chamfer_distance(q, *this, c);
      results[c] = {get_id(c), d};
    };
    // For small nc (greedy beam-search nodes typically have ~16 children),
    // the parlay::parallel_for scheduling overhead dwarfs the inner work and
    // the call already runs inside an outer parallel_for over queries.
    if (nc <= 64) {
      for (size_t c = 0; c < nc; ++c) score_one(c);
    } else {
      parlay::parallel_for(0, nc, score_one);
    }
  }

  void save(std::ofstream& out) const {
    out.write(reinterpret_cast<const char*>(&num_bytes_per_datapoint),
              sizeof(num_bytes_per_datapoint));
    out.write(reinterpret_cast<const char*>(&total_tiles), sizeof(total_tiles));
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
    in.read(reinterpret_cast<char*>(&total_tiles), sizeof(total_tiles));
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

// Quantized_Point_Cloud::size needs the full Set definition.
template<bool Metric>
inline size_t Quantized_Point_Cloud<Metric>::size() const {
  return db ? db->cloud_sizes[cloud_idx] : 0;
}

template<bool Metric>
float turboquant_mv_chamfer_distance(const Quantized_Query_Point_Cloud<Metric>& q,
                                     const Quantized_Point_Cloud_Set<Metric>& db, size_t c) {
  const size_t num_q = q.num_queries;
  if (num_q == 0) return 0.0f;

  const size_t cs = db.cloud_sizes[c];
  if (cs == 0) return std::numeric_limits<float>::max();

  const size_t panel_byte_off = db.panel_offsets[c];
  const size_t pt_off = db.point_offsets[c];
  const size_t np = (db.panel_offsets[c + 1] - panel_byte_off) / db.panel_bytes;
  const uint8_t* panel_ptr = db.panel_data.data() + panel_byte_off;
  const float* ns = db.norm_scaling_factors.data() + pt_off;
  const float* sqn = db.unquantized_squared_norms.data() + pt_off;

  float dist_sum = 0.0f;
#ifdef __AVX512F__
  dist_sum = internal::chamfer_panels<Metric>(
      q.flat_query_data.data(), q.norm_scaling_factors.data(), q.unquantized_squared_norms.data(),
      q.byte_sums.data(), q.q_stride, num_q, panel_ptr, ns, sqn, db.total_tiles, db.panel_bytes,
      np);
#else
  dist_sum = internal::chamfer_panels_scalar<Metric>(
      q.flat_query_data.data(), q.norm_scaling_factors.data(), q.unquantized_squared_norms.data(),
      q.byte_sums.data(), q.q_stride, num_q, panel_ptr, ns, sqn, db.total_tiles, db.panel_bytes,
      np, cs);
#endif
  return dist_sum / static_cast<float>(num_q);
}

// =========================================================================
// Multi-Vector Model
// =========================================================================
template<bool Metric>
class Model {
 public:
  turboquant::BaseEncoder encoder;

  Model() = default;

  template<typename PCSet>
  void train(const PCSet& pcs) {
    encoder.train(pcs.get_dims());
  }

  template<typename PCSet>
  Quantized_Point_Cloud_Set<Metric> encode(const PCSet& pcs) const {
    Quantized_Point_Cloud_Set<Metric> enc;
    enc.num_bytes_per_datapoint = encoder.num_bytes_per_datapoint;

    const size_t decoded_dim = 2 * enc.num_bytes_per_datapoint;
    const size_t padded_dim = (decoded_dim + 3) & ~3;
    enc.total_tiles = padded_dim / 4;
    constexpr size_t N = internal::kVnniPoints * 4;
    enc.panel_bytes = enc.total_tiles * N;

    auto float_offsets = pcs.get_offsets();
    const size_t n_clouds = float_offsets.size() > 0 ? float_offsets.size() - 1 : 0;

    enc.panel_offsets.resize(n_clouds + 1);
    enc.point_offsets.resize(n_clouds + 1);
    enc.cloud_sizes.resize(n_clouds);

    size_t cur_panel_bytes = 0;
    size_t cur_padded_pts = 0;
    for (size_t c = 0; c < n_clouds; ++c) {
      const size_t n_vecs = (float_offsets[c + 1] - float_offsets[c]) / encoder.dim;
      const size_t np = (n_vecs + internal::kVnniPoints - 1) / internal::kVnniPoints;
      enc.panel_offsets[c] = cur_panel_bytes;
      enc.point_offsets[c] = cur_padded_pts;
      enc.cloud_sizes[c] = n_vecs;
      cur_panel_bytes += np * enc.panel_bytes;
      cur_padded_pts += np * internal::kVnniPoints;
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

      const size_t np = (n_vecs + internal::kVnniPoints - 1) / internal::kVnniPoints;
      const size_t n_strips = (n_vecs + 63) / 64;  // 64 points per strip
      const size_t strip_bytes = enc.num_bytes_per_datapoint * 64;

      // Per-cloud temporary strip buffer (lives only inside this lambda).
      // Layout: strip_buf[strip * strip_bytes + j * 64 + lane] = byte j of point lane.
      std::vector<uint8_t> strip_buf(n_strips * strip_bytes, 0);
      std::vector<uint8_t> p_codes(enc.num_bytes_per_datapoint);
      std::vector<float> ws(encoder.padded_dim);

      const size_t src_start = float_offsets[c] / encoder.dim;
      const size_t pt_off = enc.point_offsets[c];

      for (size_t i = 0; i < n_vecs; ++i) {
        const float* p =
            reinterpret_cast<const float*>(pcs.data() + (src_start + i) * encoder.dim);
        auto [sqn, nsf] = encoder.encode_single(p, p_codes.data(), ws);

        enc.norm_scaling_factors[pt_off + i] = nsf;
        enc.unquantized_squared_norms[pt_off + i] = sqn;

        const size_t strip = i / 64;
        const size_t lane = i % 64;
        for (size_t j = 0; j < enc.num_bytes_per_datapoint; ++j) {
          strip_buf[strip * strip_bytes + j * 64 + lane] = p_codes[j];
        }
      }

      // Pre-transpose: walk panels of 16 points each, decode each one into the
      // permanent panel_data buffer. All panels live within a single strip
      // because each strip holds 64 points (4 panels per strip).
      uint8_t* cloud_panels = enc.panel_data.data() + enc.panel_offsets[c];
      for (size_t p = 0; p < np; ++p) {
        const size_t abs = p * internal::kVnniPoints;
        const size_t strip = abs / 64;
        const size_t lane = abs % 64;
        internal::decode_strip_to_panel_simd(strip_buf.data() + strip * strip_bytes, lane,
                                             enc.num_bytes_per_datapoint, enc.total_tiles,
                                             cloud_panels + p * enc.panel_bytes);
      }
    });

    return enc;
  }

  template<typename ChPoint>
  Quantized_Query_Point_Cloud<Metric> quantize_query(const ChPoint& query_cloud) const {
    Quantized_Query_Point_Cloud<Metric> res;
    res.num_queries = query_cloud.size();
    if (res.num_queries == 0) return res;

    const size_t decoded_dim = 2 * encoder.num_bytes_per_datapoint;
    res.q_stride = (decoded_dim + 3) & ~3;

    res.flat_query_data.resize(res.num_queries * res.q_stride, 0);
    res.norm_scaling_factors.resize(res.num_queries, 0.0f);
    res.unquantized_squared_norms.resize(res.num_queries, 0.0f);
    res.byte_sums.resize(res.num_queries, 0);

    const float* base_ptr = query_cloud.data();

    for (size_t qi = 0; qi < res.num_queries; ++qi) {
      std::vector<float> q_rot(encoder.padded_dim);
      encoder.rotator->rotate(base_ptr + qi * encoder.dim, q_rot.data());

      float sqr_norm = 0.0f;
      for (size_t i = 0; i < encoder.padded_dim; ++i)
        sqr_norm += q_rot[i] * q_rot[i];

      if (sqr_norm == 0.0f || !std::isfinite(sqr_norm)) continue;

      const float norm = std::sqrt(sqr_norm);
      const float scale = std::sqrt(static_cast<float>(encoder.padded_dim)) / norm;

      float max_value = 0.0f;
      for (size_t i = 0; i < encoder.padded_dim; ++i) {
        q_rot[i] = std::clamp(q_rot[i] * scale, -turboquant::internal::kValueCap,
                              turboquant::internal::kValueCap);
        max_value = std::max(max_value, std::abs(q_rot[i]));
      }

      const float sf = max_value > 0.0f ? 127.0f / max_value : 0.0f;

      int quant_norm = 0;
      int32_t byte_sum = 0;
      int8_t* q_out_ptr = res.flat_query_data.data() + qi * res.q_stride;

      for (size_t i = 0; i < encoder.padded_dim; ++i) {
        int8_t snapped =
            static_cast<int8_t>(std::clamp(std::round(q_rot[i] * sf), -127.0f, 127.0f));
        q_out_ptr[i] = snapped;
        quant_norm += snapped * snapped;
        byte_sum += snapped;
      }

      res.norm_scaling_factors[qi] = norm / std::sqrt(static_cast<float>(quant_norm));
      res.unquantized_squared_norms[qi] = sqr_norm;
      res.byte_sums[qi] = byte_sum;
    }

    return res;
  }

  void save(std::ofstream& out) const { encoder.save(out); }
  void load(std::ifstream& in) { encoder.load(in); }
};

// ------------------------------------------------------------------
// Pre-Fused Query Batch
//
// Packs many `Quantized_Query_Point_Cloud<Metric>` into one flat layout
// (flat_query_data + per-embedding norms / sqns / bsums + per-source
// emb_offsets) so the GEMM kernel can score the whole batch against any
// number of DB clouds without paying the per-call memcpy fusion cost the
// older `ManyToMany::TopKIntoUninitialized` paid.
//
// Rule of thumb: build this ONCE per (set-of-queries, batch-of-leaves) and
// reuse it across every leaf you score. Building it is parallelized over
// source clouds.
// ------------------------------------------------------------------
template<bool Metric>
struct FusedQueryBatch {
  size_t num_source_clouds = 0;
  size_t total_embeddings = 0;
  size_t q_stride = 0;

  std::vector<int8_t> flat_query_data;           // total_embeddings * q_stride
  std::vector<float> norm_scaling_factors;       // total_embeddings
  std::vector<float> unquantized_squared_norms;  // total_embeddings
  std::vector<int32_t> byte_sums;                // total_embeddings
  std::vector<size_t> emb_offsets;               // num_source_clouds + 1

  void Build(const std::vector<const Quantized_Query_Point_Cloud<Metric>*>& A) {
    num_source_clouds = A.size();
    emb_offsets.assign(num_source_clouds + 1, 0);
    if (num_source_clouds == 0) {
      total_embeddings = 0;
      q_stride = 0;
      return;
    }
    q_stride = A[0]->q_stride;
    for (size_t i = 0; i < num_source_clouds; ++i) {
      emb_offsets[i + 1] = emb_offsets[i] + A[i]->num_queries;
    }
    total_embeddings = emb_offsets[num_source_clouds];

    flat_query_data.resize(total_embeddings * q_stride);
    norm_scaling_factors.resize(total_embeddings);
    unquantized_squared_norms.resize(total_embeddings);
    byte_sums.resize(total_embeddings);

    parlay::parallel_for(0, num_source_clouds, [&](size_t i) {
      const auto* qc = A[i];
      const size_t off = emb_offsets[i];
      const size_t cnt = qc->num_queries;
      if (cnt == 0) return;
      std::memcpy(flat_query_data.data() + off * q_stride, qc->flat_query_data.data(),
                  cnt * q_stride);
      std::memcpy(norm_scaling_factors.data() + off, qc->norm_scaling_factors.data(),
                  cnt * sizeof(float));
      std::memcpy(unquantized_squared_norms.data() + off, qc->unquantized_squared_norms.data(),
                  cnt * sizeof(float));
      std::memcpy(byte_sums.data() + off, qc->byte_sums.data(), cnt * sizeof(int32_t));
    });
  }
};

// ------------------------------------------------------------------
// Fused-query Chamfer scoring against one Quantized_Point_Cloud_Set
//
// Computes per-(source-cloud, db-cloud) Chamfer distance for every source
// cloud in `fq` against every DB cloud in `db`, writing into
//   out[i * db.num_clouds() + c] = {db.get_id(c), chamfer_dist}.
//
// Parallelizes over DB clouds via parlay so each call uses many cores when
// invoked from a per-leaf serial loop. Composes cleanly with an outer
// parallel_for over leaves: parlay's nested scheduler handles it.
//
// `db_workspaces` may be nullptr; if non-null it must point to at least
// `parlay::num_workers()` `std::vector<float>` slots, used as reusable
// per-thread scratch for the per-DB-cloud per-embedding minima. Passing one
// in eliminates the per-DB-cloud allocation; passing nullptr falls back to
// per-call allocation (still correct, just slightly slower).
// ------------------------------------------------------------------
template<bool Metric>
inline void chamfer_score_all_fused(const FusedQueryBatch<Metric>& fq,
                                    const Quantized_Point_Cloud_Set<Metric>& db,
                                    std::pair<uint32_t, float>* out,
                                    std::vector<std::vector<float>>* db_workspaces = nullptr,
                                    bool parallel_db = true) {
  const size_t num_src = fq.num_source_clouds;
  const size_t num_db = db.num_clouds();
  if (num_src == 0 || num_db == 0) return;

  auto process_one_db = [&](size_t c) {
    const size_t cs = db.cloud_sizes[c];
    const uint32_t id = db.get_id(c);
    if (cs == 0) {
      const float bad = std::numeric_limits<float>::max();
      for (size_t i = 0; i < num_src; ++i) out[i * num_db + c] = {id, bad};
      return;
    }

    const size_t panel_byte_off = db.panel_offsets[c];
    const size_t pt_off = db.point_offsets[c];
    const size_t np = (db.panel_offsets[c + 1] - panel_byte_off) / db.panel_bytes;

    // Per-embedding mins scratch. Reuse a per-worker buffer if provided.
    float* emb_min;
    std::vector<float> local_buf;
    if (db_workspaces != nullptr) {
      auto& ws = (*db_workspaces)[parlay::worker_id()];
      if (ws.size() < fq.total_embeddings) ws.resize(fq.total_embeddings);
      emb_min = ws.data();
    } else {
      local_buf.resize(fq.total_embeddings);
      emb_min = local_buf.data();
    }

#ifdef __AVX512F__
    internal::chamfer_panels_multi<Metric>(
        fq.flat_query_data.data(), fq.norm_scaling_factors.data(),
        fq.unquantized_squared_norms.data(), fq.byte_sums.data(), fq.q_stride,
        fq.total_embeddings, db.panel_data.data() + panel_byte_off,
        db.norm_scaling_factors.data() + pt_off,
        db.unquantized_squared_norms.data() + pt_off, db.total_tiles, db.panel_bytes, np, emb_min);
#else
    for (size_t qi = 0; qi < fq.total_embeddings; ++qi) {
      emb_min[qi] = internal::chamfer_panels_scalar<Metric>(
          fq.flat_query_data.data() + qi * fq.q_stride, fq.norm_scaling_factors.data() + qi,
          fq.unquantized_squared_norms.data() + qi, fq.byte_sums.data() + qi, fq.q_stride, 1,
          db.panel_data.data() + panel_byte_off, db.norm_scaling_factors.data() + pt_off,
          db.unquantized_squared_norms.data() + pt_off, db.total_tiles, db.panel_bytes, np, cs);
    }
#endif

    // Aggregate per source cloud → mean Chamfer.
    for (size_t i = 0; i < num_src; ++i) {
      const size_t e_start = fq.emb_offsets[i];
      const size_t e_end = fq.emb_offsets[i + 1];
      if (e_end == e_start) {
        out[i * num_db + c] = {id, std::numeric_limits<float>::max()};
        continue;
      }
      float dist_sum = 0.0f;
      for (size_t e = e_start; e < e_end; ++e) dist_sum += emb_min[e];
      out[i * num_db + c] = {id, dist_sum / static_cast<float>(e_end - e_start)};
    }
  };

  if (parallel_db) {
    // granularity=0 lets parlay's scheduler choose; empirically this beats a
    // fixed block size on the dense (32 leaves × ~120 db clouds) workload.
    parlay::parallel_for(0, num_db, process_one_db);
  } else {
    for (size_t c = 0; c < num_db; ++c) process_one_db(c);
  }
}

// ------------------------------------------------------------------
// ManyToMany Batch Operator
// ------------------------------------------------------------------
template<typename PCS>
class ManyToMany {
 public:
  static void TopKIntoUninitialized(
      const std::vector<const Quantized_Query_Point_Cloud<PCS::is_metric()>*>& A, const PCS& B,
      uint32_t k, std::pair<uint32_t, float>* results, size_t q_block = 16,
      bool parallel_query_blocks = true) {

    const size_t num_q_clouds = A.size();
    const size_t num_db_clouds = B.num_clouds();
    if (num_q_clouds == 0 || num_db_clouds == 0 || k == 0) return;
    if (q_block == 0) q_block = 1;

    auto process_query_range = [&](size_t q_start, size_t q_end) {
      const size_t q_count = q_end - q_start;
      if (q_count == 0) return;

      std::vector<std::priority_queue<std::pair<float, uint32_t>>> heaps(q_count);

      // Fuse all per-source-cloud queries into one flat array so the GEMM can
      // saturate kernel utilization on every DB-cloud visit.
      size_t total_embeddings = 0;
      std::vector<size_t> emb_offsets(q_count + 1, 0);
      for (size_t i = 0; i < q_count; ++i) {
        total_embeddings += A[q_start + i]->num_queries;
        emb_offsets[i + 1] = total_embeddings;
      }
      const size_t q_stride = A[0]->q_stride;

      std::vector<int8_t> fused_q_data(total_embeddings * q_stride);
      std::vector<float> fused_q_norms(total_embeddings);
      std::vector<float> fused_q_sqns(total_embeddings);
      std::vector<int32_t> fused_q_bsums(total_embeddings);

      for (size_t i = 0; i < q_count; ++i) {
        const auto* qc = A[q_start + i];
        const size_t off = emb_offsets[i];
        const size_t cnt = qc->num_queries;
        std::memcpy(fused_q_data.data() + off * q_stride, qc->flat_query_data.data(),
                    cnt * q_stride);
        std::memcpy(fused_q_norms.data() + off, qc->norm_scaling_factors.data(),
                    cnt * sizeof(float));
        std::memcpy(fused_q_sqns.data() + off, qc->unquantized_squared_norms.data(),
                    cnt * sizeof(float));
        std::memcpy(fused_q_bsums.data() + off, qc->byte_sums.data(), cnt * sizeof(int32_t));
      }

      std::vector<float> emb_min_dists(total_embeddings);

      for (size_t c = 0; c < num_db_clouds; ++c) {
        const size_t cs = B.cloud_sizes[c];
        if (cs == 0) continue;

        const size_t panel_off = B.panel_offsets[c];
        const size_t pt_off = B.point_offsets[c];
        const size_t np = (B.panel_offsets[c + 1] - panel_off) / B.panel_bytes;

#ifdef __AVX512F__
        internal::chamfer_panels_multi<PCS::is_metric()>(
            fused_q_data.data(), fused_q_norms.data(), fused_q_sqns.data(), fused_q_bsums.data(),
            q_stride, total_embeddings, B.panel_data.data() + panel_off,
            B.norm_scaling_factors.data() + pt_off, B.unquantized_squared_norms.data() + pt_off,
            B.total_tiles, B.panel_bytes, np, emb_min_dists.data());
#else
        for (size_t qi = 0; qi < total_embeddings; ++qi) {
          emb_min_dists[qi] = internal::chamfer_panels_scalar<PCS::is_metric()>(
              fused_q_data.data() + qi * q_stride, fused_q_norms.data() + qi,
              fused_q_sqns.data() + qi, fused_q_bsums.data() + qi, q_stride, 1,
              B.panel_data.data() + panel_off, B.norm_scaling_factors.data() + pt_off,
              B.unquantized_squared_norms.data() + pt_off, B.total_tiles, B.panel_bytes, np, cs);
        }
#endif

        // Aggregate min sums into per-source-cloud Chamfer means and update heaps.
        for (size_t i = 0; i < q_count; ++i) {
          float dist_sum = 0.0f;
          const size_t e_start = emb_offsets[i];
          const size_t e_count = emb_offsets[i + 1] - e_start;
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
          results[global_idx * k + (cnt - 1 - ki)] = {heaps[i].top().second, heaps[i].top().first};
          heaps[i].pop();
        }
        for (size_t ki = cnt; ki < k; ++ki) {
          results[global_idx * k + ki] = {0, std::numeric_limits<float>::max()};
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

}  // namespace turboquant_mv
}  // namespace mvsic
