#pragma once

#include <vector>
#include <cstdint>
#include <immintrin.h>
#include <limits>
#include <algorithm>
#include <cstring>

#include "parlay/parallel.h"
#include "parlay/sequence.h"
#include "mvsic/core/quantization/turboquant_utils.h"

namespace mvsic {
namespace turboquant_mv {

// =========================================================================
// SIMD AVX Kernels
// =========================================================================
namespace internal {

// Centroids shifted by +128 for native vpdpbusd (unsigned × signed).
static constexpr std::array<uint8_t, 16> kCentroidsUint8 = {134, 146, 159, 172, 186, 203, 224, 255,
                                                            122, 110, 97,  84,  70,  53,  32,  1};

static constexpr size_t kVnniPoints = 16;  // int32 lanes in __m512i
static constexpr size_t kVnniMq = 8;       // queries per batch (2-panel path)
static constexpr size_t kVnniMq4 = 4;      // queries per batch (4-panel path)

// SIMD bulk decode: 16 points from a strip into one block-transposed panel.
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

  const __m128i neutral = _mm_set1_epi8(static_cast<char>(0x80));
  for (; tile < total_tiles; ++tile) {
    _mm_storeu_si128(reinterpret_cast<__m128i*>(panel + tile * N + 0), neutral);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(panel + tile * N + 16), neutral);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(panel + tile * N + 32), neutral);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(panel + tile * N + 48), neutral);
  }
}

#if defined(__AVX512BW__) && defined(__AVX512VL__)
// Fully SIMD version of gathering crossing points. Replaces the memcpy loop.
inline void decode_crossing_panel_simd(const uint8_t* s0, const uint8_t* s1, size_t base_lane,
                                       size_t num_bytes, size_t total_tiles, uint8_t* panel) {
  const __m128i codebook_u8 =
      _mm_loadu_si128(reinterpret_cast<const __m128i*>(kCentroidsUint8.data()));
  const __m128i mask_0f = _mm_set1_epi8(0x0F);
  const size_t in_this = 64 - base_lane;

  // Masks to stitch Strip 0 and Strip 1 fragments together
  const __mmask16 m_lo = (1U << in_this) - 1;
  const __mmask16 m_hi = ~m_lo;

  size_t tile = 0;
  size_t j = 0;

  // Main loop: Process 2 byte-positions (4 dimensions) at a time
  for (; j + 1 < num_bytes; j += 2) {
    // Stitch bytes for dimension j and j+1
    __m128i p_j = _mm_maskz_loadu_epi8(m_lo, s0 + j * 64 + base_lane);
    p_j = _mm_mask_loadu_epi8(p_j, m_hi, s1 + j * 64 - in_this);

    __m128i p_j1 = _mm_maskz_loadu_epi8(m_lo, s0 + (j + 1) * 64 + base_lane);
    p_j1 = _mm_mask_loadu_epi8(p_j1, m_hi, s1 + (j + 1) * 64 - in_this);

    // Standard VNNI Transpose/Unpack logic
    const __m128i ej = _mm_shuffle_epi8(codebook_u8, _mm_and_si128(p_j, mask_0f));
    const __m128i oj =
        _mm_shuffle_epi8(codebook_u8, _mm_and_si128(_mm_srli_epi16(p_j, 4), mask_0f));
    const __m128i ej1 = _mm_shuffle_epi8(codebook_u8, _mm_and_si128(p_j1, mask_0f));
    const __m128i oj1 =
        _mm_shuffle_epi8(codebook_u8, _mm_and_si128(_mm_srli_epi16(p_j1, 4), mask_0f));

    const __m128i pair_j = _mm_unpacklo_epi8(ej, oj);
    const __m128i pair_j_hi = _mm_unpackhi_epi8(ej, oj);
    const __m128i pair_j1 = _mm_unpacklo_epi8(ej1, oj1);
    const __m128i pair_j1_hi = _mm_unpackhi_epi8(ej1, oj1);

    _mm_storeu_si128(reinterpret_cast<__m128i*>(panel + tile * 64 + 0),
                     _mm_unpacklo_epi16(pair_j, pair_j1));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(panel + tile * 64 + 16),
                     _mm_unpackhi_epi16(pair_j, pair_j1));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(panel + tile * 64 + 32),
                     _mm_unpacklo_epi16(pair_j_hi, pair_j1_hi));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(panel + tile * 64 + 48),
                     _mm_unpackhi_epi16(pair_j_hi, pair_j1_hi));
    ++tile;
  }

  // Handle tail byte-position
  if (j < num_bytes) {
    __m128i p_j = _mm_maskz_loadu_epi8(m_lo, s0 + j * 64 + base_lane);
    p_j = _mm_mask_loadu_epi8(p_j, m_hi, s1 + j * 64 - in_this);

    const __m128i ej = _mm_shuffle_epi8(codebook_u8, _mm_and_si128(p_j, mask_0f));
    const __m128i oj =
        _mm_shuffle_epi8(codebook_u8, _mm_and_si128(_mm_srli_epi16(p_j, 4), mask_0f));
    const __m128i zeros = _mm_setzero_si128();

    const __m128i pair_j = _mm_unpacklo_epi8(ej, oj);
    const __m128i pair_j_hi = _mm_unpackhi_epi8(ej, oj);

    _mm_storeu_si128(reinterpret_cast<__m128i*>(panel + tile * 64 + 0),
                     _mm_unpacklo_epi16(pair_j, zeros));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(panel + tile * 64 + 16),
                     _mm_unpackhi_epi16(pair_j, zeros));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(panel + tile * 64 + 32),
                     _mm_unpacklo_epi16(pair_j_hi, zeros));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(panel + tile * 64 + 48),
                     _mm_unpackhi_epi16(pair_j_hi, zeros));
    ++tile;
  }

  // Pad remaining tiles with neutral value (-128)
  const __m128i neutral = _mm_set1_epi8(static_cast<char>(0x80));
  for (; tile < total_tiles; ++tile) {
    for (int k = 0; k < 4; ++k)
      _mm_storeu_si128(reinterpret_cast<__m128i*>(panel + tile * 64 + k * 16), neutral);
  }
}
#endif

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
    const __m512i b = _mm512_load_si512(reinterpret_cast<const __m512i*>(panel + t * N));
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
    const __m512i b0 = _mm512_load_si512(reinterpret_cast<const __m512i*>(panel0 + t * N));
    const __m512i b1 = _mm512_load_si512(reinterpret_cast<const __m512i*>(panel1 + t * N));
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
    const __m512i b0 = _mm512_load_si512(reinterpret_cast<const __m512i*>(panel0 + t * N));
    const __m512i b1 = _mm512_load_si512(reinterpret_cast<const __m512i*>(panel1 + t * N));
    const __m512i b2 = _mm512_load_si512(reinterpret_cast<const __m512i*>(panel2 + t * N));
    const __m512i b3 = _mm512_load_si512(reinterpret_cast<const __m512i*>(panel3 + t * N));
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

template<bool Metric>
inline float chamfer_vnni_gemm(const int8_t* q_flat_data, const float* q_norms, const float* q_sqns,
                               const int32_t* q_bsums, size_t q_stride, size_t num_queries,
                               const uint8_t* strip_data, const float* norms,
                               const float* squared_norms, size_t strip_stride, size_t n_strips,
                               size_t num_bytes_per_point, size_t cloud_size,
                               size_t lane_offset = 0) {
  const size_t decoded_dim = 2 * num_bytes_per_point;
  const size_t padded_dim = (decoded_dim + 3) & ~3;
  const size_t total_tiles = padded_dim / 4;
  constexpr size_t N = kVnniPoints * 4;
  const size_t panel_bytes = total_tiles * N;
  const size_t n_panels = (cloud_size + kVnniPoints - 1) / kVnniPoints;

  // Step 0 (Decode Queries): Already done during quantize_query using optimized flat arrays
  // directly!

  // Step 1: Decode DB cloud to block-transposed panels
  thread_local std::vector<uint8_t> panels;
  panels.resize(n_panels * panel_bytes + 64);
  uint8_t* panels_aligned =
      reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(panels.data()) + 63) & ~63);
  thread_local std::vector<uint8_t> gather_buf;

  for (size_t p = 0; p < n_panels; ++p) {
    const size_t abs_point = lane_offset + p * kVnniPoints;
    const size_t strip = abs_point / 64;
    const size_t base_lane = abs_point % 64;

    if (base_lane + kVnniPoints <= 64) {
      decode_strip_to_panel_simd(strip_data + strip * strip_stride, base_lane, num_bytes_per_point,
                                 total_tiles, panels_aligned + p * panel_bytes);
    } else {
#if defined(__AVX512BW__) && defined(__AVX512VL__)
      const uint8_t* s0 = strip_data + strip * strip_stride;
      const uint8_t* s1 = strip_data + (strip + 1) * strip_stride;
      // HIT THE RAW KERNEL: No more gather_buf, no more memcpy overhead.
      decode_crossing_panel_simd(s0, s1, base_lane, num_bytes_per_point, total_tiles,
                                 panels_aligned + p * panel_bytes);
#else
      // Scalar/Memcpy fallback for machines without AVX-512BW
      const size_t in_this = 64 - base_lane;
      const size_t in_next = kVnniPoints - in_this;
      gather_buf.resize(num_bytes_per_point * 64);
      const uint8_t* s0 = strip_data + strip * strip_stride;
      const uint8_t* s1 = strip_data + (strip + 1) * strip_stride;
      for (size_t j = 0; j < num_bytes_per_point; ++j) {
        std::memcpy(gather_buf.data() + j * 64, s0 + j * 64 + base_lane, in_this);
        std::memcpy(gather_buf.data() + j * 64 + in_this, s1 + j * 64, in_next);
      }
      decode_strip_to_panel_simd(gather_buf.data(), 0, num_bytes_per_point, total_tiles,
                                 panels_aligned + p * panel_bytes);
#endif
    }
  }

  // Step 2: Pad norm arrays
  const size_t padded_pts = n_panels * kVnniPoints;
  thread_local std::vector<float> padded_norms;
  thread_local std::vector<float> padded_sqn;
  padded_norms.resize(padded_pts);
  std::memcpy(padded_norms.data(), norms, cloud_size * sizeof(float));
  std::memset(padded_norms.data() + cloud_size, 0, (padded_pts - cloud_size) * sizeof(float));
  if constexpr (Metric) {
    padded_sqn.resize(padded_pts);
    std::memcpy(padded_sqn.data(), squared_norms, cloud_size * sizeof(float));
    std::memset(padded_sqn.data() + cloud_size, 0, (padded_pts - cloud_size) * sizeof(float));
  }

  // Step 3: Score all queries against all panels
  float total_chamfer = 0.0f;
  size_t qi = 0;

  for (; qi + kVnniMq4 <= num_queries; qi += kVnniMq4) {
    const int8_t* q_batch[kVnniMq4];
    for (size_t q = 0; q < kVnniMq4; ++q) {
      q_batch[q] = q_flat_data + (qi + q) * q_stride;
    }

    __m512 mins[kVnniMq4];
    for (size_t q = 0; q < kVnniMq4; ++q)
      mins[q] = _mm512_set1_ps(std::numeric_limits<float>::max());

    size_t p = 0;
    for (; p + 4 <= n_panels; p += 4) {
      __m512i a0[kVnniMq4], a1[kVnniMq4], a2[kVnniMq4], a3[kVnniMq4];
      vnni_micro_kernel_4panel<kVnniMq4>(
          q_batch, panels_aligned + p * panel_bytes, panels_aligned + (p + 1) * panel_bytes,
          panels_aligned + (p + 2) * panel_bytes, panels_aligned + (p + 3) * panel_bytes,
          total_tiles, a0, a1, a2, a3);

      for (size_t q = 0; q < kVnniMq4; ++q) {
        size_t query_idx = qi + q;
        vnni_chamfer_epilogue<Metric>(a0[q], q_bsums[query_idx], q_norms[query_idx],
                                      q_sqns[query_idx], padded_norms.data() + p * kVnniPoints,
                                      padded_sqn.data() + p * kVnniPoints, mins[q]);
        vnni_chamfer_epilogue<Metric>(a1[q], q_bsums[query_idx], q_norms[query_idx],
                                      q_sqns[query_idx],
                                      padded_norms.data() + (p + 1) * kVnniPoints,
                                      padded_sqn.data() + (p + 1) * kVnniPoints, mins[q]);
        vnni_chamfer_epilogue<Metric>(a2[q], q_bsums[query_idx], q_norms[query_idx],
                                      q_sqns[query_idx],
                                      padded_norms.data() + (p + 2) * kVnniPoints,
                                      padded_sqn.data() + (p + 2) * kVnniPoints, mins[q]);
        vnni_chamfer_epilogue<Metric>(a3[q], q_bsums[query_idx], q_norms[query_idx],
                                      q_sqns[query_idx],
                                      padded_norms.data() + (p + 3) * kVnniPoints,
                                      padded_sqn.data() + (p + 3) * kVnniPoints, mins[q]);
      }
    }
    for (; p + 2 <= n_panels; p += 2) {
      __m512i ac0[kVnniMq4], ac1[kVnniMq4];
      vnni_micro_kernel_2panel<kVnniMq4>(q_batch, panels_aligned + p * panel_bytes,
                                         panels_aligned + (p + 1) * panel_bytes, total_tiles, ac0,
                                         ac1);

      for (size_t q = 0; q < kVnniMq4; ++q) {
        size_t query_idx = qi + q;
        vnni_chamfer_epilogue<Metric>(ac0[q], q_bsums[query_idx], q_norms[query_idx],
                                      q_sqns[query_idx], padded_norms.data() + p * kVnniPoints,
                                      padded_sqn.data() + p * kVnniPoints, mins[q]);
        vnni_chamfer_epilogue<Metric>(ac1[q], q_bsums[query_idx], q_norms[query_idx],
                                      q_sqns[query_idx],
                                      padded_norms.data() + (p + 1) * kVnniPoints,
                                      padded_sqn.data() + (p + 1) * kVnniPoints, mins[q]);
      }
    }
    for (; p < n_panels; ++p) {
      __m512i acc[kVnniMq4];
      vnni_micro_kernel_1panel<kVnniMq4>(q_batch, panels_aligned + p * panel_bytes, total_tiles,
                                         acc);

      for (size_t q = 0; q < kVnniMq4; ++q) {
        size_t query_idx = qi + q;
        vnni_chamfer_epilogue<Metric>(acc[q], q_bsums[query_idx], q_norms[query_idx],
                                      q_sqns[query_idx], padded_norms.data() + p * kVnniPoints,
                                      padded_sqn.data() + p * kVnniPoints, mins[q]);
      }
    }

    for (size_t q = 0; q < kVnniMq4; ++q)
      total_chamfer += _mm512_reduce_min_ps(mins[q]);
  }

  // Tail: remaining queries.
  for (; qi < num_queries; ++qi) {
    const int8_t* qp = q_flat_data + qi * q_stride;
    __m512 running_min = _mm512_set1_ps(std::numeric_limits<float>::max());

    for (size_t p = 0; p < n_panels; ++p) {
      __m512i acc;
      vnni_micro_kernel_1panel<1>(&qp, panels_aligned + p * panel_bytes, total_tiles, &acc);
      vnni_chamfer_epilogue<Metric>(acc, q_bsums[qi], q_norms[qi], q_sqns[qi],
                                    padded_norms.data() + p * kVnniPoints,
                                    padded_sqn.data() + p * kVnniPoints, running_min);
    }
    total_chamfer += _mm512_reduce_min_ps(running_min);
  }

  return total_chamfer;
}
#endif  // __AVX512F__

// ------------------------------------------------------------------
// Fused SIMD Kernel: Scores a flat array of embeddings against a DB cloud
// ------------------------------------------------------------------
#ifdef __AVX512F__
template<bool Metric>
inline void chamfer_vnni_gemm_fused(const int8_t* q_flat_data, const float* q_norms,
                                    const float* q_sqns, const int32_t* q_bsums, size_t q_stride,
                                    size_t num_fused_embeddings, const uint8_t* strip_data,
                                    const float* norms, const float* squared_norms,
                                    size_t strip_stride, size_t n_strips,
                                    size_t num_bytes_per_point, size_t cloud_size,
                                    size_t lane_offset, float* out_dists) {

  const size_t decoded_dim = 2 * num_bytes_per_point;
  const size_t padded_dim = (decoded_dim + 3) & ~3;
  const size_t total_tiles = padded_dim / 4;
  constexpr size_t N = kVnniPoints * 4;
  const size_t panel_bytes = total_tiles * N;
  const size_t n_panels = (cloud_size + kVnniPoints - 1) / kVnniPoints;

  // 1. Decode DB cloud to block-transposed panels (Done ONCE per cloud)
  thread_local std::vector<uint8_t> panels;
  panels.resize(n_panels * panel_bytes + 64);
  uint8_t* panels_aligned =
      reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(panels.data()) + 63) & ~63);
  thread_local std::vector<uint8_t> gather_buf;

  for (size_t p = 0; p < n_panels; ++p) {
    const size_t abs_point = lane_offset + p * kVnniPoints;
    const size_t strip = abs_point / 64;
    const size_t base_lane = abs_point % 64;

    if (base_lane + kVnniPoints <= 64) {
      decode_strip_to_panel_simd(strip_data + strip * strip_stride, base_lane, num_bytes_per_point,
                                 total_tiles, panels_aligned + p * panel_bytes);
    } else {
#if defined(__AVX512BW__) && defined(__AVX512VL__)
      const uint8_t* s0 = strip_data + strip * strip_stride;
      const uint8_t* s1 = strip_data + (strip + 1) * strip_stride;
      decode_crossing_panel_simd(s0, s1, base_lane, num_bytes_per_point, total_tiles,
                                 panels_aligned + p * panel_bytes);
#else
      const size_t in_this = 64 - base_lane;
      const size_t in_next = kVnniPoints - in_this;
      gather_buf.resize(num_bytes_per_point * 64);
      const uint8_t* s0 = strip_data + strip * strip_stride;
      const uint8_t* s1 = strip_data + (strip + 1) * strip_stride;
      for (size_t j = 0; j < num_bytes_per_point; ++j) {
        std::memcpy(gather_buf.data() + j * 64, s0 + j * 64 + base_lane, in_this);
        std::memcpy(gather_buf.data() + j * 64 + in_this, s1 + j * 64, in_next);
      }
      decode_strip_to_panel_simd(gather_buf.data(), 0, num_bytes_per_point, total_tiles,
                                 panels_aligned + p * panel_bytes);
#endif
    }
  }

  const size_t padded_pts = n_panels * kVnniPoints;
  thread_local std::vector<float> padded_norms;
  thread_local std::vector<float> padded_sqn;
  padded_norms.resize(padded_pts);
  std::memcpy(padded_norms.data(), norms, cloud_size * sizeof(float));
  std::memset(padded_norms.data() + cloud_size, 0, (padded_pts - cloud_size) * sizeof(float));
  if constexpr (Metric) {
    padded_sqn.resize(padded_pts);
    std::memcpy(padded_sqn.data(), squared_norms, cloud_size * sizeof(float));
    std::memset(padded_sqn.data() + cloud_size, 0, (padded_pts - cloud_size) * sizeof(float));
  }

  // 2. Score ALL fused embeddings against the decoded panels
  size_t qi = 0;
  for (; qi + kVnniMq4 <= num_fused_embeddings; qi += kVnniMq4) {
    const int8_t* q_batch[kVnniMq4];
    for (size_t q = 0; q < kVnniMq4; ++q)
      q_batch[q] = q_flat_data + (qi + q) * q_stride;

    __m512 mins[kVnniMq4];
    for (size_t q = 0; q < kVnniMq4; ++q)
      mins[q] = _mm512_set1_ps(std::numeric_limits<float>::max());

    size_t p = 0;
    for (; p + 4 <= n_panels; p += 4) {
      __m512i a0[kVnniMq4], a1[kVnniMq4], a2[kVnniMq4], a3[kVnniMq4];
      vnni_micro_kernel_4panel<kVnniMq4>(
          q_batch, panels_aligned + p * panel_bytes, panels_aligned + (p + 1) * panel_bytes,
          panels_aligned + (p + 2) * panel_bytes, panels_aligned + (p + 3) * panel_bytes,
          total_tiles, a0, a1, a2, a3);

      for (size_t q = 0; q < kVnniMq4; ++q) {
        size_t q_idx = qi + q;
        vnni_chamfer_epilogue<Metric>(a0[q], q_bsums[q_idx], q_norms[q_idx], q_sqns[q_idx],
                                      padded_norms.data() + p * kVnniPoints,
                                      padded_sqn.data() + p * kVnniPoints, mins[q]);
        vnni_chamfer_epilogue<Metric>(a1[q], q_bsums[q_idx], q_norms[q_idx], q_sqns[q_idx],
                                      padded_norms.data() + (p + 1) * kVnniPoints,
                                      padded_sqn.data() + (p + 1) * kVnniPoints, mins[q]);
        vnni_chamfer_epilogue<Metric>(a2[q], q_bsums[q_idx], q_norms[q_idx], q_sqns[q_idx],
                                      padded_norms.data() + (p + 2) * kVnniPoints,
                                      padded_sqn.data() + (p + 2) * kVnniPoints, mins[q]);
        vnni_chamfer_epilogue<Metric>(a3[q], q_bsums[q_idx], q_norms[q_idx], q_sqns[q_idx],
                                      padded_norms.data() + (p + 3) * kVnniPoints,
                                      padded_sqn.data() + (p + 3) * kVnniPoints, mins[q]);
      }
    }
    for (; p + 2 <= n_panels; p += 2) {
      __m512i ac0[kVnniMq4], ac1[kVnniMq4];
      vnni_micro_kernel_2panel<kVnniMq4>(q_batch, panels_aligned + p * panel_bytes,
                                         panels_aligned + (p + 1) * panel_bytes, total_tiles, ac0,
                                         ac1);
      for (size_t q = 0; q < kVnniMq4; ++q) {
        size_t q_idx = qi + q;
        vnni_chamfer_epilogue<Metric>(ac0[q], q_bsums[q_idx], q_norms[q_idx], q_sqns[q_idx],
                                      padded_norms.data() + p * kVnniPoints,
                                      padded_sqn.data() + p * kVnniPoints, mins[q]);
        vnni_chamfer_epilogue<Metric>(ac1[q], q_bsums[q_idx], q_norms[q_idx], q_sqns[q_idx],
                                      padded_norms.data() + (p + 1) * kVnniPoints,
                                      padded_sqn.data() + (p + 1) * kVnniPoints, mins[q]);
      }
    }
    for (; p < n_panels; ++p) {
      __m512i acc[kVnniMq4];
      vnni_micro_kernel_1panel<kVnniMq4>(q_batch, panels_aligned + p * panel_bytes, total_tiles,
                                         acc);
      for (size_t q = 0; q < kVnniMq4; ++q) {
        size_t q_idx = qi + q;
        vnni_chamfer_epilogue<Metric>(acc[q], q_bsums[q_idx], q_norms[q_idx], q_sqns[q_idx],
                                      padded_norms.data() + p * kVnniPoints,
                                      padded_sqn.data() + p * kVnniPoints, mins[q]);
      }
    }

    for (size_t q = 0; q < kVnniMq4; ++q)
      out_dists[qi + q] = _mm512_reduce_min_ps(mins[q]);
  }

  // Tail queries
  for (; qi < num_fused_embeddings; ++qi) {
    const int8_t* qp = q_flat_data + qi * q_stride;
    __m512 running_min = _mm512_set1_ps(std::numeric_limits<float>::max());
    for (size_t p = 0; p < n_panels; ++p) {
      __m512i acc;
      vnni_micro_kernel_1panel<1>(&qp, panels_aligned + p * panel_bytes, total_tiles, &acc);
      vnni_chamfer_epilogue<Metric>(acc, q_bsums[qi], q_norms[qi], q_sqns[qi],
                                    padded_norms.data() + p * kVnniPoints,
                                    padded_sqn.data() + p * kVnniPoints, running_min);
    }
    out_dists[qi] = _mm512_reduce_min_ps(running_min);
  }
}
#endif  // __AVX512F__

#if !defined(__AVX512F__) && defined(__AVX2__)
static constexpr size_t kAvx2Points = 8;
static constexpr size_t kAvx2Mq4 = 4;

inline __m256i avx2_dpbusd(__m256i acc, __m256i a_unsigned, __m256i b_signed) {
  const __m256i prod16 = _mm256_maddubs_epi16(a_unsigned, b_signed);
  const __m256i prod32 = _mm256_madd_epi16(prod16, _mm256_set1_epi16(1));
  return _mm256_add_epi32(acc, prod32);
}

template<size_t Mq>
inline void avx2_micro_kernel_1panel(const int8_t* const* query_ptrs, const uint8_t* panel,
                                     size_t total_tiles, __m256i* acc_lo, __m256i* acc_hi) {
  constexpr size_t N = kVnniPoints * 4;
  for (size_t q = 0; q < Mq; ++q) {
    acc_lo[q] = _mm256_setzero_si256();
    acc_hi[q] = _mm256_setzero_si256();
  }

  for (size_t t = 0; t < total_tiles; ++t) {
    const __m256i b_lo = _mm256_load_si256(reinterpret_cast<const __m256i*>(panel + t * N));
    const __m256i b_hi = _mm256_load_si256(reinterpret_cast<const __m256i*>(panel + t * N + 32));
    for (size_t q = 0; q < Mq; ++q) {
      const __m256i qv = _mm256_set1_epi32(reinterpret_cast<const int32_t*>(query_ptrs[q])[t]);
      acc_lo[q] = avx2_dpbusd(acc_lo[q], b_lo, qv);
      acc_hi[q] = avx2_dpbusd(acc_hi[q], b_hi, qv);
    }
  }
}

template<size_t Mq>
inline void avx2_micro_kernel_4panel(const int8_t* const* query_ptrs, const uint8_t* panel0,
                                     const uint8_t* panel1, const uint8_t* panel2,
                                     const uint8_t* panel3, size_t total_tiles, __m256i* a0_lo,
                                     __m256i* a0_hi, __m256i* a1_lo, __m256i* a1_hi, __m256i* a2_lo,
                                     __m256i* a2_hi, __m256i* a3_lo, __m256i* a3_hi) {
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
      const __m256i qv = _mm256_set1_epi32(reinterpret_cast<const int32_t*>(query_ptrs[q])[t]);
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

template<bool Metric>
inline void avx2_chamfer_epilogue(__m256i acc, int32_t byte_sum, float q_nsf, float q_sqn,
                                  const float* norms8, const float* sqn8, __m256& running_min) {
  const __m256i bias = _mm256_set1_epi32(128 * byte_sum);
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
    dist = _mm256_add_ps(sqn_v, _mm256_add_ps(_mm256_add_ps(neg_dot, neg_dot), sqn_q));
  } else {
    dist = neg_dot;
  }
  running_min = _mm256_min_ps(running_min, dist);
}

inline float avx2_reduce_min_ps(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v);
  __m128 hi = _mm256_extractf128_ps(v, 1);
  __m128 m = _mm_min_ps(lo, hi);
  __m128 m2 = _mm_shuffle_ps(m, m, _MM_SHUFFLE(1, 0, 3, 2));
  m = _mm_min_ps(m, m2);
  __m128 m3 = _mm_shuffle_ps(m, m, _MM_SHUFFLE(0, 1, 0, 1));
  m = _mm_min_ps(m, m3);
  return _mm_cvtss_f32(m);
}

template<bool Metric>
inline float chamfer_avx2_gemm(const int8_t* q_flat_data, const float* q_norms, const float* q_sqns,
                               const int32_t* q_bsums, size_t q_stride, size_t num_queries,
                               const uint8_t* strip_data, const float* norms,
                               const float* squared_norms, size_t strip_stride, size_t n_strips,
                               size_t num_bytes_per_point, size_t cloud_size,
                               size_t lane_offset = 0) {
  const size_t decoded_dim = 2 * num_bytes_per_point;
  const size_t padded_dim = (decoded_dim + 3) & ~3;
  const size_t total_tiles = padded_dim / 4;
  constexpr size_t N = kVnniPoints * 4;
  const size_t panel_bytes = total_tiles * N;
  const size_t n_panels = (cloud_size + kVnniPoints - 1) / kVnniPoints;

  thread_local std::vector<uint8_t> panels;
  panels.resize(n_panels * panel_bytes + 64);
  uint8_t* panels_aligned =
      reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(panels.data()) + 63) & ~63);

  thread_local std::vector<uint8_t> gather_buf;

  for (size_t p = 0; p < n_panels; ++p) {
    const size_t abs_point = lane_offset + p * kVnniPoints;
    const size_t strip = abs_point / 64;
    const size_t base_lane = abs_point % 64;

    if (base_lane + kVnniPoints <= 64) {
      decode_strip_to_panel_simd(strip_data + strip * strip_stride, base_lane, num_bytes_per_point,
                                 total_tiles, panels_aligned + p * panel_bytes);
    } else {
      const size_t in_this = 64 - base_lane;
      const size_t in_next = kVnniPoints - in_this;
      gather_buf.resize(num_bytes_per_point * 64);
      const uint8_t* s0 = strip_data + strip * strip_stride;
      const uint8_t* s1 = strip_data + (strip + 1) * strip_stride;
      for (size_t j = 0; j < num_bytes_per_point; ++j) {
        std::memcpy(gather_buf.data() + j * 64, s0 + j * 64 + base_lane, in_this);
        std::memcpy(gather_buf.data() + j * 64 + in_this, s1 + j * 64, in_next);
      }
      decode_strip_to_panel_simd(gather_buf.data(), 0, num_bytes_per_point, total_tiles,
                                 panels_aligned + p * panel_bytes);
    }
  }

  const size_t padded_pts = n_panels * kVnniPoints;
  thread_local std::vector<float> padded_norms;
  thread_local std::vector<float> padded_sqn;
  padded_norms.resize(padded_pts);
  std::memcpy(padded_norms.data(), norms, cloud_size * sizeof(float));
  std::memset(padded_norms.data() + cloud_size, 0, (padded_pts - cloud_size) * sizeof(float));
  if constexpr (Metric) {
    padded_sqn.resize(padded_pts);
    std::memcpy(padded_sqn.data(), squared_norms, cloud_size * sizeof(float));
    std::memset(padded_sqn.data() + cloud_size, 0, (padded_pts - cloud_size) * sizeof(float));
  }

  float total_chamfer = 0.0f;
  size_t qi = 0;

  for (; qi + kAvx2Mq4 <= num_queries; qi += kAvx2Mq4) {
    const int8_t* q_batch[kAvx2Mq4];
    for (size_t q = 0; q < kAvx2Mq4; ++q) {
      q_batch[q] = q_flat_data + (qi + q) * q_stride;
    }

    __m256 mins[kAvx2Mq4];
    for (size_t q = 0; q < kAvx2Mq4; ++q)
      mins[q] = _mm256_set1_ps(std::numeric_limits<float>::max());

    size_t p = 0;
    for (; p + 4 <= n_panels; p += 4) {
      __m256i a0_lo[kAvx2Mq4], a0_hi[kAvx2Mq4];
      __m256i a1_lo[kAvx2Mq4], a1_hi[kAvx2Mq4];
      __m256i a2_lo[kAvx2Mq4], a2_hi[kAvx2Mq4];
      __m256i a3_lo[kAvx2Mq4], a3_hi[kAvx2Mq4];
      avx2_micro_kernel_4panel<kAvx2Mq4>(
          q_batch, panels_aligned + p * panel_bytes, panels_aligned + (p + 1) * panel_bytes,
          panels_aligned + (p + 2) * panel_bytes, panels_aligned + (p + 3) * panel_bytes,
          total_tiles, a0_lo, a0_hi, a1_lo, a1_hi, a2_lo, a2_hi, a3_lo, a3_hi);

      for (size_t q = 0; q < kAvx2Mq4; ++q) {
        size_t query_idx = qi + q;
        avx2_chamfer_epilogue<Metric>(a0_lo[q], q_bsums[query_idx], q_norms[query_idx],
                                      q_sqns[query_idx], padded_norms.data() + p * kVnniPoints,
                                      padded_sqn.data() + p * kVnniPoints, mins[q]);
        avx2_chamfer_epilogue<Metric>(a0_hi[q], q_bsums[query_idx], q_norms[query_idx],
                                      q_sqns[query_idx],
                                      padded_norms.data() + p * kVnniPoints + kAvx2Points,
                                      padded_sqn.data() + p * kVnniPoints + kAvx2Points, mins[q]);

        avx2_chamfer_epilogue<Metric>(a1_lo[q], q_bsums[query_idx], q_norms[query_idx],
                                      q_sqns[query_idx],
                                      padded_norms.data() + (p + 1) * kVnniPoints,
                                      padded_sqn.data() + (p + 1) * kVnniPoints, mins[q]);
        avx2_chamfer_epilogue<Metric>(
            a1_hi[q], q_bsums[query_idx], q_norms[query_idx], q_sqns[query_idx],
            padded_norms.data() + (p + 1) * kVnniPoints + kAvx2Points,
            padded_sqn.data() + (p + 1) * kVnniPoints + kAvx2Points, mins[q]);

        avx2_chamfer_epilogue<Metric>(a2_lo[q], q_bsums[query_idx], q_norms[query_idx],
                                      q_sqns[query_idx],
                                      padded_norms.data() + (p + 2) * kVnniPoints,
                                      padded_sqn.data() + (p + 2) * kVnniPoints, mins[q]);
        avx2_chamfer_epilogue<Metric>(
            a2_hi[q], q_bsums[query_idx], q_norms[query_idx], q_sqns[query_idx],
            padded_norms.data() + (p + 2) * kVnniPoints + kAvx2Points,
            padded_sqn.data() + (p + 2) * kVnniPoints + kAvx2Points, mins[q]);

        avx2_chamfer_epilogue<Metric>(a3_lo[q], q_bsums[query_idx], q_norms[query_idx],
                                      q_sqns[query_idx],
                                      padded_norms.data() + (p + 3) * kVnniPoints,
                                      padded_sqn.data() + (p + 3) * kVnniPoints, mins[q]);
        avx2_chamfer_epilogue<Metric>(
            a3_hi[q], q_bsums[query_idx], q_norms[query_idx], q_sqns[query_idx],
            padded_norms.data() + (p + 3) * kVnniPoints + kAvx2Points,
            padded_sqn.data() + (p + 3) * kVnniPoints + kAvx2Points, mins[q]);
      }
    }
    for (; p < n_panels; ++p) {
      __m256i acc_lo[kAvx2Mq4], acc_hi[kAvx2Mq4];
      avx2_micro_kernel_1panel<kAvx2Mq4>(q_batch, panels_aligned + p * panel_bytes, total_tiles,
                                         acc_lo, acc_hi);

      for (size_t q = 0; q < kAvx2Mq4; ++q) {
        size_t query_idx = qi + q;
        avx2_chamfer_epilogue<Metric>(acc_lo[q], q_bsums[query_idx], q_norms[query_idx],
                                      q_sqns[query_idx], padded_norms.data() + p * kVnniPoints,
                                      padded_sqn.data() + p * kVnniPoints, mins[q]);
        avx2_chamfer_epilogue<Metric>(acc_hi[q], q_bsums[query_idx], q_norms[query_idx],
                                      q_sqns[query_idx],
                                      padded_norms.data() + p * kVnniPoints + kAvx2Points,
                                      padded_sqn.data() + p * kVnniPoints + kAvx2Points, mins[q]);
      }
    }

    for (size_t q = 0; q < kAvx2Mq4; ++q)
      total_chamfer += avx2_reduce_min_ps(mins[q]);
  }

  // Tail: remaining queries.
  for (; qi < num_queries; ++qi) {
    const int8_t* qp = q_flat_data + qi * q_stride;
    __m256 running_min = _mm256_set1_ps(std::numeric_limits<float>::max());

    for (size_t p = 0; p < n_panels; ++p) {
      __m256i acc_lo, acc_hi;
      avx2_micro_kernel_1panel<1>(&qp, panels_aligned + p * panel_bytes, total_tiles, &acc_lo,
                                  &acc_hi);
      avx2_chamfer_epilogue<Metric>(acc_lo, q_bsums[qi], q_norms[qi], q_sqns[qi],
                                    padded_norms.data() + p * kVnniPoints,
                                    padded_sqn.data() + p * kVnniPoints, running_min);
      avx2_chamfer_epilogue<Metric>(acc_hi, q_bsums[qi], q_norms[qi], q_sqns[qi],
                                    padded_norms.data() + p * kVnniPoints + kAvx2Points,
                                    padded_sqn.data() + p * kVnniPoints + kAvx2Points, running_min);
    }

    total_chamfer += avx2_reduce_min_ps(running_min);
  }

  return total_chamfer;
}
#endif  // !__AVX512F__ && __AVX2__

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
                                     const Quantized_Point_Cloud_Set<Metric>& db, size_t start_vec,
                                     size_t end_vec);

// =========================================================================
// Proxy: flat vector index range [start_idx, end_idx) into a quantized set
// =========================================================================
template<bool Metric>
class Quantized_Point_Cloud {
 public:
  const Quantized_Point_Cloud_Set<Metric>* db = nullptr;
  size_t start_idx = 0;
  size_t end_idx = 0;

  Quantized_Point_Cloud() = default;
  Quantized_Point_Cloud(const Quantized_Point_Cloud_Set<Metric>* d, size_t s, size_t e) :
      db(d), start_idx(s), end_idx(e) {}

  size_t size() const { return end_idx - start_idx; }

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
    return turboquant_mv_chamfer_distance(*this, *cloud.db, cloud.start_idx, cloud.end_idx);
  }

  template<typename CloudHandle>
  std::pair<float, size_t> distance_w_cmps(const CloudHandle& cloud) const {
    const size_t cloud_size = cloud.end_idx - cloud.start_idx;
    const size_t bytes_per_vec = cloud.db->num_bytes_per_datapoint + sizeof(float) +
                                 (Metric ? sizeof(float) : 0);
    return {this->distance(cloud), cloud_size * bytes_per_vec};
  }

  static constexpr bool is_metric() { return Metric; }
};

// =========================================================================
// Multi-Vector Set (Strip Layout)
// =========================================================================
template<bool Metric>
class Quantized_Point_Cloud_Set {
 public:
  size_t num_bytes_per_datapoint = 0;
  size_t stride = 0;  // strip stride = num_bytes * 64

  parlay::sequence<uint8_t> packed_codes;
  parlay::sequence<float> norm_scaling_factors;
  parlay::sequence<float> unquantized_squared_norms;

  parlay::sequence<size_t> offsets;
  parlay::sequence<uint32_t> ids;

  Quantized_Point_Cloud_Set() = default;
  static constexpr bool is_metric() noexcept { return Metric; }

  Quantized_Point_Cloud<Metric> operator[](size_t i) const {
    const size_t start = offsets[i];
    const size_t end = offsets[i + 1];
    return Quantized_Point_Cloud<Metric>(this, start, end);
  }

  inline uint32_t get_id(size_t i) const noexcept {
    return (ids.size() > 0) ? ids[i] : static_cast<uint32_t>(i);
  }
  inline size_t num_bytes() const noexcept { return packed_codes.size() * sizeof(uint8_t); }

  void distances_all(const Quantized_Query_Point_Cloud<Metric>& q,
                     std::pair<uint32_t, float>* results) const {
    const size_t num_q = q.num_queries;
    const size_t num_clouds = (offsets.size() > 0) ? offsets.size() - 1 : 0;

    if (num_q == 0 || num_clouds == 0) return;

    parlay::parallel_for(0, num_clouds, [&](size_t c) {
      const size_t start_vec = offsets[c];
      const size_t end_vec = offsets[c + 1];
      const size_t cloud_size = end_vec - start_vec;

      if (cloud_size == 0) {
        results[c] = {get_id(c), std::numeric_limits<float>::max()};
        return;
      }

      const float d = turboquant_mv_chamfer_distance(q, *this, start_vec, end_vec);
      results[c] = {get_id(c), d};
    });
  }

  void save(std::ofstream& out) const {
    out.write(reinterpret_cast<const char*>(&num_bytes_per_datapoint),
              sizeof(num_bytes_per_datapoint));
    out.write(reinterpret_cast<const char*>(&stride), sizeof(stride));

    size_t sz = packed_codes.size();
    out.write(reinterpret_cast<const char*>(&sz), sizeof(sz));
    if (sz) out.write(reinterpret_cast<const char*>(packed_codes.data()), sz);

    size_t nsz = norm_scaling_factors.size();
    out.write(reinterpret_cast<const char*>(&nsz), sizeof(nsz));
    if (nsz)
      out.write(reinterpret_cast<const char*>(norm_scaling_factors.data()), nsz * sizeof(float));

    size_t ssz = unquantized_squared_norms.size();
    out.write(reinterpret_cast<const char*>(&ssz), sizeof(ssz));
    if (ssz)
      out.write(reinterpret_cast<const char*>(unquantized_squared_norms.data()),
                ssz * sizeof(float));

    size_t osz = offsets.size();
    out.write(reinterpret_cast<const char*>(&osz), sizeof(osz));
    if (osz) out.write(reinterpret_cast<const char*>(offsets.data()), osz * sizeof(size_t));

    size_t isz = ids.size();
    out.write(reinterpret_cast<const char*>(&isz), sizeof(isz));
    if (isz) out.write(reinterpret_cast<const char*>(ids.data()), isz * sizeof(uint32_t));
  }

  void load(std::ifstream& in) {
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
    if (ssz)
      in.read(reinterpret_cast<char*>(unquantized_squared_norms.data()), ssz * sizeof(float));

    size_t osz = 0;
    in.read(reinterpret_cast<char*>(&osz), sizeof(osz));
    offsets.resize(osz);
    if (osz) in.read(reinterpret_cast<char*>(offsets.data()), osz * sizeof(size_t));

    size_t isz = 0;
    in.read(reinterpret_cast<char*>(&isz), sizeof(isz));
    ids.resize(isz);
    if (isz) in.read(reinterpret_cast<char*>(ids.data()), isz * sizeof(uint32_t));
  }
};

template<bool Metric>
float turboquant_mv_chamfer_distance(const Quantized_Query_Point_Cloud<Metric>& q,
                                     const Quantized_Point_Cloud_Set<Metric>& db, size_t start_vec,
                                     size_t end_vec) {
  const size_t num_q = q.num_queries;
  if (num_q == 0) return 0.0f;
  if (end_vec <= start_vec) return std::numeric_limits<float>::max();

  const size_t cloud_size = end_vec - start_vec;
  const size_t strip_idx = start_vec / 64;
  const size_t lane_offset = start_vec % 64;
  const size_t n_strips = (lane_offset + cloud_size + 63) / 64;

  const uint8_t* s_data = db.packed_codes.data() + strip_idx * db.stride;
  const float* ns = db.norm_scaling_factors.data() + start_vec;
  const float* sqn = db.unquantized_squared_norms.data() + start_vec;

  float dist_sum = 0.0f;

#ifdef __AVX512F__
  dist_sum = internal::chamfer_vnni_gemm<Metric>(
      q.flat_query_data.data(), q.norm_scaling_factors.data(), q.unquantized_squared_norms.data(),
      q.byte_sums.data(), q.q_stride, num_q, s_data, ns, sqn, db.stride, n_strips,
      db.num_bytes_per_datapoint, cloud_size, lane_offset);
#elif defined(__AVX2__)
  dist_sum = internal::chamfer_avx2_gemm<Metric>(
      q.flat_query_data.data(), q.norm_scaling_factors.data(), q.unquantized_squared_norms.data(),
      q.byte_sums.data(), q.q_stride, num_q, s_data, ns, sqn, db.stride, n_strips,
      db.num_bytes_per_datapoint, cloud_size, lane_offset);
#else
  float total_chamfer = 0.0f;
  for (size_t qi = 0; qi < num_q; ++qi) {
    float min_d = std::numeric_limits<float>::max();
    const int8_t* q_ptr = q.get_q_ptr(qi);

    for (size_t i = 0; i < cloud_size; ++i) {
      int32_t dot = 0;
      const size_t pt_lane = (lane_offset + i) % 64;
      const size_t pt_strip = (lane_offset + i) / 64;
      const uint8_t* code_ptr = s_data + pt_strip * db.stride + pt_lane;

      for (size_t j = 0; j < db.num_bytes_per_datapoint; ++j) {
        const uint8_t byte = code_ptr[j * 64];
        const uint8_t b_even = byte & 0xF;
        const uint8_t b_odd = byte >> 4;
        dot += static_cast<int32_t>(internal::kCentroidsUint8[b_even]) *
               q_ptr[static_cast<size_t>(2 * j)];
        dot += static_cast<int32_t>(internal::kCentroidsUint8[b_odd]) *
               q_ptr[static_cast<size_t>(2 * j + 1)];
      }

      dot -= 128 * q.byte_sums[qi];
      float neg_dot = -static_cast<float>(dot) * ns[i] * q.norm_scaling_factors[qi];
      float d;
      if constexpr (Metric) {
        d = sqn[i] + 2.0f * neg_dot + q.unquantized_squared_norms[qi];
      } else {
        d = neg_dot;
      }
      if (d < min_d) min_d = d;
    }
    total_chamfer += min_d;
  }
  dist_sum = total_chamfer;
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
    enc.stride = enc.num_bytes_per_datapoint * 64;

    auto float_offsets = pcs.get_offsets();
    size_t n_clouds = float_offsets.size() > 0 ? float_offsets.size() - 1 : 0;

    enc.offsets.resize(n_clouds + 1);
    enc.offsets[0] = 0;

    size_t cur_padded = 0;
    for (size_t c = 0; c < n_clouds; ++c) {
      enc.offsets[c] = cur_padded;
      size_t n_vecs = (float_offsets[c + 1] - float_offsets[c]) / encoder.dim;
      cur_padded += ((n_vecs + 63) / 64) * 64;  // Pad each cloud to 64!
    }
    enc.offsets[n_clouds] = cur_padded;

    size_t n_strips = cur_padded / 64;
    enc.packed_codes.resize(n_strips * enc.stride, 0);
    enc.norm_scaling_factors.resize(cur_padded, 0.0f);
    enc.unquantized_squared_norms.resize(cur_padded, 0.0f);

    auto pcs_ids = pcs.get_ids();
    enc.ids = parlay::sequence<uint32_t>(pcs_ids.begin(), pcs_ids.end());

    for (size_t c = 0; c < n_clouds; ++c) {
      size_t src_start = float_offsets[c] / encoder.dim;
      size_t n_vecs = (float_offsets[c + 1] - float_offsets[c]) / encoder.dim;
      size_t dst_start = enc.offsets[c];

      parlay::parallel_for(0, n_vecs, [&](size_t i) {
        std::vector<float> ws(encoder.padded_dim);
        std::vector<uint8_t> p_codes(enc.num_bytes_per_datapoint);

        const float* p = reinterpret_cast<const float*>(pcs.data() + (src_start + i) * encoder.dim);
        auto [sqn, nsf] = encoder.encode_single(p, p_codes.data(), ws);

        size_t dst_idx = dst_start + i;
        enc.norm_scaling_factors[dst_idx] = nsf;
        enc.unquantized_squared_norms[dst_idx] = sqn;

        size_t strip = dst_idx / 64;
        size_t lane = dst_idx % 64;
        for (size_t j = 0; j < enc.num_bytes_per_datapoint; ++j) {
          enc.packed_codes[strip * enc.stride + j * 64 + lane] = p_codes[j];
        }
      });
    }
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
// ManyToMany Batch Operator
// ------------------------------------------------------------------
template<typename PCS>
class ManyToMany {
 public:
  static void TopKIntoUninitialized(
      const std::vector<const Quantized_Query_Point_Cloud<PCS::is_metric()>*>& A, const PCS& B,
      uint32_t k, std::pair<uint32_t, float>* results) {

    const size_t num_q_clouds = A.size();
    const size_t num_db_clouds = (B.offsets.size() > 0) ? B.offsets.size() - 1 : 0;
    if (num_q_clouds == 0 || num_db_clouds == 0) return;

    // 16 clouds per thread. If each cloud has 32 vectors, this is a 512-vector SIMD block.
    const size_t Q_BLOCK = 16;

    parlay::parallel_for(0, (num_q_clouds + Q_BLOCK - 1) / Q_BLOCK, [&](size_t qb) {
      size_t q_start = qb * Q_BLOCK;
      size_t q_end = std::min(q_start + Q_BLOCK, num_q_clouds);
      size_t q_count = q_end - q_start;

      std::vector<std::priority_queue<std::pair<float, uint32_t>>> heaps(q_count);

      // --- 1. FUSE QUERIES ---
      size_t total_embeddings = 0;
      std::vector<size_t> emb_offsets(q_count + 1, 0);
      for (size_t i = 0; i < q_count; ++i) {
        total_embeddings += A[q_start + i]->num_queries;
        emb_offsets[i + 1] = total_embeddings;
      }

      size_t q_stride = A[0]->q_stride;
      std::vector<int8_t> fused_q_data(total_embeddings * q_stride);
      std::vector<float> fused_q_norms(total_embeddings);
      std::vector<float> fused_q_sqns(total_embeddings);
      std::vector<int32_t> fused_q_bsums(total_embeddings);

      for (size_t i = 0; i < q_count; ++i) {
        const auto* qc = A[q_start + i];
        size_t off = emb_offsets[i];
        size_t count = qc->num_queries;
        std::memcpy(fused_q_data.data() + off * q_stride, qc->flat_query_data.data(),
                    count * q_stride);
        std::memcpy(fused_q_norms.data() + off, qc->norm_scaling_factors.data(),
                    count * sizeof(float));
        std::memcpy(fused_q_sqns.data() + off, qc->unquantized_squared_norms.data(),
                    count * sizeof(float));
        std::memcpy(fused_q_bsums.data() + off, qc->byte_sums.data(), count * sizeof(int32_t));
      }

      std::vector<float> emb_min_dists(total_embeddings);

      // --- 2. DATABASE PROBING ---
      for (size_t c = 0; c < num_db_clouds; ++c) {
        const size_t start_vec = B.offsets[c];
        const size_t end_vec = B.offsets[c + 1];
        const size_t cloud_size = end_vec - start_vec;

        if (cloud_size == 0) continue;

// The Magic: One decode, full saturation.
#ifdef __AVX512F__
        internal::chamfer_vnni_gemm_fused<PCS::is_metric()>(
            fused_q_data.data(), fused_q_norms.data(), fused_q_sqns.data(), fused_q_bsums.data(),
            q_stride, total_embeddings, B.packed_codes.data() + (start_vec / 64) * B.stride,
            B.norm_scaling_factors.data() + start_vec,
            B.unquantized_squared_norms.data() + start_vec, B.stride,
            (start_vec % 64 + cloud_size + 63) / 64, B.num_bytes_per_datapoint, cloud_size,
            start_vec % 64, emb_min_dists.data());
#endif

        // Aggregate Chamfer distance for each query cloud
        for (size_t i = 0; i < q_count; ++i) {
          float dist_sum = 0.0f;
          size_t e_start = emb_offsets[i];
          size_t e_count = emb_offsets[i + 1] - e_start;

          for (size_t e = 0; e < e_count; ++e) {
            dist_sum += emb_min_dists[e_start + e];
          }
          float chamfer_dist = dist_sum / static_cast<float>(e_count);

          if (heaps[i].size() < k) {
            heaps[i].push({chamfer_dist, B.get_id(c)});
          } else if (chamfer_dist < heaps[i].top().first) {
            heaps[i].pop();
            heaps[i].push({chamfer_dist, B.get_id(c)});
          }
        }
      }

      // --- 3. WRITE TO OUTPUT ---
      for (size_t i = 0; i < q_count; ++i) {
        size_t count = heaps[i].size();
        size_t global_idx = q_start + i;
        for (size_t ki = 0; ki < count; ++ki) {
          results[global_idx * k + (count - 1 - ki)] = {heaps[i].top().second,
                                                        heaps[i].top().first};
          heaps[i].pop();
        }
        for (size_t ki = count; ki < k; ++ki) {
          results[global_idx * k + ki] = {0, std::numeric_limits<float>::max()};
        }
      }
    });
  }
};

}  // namespace turboquant_mv
}  // namespace mvsic