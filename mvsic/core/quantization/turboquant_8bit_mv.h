#pragma once

// Multi-vector 8-bit TurboQuant Chamfer with PRE-TRANSPOSED panel storage.
//
// Same overall layout as `turboquant_mv.h` (4-bit), but each point stores 1
// byte per (rotated, padded) dimension instead of a packed nibble. The panel
// format and AVX-512 micro-kernels are identical: every panel holds
// kVnniPoints = 16 consecutive points laid out so one VPDPBUSD produces one
// int32 lane per point. The only differences vs the 4-bit kernel are:
//
//   * Encoding: per-point adaptive max-abs scaling to fill [-127, 127]. After
//     the FhtKac rotation the coordinates are tiny (~1/sqrt(padded_dim)), so a
//     fixed cap (the 4-bit `kValueCap=3.91724`) would waste almost all int8
//     resolution. Instead each point stretches its own [-max_abs, max_abs] to
//     [-127, 127] and stores the scale separately as `norm_scaling_factor`.
//
//   * Storage: num_bytes_per_datapoint = padded_dim (vs padded_dim/2 for 4-bit).
//     Stored as uint8 = int8 ^ 0x80 so the GEMM can use VPDPBUSD natively
//     (unsigned * signed -> int32) the same way 4-bit does.
//
//   * decode_strip_to_panel: no nibble unpack. Just XOR 0x80 + the same 4x16
//     -> 16x4 byte transpose used by the 4-bit path.
//
// Everything else (`vnni_micro_kernel_*panel`, `chamfer_panels`,
// `chamfer_panels_multi`, `FusedQueryBatch`, `ManyToMany`) is identical to
// `turboquant_mv.h`. Reference C/Highway kernels are in `8bit_code/`.

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
namespace turboquant_8bit_mv {

namespace internal {

static constexpr size_t kVnniPoints = 16;  // int32 lanes in __m512i (= panel point count)
static constexpr size_t kVnniMq = 6;       // queries per batch (matches 4-bit MV)

// =========================================================================
// Panel build-time helpers
// =========================================================================

// Builds one block-transposed panel from a strip-laid-out source buffer
// `strip_ptr` (byte j of point lane => strip_ptr[j*64 + lane]). Reads
// kVnniPoints (= 16) consecutive lanes starting at `base_lane`. `num_bytes` is
// the number of stored bytes per point (= padded_dim for 8-bit). Pads any
// trailing tile slots with the neutral byte (0x80) so they don't move the dot.
inline void decode_strip_to_panel_simd(const uint8_t* strip_ptr, size_t base_lane, size_t num_bytes,
                                       size_t total_tiles, uint8_t* panel) {
  constexpr size_t N = kVnniPoints * 4;
  const __m128i bias = _mm_set1_epi8(static_cast<char>(0x80));

  // Process 4 source bytes per point per tile (== 4 dims). For 8-bit the
  // panel layout is `panel[tile*64 + lane*4 + b] = src[(tile*4+b) * 64 + lane] ^ 0x80`,
  // i.e. a 4x16 -> 16x4 byte transpose with a one-vector XOR.
  size_t tile = 0;
  size_t j = 0;
  for (; j + 4 <= num_bytes; j += 4) {
    const __m128i a =
        _mm_xor_si128(_mm_loadu_si128(reinterpret_cast<const __m128i*>(strip_ptr + (j + 0) * 64 + base_lane)), bias);
    const __m128i b =
        _mm_xor_si128(_mm_loadu_si128(reinterpret_cast<const __m128i*>(strip_ptr + (j + 1) * 64 + base_lane)), bias);
    const __m128i c =
        _mm_xor_si128(_mm_loadu_si128(reinterpret_cast<const __m128i*>(strip_ptr + (j + 2) * 64 + base_lane)), bias);
    const __m128i d =
        _mm_xor_si128(_mm_loadu_si128(reinterpret_cast<const __m128i*>(strip_ptr + (j + 3) * 64 + base_lane)), bias);

    const __m128i pair_ab_lo = _mm_unpacklo_epi8(a, b);  // lanes 0..7  pairs (a,b)
    const __m128i pair_ab_hi = _mm_unpackhi_epi8(a, b);  // lanes 8..15
    const __m128i pair_cd_lo = _mm_unpacklo_epi8(c, d);
    const __m128i pair_cd_hi = _mm_unpackhi_epi8(c, d);

    const __m128i tile_lo     = _mm_unpacklo_epi16(pair_ab_lo, pair_cd_lo);  // lanes 0..3
    const __m128i tile_mid_lo = _mm_unpackhi_epi16(pair_ab_lo, pair_cd_lo);  // lanes 4..7
    const __m128i tile_mid_hi = _mm_unpacklo_epi16(pair_ab_hi, pair_cd_hi);  // lanes 8..11
    const __m128i tile_hi     = _mm_unpackhi_epi16(pair_ab_hi, pair_cd_hi);  // lanes 12..15

    _mm_storeu_si128(reinterpret_cast<__m128i*>(panel + tile * N + 0),  tile_lo);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(panel + tile * N + 16), tile_mid_lo);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(panel + tile * N + 32), tile_mid_hi);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(panel + tile * N + 48), tile_hi);
    ++tile;
  }

  // Tail: 1..3 leftover source bytes per point (only when num_bytes % 4 != 0).
  // Read what's available, treat the rest as 0x80 (neutral). We synthesize the
  // missing dim slices with a neutral vector.
  if (j < num_bytes) {
    const __m128i neutral = _mm_set1_epi8(static_cast<char>(0x80));
    auto load_or_neutral = [&](size_t off) -> __m128i {
      if (j + off < num_bytes) {
        return _mm_xor_si128(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(strip_ptr + (j + off) * 64 + base_lane)),
            bias);
      }
      return neutral;
    };
    const __m128i a = load_or_neutral(0);
    const __m128i b = load_or_neutral(1);
    const __m128i c = load_or_neutral(2);
    const __m128i d = load_or_neutral(3);

    const __m128i pair_ab_lo = _mm_unpacklo_epi8(a, b);
    const __m128i pair_ab_hi = _mm_unpackhi_epi8(a, b);
    const __m128i pair_cd_lo = _mm_unpacklo_epi8(c, d);
    const __m128i pair_cd_hi = _mm_unpackhi_epi8(c, d);

    const __m128i tile_lo     = _mm_unpacklo_epi16(pair_ab_lo, pair_cd_lo);
    const __m128i tile_mid_lo = _mm_unpackhi_epi16(pair_ab_lo, pair_cd_lo);
    const __m128i tile_mid_hi = _mm_unpacklo_epi16(pair_ab_hi, pair_cd_hi);
    const __m128i tile_hi     = _mm_unpackhi_epi16(pair_ab_hi, pair_cd_hi);

    _mm_storeu_si128(reinterpret_cast<__m128i*>(panel + tile * N + 0),  tile_lo);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(panel + tile * N + 16), tile_mid_lo);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(panel + tile * N + 32), tile_mid_hi);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(panel + tile * N + 48), tile_hi);
    ++tile;
  }

  // Neutral pad: 0x80 = unsigned form of int8 0, zero contribution to dot product.
  const __m128i neutral = _mm_set1_epi8(static_cast<char>(0x80));
  for (; tile < total_tiles; ++tile) {
    _mm_storeu_si128(reinterpret_cast<__m128i*>(panel + tile * N + 0),  neutral);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(panel + tile * N + 16), neutral);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(panel + tile * N + 32), neutral);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(panel + tile * N + 48), neutral);
  }
}

// =========================================================================
// AVX-512 GEMM micro-kernels (identical to the 4-bit MV kernels — the kernel
// is agnostic to whether the panel was produced by nibble-decode or by raw
// 8-bit XOR transpose; both end up in the same layout).
// =========================================================================
#ifdef __AVX512F__
inline __m512i tq8_dpbusd(__m512i acc, __m512i a_unsigned, __m512i b_signed) {
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
      acc[q] = tq8_dpbusd(acc[q], b, qv);
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
      acc0[q] = tq8_dpbusd(acc0[q], b0, qv);
      acc1[q] = tq8_dpbusd(acc1[q], b1, qv);
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
      acc0[q] = tq8_dpbusd(acc0[q], b0, qv);
      acc1[q] = tq8_dpbusd(acc1[q], b1, qv);
      acc2[q] = tq8_dpbusd(acc2[q], b2, qv);
      acc3[q] = tq8_dpbusd(acc3[q], b3, qv);
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
          panel_data + (p + 2) * panel_bytes, panel_data + (p + 3) * panel_bytes, total_tiles, a0,
          a1, a2, a3);

      for (size_t q = 0; q < kVnniMq; ++q) {
        const size_t qg = qi + q;
        vnni_chamfer_epilogue<Metric>(a0[q], q_bsums[qg], q_norms[qg], q_sqns[qg],
                                      padded_norms + p * kVnniPoints, padded_sqn + p * kVnniPoints,
                                      mins[q]);
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
                                      padded_norms + p * kVnniPoints, padded_sqn + p * kVnniPoints,
                                      mins[q]);
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
                                      padded_norms + p * kVnniPoints, padded_sqn + p * kVnniPoints,
                                      mins[q]);
      }
    }

    for (size_t q = 0; q < kVnniMq; ++q)
      total_chamfer += _mm512_reduce_min_ps(mins[q]);
  }

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
          panel_data + (p + 2) * panel_bytes, panel_data + (p + 3) * panel_bytes, total_tiles, a0,
          a1, a2, a3);
      for (size_t q = 0; q < kVnniMq; ++q) {
        const size_t qg = qi + q;
        vnni_chamfer_epilogue<Metric>(a0[q], q_bsums[qg], q_norms[qg], q_sqns[qg],
                                      padded_norms + p * kVnniPoints, padded_sqn + p * kVnniPoints,
                                      mins[q]);
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
                                      padded_norms + p * kVnniPoints, padded_sqn + p * kVnniPoints,
                                      mins[q]);
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
                                      padded_norms + p * kVnniPoints, padded_sqn + p * kVnniPoints,
                                      mins[q]);
      }
    }

    for (size_t q = 0; q < kVnniMq; ++q)
      out_dists[qi + q] = _mm512_reduce_min_ps(mins[q]);
  }

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
// Scalar fallback (no AVX-512). Decodes each panel point on-the-fly: panel
// holds (lane, byte) = panel[t*64 + lane*4 + b], stored as uint8 = int8+128.
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
float turboquant_8bit_mv_chamfer_distance(const Quantized_Query_Point_Cloud<Metric>& q,
                                          const Quantized_Point_Cloud_Set<Metric>& db,
                                          size_t cloud_idx);

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

  std::vector<int8_t> flat_query_data;           // Size: num_queries * q_stride
  std::vector<float> norm_scaling_factors;       // Size: num_queries
  std::vector<float> unquantized_squared_norms;  // Size: num_queries
  std::vector<int32_t> byte_sums;                // Size: num_queries

  Quantized_Query_Point_Cloud() = default;

  inline const int8_t* get_q_ptr(size_t i) const { return flat_query_data.data() + i * q_stride; }

  template<typename CloudView>
  float distance(const CloudView& cloud) const {
    return turboquant_8bit_mv_chamfer_distance(*this, *cloud.db, cloud.cloud_idx);
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
// Multi-Vector Set (Pre-Transposed Panel Layout)
// =========================================================================
template<bool Metric>
class Quantized_Point_Cloud_Set {
 public:
  size_t num_bytes_per_datapoint = 0;  // = padded_dim for 8-bit (1 byte per dim)
  size_t total_tiles = 0;              // = ceil(num_bytes / 4) -- one VPDPBUSD per tile
  size_t panel_bytes = 0;              // = total_tiles * (kVnniPoints * 4)

  parlay::sequence<uint8_t> panel_data;
  parlay::sequence<size_t> panel_offsets;  // size: n_clouds + 1, in BYTES

  parlay::sequence<size_t> point_offsets;  // size: n_clouds + 1, in points
  parlay::sequence<float> norm_scaling_factors;
  parlay::sequence<float> unquantized_squared_norms;

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
    return panel_data.size() * sizeof(uint8_t) + norm_scaling_factors.size() * sizeof(float) +
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
      const float d = turboquant_8bit_mv_chamfer_distance(q, *this, c);
      results[c] = {get_id(c), d};
    };
    if (nc <= 64) {
      for (size_t c = 0; c < nc; ++c)
        score_one(c);
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

template<bool Metric>
inline size_t Quantized_Point_Cloud<Metric>::size() const {
  return db ? db->cloud_sizes[cloud_idx] : 0;
}

template<bool Metric>
float turboquant_8bit_mv_chamfer_distance(const Quantized_Query_Point_Cloud<Metric>& q,
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
      q.byte_sums.data(), q.q_stride, num_q, panel_ptr, ns, sqn, db.total_tiles, db.panel_bytes, np,
      cs);
#endif
  return dist_sum / static_cast<float>(num_q);
}

// =========================================================================
// Multi-Vector Model
// =========================================================================
template<bool Metric>
class Model {
 public:
  static constexpr uint32_t kClassId = 7;         // QuantizerTag::kEightBitTQ
  static constexpr uint32_t kBatchAlignment = 6;  // kVnniMq
  static constexpr const char* kName = "8bit_tq";
  using EncodedSet = ::mvsic::turboquant_8bit_mv::Quantized_Point_Cloud_Set<Metric>;
  using EncodedQuery = ::mvsic::turboquant_8bit_mv::Quantized_Query_Point_Cloud<Metric>;
  struct Params {};

  // Reuses the TQ rotator (FhtKacRotator). We don't use BaseEncoder's 4-bit
  // bucket encoding — only its rotator setup.
  turboquant::BaseEncoder encoder;

  Model() = default;

  template<typename PCSet>
  void train(const PCSet& pcs, const Params& /*p*/) {
    train(pcs);
  }

  template<typename PCSet>
  void train(const PCSet& pcs) {
    encoder.train(pcs.get_dims());
  }

  // Encode a single rotated vector into `padded_dim` bytes (each = uint8 form
  // of int8 value, i.e. int8 ^ 0x80). Returns (unquantized_sqr_norm,
  // norm_scaling_factor). The scaling factor combines the unquantized norm
  // with the quantization scale so the GEMM dot product reconstructs the
  // unquantized inner product up to per-coordinate quantization error.
  inline std::pair<float, float> encode_single(const float* p, uint8_t* output,
                                               std::vector<float>& ws) const {
    const size_t pdim = encoder.padded_dim;
    encoder.rotator->rotate(p, ws.data());

    float sqr_norm = 0.0f;
    float max_abs = 0.0f;
    for (size_t i = 0; i < pdim; ++i) {
      sqr_norm += ws[i] * ws[i];
      const float a = std::abs(ws[i]);
      if (a > max_abs) max_abs = a;
    }
    if (sqr_norm == 0.0f || max_abs == 0.0f) {
      // Raw int8 zeros — decode_strip XORs with 0x80 to produce the neutral
      // bias-form byte that contributes 0 to dot products.
      std::memset(output, 0, pdim);
      return {0.0f, 0.0f};
    }

    const float norm = std::sqrt(sqr_norm);
    const float scale = 127.0f / max_abs;

    int64_t q_sqr_norm = 0;
    for (size_t i = 0; i < pdim; ++i) {
      const int snapped = static_cast<int>(std::lround(ws[i] * scale));
      const int8_t iv =
          static_cast<int8_t>(snapped < -127 ? -127 : (snapped > 127 ? 127 : snapped));
      // Stored as raw int8 bytes (cast to uint8). decode_strip_to_panel_simd
      // applies the XOR 0x80 bias to produce the uint8 form required by VPDPBUSD.
      output[i] = static_cast<uint8_t>(iv);
      q_sqr_norm += static_cast<int64_t>(iv) * static_cast<int64_t>(iv);
    }

    if (q_sqr_norm == 0) return {sqr_norm, 0.0f};
    const float nsf = norm / std::sqrt(static_cast<float>(q_sqr_norm));
    return {sqr_norm, nsf};
  }

  template<typename PCSet>
  Quantized_Point_Cloud_Set<Metric> encode(const PCSet& pcs) const {
    Quantized_Point_Cloud_Set<Metric> enc;
    enc.num_bytes_per_datapoint = encoder.padded_dim;  // 1 byte per dim

    const size_t padded_dim_quad = (encoder.padded_dim + 3) & ~3;
    enc.total_tiles = padded_dim_quad / 4;
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
      const size_t n_strips = (n_vecs + 63) / 64;
      const size_t strip_bytes = enc.num_bytes_per_datapoint * 64;

      // Per-cloud strip buffer: strip_buf[strip * strip_bytes + j * 64 + lane]
      // = raw int8 byte j of point lane within a 64-point strip. Pre-fill with
      // 0 so missing trailing lanes (n_vecs not a multiple of 64) and unused
      // trailing bytes decode to 0x80 (= bias form of int8 0, neutral for dot).
      std::vector<uint8_t> strip_buf(n_strips * strip_bytes, 0);
      std::vector<uint8_t> p_codes(enc.num_bytes_per_datapoint);
      std::vector<float> ws(encoder.padded_dim);

      const size_t src_start = float_offsets[c] / encoder.dim;
      const size_t pt_off = enc.point_offsets[c];

      for (size_t i = 0; i < n_vecs; ++i) {
        const float* p = reinterpret_cast<const float*>(pcs.data() + (src_start + i) * encoder.dim);
        auto [sqn, nsf] = encode_single(p, p_codes.data(), ws);

        enc.norm_scaling_factors[pt_off + i] = nsf;
        enc.unquantized_squared_norms[pt_off + i] = sqn;

        const size_t strip = i / 64;
        const size_t lane = i % 64;
        for (size_t j = 0; j < enc.num_bytes_per_datapoint; ++j) {
          strip_buf[strip * strip_bytes + j * 64 + lane] = p_codes[j];
        }
      }

      // Pre-transpose strips into permanent panel layout (16 points per panel,
      // 4 panels per 64-point strip).
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

    const size_t decoded_dim = encoder.padded_dim;
    res.q_stride = (decoded_dim + 3) & ~3;

    res.flat_query_data.resize(res.num_queries * res.q_stride, 0);
    res.norm_scaling_factors.resize(res.num_queries, 0.0f);
    res.unquantized_squared_norms.resize(res.num_queries, 0.0f);
    res.byte_sums.resize(res.num_queries, 0);

    const float* base_ptr = query_cloud.data();
    // Hoisted scratch: rotate writes into the same buffer for every qi instead
    // of heap-allocating a fresh `q_rot` vector per query.  For the build-top
    // 8BTQ cache (500k clouds × ~22 vecs each on nq500k) this removes ~11M
    // small heap allocs across the parallel encode.
    std::vector<float> q_rot(encoder.padded_dim);

    for (size_t qi = 0; qi < res.num_queries; ++qi) {
      encoder.rotator->rotate(base_ptr + qi * encoder.dim, q_rot.data());

      float sqr_norm = 0.0f;
      float max_value = 0.0f;
      for (size_t i = 0; i < encoder.padded_dim; ++i) {
        sqr_norm += q_rot[i] * q_rot[i];
        const float a = std::abs(q_rot[i]);
        if (a > max_value) max_value = a;
      }

      if (sqr_norm == 0.0f || !std::isfinite(sqr_norm) || max_value == 0.0f) continue;

      const float norm = std::sqrt(sqr_norm);
      const float sf = 127.0f / max_value;

      int64_t quant_norm = 0;
      int32_t byte_sum = 0;
      int8_t* q_out_ptr = res.flat_query_data.data() + qi * res.q_stride;

      for (size_t i = 0; i < encoder.padded_dim; ++i) {
        const int snapped = static_cast<int>(std::lround(q_rot[i] * sf));
        const int8_t iv =
            static_cast<int8_t>(snapped < -127 ? -127 : (snapped > 127 ? 127 : snapped));
        q_out_ptr[i] = iv;
        quant_norm += static_cast<int64_t>(iv) * static_cast<int64_t>(iv);
        byte_sum += iv;
      }

      res.norm_scaling_factors[qi] =
          quant_norm > 0 ? norm / std::sqrt(static_cast<float>(quant_norm)) : 0.0f;
      res.unquantized_squared_norms[qi] = sqr_norm;
      res.byte_sums[qi] = byte_sum;
    }

    return res;
  }

  void save(std::ofstream& out) const { encoder.save(out); }
  void load(std::ifstream& in) { encoder.load(in); }
};

// =========================================================================
// Pre-Fused Query Batch — same structure as turboquant_mv::FusedQueryBatch.
// =========================================================================
template<bool Metric>
struct FusedQueryBatch {
  size_t num_source_clouds = 0;
  size_t total_embeddings = 0;
  size_t q_stride = 0;

  std::vector<int8_t> flat_query_data;
  std::vector<float> norm_scaling_factors;
  std::vector<float> unquantized_squared_norms;
  std::vector<int32_t> byte_sums;
  std::vector<size_t> emb_offsets;

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

// =========================================================================
// Fused-query Chamfer scoring against one Quantized_Point_Cloud_Set.
// Same shape as turboquant_mv::chamfer_score_all_fused.
// =========================================================================
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
      for (size_t i = 0; i < num_src; ++i)
        out[i * num_db + c] = {id, bad};
      return;
    }

    const size_t panel_byte_off = db.panel_offsets[c];
    const size_t pt_off = db.point_offsets[c];
    const size_t np = (db.panel_offsets[c + 1] - panel_byte_off) / db.panel_bytes;

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
        fq.unquantized_squared_norms.data(), fq.byte_sums.data(), fq.q_stride, fq.total_embeddings,
        db.panel_data.data() + panel_byte_off, db.norm_scaling_factors.data() + pt_off,
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

    for (size_t i = 0; i < num_src; ++i) {
      const size_t e_start = fq.emb_offsets[i];
      const size_t e_end = fq.emb_offsets[i + 1];
      if (e_end == e_start) {
        out[i * num_db + c] = {id, std::numeric_limits<float>::max()};
        continue;
      }
      float dist_sum = 0.0f;
      for (size_t e = e_start; e < e_end; ++e)
        dist_sum += emb_min[e];
      out[i * num_db + c] = {id, dist_sum / static_cast<float>(e_end - e_start)};
    }
  };

  if (parallel_db) {
    parlay::parallel_for(0, num_db, process_one_db);
  } else {
    for (size_t c = 0; c < num_db; ++c)
      process_one_db(c);
  }
}

// =========================================================================
// ManyToMany Batch Operator (mirrors turboquant_mv::ManyToMany).
// =========================================================================
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

        for (size_t i = 0; i < q_count; ++i) {
          float dist_sum = 0.0f;
          const size_t e_start = emb_offsets[i];
          const size_t e_count = emb_offsets[i + 1] - e_start;
          for (size_t e = 0; e < e_count; ++e)
            dist_sum += emb_min_dists[e_start + e];
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

// =========================================================================
// SingleCloudArgmin: per-query argmin lane against ONE multi-vector DB cloud.
//
// Use case: inner Lloyd's k-means assignment.  Each query is a single rotated
// vector (Quantized_Query_Point_Cloud with num_queries=1); the DB is k
// centers packed into one Quantized_Point_Cloud_Set cloud of size k.  Unlike
// ManyToMany (which iterates over n_clouds 1-vec DB "clouds" and wastes 15/16
// of each panel), this kernel keeps every panel full: the n_panels = ceil(k/16)
// panels of the single DB cloud are each 16 lanes of real centers (with the
// last panel possibly partial), so VPDPBUSD throughput is fully utilized.
//
// The kernel returns, per query, the *lane index* in [0, k) of the closest
// center.  This requires extending the standard chamfer epilogue to track an
// argmin lane: alongside running_min we keep running_argmin_panel, the panel
// index that produced each lane's current best.  Final reduction picks the
// lane with the global min and combines (winner_lane, winner_panel) into the
// flat center index.
//
// Mq=6 query-batched for panel reuse (load each panel once, dot it against 6
// queries).  Padded lanes in the last partial panel are masked to +inf so they
// can't beat real centers under IP (where their natural distance is 0).
// =========================================================================
template <bool Metric>
class SingleCloudArgmin {
 public:
  // queries: N pointers to single-vector encoded queries (num_queries==1 each,
  //          all sharing the same q_stride coming from the same Model).
  // centers_db: a Quantized_Point_Cloud_Set with exactly one cloud of size k.
  // out_argmin: array of N uint32_t; filled with the argmin lane in [0, k)
  //             for each query.
  static void Run(const std::vector<const Quantized_Query_Point_Cloud<Metric>*>& queries,
                  const Quantized_Point_Cloud_Set<Metric>& centers_db,
                  uint32_t* out_argmin) {
    const size_t N = queries.size();
    if (N == 0) return;
    // Flatten the query-pointer array into the same layout RunFlat expects.
    // This wrapper keeps the original Run callable while the flat path is
    // faster (no per-query EncodedQuery struct churn).
    const size_t q_stride = queries[0]->q_stride;
    std::vector<int8_t> flat_data(N * q_stride);
    std::vector<float> flat_nsf(N), flat_sqn(N);
    std::vector<int32_t> flat_bsum(N);
    for (size_t i = 0; i < N; ++i) {
      const auto* q = queries[i];
      std::memcpy(flat_data.data() + i * q_stride, q->flat_query_data.data(), q_stride);
      flat_nsf[i] = q->norm_scaling_factors[0];
      flat_sqn[i] = q->unquantized_squared_norms[0];
      flat_bsum[i] = q->byte_sums[0];
    }
    RunFlat(flat_data.data(), flat_nsf.data(), flat_sqn.data(), flat_bsum.data(), q_stride, N,
            centers_db, out_argmin);
  }

  // Faster entry: queries provided as flat parallel arrays (no per-query
  // struct).  This is what TQ8LloydsBackend uses in the inner-kmeans hot path.
  static void RunFlat(const int8_t* q_flat_data, const float* q_nsf, const float* q_sqn,
                      const int32_t* q_bsum, size_t q_stride, size_t N,
                      const Quantized_Point_Cloud_Set<Metric>& centers_db,
                      uint32_t* out_argmin) {
    if (N == 0) return;
    if (centers_db.num_clouds() != 1) {
      std::cerr << "[SingleCloudArgmin] expected exactly 1 DB cloud, got "
                << centers_db.num_clouds() << std::endl;
      std::abort();
    }
    const size_t k = centers_db.cloud_sizes[0];
    if (k == 0) {
      std::memset(out_argmin, 0, N * sizeof(uint32_t));
      return;
    }

    const size_t panel_byte_off = centers_db.panel_offsets[0];
    const size_t pt_off = centers_db.point_offsets[0];
    const size_t n_panels =
        (centers_db.panel_offsets[1] - panel_byte_off) / centers_db.panel_bytes;
    const uint8_t* panel_data = centers_db.panel_data.data() + panel_byte_off;
    const float* db_norms = centers_db.norm_scaling_factors.data() + pt_off;
    const float* db_sqn = centers_db.unquantized_squared_norms.data() + pt_off;
    const size_t total_tiles = centers_db.total_tiles;
    const size_t panel_bytes = centers_db.panel_bytes;

    const size_t valid_in_last = k - (n_panels - 1) * internal::kVnniPoints;
    const uint16_t last_panel_valid_mask =
        (valid_in_last == internal::kVnniPoints)
            ? static_cast<uint16_t>(0xFFFFu)
            : static_cast<uint16_t>((1u << valid_in_last) - 1u);

    constexpr size_t QBlock = 64;
    const size_t num_blocks = (N + QBlock - 1) / QBlock;
    parlay::parallel_for(0, num_blocks, [&](size_t bi) {
      const size_t q_start = bi * QBlock;
      const size_t q_end = std::min(q_start + QBlock, N);
      ProcessRangeFlat_(q_flat_data, q_nsf, q_sqn, q_bsum, q_stride, panel_data, db_norms,
                        db_sqn, total_tiles, panel_bytes, n_panels, last_panel_valid_mask,
                        valid_in_last, q_start, q_end, out_argmin);
    });
  }

 private:
  static void ProcessRangeFlat_(const int8_t* q_flat_data, const float* q_nsf,
                                const float* q_sqn, const int32_t* q_bsum, size_t q_stride,
                                const uint8_t* panel_data, const float* db_norms,
                                const float* db_sqn, size_t total_tiles, size_t panel_bytes,
                                size_t n_panels, uint16_t last_panel_valid_mask,
                                size_t valid_in_last, size_t q_start, size_t q_end,
                                uint32_t* out_argmin) {
#ifdef __AVX512F__
    constexpr size_t Mq = internal::kVnniMq;  // = 6

    size_t qi = q_start;
    for (; qi + Mq <= q_end; qi += Mq) {
      const int8_t* q_ptrs[Mq];
      float q_nsfs[Mq];
      float q_sqns[Mq];
      int32_t q_bsums[Mq];
      for (size_t q = 0; q < Mq; ++q) {
        q_ptrs[q] = q_flat_data + (qi + q) * q_stride;
        q_nsfs[q] = q_nsf[qi + q];
        q_sqns[q] = q_sqn[qi + q];
        q_bsums[q] = q_bsum[qi + q];
      }

      __m512 mins[Mq];
      __m512i argmin_panels[Mq];
      for (size_t q = 0; q < Mq; ++q) {
        mins[q] = _mm512_set1_ps(std::numeric_limits<float>::max());
        argmin_panels[q] = _mm512_setzero_si512();
      }

      for (size_t p = 0; p < n_panels; ++p) {
        // 16-lane VPDPBUSD accumulator per query.
        __m512i accs[Mq];
        for (size_t q = 0; q < Mq; ++q) accs[q] = _mm512_setzero_si512();
        for (size_t t = 0; t < total_tiles; ++t) {
          const __m512i b = _mm512_loadu_si512(
              reinterpret_cast<const __m512i*>(panel_data + p * panel_bytes + t * (internal::kVnniPoints * 4)));
          for (size_t q = 0; q < Mq; ++q) {
            const __m512i qv =
                _mm512_set1_epi32(reinterpret_cast<const int32_t*>(q_ptrs[q])[t]);
            accs[q] = internal::tq8_dpbusd(accs[q], b, qv);
          }
        }

        const __m512 db_norm_v = _mm512_loadu_ps(db_norms + p * internal::kVnniPoints);
        __m512 db_sqn_v;
        if constexpr (Metric) {
          db_sqn_v = _mm512_loadu_ps(db_sqn + p * internal::kVnniPoints);
        }

        const bool need_mask = (p == n_panels - 1) && (valid_in_last < internal::kVnniPoints);
        const __m512i p_vec = _mm512_set1_epi32(static_cast<int>(p));

        for (size_t q = 0; q < Mq; ++q) {
          // Bias correction: panel bytes are stored as uint8 = int8 ^ 0x80, so each
          // tile dot product picks up a 128 * sum_of_query_bytes offset to undo.
          const __m512i correction = _mm512_set1_epi32(128 * q_bsums[q]);
          const __m512i acc_corr = _mm512_sub_epi32(accs[q], correction);

          const __m512 fdot = _mm512_cvtepi32_ps(acc_corr);
          __m512 neg_dot = _mm512_mul_ps(fdot, db_norm_v);
          neg_dot = _mm512_mul_ps(neg_dot, _mm512_set1_ps(q_nsfs[q]));
          neg_dot = _mm512_sub_ps(_mm512_setzero_ps(), neg_dot);

          __m512 dist;
          if constexpr (Metric) {
            dist = _mm512_add_ps(
                db_sqn_v,
                _mm512_add_ps(_mm512_add_ps(neg_dot, neg_dot), _mm512_set1_ps(q_sqns[q])));
          } else {
            // IP: padded lanes have db_norm=0 so neg_dot is 0, which can falsely tie
            // the argmin against centers with positive distance. Mask handles this.
            dist = neg_dot;
          }

          if (need_mask) {
            dist = _mm512_mask_blend_ps(last_panel_valid_mask,
                                        _mm512_set1_ps(std::numeric_limits<float>::max()), dist);
          }

          // Keep both the running per-lane min and the panel index that produced it.
          const __mmask16 better = _mm512_cmp_ps_mask(dist, mins[q], _CMP_LT_OQ);
          mins[q] = _mm512_mask_blend_ps(better, mins[q], dist);
          argmin_panels[q] = _mm512_mask_blend_epi32(better, argmin_panels[q], p_vec);
        }
      }

      for (size_t q = 0; q < Mq; ++q) {
        out_argmin[qi + q] = ExtractArgmin_(mins[q], argmin_panels[q]);
      }
    }

    // Tail: < Mq queries left in this block.
    for (; qi < q_end; ++qi) {
      const int8_t* qp = q_flat_data + qi * q_stride;
      const float q_nsf_v = q_nsf[qi];
      const float q_sqn_v = q_sqn[qi];
      const int32_t q_bsum_v = q_bsum[qi];

      __m512 running_min = _mm512_set1_ps(std::numeric_limits<float>::max());
      __m512i running_argmin_panel = _mm512_setzero_si512();

      for (size_t p = 0; p < n_panels; ++p) {
        __m512i acc = _mm512_setzero_si512();
        for (size_t t = 0; t < total_tiles; ++t) {
          const __m512i b = _mm512_loadu_si512(
              reinterpret_cast<const __m512i*>(panel_data + p * panel_bytes + t * (internal::kVnniPoints * 4)));
          const __m512i qv = _mm512_set1_epi32(reinterpret_cast<const int32_t*>(qp)[t]);
          acc = internal::tq8_dpbusd(acc, b, qv);
        }
        const __m512i correction = _mm512_set1_epi32(128 * q_bsum_v);
        acc = _mm512_sub_epi32(acc, correction);

        const __m512 fdot = _mm512_cvtepi32_ps(acc);
        const __m512 db_norm_v = _mm512_loadu_ps(db_norms + p * internal::kVnniPoints);
        __m512 neg_dot = _mm512_mul_ps(fdot, db_norm_v);
        neg_dot = _mm512_mul_ps(neg_dot, _mm512_set1_ps(q_nsf_v));
        neg_dot = _mm512_sub_ps(_mm512_setzero_ps(), neg_dot);

        __m512 dist;
        if constexpr (Metric) {
          const __m512 db_sqn_v = _mm512_loadu_ps(db_sqn + p * internal::kVnniPoints);
          dist = _mm512_add_ps(
              db_sqn_v, _mm512_add_ps(_mm512_add_ps(neg_dot, neg_dot), _mm512_set1_ps(q_sqn_v)));
        } else {
          dist = neg_dot;
        }
        if (p == n_panels - 1 && valid_in_last < internal::kVnniPoints) {
          dist = _mm512_mask_blend_ps(last_panel_valid_mask,
                                      _mm512_set1_ps(std::numeric_limits<float>::max()), dist);
        }
        const __mmask16 better = _mm512_cmp_ps_mask(dist, running_min, _CMP_LT_OQ);
        running_min = _mm512_mask_blend_ps(better, running_min, dist);
        running_argmin_panel = _mm512_mask_blend_epi32(better, running_argmin_panel,
                                                       _mm512_set1_epi32(static_cast<int>(p)));
      }
      out_argmin[qi] = ExtractArgmin_(running_min, running_argmin_panel);
    }
#else
    // Scalar fallback.
    for (size_t qi = q_start; qi < q_end; ++qi) {
      const int8_t* qp = q_flat_data + qi * q_stride;
      const float q_nsf_v = q_nsf[qi];
      const float q_sqn_v = q_sqn[qi];
      const int32_t q_bsum_v = q_bsum[qi];

      uint32_t best_lane = 0;
      float best_dist = std::numeric_limits<float>::max();
      const size_t k_total = (n_panels - 1) * internal::kVnniPoints + valid_in_last;
      for (size_t i = 0; i < k_total; ++i) {
        const size_t panel = i / internal::kVnniPoints;
        const size_t lane = i % internal::kVnniPoints;
        const uint8_t* base = panel_data + panel * panel_bytes;
        constexpr size_t TileN = internal::kVnniPoints * 4;
        int32_t dot = 0;
        for (size_t t = 0; t < total_tiles; ++t) {
          const uint8_t* tile = base + t * TileN + lane * 4;
          for (size_t b = 0; b < 4; ++b) {
            dot += static_cast<int32_t>(tile[b]) * static_cast<int32_t>(qp[t * 4 + b]);
          }
        }
        dot -= 128 * q_bsum_v;
        const float neg_dot =
            -static_cast<float>(dot) * db_norms[panel * internal::kVnniPoints + lane] * q_nsf_v;
        float d;
        if constexpr (Metric) {
          d = db_sqn[panel * internal::kVnniPoints + lane] + 2.0f * neg_dot + q_sqn_v;
        } else {
          d = neg_dot;
        }
        if (d < best_dist) {
          best_dist = d;
          best_lane = static_cast<uint32_t>(i);
        }
      }
      out_argmin[qi] = best_lane;
    }
#endif
  }

#ifdef __AVX512F__
  // Combine running_min[16] + running_argmin_panel[16] into a flat lane index
  // in [0, k).  Picks the lowest set bit on ties (deterministic).
  static inline uint32_t ExtractArgmin_(__m512 mins, __m512i argmin_panels) {
    const float gmin = _mm512_reduce_min_ps(mins);
    const __mmask16 winner = _mm512_cmp_ps_mask(mins, _mm512_set1_ps(gmin), _CMP_EQ_OQ);
    const uint32_t winner_lane = static_cast<uint32_t>(__builtin_ctz(static_cast<uint32_t>(winner)));
    alignas(64) uint32_t panels[internal::kVnniPoints];
    _mm512_store_si512(reinterpret_cast<__m512i*>(panels), argmin_panels);
    return panels[winner_lane] * static_cast<uint32_t>(internal::kVnniPoints) + winner_lane;
  }
#endif
};

}  // namespace turboquant_8bit_mv
}  // namespace mvsic
