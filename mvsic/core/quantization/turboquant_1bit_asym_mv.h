#pragma once

// Multi-vector asymmetric 1-bit TurboQuant (1BTQAsym): 1-bit DB sign codes
// scored against an int4-quantized query. The estimator is structurally
// identical to PQ FastScan — every group of 4 sign bits forms a 4-bit "code",
// and the per-(query, nibble-position) contribution is read from a 16-entry
// int8 LUT keyed by the code value. Chamfer is the sum over query embeddings
// of the per-embedding maximum integer score `s_q` (which corresponds to the
// closest DB point under the unit-norm assumption).
//
// DB layout: FastScan VNNI+VBMI strip layout.
//   For each cloud (padded up to a multiple of 64 db points), the codes are
//   packed in 64-point strips. Per strip (`strip_stride` bytes):
//     byte[g*128 + lp*4 + j] = (code_even_{g*4+j} & 0x0F) | (code_odd_{g*4+j} << 4)
//   where g indexes a group of 4 nibble-positions, lp ∈ [0,32) is a lane pair
//   covering 2 db points (even/odd), and j ∈ [0,4) selects one of the 4
//   nibble-positions in the group. See fastscan_mv.h:53-60.
//
// Query LUT: per query embedding, num_nibble_positions × 16 int8 entries,
// laid out as `lut[g*64 + lp_quad*16 + v]` to match
// `_mm512_permutexvar_epi8(idx, lut512)` with the 0x30201000 inter-row offset.
// Each entry is the signed sum of ±q_int8 over 4 dims and lies in [-28, +28]
// (with int4 query in [-7, +7]).
//
// Inner kernel: nearly verbatim copy of the FastScan
// `scan_64_running_min` / `scan_64_6q_running_min` kernels, with two changes:
//   (1) running reduction is `_mm512_max_epi32` (signed max) instead of
//       `_mm512_min_epu32` — we want the LARGEST integer score per
//       (query, cloud), corresponding to the MIN distance under the
//       monotone-decreasing affine map.
//   (2) VPDPBUSD is called with `ones_i8` as the 1st (u8) operand and the
//       LUT-permuted bytes as the 2nd (s8) operand, so the LUT entries are
//       interpreted as signed.
//
// Per (query, cloud) at the end of the strip sweep, we apply the affine map
// once:
//     IP (non-Metric): dist = -p_nsf * q_nsf * max_s_q
//     L2 (Metric)    : dist = ||q||^2 + ||x||^2 - 2 * p_nsf * q_nsf * max_s_q
//                            = 2 - 2 * p_nsf * q_nsf * max_s_q under unit norms
// where p_nsf = 1/sqrt(D) is constant (all DB vectors are unit-norm) and
// q_nsf = ||q|| / sqrt(sum q_int8^2) is per-query.

#include <vector>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <limits>
#include <algorithm>
#include <fstream>
#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

#include "parlay/parallel.h"
#include "parlay/sequence.h"
#include "mvsic/core/quantization/turboquant_1bit_asym.h"

namespace mvsic {
namespace turboquant_1bit_asym_mv {

namespace internal {

// Match the FastScan VNNI strip stride: 32 bytes per (lp, g) tile of 4 nibble
// positions × 32 lane pairs × 2 db points/pair.
inline size_t compute_strip_stride(uint32_t num_nibble_positions) {
  return static_cast<size_t>(((num_nibble_positions + 3) / 4) * 4) * 32;
}

inline uint32_t lut_padded_blocks(uint32_t num_nibble_positions) {
  return ((num_nibble_positions + 3) / 4) * 4;
}

// Number of queries to batch together in the inner loop; matches
// fastscan_mv::internal::SCAN_Q_BATCH on VNNI.
static constexpr size_t SCAN_Q_BATCH = 6;
static constexpr size_t NQ_BATCH = 64;

// Build the 16-entry int8 LUT for ONE nibble-position from the 4 query int8
// values covering dims [base, base+4). LUT[v] = sum_{k=0..3} sign_k(v) * q[k]
// where sign_k(v) = +1 if bit k of v is 0 (DB coordinate non-negative), -1 if
// bit k is 1 (DB coordinate negative). Result fits in int8 ([-28, 28] given
// int4 query in [-7, 7]).
inline void build_lut_one_position(const int8_t* q4, int8_t* lut16) {
  for (int v = 0; v < 16; ++v) {
    int s = 0;
    for (int k = 0; k < 4; ++k) {
      const int sgn = ((v >> k) & 1) ? -1 : +1;
      s += sgn * static_cast<int>(q4[k]);
    }
    lut16[v] = static_cast<int8_t>(s);
  }
}

// Pack one db point's `padded_dim` rotated values into `num_nibble_positions`
// nibble codes. Nibble at position p (covering dims [p*4, p*4+3]) has bit k
// set iff `rotated[p*4 + k] < 0` (sign convention matches the single-vector
// kernel: bit set = negative).
inline void rotated_to_nibbles(const float* rotated, size_t padded_dim, uint8_t* nibbles) {
  const size_t num_pos = padded_dim / 4;
  for (size_t p = 0; p < num_pos; ++p) {
    uint8_t code = 0;
    for (size_t k = 0; k < 4; ++k) {
      if (rotated[p * 4 + k] < 0.0f) code |= static_cast<uint8_t>(1u << k);
    }
    nibbles[p] = code;
  }
}

#if defined(__AVX512VNNI__) && defined(__AVX512VBMI__)

// Per-strip running max accumulators (lo / hi each hold 16 int32 lanes,
// covering 32 of the 64 db points in the strip).
struct RunningMaxV {
  __m512i lo, hi;
  void set_min() {
    const __m512i kIntMin = _mm512_set1_epi32(std::numeric_limits<int32_t>::min());
    lo = kIntMin;
    hi = kIntMin;
  }
  static RunningMaxV min() {
    RunningMaxV r;
    r.set_min();
    return r;
  }
};

// Horizontal max over 16 signed int32 lanes of a __m512i.
inline int32_t hmax_512_epi32(__m512i v) {
  v = _mm512_max_epi32(v, _mm512_shuffle_i32x4(v, v, _MM_SHUFFLE(1, 0, 3, 2)));
  v = _mm512_max_epi32(v, _mm512_shuffle_i32x4(v, v, _MM_SHUFFLE(0, 0, 1, 1)));
  __m128i v128 = _mm512_castsi512_si128(v);
  v128 = _mm_max_epi32(v128, _mm_srli_si128(v128, 8));
  v128 = _mm_max_epi32(v128, _mm_srli_si128(v128, 4));
  return _mm_extract_epi32(v128, 0);
}

inline int32_t reduce_running_max(const RunningMaxV& r) {
  return hmax_512_epi32(_mm512_max_epi32(r.lo, r.hi));
}

// Single-query scan kernel: sweep `num_blocks` nibble positions across one
// 64-point strip and accumulate signed-int32 running max into
// `current_max_v`. Mirrors fastscan_mv::scan_64_running_min, with the
// signed-max reduction and the LUT-as-s8 VPDPBUSD operand swap.
inline void scan_64_running_max(const int8_t* lut, const uint8_t* codes_ptr, uint32_t num_blocks,
                                RunningMaxV& current_max_v) {
  __m512i ae_lo = _mm512_setzero_si512(), ae_hi = _mm512_setzero_si512();
  __m512i ao_lo = _mm512_setzero_si512(), ao_hi = _mm512_setzero_si512();
  const __m512i low_mask = _mm512_set1_epi8(0x0F);
  const __m512i ones_i8 = _mm512_set1_epi8(1);
  const __m512i blk_off = _mm512_set1_epi32(0x30201000);
  const uint32_t nb4 = ((num_blocks + 3) / 4) * 4;

  for (uint32_t g = 0; g < nb4 / 4; ++g) {
    const __m512i p_lo = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(codes_ptr));
    const __m512i p_hi = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(codes_ptr + 64));
    codes_ptr += 128;
    const __m512i lut512 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&lut[g * 64]));

    __m512i ce_lo = _mm512_or_si512(_mm512_and_si512(p_lo, low_mask), blk_off);
    __m512i ce_hi = _mm512_or_si512(_mm512_and_si512(p_hi, low_mask), blk_off);
    __m512i co_lo = _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(p_lo, 4), low_mask),
                                    blk_off);
    __m512i co_hi = _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(p_hi, 4), low_mask),
                                    blk_off);

    // ones is u8 (1st operand); LUT-permuted bytes are s8 (2nd operand).
    // Each vpdpbusd sums 4 signed-i8 LUT entries into one i32 lane.
    ae_lo = _mm512_dpbusd_epi32(ae_lo, ones_i8, _mm512_permutexvar_epi8(ce_lo, lut512));
    ae_hi = _mm512_dpbusd_epi32(ae_hi, ones_i8, _mm512_permutexvar_epi8(ce_hi, lut512));
    ao_lo = _mm512_dpbusd_epi32(ao_lo, ones_i8, _mm512_permutexvar_epi8(co_lo, lut512));
    ao_hi = _mm512_dpbusd_epi32(ao_hi, ones_i8, _mm512_permutexvar_epi8(co_hi, lut512));
  }
  current_max_v.lo = _mm512_max_epi32(current_max_v.lo, _mm512_max_epi32(ae_lo, ao_lo));
  current_max_v.hi = _mm512_max_epi32(current_max_v.hi, _mm512_max_epi32(ae_hi, ao_hi));
}

// Single-query masked variant: only db lanes in [lo, hi) contribute to the
// running max. Used on the trailing partial strip when cloud_size % 64 != 0.
inline int32_t scan_64_chunk_max_masked(const int8_t* lut, const uint8_t* codes_ptr,
                                        uint32_t num_blocks, int lo, int hi) {
  lo = std::max(lo, 0);
  hi = std::min(hi, 64);
  if (hi <= lo) return std::numeric_limits<int32_t>::min();

  __m512i ae_lo = _mm512_setzero_si512(), ae_hi = _mm512_setzero_si512();
  __m512i ao_lo = _mm512_setzero_si512(), ao_hi = _mm512_setzero_si512();
  const __m512i low_mask = _mm512_set1_epi8(0x0F);
  const __m512i ones_i8 = _mm512_set1_epi8(1);
  const __m512i blk_off = _mm512_set1_epi32(0x30201000);
  const uint32_t nb4 = ((num_blocks + 3) / 4) * 4;

  for (uint32_t g = 0; g < nb4 / 4; ++g) {
    const __m512i p_lo = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(codes_ptr));
    const __m512i p_hi = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(codes_ptr + 64));
    codes_ptr += 128;
    const __m512i lut512 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&lut[g * 64]));

    __m512i ce_lo = _mm512_or_si512(_mm512_and_si512(p_lo, low_mask), blk_off);
    __m512i ce_hi = _mm512_or_si512(_mm512_and_si512(p_hi, low_mask), blk_off);
    __m512i co_lo = _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(p_lo, 4), low_mask),
                                    blk_off);
    __m512i co_hi = _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(p_hi, 4), low_mask),
                                    blk_off);

    ae_lo = _mm512_dpbusd_epi32(ae_lo, ones_i8, _mm512_permutexvar_epi8(ce_lo, lut512));
    ae_hi = _mm512_dpbusd_epi32(ae_hi, ones_i8, _mm512_permutexvar_epi8(ce_hi, lut512));
    ao_lo = _mm512_dpbusd_epi32(ao_lo, ones_i8, _mm512_permutexvar_epi8(co_lo, lut512));
    ao_hi = _mm512_dpbusd_epi32(ao_hi, ones_i8, _mm512_permutexvar_epi8(co_hi, lut512));
  }

  // Lane-mask construction: each 16-lane accumulator covers 16 of the 32
  // even (or odd) db points in this strip; the lane→db_point mapping matches
  // FastScan's masked variant. Masked-out lanes get INT32_MIN so they lose
  // the max.
  const __m512i NEG_INF = _mm512_set1_epi32(std::numeric_limits<int32_t>::min());
  __mmask16 me_lo = 0, me_hi = 0, mo_lo = 0, mo_hi = 0;
  for (int lp = 0; lp < 16; ++lp) {
    if (lo <= lp * 2 && lp * 2 < hi) me_lo |= (__mmask16(1) << lp);
    if (lo <= (lp + 16) * 2 && (lp + 16) * 2 < hi) me_hi |= (__mmask16(1) << lp);
    if (lo <= lp * 2 + 1 && lp * 2 + 1 < hi) mo_lo |= (__mmask16(1) << lp);
    if (lo <= (lp + 16) * 2 + 1 && (lp + 16) * 2 + 1 < hi) mo_hi |= (__mmask16(1) << lp);
  }
  ae_lo = _mm512_mask_mov_epi32(NEG_INF, me_lo, ae_lo);
  ae_hi = _mm512_mask_mov_epi32(NEG_INF, me_hi, ae_hi);
  ao_lo = _mm512_mask_mov_epi32(NEG_INF, mo_lo, ao_lo);
  ao_hi = _mm512_mask_mov_epi32(NEG_INF, mo_hi, ao_hi);

  __m512i m = _mm512_max_epi32(_mm512_max_epi32(ae_lo, ao_lo), _mm512_max_epi32(ae_hi, ao_hi));
  return hmax_512_epi32(m);
}

// 6-query batched scan kernel. Streams one 64-point strip, computes 6
// per-query LUT lookups + accumulators per (g, lp_quad). Mirrors
// fastscan_mv::scan_64_6q_running_min with the same signed-max changes.
inline void scan_64_6q_running_max(
    const int8_t* lut0, const int8_t* lut1, const int8_t* lut2,
    const int8_t* lut3, const int8_t* lut4, const int8_t* lut5,
    const uint8_t* codes_ptr, uint32_t num_blocks,
    RunningMaxV& rm0, RunningMaxV& rm1, RunningMaxV& rm2,
    RunningMaxV& rm3, RunningMaxV& rm4, RunningMaxV& rm5) {

  __m512i a0el = _mm512_setzero_si512(), a0ol = _mm512_setzero_si512();
  __m512i a0eh = _mm512_setzero_si512(), a0oh = _mm512_setzero_si512();
  __m512i a1el = _mm512_setzero_si512(), a1ol = _mm512_setzero_si512();
  __m512i a1eh = _mm512_setzero_si512(), a1oh = _mm512_setzero_si512();
  __m512i a2el = _mm512_setzero_si512(), a2ol = _mm512_setzero_si512();
  __m512i a2eh = _mm512_setzero_si512(), a2oh = _mm512_setzero_si512();
  __m512i a3el = _mm512_setzero_si512(), a3ol = _mm512_setzero_si512();
  __m512i a3eh = _mm512_setzero_si512(), a3oh = _mm512_setzero_si512();
  __m512i a4el = _mm512_setzero_si512(), a4ol = _mm512_setzero_si512();
  __m512i a4eh = _mm512_setzero_si512(), a4oh = _mm512_setzero_si512();
  __m512i a5el = _mm512_setzero_si512(), a5ol = _mm512_setzero_si512();
  __m512i a5eh = _mm512_setzero_si512(), a5oh = _mm512_setzero_si512();

  const __m512i low_mask = _mm512_set1_epi8(0x0F);
  const __m512i ones_i8 = _mm512_set1_epi8(1);
  const __m512i blk_off = _mm512_set1_epi32(0x30201000);
  const uint32_t nb4 = ((num_blocks + 3) / 4) * 4;

  for (uint32_t g = 0; g < nb4 / 4; ++g) {
    const __m512i p_lo = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(codes_ptr));
    const __m512i p_hi = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(codes_ptr + 64));
    codes_ptr += 128;

    const __m512i ce_lo = _mm512_or_si512(_mm512_and_si512(p_lo, low_mask), blk_off);
    const __m512i ce_hi = _mm512_or_si512(_mm512_and_si512(p_hi, low_mask), blk_off);
    const __m512i co_lo = _mm512_or_si512(
        _mm512_and_si512(_mm512_srli_epi16(p_lo, 4), low_mask), blk_off);
    const __m512i co_hi = _mm512_or_si512(
        _mm512_and_si512(_mm512_srli_epi16(p_hi, 4), low_mask), blk_off);

#define TQ1BTQA_VNNI_Q(Q, lp)                                                                     \
    {                                                                                              \
      const __m512i lut512 =                                                                       \
          _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&lp[g * 64]));                       \
      a##Q##el = _mm512_dpbusd_epi32(a##Q##el, ones_i8, _mm512_permutexvar_epi8(ce_lo, lut512));  \
      a##Q##eh = _mm512_dpbusd_epi32(a##Q##eh, ones_i8, _mm512_permutexvar_epi8(ce_hi, lut512));  \
      a##Q##ol = _mm512_dpbusd_epi32(a##Q##ol, ones_i8, _mm512_permutexvar_epi8(co_lo, lut512));  \
      a##Q##oh = _mm512_dpbusd_epi32(a##Q##oh, ones_i8, _mm512_permutexvar_epi8(co_hi, lut512));  \
    }
    TQ1BTQA_VNNI_Q(0, lut0)
    TQ1BTQA_VNNI_Q(1, lut1)
    TQ1BTQA_VNNI_Q(2, lut2)
    TQ1BTQA_VNNI_Q(3, lut3)
    TQ1BTQA_VNNI_Q(4, lut4)
    TQ1BTQA_VNNI_Q(5, lut5)
#undef TQ1BTQA_VNNI_Q
  }

#define TQ1BTQA_VNNI_REDUCE(Q, rm)                                                       \
  rm.lo = _mm512_max_epi32(rm.lo, _mm512_max_epi32(a##Q##el, a##Q##ol));                 \
  rm.hi = _mm512_max_epi32(rm.hi, _mm512_max_epi32(a##Q##eh, a##Q##oh));
  TQ1BTQA_VNNI_REDUCE(0, rm0)
  TQ1BTQA_VNNI_REDUCE(1, rm1)
  TQ1BTQA_VNNI_REDUCE(2, rm2)
  TQ1BTQA_VNNI_REDUCE(3, rm3)
  TQ1BTQA_VNNI_REDUCE(4, rm4)
  TQ1BTQA_VNNI_REDUCE(5, rm5)
#undef TQ1BTQA_VNNI_REDUCE
}

inline void scan_64_6q_running_max_masked(
    const int8_t* lut0, const int8_t* lut1, const int8_t* lut2,
    const int8_t* lut3, const int8_t* lut4, const int8_t* lut5,
    const uint8_t* codes_ptr, uint32_t num_blocks,
    int lo, int hi,
    RunningMaxV& rm0, RunningMaxV& rm1, RunningMaxV& rm2,
    RunningMaxV& rm3, RunningMaxV& rm4, RunningMaxV& rm5) {

  lo = std::max(lo, 0);
  hi = std::min(hi, 64);
  if (hi <= lo) return;

  __m512i a0el = _mm512_setzero_si512(), a0ol = _mm512_setzero_si512();
  __m512i a0eh = _mm512_setzero_si512(), a0oh = _mm512_setzero_si512();
  __m512i a1el = _mm512_setzero_si512(), a1ol = _mm512_setzero_si512();
  __m512i a1eh = _mm512_setzero_si512(), a1oh = _mm512_setzero_si512();
  __m512i a2el = _mm512_setzero_si512(), a2ol = _mm512_setzero_si512();
  __m512i a2eh = _mm512_setzero_si512(), a2oh = _mm512_setzero_si512();
  __m512i a3el = _mm512_setzero_si512(), a3ol = _mm512_setzero_si512();
  __m512i a3eh = _mm512_setzero_si512(), a3oh = _mm512_setzero_si512();
  __m512i a4el = _mm512_setzero_si512(), a4ol = _mm512_setzero_si512();
  __m512i a4eh = _mm512_setzero_si512(), a4oh = _mm512_setzero_si512();
  __m512i a5el = _mm512_setzero_si512(), a5ol = _mm512_setzero_si512();
  __m512i a5eh = _mm512_setzero_si512(), a5oh = _mm512_setzero_si512();

  const __m512i low_mask = _mm512_set1_epi8(0x0F);
  const __m512i ones_i8 = _mm512_set1_epi8(1);
  const __m512i blk_off = _mm512_set1_epi32(0x30201000);
  const uint32_t nb4 = ((num_blocks + 3) / 4) * 4;

  for (uint32_t g = 0; g < nb4 / 4; ++g) {
    const __m512i p_lo = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(codes_ptr));
    const __m512i p_hi = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(codes_ptr + 64));
    codes_ptr += 128;

    const __m512i ce_lo = _mm512_or_si512(_mm512_and_si512(p_lo, low_mask), blk_off);
    const __m512i ce_hi = _mm512_or_si512(_mm512_and_si512(p_hi, low_mask), blk_off);
    const __m512i co_lo = _mm512_or_si512(
        _mm512_and_si512(_mm512_srli_epi16(p_lo, 4), low_mask), blk_off);
    const __m512i co_hi = _mm512_or_si512(
        _mm512_and_si512(_mm512_srli_epi16(p_hi, 4), low_mask), blk_off);

#define TQ1BTQA_VNNI_QM(Q, lp)                                                                    \
    {                                                                                              \
      const __m512i lut512 =                                                                       \
          _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&lp[g * 64]));                       \
      a##Q##el = _mm512_dpbusd_epi32(a##Q##el, ones_i8, _mm512_permutexvar_epi8(ce_lo, lut512));  \
      a##Q##eh = _mm512_dpbusd_epi32(a##Q##eh, ones_i8, _mm512_permutexvar_epi8(ce_hi, lut512));  \
      a##Q##ol = _mm512_dpbusd_epi32(a##Q##ol, ones_i8, _mm512_permutexvar_epi8(co_lo, lut512));  \
      a##Q##oh = _mm512_dpbusd_epi32(a##Q##oh, ones_i8, _mm512_permutexvar_epi8(co_hi, lut512));  \
    }
    TQ1BTQA_VNNI_QM(0, lut0)
    TQ1BTQA_VNNI_QM(1, lut1)
    TQ1BTQA_VNNI_QM(2, lut2)
    TQ1BTQA_VNNI_QM(3, lut3)
    TQ1BTQA_VNNI_QM(4, lut4)
    TQ1BTQA_VNNI_QM(5, lut5)
#undef TQ1BTQA_VNNI_QM
  }

  const __m512i NEG_INF = _mm512_set1_epi32(std::numeric_limits<int32_t>::min());
  __mmask16 me_lo = 0, me_hi = 0, mo_lo = 0, mo_hi = 0;
  for (int lp = 0; lp < 16; ++lp) {
    if (lo <= lp * 2 && lp * 2 < hi) me_lo |= (__mmask16(1) << lp);
    if (lo <= (lp + 16) * 2 && (lp + 16) * 2 < hi) me_hi |= (__mmask16(1) << lp);
    if (lo <= lp * 2 + 1 && lp * 2 + 1 < hi) mo_lo |= (__mmask16(1) << lp);
    if (lo <= (lp + 16) * 2 + 1 && (lp + 16) * 2 + 1 < hi) mo_hi |= (__mmask16(1) << lp);
  }

#define TQ1BTQA_VNNI_MREDUCE(Q, rm)                                                                \
  rm.lo = _mm512_max_epi32(rm.lo, _mm512_max_epi32(                                               \
      _mm512_mask_mov_epi32(NEG_INF, me_lo, a##Q##el),                                            \
      _mm512_mask_mov_epi32(NEG_INF, mo_lo, a##Q##ol)));                                          \
  rm.hi = _mm512_max_epi32(rm.hi, _mm512_max_epi32(                                               \
      _mm512_mask_mov_epi32(NEG_INF, me_hi, a##Q##eh),                                            \
      _mm512_mask_mov_epi32(NEG_INF, mo_hi, a##Q##oh)));
  TQ1BTQA_VNNI_MREDUCE(0, rm0)
  TQ1BTQA_VNNI_MREDUCE(1, rm1)
  TQ1BTQA_VNNI_MREDUCE(2, rm2)
  TQ1BTQA_VNNI_MREDUCE(3, rm3)
  TQ1BTQA_VNNI_MREDUCE(4, rm4)
  TQ1BTQA_VNNI_MREDUCE(5, rm5)
#undef TQ1BTQA_VNNI_MREDUCE
}

inline void scan_64_batched(const int8_t* const* luts, size_t nq, const uint8_t* codes_ptr,
                            uint32_t num_blocks, RunningMaxV* rmaxs) {
  size_t qi = 0;
  for (; qi + 5 < nq; qi += 6)
    scan_64_6q_running_max(luts[qi], luts[qi + 1], luts[qi + 2], luts[qi + 3], luts[qi + 4],
                           luts[qi + 5], codes_ptr, num_blocks, rmaxs[qi], rmaxs[qi + 1],
                           rmaxs[qi + 2], rmaxs[qi + 3], rmaxs[qi + 4], rmaxs[qi + 5]);
  for (; qi < nq; ++qi)
    scan_64_running_max(luts[qi], codes_ptr, num_blocks, rmaxs[qi]);
}

inline void scan_64_batched_masked(const int8_t* const* luts, size_t nq,
                                   const uint8_t* codes_ptr, uint32_t num_blocks, int lo, int hi,
                                   RunningMaxV* rmaxs) {
  size_t qi = 0;
  for (; qi + 5 < nq; qi += 6)
    scan_64_6q_running_max_masked(luts[qi], luts[qi + 1], luts[qi + 2], luts[qi + 3], luts[qi + 4],
                                  luts[qi + 5], codes_ptr, num_blocks, lo, hi, rmaxs[qi],
                                  rmaxs[qi + 1], rmaxs[qi + 2], rmaxs[qi + 3], rmaxs[qi + 4],
                                  rmaxs[qi + 5]);
  for (; qi < nq; ++qi) {
    int32_t t = scan_64_chunk_max_masked(luts[qi], codes_ptr, num_blocks, lo, hi);
    __m512i tv = _mm512_set1_epi32(t);
    rmaxs[qi].lo = _mm512_max_epi32(rmaxs[qi].lo, tv);
    rmaxs[qi].hi = _mm512_max_epi32(rmaxs[qi].hi, tv);
  }
}

#else  // !VNNI+VBMI

// Scalar fallback running-max accumulator: one int32 per db lane in a 64-pt
// strip (only used when running on a non-VNNI/VBMI CPU; correctness only).
struct RunningMaxV {
  int32_t data[64];
  void set_min() {
    for (int i = 0; i < 64; ++i) data[i] = std::numeric_limits<int32_t>::min();
  }
  static RunningMaxV min() {
    RunningMaxV r;
    r.set_min();
    return r;
  }
};

inline int32_t reduce_running_max(const RunningMaxV& r) {
  int32_t best = std::numeric_limits<int32_t>::min();
  for (int i = 0; i < 64; ++i) best = std::max(best, r.data[i]);
  return best;
}

#endif

}  // namespace internal

// =========================================================================
// Forward declarations
// =========================================================================
template<bool Metric>
class Quantized_Point_Cloud_Set;
template<bool Metric>
class Quantized_Query_Point_Cloud;

// =========================================================================
// Cloud handle (lightweight view)
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

template<bool Metric>
float chamfer_distance(const Quantized_Query_Point_Cloud<Metric>& q,
                       const Quantized_Point_Cloud_Set<Metric>& db, size_t start_vec,
                       size_t end_vec);

// =========================================================================
// Multi-Vector Query
// =========================================================================
template<bool Metric>
class Quantized_Query_Point_Cloud {
 public:
  using distanceType = float;

  size_t num_queries = 0;
  uint32_t num_nibble_positions = 0;  // padded_dim / 4
  uint32_t padded_dim = 0;

  // Flat int8 LUTs: layout matches scan_64_running_max's expectation.
  // size = num_queries * lut_padded_blocks(num_nibble_positions) * 16
  std::vector<int8_t> flat_int_luts;

  // Per-query "scaled" norm scaling factor: q_nsf * (1/sqrt(D)). Multiplies
  // max_s_q to give the IP estimate for that query against the closest DB
  // point (under the unit-norm DB assumption, so p_nsf = 1/sqrt(D) is folded
  // in here).
  std::vector<float> scaled_q_nsf;

  // Per-query unquantized squared norm (||q||^2). Only consumed in Metric=L2
  // path; left zero on the IP path.
  std::vector<float> unquantized_squared_norms;

  Quantized_Query_Point_Cloud() = default;

  inline const int8_t* get_lut(size_t qi) const {
    const uint32_t lpb = internal::lut_padded_blocks(num_nibble_positions);
    return flat_int_luts.data() + qi * static_cast<size_t>(lpb) * 16;
  }

  // Decode one (query, cloud) running max into a float distance under the
  // unit-norm assumption:
  //   IP   :  dist = -scaled_q_nsf[qi] * max_s_q
  //   L2   :  dist = ||q||^2 + 1 - 2 * scaled_q_nsf[qi] * max_s_q
  //              ≈ 2 - 2 * scaled_q_nsf[qi] * max_s_q (DB ||x||=1)
  inline float decode(size_t qi, int32_t max_s_q) const {
    const float k = scaled_q_nsf[qi] * static_cast<float>(max_s_q);
    if constexpr (Metric) {
      // ||x||^2 ≈ 1 (unit-norm DB); ||q||^2 stored exactly.
      return unquantized_squared_norms[qi] + 1.0f - 2.0f * k;
    } else {
      return -k;
    }
  }

  template<typename CloudView>
  float distance(const CloudView& cloud) const {
    return chamfer_distance(*this, *cloud.db, cloud.start_idx, cloud.end_idx);
  }

  template<typename CloudHandle>
  std::pair<float, size_t> distance_w_cmps(const CloudHandle& cloud) const {
    const size_t cloud_size = cloud.end_idx - cloud.start_idx;
    const size_t bytes_per_vec = (cloud.db->num_nibble_positions + 1) / 2;
    return {this->distance(cloud), cloud_size * bytes_per_vec};
  }

  static constexpr bool is_metric() { return Metric; }
};

// =========================================================================
// Multi-Vector Set (FastScan strip layout)
// =========================================================================
template<bool Metric>
class Quantized_Point_Cloud_Set {
 public:
  parlay::sequence<uint8_t> packed_codes;
  uint32_t num_nibble_positions = 0;  // padded_dim / 4
  uint32_t padded_dim = 0;

  // One offset per cloud (in units of padded db points = strip lanes).
  // offsets[c+1] - offsets[c] is a multiple of 64 (one strip).
  parlay::sequence<size_t> offsets;
  parlay::sequence<uint32_t> sizes_unpadded;  // actual cloud size, ≤ offset diff
  parlay::sequence<uint32_t> ids;

  Quantized_Point_Cloud_Set() = default;
  static constexpr bool is_metric() noexcept { return Metric; }

  Quantized_Point_Cloud<Metric> operator[](size_t i) const {
    const size_t n_clouds = (offsets.size() > 0) ? offsets.size() - 1 : 0;
    const size_t start = offsets[i];
    size_t cloud_size = 0;
    if (sizes_unpadded.size() == static_cast<size_t>(n_clouds)) {
      cloud_size = static_cast<size_t>(sizes_unpadded[i]);
    } else {
      const size_t end_padded = offsets[i + 1];
      cloud_size = (end_padded > start) ? (end_padded - start) : 0;
    }
    return Quantized_Point_Cloud<Metric>(this, start, start + cloud_size);
  }

  inline uint32_t get_id(size_t i) const noexcept { return (ids.size() > 0) ? ids[i] : i; }
  inline size_t num_bytes() const noexcept { return packed_codes.size() * sizeof(uint8_t); }

  inline size_t num_clouds() const noexcept {
    return (offsets.size() > 0) ? offsets.size() - 1 : 0;
  }

  void distances_all(const Quantized_Query_Point_Cloud<Metric>& q,
                     std::pair<uint32_t, float>* results) const {
    const size_t num_q = q.num_queries;
    const size_t nc = num_clouds();
    if (num_q == 0 || nc == 0) return;

    parlay::parallel_for(0, nc, [&](size_t cid) {
      const size_t start = offsets[cid];
      size_t cloud_size = 0;
      if (sizes_unpadded.size() == static_cast<size_t>(nc)) {
        cloud_size = static_cast<size_t>(sizes_unpadded[cid]);
      } else {
        const size_t end_padded = offsets[cid + 1];
        cloud_size = (end_padded > start) ? (end_padded - start) : 0;
      }
      const float d = chamfer_distance(q, *this, start, start + cloud_size);
      results[cid] = {get_id(static_cast<uint32_t>(cid)), d};
    });
  }

  void save(std::ofstream& out) const {
    out.write(reinterpret_cast<const char*>(&num_nibble_positions),
              sizeof(num_nibble_positions));
    out.write(reinterpret_cast<const char*>(&padded_dim), sizeof(padded_dim));
    size_t pc_sz = packed_codes.size();
    out.write(reinterpret_cast<const char*>(&pc_sz), sizeof(pc_sz));
    if (pc_sz)
      out.write(reinterpret_cast<const char*>(packed_codes.data()), pc_sz * sizeof(uint8_t));

    size_t n_clouds = num_clouds();
    out.write(reinterpret_cast<const char*>(&n_clouds), sizeof(n_clouds));

    size_t off_size = offsets.size();
    out.write(reinterpret_cast<const char*>(&off_size), sizeof(off_size));
    if (off_size)
      out.write(reinterpret_cast<const char*>(offsets.data()), off_size * sizeof(size_t));

    size_t sz_size = sizes_unpadded.size();
    out.write(reinterpret_cast<const char*>(&sz_size), sizeof(sz_size));
    if (sz_size)
      out.write(reinterpret_cast<const char*>(sizes_unpadded.data()), sz_size * sizeof(uint32_t));

    size_t ids_size = ids.size();
    out.write(reinterpret_cast<const char*>(&ids_size), sizeof(ids_size));
    if (ids_size)
      out.write(reinterpret_cast<const char*>(ids.data()), ids_size * sizeof(uint32_t));
  }

  void load(std::ifstream& in) {
    in.read(reinterpret_cast<char*>(&num_nibble_positions), sizeof(num_nibble_positions));
    in.read(reinterpret_cast<char*>(&padded_dim), sizeof(padded_dim));
    size_t pc_sz = 0;
    in.read(reinterpret_cast<char*>(&pc_sz), sizeof(pc_sz));
    packed_codes.resize(pc_sz);
    if (pc_sz) in.read(reinterpret_cast<char*>(packed_codes.data()), pc_sz * sizeof(uint8_t));

    size_t n_clouds = 0;
    in.read(reinterpret_cast<char*>(&n_clouds), sizeof(n_clouds));

    size_t off_size = 0;
    in.read(reinterpret_cast<char*>(&off_size), sizeof(off_size));
    offsets.resize(off_size);
    if (off_size) in.read(reinterpret_cast<char*>(offsets.data()), off_size * sizeof(size_t));

    size_t sz_size = 0;
    in.read(reinterpret_cast<char*>(&sz_size), sizeof(sz_size));
    sizes_unpadded.resize(sz_size);
    if (sz_size)
      in.read(reinterpret_cast<char*>(sizes_unpadded.data()), sz_size * sizeof(uint32_t));

    size_t ids_size = 0;
    in.read(reinterpret_cast<char*>(&ids_size), sizeof(ids_size));
    ids.resize(ids_size);
    if (ids_size) in.read(reinterpret_cast<char*>(ids.data()), ids_size * sizeof(uint32_t));
  }
};

// =========================================================================
// Per-leaf chamfer distance: one query cloud × one db cloud.
// =========================================================================
template<bool Metric>
float chamfer_distance(const Quantized_Query_Point_Cloud<Metric>& q,
                       const Quantized_Point_Cloud_Set<Metric>& db, size_t start_vec,
                       size_t end_vec) {
  const size_t num_q = q.num_queries;
  if (num_q == 0) return 0.0f;
  if (end_vec <= start_vec) return std::numeric_limits<float>::max();

  const size_t cloud_size = end_vec - start_vec;

#if defined(__AVX512VNNI__) && defined(__AVX512VBMI__)
  const uint32_t num_blocks = db.num_nibble_positions;
  const size_t strip_stride = internal::compute_strip_stride(num_blocks);
  const size_t strip0 = start_vec / 64;
  const size_t num_full_strips = cloud_size / 64;
  const int tail = static_cast<int>(cloud_size % 64);
  const size_t total_strips = num_full_strips + (tail > 0 ? 1 : 0);

  auto strip_ptr = [&](size_t s) -> const uint8_t* {
    return &db.packed_codes[s * strip_stride];
  };

  float total_chamfer = 0.0f;

  for (size_t q_base = 0; q_base < num_q; q_base += internal::NQ_BATCH) {
    const size_t q_count = std::min(internal::NQ_BATCH, num_q - q_base);

    const int8_t* lut_ptrs[internal::NQ_BATCH];
    for (size_t qi = 0; qi < q_count; ++qi)
      lut_ptrs[qi] = q.get_lut(q_base + qi);

    internal::RunningMaxV rmaxs[internal::NQ_BATCH];
    for (size_t qi = 0; qi < q_count; ++qi)
      rmaxs[qi] = internal::RunningMaxV::min();

    for (size_t s = 0; s < num_full_strips; ++s) {
      const uint8_t* sp = strip_ptr(strip0 + s);
      if (s + 2 < total_strips) __builtin_prefetch(strip_ptr(strip0 + s + 2), 0, 3);
      internal::scan_64_batched(lut_ptrs, q_count, sp, num_blocks, rmaxs);
    }

    if (tail > 0) {
      internal::scan_64_batched_masked(lut_ptrs, q_count, strip_ptr(strip0 + num_full_strips),
                                       num_blocks, 0, tail, rmaxs);
    }

    for (size_t qi = 0; qi < q_count; ++qi) {
      const int32_t mx = internal::reduce_running_max(rmaxs[qi]);
      total_chamfer += q.decode(q_base + qi, mx);
    }
  }

  return total_chamfer / static_cast<float>(num_q);
#else
  // Scalar fallback (correctness only): for each query, for each db point in
  // the cloud, compute s_q from the per-query LUT and the per-point nibbles
  // by re-extracting them from the strip layout.
  const uint32_t num_blocks = db.num_nibble_positions;
  const size_t strip_stride = internal::compute_strip_stride(num_blocks);
  const uint32_t nb4 = ((num_blocks + 3) / 4) * 4;
  const uint32_t lpb = internal::lut_padded_blocks(num_blocks);
  const size_t strip0 = start_vec / 64;

  float total_chamfer = 0.0f;
  for (size_t qi = 0; qi < num_q; ++qi) {
    const int8_t* lut = q.get_lut(qi);
    int32_t max_s_q = std::numeric_limits<int32_t>::min();
    for (size_t i = 0; i < cloud_size; ++i) {
      const size_t lp = (i / 2) % 32;
      const size_t parity = i & 1u;  // 0 = even, 1 = odd
      const size_t strip_idx = strip0 + (i / 64);
      const uint8_t* strip_base = &db.packed_codes[strip_idx * strip_stride];
      int32_t s_q = 0;
      for (uint32_t g = 0; g < nb4 / 4; ++g) {
        const uint8_t* tile = strip_base + g * 128 + lp * 4;
        for (uint32_t j = 0; j < 4; ++j) {
          const uint32_t pos = g * 4 + j;
          if (pos >= num_blocks) continue;
          const uint8_t code = parity == 0 ? (tile[j] & 0x0F) : (tile[j] >> 4);
          s_q += static_cast<int32_t>(lut[g * 64 + j * 16 + code]);
        }
      }
      // Note: lp covers only (i/2)%32, but the strip holds 64 db points across
      // 32 lane pairs split by even/odd. The above already encodes that via
      // `parity`, but we must also account for the second half of the strip
      // (lp+16 worth of points). Use the running cloud index `i` directly so
      // we don't need to re-derive lp_in_strip.
      (void)lpb;
      if (s_q > max_s_q) max_s_q = s_q;
    }
    total_chamfer += q.decode(qi, max_s_q);
  }
  return total_chamfer / static_cast<float>(num_q);
#endif
}

// =========================================================================
// Model: rotator + DB encoder + query LUT builder.
// =========================================================================
template<bool Metric>
class Model {
 public:
  static constexpr uint32_t kClassId = 7;  // OneBitTQAsym
  static constexpr uint32_t kBatchAlignment = 6;  // SCAN_Q_BATCH
  static constexpr const char* kName = "1bit_tqa";
  using EncodedSet = ::mvsic::turboquant_1bit_asym_mv::Quantized_Point_Cloud_Set<Metric>;
  using EncodedQuery = ::mvsic::turboquant_1bit_asym_mv::Quantized_Query_Point_Cloud<Metric>;
  struct Params {};

  // Reuses the single-vector kernel's encoder (rotator + sign packing).
  ::mvsic::turboquant_1bit_asym::BaseEncoder1BitAsym encoder;

  Model() = default;

  template<typename PCSet>
  void train(const PCSet& pcs, const Params& /*p*/) {
    train(pcs);
  }

  template<typename PCSet>
  void train(const PCSet& pcs) {
    encoder.train(pcs.get_dims());
  }

  // Encode a PointCloudSet into the strip layout. Each cloud is padded up to
  // a multiple of 64 db points; padding lanes hold the all-zero nibble code
  // (0 = "all positive" sign pattern) and are excluded from the running max
  // by the masked tail scan in the per-cloud kernel.
  template<typename PCSet>
  Quantized_Point_Cloud_Set<Metric> encode(const PCSet& pcs) const {
    Quantized_Point_Cloud_Set<Metric> res;
    res.padded_dim = static_cast<uint32_t>(encoder.padded_dim);
    res.num_nibble_positions = static_cast<uint32_t>(encoder.padded_dim / 4);

    auto float_offsets = pcs.get_offsets();
    const size_t n_clouds = float_offsets.size() > 0 ? float_offsets.size() - 1 : 0;

    res.offsets.resize(n_clouds + 1);
    res.offsets[0] = 0;
    res.sizes_unpadded = parlay::sequence<uint32_t>(n_clouds);

    size_t cur_padded = 0;
    for (size_t c = 0; c < n_clouds; ++c) {
      res.offsets[c] = cur_padded;
      const size_t n_vecs = (float_offsets[c + 1] - float_offsets[c]) / encoder.dim;
      res.sizes_unpadded[c] = static_cast<uint32_t>(n_vecs);
      cur_padded += ((n_vecs + 63) / 64) * 64;
    }
    res.offsets[n_clouds] = cur_padded;

    const size_t n_strips = cur_padded / 64;
    const size_t strip_stride = internal::compute_strip_stride(res.num_nibble_positions);
    res.packed_codes.resize(n_strips * strip_stride, 0);

    auto pcs_ids = pcs.get_ids();
    res.ids = parlay::sequence<uint32_t>(pcs_ids.begin(), pcs_ids.end());

    if (encoder.padded_dim == 0) return res;

    parlay::parallel_for(0, n_clouds, [&](size_t c) {
      const size_t n_vecs = res.sizes_unpadded[c];
      if (n_vecs == 0) return;

      const size_t src_start = float_offsets[c] / encoder.dim;
      const size_t dst_start = res.offsets[c];
      const size_t padded_sz = res.offsets[c + 1] - res.offsets[c];
      const size_t n_strips_c = padded_sz / 64;
      const size_t strip0_dst = dst_start / 64;
      const uint32_t num_blocks = res.num_nibble_positions;
      const uint32_t nb4 = ((num_blocks + 3) / 4) * 4;

      // Per-cloud temp: one nibble byte per (point, nibble-position).
      std::vector<uint8_t> nibbles_per_pt(n_vecs * num_blocks, 0);
      std::vector<float> ws(encoder.padded_dim);
      for (size_t i = 0; i < n_vecs; ++i) {
        const float* p = reinterpret_cast<const float*>(pcs.data()
                                                        + (src_start + i) * encoder.dim);
        encoder.rotator->rotate(p, ws.data());
        internal::rotated_to_nibbles(ws.data(), encoder.padded_dim,
                                     nibbles_per_pt.data() + i * num_blocks);
      }

      // Repack into FastScan strip layout. Padding points contribute the
      // zero nibble (all-positive sign pattern); per-strip masking at scoring
      // time excludes them from the running max.
      for (size_t s = 0; s < n_strips_c; ++s) {
        uint8_t* strip_base = res.packed_codes.data() + (strip0_dst + s) * strip_stride;
        for (uint32_t g = 0; g < nb4 / 4; ++g) {
          for (size_t lp = 0; lp < 32; ++lp) {
            const size_t v_even_in_cloud = s * 64 + lp * 2;
            const size_t v_odd_in_cloud = v_even_in_cloud + 1;
            for (uint32_t j = 0; j < 4; ++j) {
              const uint32_t b = g * 4 + j;
              uint8_t c_e = 0, c_o = 0;
              if (b < num_blocks) {
                if (v_even_in_cloud < n_vecs)
                  c_e = nibbles_per_pt[v_even_in_cloud * num_blocks + b];
                if (v_odd_in_cloud < n_vecs)
                  c_o = nibbles_per_pt[v_odd_in_cloud * num_blocks + b];
              }
              strip_base[g * 128 + lp * 4 + j] =
                  static_cast<uint8_t>((c_e & 0x0F) | ((c_o & 0x0F) << 4));
            }
          }
        }
      }
    });

    return res;
  }

  // Quantize one query cloud: rotate every embedding, max-abs scale to int4
  // [-7, +7], then build the per-embedding LUTs in the strip layout.
  template<typename ChPoint>
  Quantized_Query_Point_Cloud<Metric> quantize_query(const ChPoint& query_cloud) const {
    Quantized_Query_Point_Cloud<Metric> res;
    res.num_queries = query_cloud.size();
    res.padded_dim = static_cast<uint32_t>(encoder.padded_dim);
    res.num_nibble_positions = static_cast<uint32_t>(encoder.padded_dim / 4);

    if (res.num_queries == 0) return res;

    const uint32_t lpb = internal::lut_padded_blocks(res.num_nibble_positions);
    res.flat_int_luts.assign(res.num_queries * static_cast<size_t>(lpb) * 16, 0);
    res.scaled_q_nsf.assign(res.num_queries, 0.0f);
    if constexpr (Metric) res.unquantized_squared_norms.assign(res.num_queries, 0.0f);

    if (encoder.padded_dim == 0) return res;
    const float inv_sqrt_D = 1.0f / std::sqrt(static_cast<float>(encoder.padded_dim));

    const float* base_ptr = query_cloud.data();
    const size_t q_dim = encoder.dim;

    std::vector<float> ws(encoder.padded_dim);
    std::vector<int8_t> q_int8(encoder.padded_dim, 0);

    for (size_t qi = 0; qi < res.num_queries; ++qi) {
      const float* qptr = base_ptr + qi * q_dim;
      encoder.rotator->rotate(qptr, ws.data());

      float sqr_norm = 0.0f;
      float max_abs = 0.0f;
      for (size_t i = 0; i < encoder.padded_dim; ++i) {
        const float v = ws[i];
        sqr_norm += v * v;
        const float a = std::abs(v);
        if (a > max_abs) max_abs = a;
      }
      if (sqr_norm == 0.0f || !std::isfinite(sqr_norm) || max_abs == 0.0f) {
        // Empty or zero query embedding: leave LUT zeros; scaled_q_nsf=0;
        // ||q||^2=0 — the running max will still be 0 (all LUTs are 0) and
        // decode will return 0 (IP) or 1 (L2 → ||q||^2 + 1 - 0 = 1).
        continue;
      }

      const float sf = static_cast<float>(::mvsic::turboquant_1bit_asym::internal::kQMax) /
                       max_abs;
      int64_t q_quant_sqr = 0;
      for (size_t i = 0; i < encoder.padded_dim; ++i) {
        int v = static_cast<int>(std::lround(ws[i] * sf));
        if (v > ::mvsic::turboquant_1bit_asym::internal::kQMax)
          v = ::mvsic::turboquant_1bit_asym::internal::kQMax;
        else if (v < ::mvsic::turboquant_1bit_asym::internal::kQMin)
          v = ::mvsic::turboquant_1bit_asym::internal::kQMin;
        q_int8[i] = static_cast<int8_t>(v);
        q_quant_sqr += static_cast<int64_t>(v) * static_cast<int64_t>(v);
      }

      const float q_nsf = (q_quant_sqr > 0)
                              ? std::sqrt(sqr_norm) / std::sqrt(static_cast<float>(q_quant_sqr))
                              : 0.0f;
      res.scaled_q_nsf[qi] = q_nsf * inv_sqrt_D;
      if constexpr (Metric) res.unquantized_squared_norms[qi] = sqr_norm;

      // Build per-nibble-position LUTs into the strip layout.
      // num_nibble_positions = padded_dim/4. Pad up to lpb = ceil(npos/4)*4
      // so the layout matches the FastScan VNNI scan kernel — padding LUT
      // rows stay zero and contribute nothing to the running max.
      int8_t* qlut = res.flat_int_luts.data() + qi * static_cast<size_t>(lpb) * 16;
      const uint32_t nb4 = lpb;
      for (uint32_t g = 0; g < nb4 / 4; ++g) {
        for (uint32_t j = 0; j < 4; ++j) {
          const uint32_t pos = g * 4 + j;
          if (pos >= res.num_nibble_positions) continue;  // padding LUT row stays zero
          internal::build_lut_one_position(q_int8.data() + pos * 4,
                                           qlut + g * 64 + j * 16);
        }
      }
    }

    return res;
  }

  void save(std::ofstream& out) const { encoder.save(out); }
  void load(std::ifstream& in) { encoder.load(in); }
};

}  // namespace turboquant_1bit_asym_mv
}  // namespace mvsic
