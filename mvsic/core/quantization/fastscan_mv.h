#pragma once

#include <vector>
#include <cstdint>
#include <immintrin.h>
#include <limits>
#include <algorithm>
#include <cstring>
#include <random>
#include <iostream>
#include <fstream>
#include <Eigen/Core>

#include "parlay/parallel.h"
#include "parlay/sequence.h"
#include "mvsic/core/utils/kmeans_util.h"

namespace mvsic {
namespace fastscan_mv {

// =========================================================================
// SIMD AVX Kernels
// =========================================================================
namespace internal {

static constexpr size_t NQ_BATCH = 64;

// -------------------------------------------------------------------------
// Helper: compile-time strip stride and LUT padding
// -------------------------------------------------------------------------
inline size_t compute_strip_stride(uint32_t num_blocks) {
#if defined(__AVX512VNNI__) && defined(__AVX512VBMI__)
  return static_cast<size_t>(((num_blocks + 3) / 4) * 4) * 32;
#elif defined(__AVX512VBMI__)
  return static_cast<size_t>((num_blocks + 1) / 2) * 64;
#else
  return static_cast<size_t>(num_blocks) * 32;
#endif
}

inline uint32_t lut_padded_blocks(uint32_t num_blocks) {
#if defined(__AVX512VNNI__) && defined(__AVX512VBMI__)
  return ((num_blocks + 3) / 4) * 4;
#else
  return num_blocks;
#endif
}

// =========================================================================
// Platform-specific scan kernels
// =========================================================================

#if defined(__AVX512VNNI__) && defined(__AVX512VBMI__)
// =========================================================================
// 0. AVX512-VNNI + VBMI  (4-Block Interleaved, vpdpbusd + vpermb)
// =========================================================================
// Strip layout (interleaved-4): for each group of 4 blocks (g), 32 lane
// pairs have 4 consecutive bytes:
//   byte[g*128 + lp*4 + j] = (code_even_{g*4+j} & 0xF) | (code_odd << 4)
// Accumulation uses 32-bit via vpdpbusd; final values fit in uint16.

struct RunningMinV {
  __m512i lo, hi;
  void set_max() {
    lo = _mm512_set1_epi32(-1);
    hi = _mm512_set1_epi32(-1);
  }
  static RunningMinV max() {
    RunningMinV r;
    r.set_max();
    return r;
  }
};

inline uint32_t hmin_512_epu32(__m512i v) {
  v = _mm512_min_epu32(v, _mm512_shuffle_i32x4(v, v, _MM_SHUFFLE(1, 0, 3, 2)));
  v = _mm512_min_epu32(v, _mm512_shuffle_i32x4(v, v, _MM_SHUFFLE(0, 0, 1, 1)));
  __m128i v128 = _mm512_castsi512_si128(v);
  v128 = _mm_min_epu32(v128, _mm_srli_si128(v128, 8));
  v128 = _mm_min_epu32(v128, _mm_srli_si128(v128, 4));
  return static_cast<uint32_t>(_mm_extract_epi32(v128, 0));
}

inline uint16_t reduce_running_min(const RunningMinV& r) {
  return static_cast<uint16_t>(hmin_512_epu32(_mm512_min_epu32(r.lo, r.hi)));
}

inline void scan_64_running_min(const uint8_t* lut, const uint8_t* codes_ptr, uint32_t num_blocks,
                                RunningMinV& current_min_v) {
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
    __m512i co_lo = _mm512_or_si512(
        _mm512_and_si512(_mm512_srli_epi16(p_lo, 4), low_mask), blk_off);
    __m512i co_hi = _mm512_or_si512(
        _mm512_and_si512(_mm512_srli_epi16(p_hi, 4), low_mask), blk_off);

    ae_lo = _mm512_dpbusd_epi32(ae_lo, _mm512_permutexvar_epi8(ce_lo, lut512), ones_i8);
    ae_hi = _mm512_dpbusd_epi32(ae_hi, _mm512_permutexvar_epi8(ce_hi, lut512), ones_i8);
    ao_lo = _mm512_dpbusd_epi32(ao_lo, _mm512_permutexvar_epi8(co_lo, lut512), ones_i8);
    ao_hi = _mm512_dpbusd_epi32(ao_hi, _mm512_permutexvar_epi8(co_hi, lut512), ones_i8);
  }
  current_min_v.lo = _mm512_min_epu32(current_min_v.lo, _mm512_min_epu32(ae_lo, ao_lo));
  current_min_v.hi = _mm512_min_epu32(current_min_v.hi, _mm512_min_epu32(ae_hi, ao_hi));
}

inline uint16_t scan_64_chunk_min_masked(const uint8_t* lut, const uint8_t* codes_ptr,
                                         uint32_t num_blocks, int lo, int hi) {
  lo = std::max(lo, 0);
  hi = std::min(hi, 64);
  if (hi <= lo) return 0xFFFF;

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
    __m512i co_lo = _mm512_or_si512(
        _mm512_and_si512(_mm512_srli_epi16(p_lo, 4), low_mask), blk_off);
    __m512i co_hi = _mm512_or_si512(
        _mm512_and_si512(_mm512_srli_epi16(p_hi, 4), low_mask), blk_off);

    ae_lo = _mm512_dpbusd_epi32(ae_lo, _mm512_permutexvar_epi8(ce_lo, lut512), ones_i8);
    ae_hi = _mm512_dpbusd_epi32(ae_hi, _mm512_permutexvar_epi8(ce_hi, lut512), ones_i8);
    ao_lo = _mm512_dpbusd_epi32(ao_lo, _mm512_permutexvar_epi8(co_lo, lut512), ones_i8);
    ao_hi = _mm512_dpbusd_epi32(ao_hi, _mm512_permutexvar_epi8(co_hi, lut512), ones_i8);
  }

  const __m512i INF = _mm512_set1_epi32(-1);
  __mmask16 me_lo = 0, me_hi = 0, mo_lo = 0, mo_hi = 0;
  for (int lp = 0; lp < 16; ++lp) {
    if (lo <= lp * 2 && lp * 2 < hi) me_lo |= (__mmask16(1) << lp);
    if (lo <= (lp + 16) * 2 && (lp + 16) * 2 < hi) me_hi |= (__mmask16(1) << lp);
    if (lo <= lp * 2 + 1 && lp * 2 + 1 < hi) mo_lo |= (__mmask16(1) << lp);
    if (lo <= (lp + 16) * 2 + 1 && (lp + 16) * 2 + 1 < hi) mo_hi |= (__mmask16(1) << lp);
  }
  ae_lo = _mm512_mask_mov_epi32(INF, me_lo, ae_lo);
  ae_hi = _mm512_mask_mov_epi32(INF, me_hi, ae_hi);
  ao_lo = _mm512_mask_mov_epi32(INF, mo_lo, ao_lo);
  ao_hi = _mm512_mask_mov_epi32(INF, mo_hi, ao_hi);

  __m512i m = _mm512_min_epu32(_mm512_min_epu32(ae_lo, ao_lo), _mm512_min_epu32(ae_hi, ao_hi));
  return static_cast<uint16_t>(hmin_512_epu32(m));
}

static constexpr size_t SCAN_Q_BATCH = 6;

inline void scan_64_6q_running_min(
    const uint8_t* lut0, const uint8_t* lut1, const uint8_t* lut2,
    const uint8_t* lut3, const uint8_t* lut4, const uint8_t* lut5,
    const uint8_t* codes_ptr, uint32_t num_blocks,
    RunningMinV& rm0, RunningMinV& rm1, RunningMinV& rm2,
    RunningMinV& rm3, RunningMinV& rm4, RunningMinV& rm5) {

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

#define FSMV_VNNI_Q(Q, lp)                                                                        \
    {                                                                                              \
      const __m512i lut512 =                                                                       \
          _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&lp[g * 64]));                       \
      a##Q##el = _mm512_dpbusd_epi32(a##Q##el, _mm512_permutexvar_epi8(ce_lo, lut512), ones_i8);  \
      a##Q##eh = _mm512_dpbusd_epi32(a##Q##eh, _mm512_permutexvar_epi8(ce_hi, lut512), ones_i8);  \
      a##Q##ol = _mm512_dpbusd_epi32(a##Q##ol, _mm512_permutexvar_epi8(co_lo, lut512), ones_i8);  \
      a##Q##oh = _mm512_dpbusd_epi32(a##Q##oh, _mm512_permutexvar_epi8(co_hi, lut512), ones_i8);  \
    }
    FSMV_VNNI_Q(0, lut0)
    FSMV_VNNI_Q(1, lut1)
    FSMV_VNNI_Q(2, lut2)
    FSMV_VNNI_Q(3, lut3)
    FSMV_VNNI_Q(4, lut4)
    FSMV_VNNI_Q(5, lut5)
#undef FSMV_VNNI_Q
  }

#define FSMV_VNNI_REDUCE(Q, rm)                                                        \
  rm.lo = _mm512_min_epu32(rm.lo, _mm512_min_epu32(a##Q##el, a##Q##ol));               \
  rm.hi = _mm512_min_epu32(rm.hi, _mm512_min_epu32(a##Q##eh, a##Q##oh));
  FSMV_VNNI_REDUCE(0, rm0)
  FSMV_VNNI_REDUCE(1, rm1)
  FSMV_VNNI_REDUCE(2, rm2)
  FSMV_VNNI_REDUCE(3, rm3)
  FSMV_VNNI_REDUCE(4, rm4)
  FSMV_VNNI_REDUCE(5, rm5)
#undef FSMV_VNNI_REDUCE
}

inline void scan_64_batched(const uint8_t* const* luts, size_t nq,
                            const uint8_t* codes_ptr, uint32_t num_blocks,
                            RunningMinV* rmins) {
  size_t qi = 0;
  for (; qi + 5 < nq; qi += 6)
    scan_64_6q_running_min(luts[qi], luts[qi + 1], luts[qi + 2], luts[qi + 3], luts[qi + 4],
                           luts[qi + 5], codes_ptr, num_blocks, rmins[qi], rmins[qi + 1],
                           rmins[qi + 2], rmins[qi + 3], rmins[qi + 4], rmins[qi + 5]);
  for (; qi < nq; ++qi)
    scan_64_running_min(luts[qi], codes_ptr, num_blocks, rmins[qi]);
}

inline void scan_64_6q_running_min_masked(
    const uint8_t* lut0, const uint8_t* lut1, const uint8_t* lut2,
    const uint8_t* lut3, const uint8_t* lut4, const uint8_t* lut5,
    const uint8_t* codes_ptr, uint32_t num_blocks,
    int lo, int hi,
    RunningMinV& rm0, RunningMinV& rm1, RunningMinV& rm2,
    RunningMinV& rm3, RunningMinV& rm4, RunningMinV& rm5) {

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

#define FSMV_VNNI_QM(Q, lp)                                                                       \
    {                                                                                              \
      const __m512i lut512 =                                                                       \
          _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&lp[g * 64]));                       \
      a##Q##el = _mm512_dpbusd_epi32(a##Q##el, _mm512_permutexvar_epi8(ce_lo, lut512), ones_i8);  \
      a##Q##eh = _mm512_dpbusd_epi32(a##Q##eh, _mm512_permutexvar_epi8(ce_hi, lut512), ones_i8);  \
      a##Q##ol = _mm512_dpbusd_epi32(a##Q##ol, _mm512_permutexvar_epi8(co_lo, lut512), ones_i8);  \
      a##Q##oh = _mm512_dpbusd_epi32(a##Q##oh, _mm512_permutexvar_epi8(co_hi, lut512), ones_i8);  \
    }
    FSMV_VNNI_QM(0, lut0)
    FSMV_VNNI_QM(1, lut1)
    FSMV_VNNI_QM(2, lut2)
    FSMV_VNNI_QM(3, lut3)
    FSMV_VNNI_QM(4, lut4)
    FSMV_VNNI_QM(5, lut5)
#undef FSMV_VNNI_QM
  }

  const __m512i INF = _mm512_set1_epi32(-1);
  __mmask16 me_lo = 0, me_hi = 0, mo_lo = 0, mo_hi = 0;
  for (int lp = 0; lp < 16; ++lp) {
    if (lo <= lp * 2 && lp * 2 < hi) me_lo |= (__mmask16(1) << lp);
    if (lo <= (lp + 16) * 2 && (lp + 16) * 2 < hi) me_hi |= (__mmask16(1) << lp);
    if (lo <= lp * 2 + 1 && lp * 2 + 1 < hi) mo_lo |= (__mmask16(1) << lp);
    if (lo <= (lp + 16) * 2 + 1 && (lp + 16) * 2 + 1 < hi) mo_hi |= (__mmask16(1) << lp);
  }

#define FSMV_VNNI_MREDUCE(Q, rm)                                                                   \
  rm.lo = _mm512_min_epu32(rm.lo, _mm512_min_epu32(                                               \
      _mm512_mask_mov_epi32(INF, me_lo, a##Q##el),                                                \
      _mm512_mask_mov_epi32(INF, mo_lo, a##Q##ol)));                                              \
  rm.hi = _mm512_min_epu32(rm.hi, _mm512_min_epu32(                                               \
      _mm512_mask_mov_epi32(INF, me_hi, a##Q##eh),                                                \
      _mm512_mask_mov_epi32(INF, mo_hi, a##Q##oh)));
  FSMV_VNNI_MREDUCE(0, rm0)
  FSMV_VNNI_MREDUCE(1, rm1)
  FSMV_VNNI_MREDUCE(2, rm2)
  FSMV_VNNI_MREDUCE(3, rm3)
  FSMV_VNNI_MREDUCE(4, rm4)
  FSMV_VNNI_MREDUCE(5, rm5)
#undef FSMV_VNNI_MREDUCE
}

inline void scan_64_batched_masked(const uint8_t* const* luts, size_t nq,
                                   const uint8_t* codes_ptr, uint32_t num_blocks,
                                   int lo, int hi, RunningMinV* rmins) {
  size_t qi = 0;
  for (; qi + 5 < nq; qi += 6)
    scan_64_6q_running_min_masked(luts[qi], luts[qi + 1], luts[qi + 2], luts[qi + 3], luts[qi + 4],
                                  luts[qi + 5], codes_ptr, num_blocks, lo, hi,
                                  rmins[qi], rmins[qi + 1], rmins[qi + 2],
                                  rmins[qi + 3], rmins[qi + 4], rmins[qi + 5]);
  for (; qi < nq; ++qi) {
    uint16_t t = scan_64_chunk_min_masked(luts[qi], codes_ptr, num_blocks, lo, hi);
    __m512i tv = _mm512_set1_epi32(static_cast<int>(static_cast<uint32_t>(t)));
    rmins[qi].lo = _mm512_min_epu32(rmins[qi].lo, tv);
    rmins[qi].hi = _mm512_min_epu32(rmins[qi].hi, tv);
  }
}

#elif defined(__AVX512VBMI__)
// =========================================================================
// 1. AVX512-VBMI (2-Block Layout, 512-bit Registers)
// =========================================================================
struct RunningMinV {
  __m512i v;
  void set_max() { v = _mm512_set1_epi16(0xFFFF); }
  static RunningMinV max() {
    RunningMinV r;
    r.v = _mm512_set1_epi16(0xFFFF);
    return r;
  }
};

inline uint16_t hmin_512_epu16(__m512i v) {
  v = _mm512_min_epu16(v, _mm512_shuffle_i32x4(v, v, _MM_SHUFFLE(1, 0, 3, 2)));
  v = _mm512_min_epu16(v, _mm512_shuffle_i32x4(v, v, _MM_SHUFFLE(0, 0, 1, 1)));
  __m128i v128 = _mm512_castsi512_si128(v);
  v128 = _mm_min_epu16(v128, _mm_srli_si128(v128, 8));
  v128 = _mm_min_epu16(v128, _mm_srli_si128(v128, 4));
  v128 = _mm_min_epu16(v128, _mm_srli_si128(v128, 2));
  return static_cast<uint16_t>(_mm_extract_epi16(v128, 0));
}

inline void scan_64_running_min(const uint8_t* lut, const uint8_t* codes_ptr, uint32_t num_blocks,
                                RunningMinV& current_min_v) {
  __m512i acc_even = _mm512_setzero_si512();
  __m512i acc_odd = _mm512_setzero_si512();
  const __m512i low_mask = _mm512_set1_epi8(0x0F);
  const __m512i offset_mask = _mm512_set1_epi16(0x1000);
  const __m512i ones = _mm512_set1_epi8(1);

  for (uint32_t b = 0; b < num_blocks; b += 2) {
    const __m512i packed = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(codes_ptr));
    codes_ptr += 64;

    __m512i codes_even = _mm512_or_si512(_mm512_and_si512(packed, low_mask), offset_mask);
    __m512i codes_odd =
        _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(packed, 4), low_mask), offset_mask);

    const __m256i lut256 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&lut[b * 16]));
    const __m512i lut512 = _mm512_castsi256_si512(lut256);

    __m512i scores_even_u8 = _mm512_permutexvar_epi8(codes_even, lut512);
    __m512i scores_odd_u8 = _mm512_permutexvar_epi8(codes_odd, lut512);

    acc_even = _mm512_add_epi16(acc_even, _mm512_maddubs_epi16(scores_even_u8, ones));
    acc_odd = _mm512_add_epi16(acc_odd, _mm512_maddubs_epi16(scores_odd_u8, ones));
  }

  current_min_v.v = _mm512_min_epu16(current_min_v.v, acc_even);
  current_min_v.v = _mm512_min_epu16(current_min_v.v, acc_odd);
}

inline uint16_t reduce_running_min(const RunningMinV& running_min_v) {
  return hmin_512_epu16(running_min_v.v);
}

inline __mmask32 mask_even_lanes(int lo, int hi) {
  __mmask32 m = 0;
  for (int i = 0; i < 32; ++i) {
    if (lo <= 2 * i && 2 * i < hi) m |= (__mmask32(1) << i);
  }
  return m;
}

inline __mmask32 mask_odd_lanes(int lo, int hi) {
  __mmask32 m = 0;
  for (int i = 0; i < 32; ++i) {
    if (lo <= 2 * i + 1 && 2 * i + 1 < hi) m |= (__mmask32(1) << i);
  }
  return m;
}

inline uint16_t scan_64_chunk_min_masked(const uint8_t* lut, const uint8_t* codes_ptr,
                                         uint32_t num_blocks, int lo, int hi) {
  lo = std::max(lo, 0);
  hi = std::min(hi, 64);
  if (hi <= lo) return 0xFFFF;

  __m512i acc_even = _mm512_setzero_si512();
  __m512i acc_odd = _mm512_setzero_si512();
  const __m512i low_mask = _mm512_set1_epi8(0x0F);
  const __m512i offset_mask = _mm512_set1_epi16(0x1000);
  const __m512i ones = _mm512_set1_epi8(1);

  for (uint32_t b = 0; b < num_blocks; b += 2) {
    const __m512i packed = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(codes_ptr));
    codes_ptr += 64;

    __m512i codes_even = _mm512_or_si512(_mm512_and_si512(packed, low_mask), offset_mask);
    __m512i codes_odd =
        _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(packed, 4), low_mask), offset_mask);

    const __m256i lut256 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&lut[b * 16]));
    const __m512i lut512 = _mm512_castsi256_si512(lut256);

    __m512i scores_even_u8 = _mm512_permutexvar_epi8(codes_even, lut512);
    __m512i scores_odd_u8 = _mm512_permutexvar_epi8(codes_odd, lut512);

    acc_even = _mm512_add_epi16(acc_even, _mm512_maddubs_epi16(scores_even_u8, ones));
    acc_odd = _mm512_add_epi16(acc_odd, _mm512_maddubs_epi16(scores_odd_u8, ones));
  }

  const __m512i INF = _mm512_set1_epi16(0xFFFF);
  const __mmask32 me = mask_even_lanes(lo, hi);
  const __mmask32 mo = mask_odd_lanes(lo, hi);
  acc_even = _mm512_mask_mov_epi16(INF, me, acc_even);
  acc_odd = _mm512_mask_mov_epi16(INF, mo, acc_odd);
  const __m512i vmin = _mm512_min_epu16(acc_even, acc_odd);
  return hmin_512_epu16(vmin);
}

static constexpr size_t SCAN_Q_BATCH = 6;

inline void scan_64_6q_running_min(
    const uint8_t* lut0, const uint8_t* lut1, const uint8_t* lut2,
    const uint8_t* lut3, const uint8_t* lut4, const uint8_t* lut5,
    const uint8_t* codes_ptr, uint32_t num_blocks,
    RunningMinV& rm0, RunningMinV& rm1, RunningMinV& rm2,
    RunningMinV& rm3, RunningMinV& rm4, RunningMinV& rm5) {

  __m512i a0e = _mm512_setzero_si512(), a0o = _mm512_setzero_si512();
  __m512i a1e = _mm512_setzero_si512(), a1o = _mm512_setzero_si512();
  __m512i a2e = _mm512_setzero_si512(), a2o = _mm512_setzero_si512();
  __m512i a3e = _mm512_setzero_si512(), a3o = _mm512_setzero_si512();
  __m512i a4e = _mm512_setzero_si512(), a4o = _mm512_setzero_si512();
  __m512i a5e = _mm512_setzero_si512(), a5o = _mm512_setzero_si512();

  const __m512i low_mask = _mm512_set1_epi8(0x0F);
  const __m512i offset_mask = _mm512_set1_epi16(0x1000);
  const __m512i ones = _mm512_set1_epi8(1);

  for (uint32_t b = 0; b < num_blocks; b += 2) {
    const __m512i packed = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(codes_ptr));
    codes_ptr += 64;
    const __m512i ce = _mm512_or_si512(_mm512_and_si512(packed, low_mask), offset_mask);
    const __m512i co =
        _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(packed, 4), low_mask), offset_mask);

#define FSMV_VBMI_Q(Q, lp)                                                                        \
    {                                                                                              \
      const __m256i l256 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&lp[b * 16]));      \
      const __m512i l512 = _mm512_castsi256_si512(l256);                                           \
      a##Q##e = _mm512_add_epi16(                                                                  \
          a##Q##e, _mm512_maddubs_epi16(_mm512_permutexvar_epi8(ce, l512), ones));                 \
      a##Q##o = _mm512_add_epi16(                                                                  \
          a##Q##o, _mm512_maddubs_epi16(_mm512_permutexvar_epi8(co, l512), ones));                 \
    }
    FSMV_VBMI_Q(0, lut0)
    FSMV_VBMI_Q(1, lut1)
    FSMV_VBMI_Q(2, lut2)
    FSMV_VBMI_Q(3, lut3)
    FSMV_VBMI_Q(4, lut4)
    FSMV_VBMI_Q(5, lut5)
#undef FSMV_VBMI_Q
  }

  rm0.v = _mm512_min_epu16(rm0.v, _mm512_min_epu16(a0e, a0o));
  rm1.v = _mm512_min_epu16(rm1.v, _mm512_min_epu16(a1e, a1o));
  rm2.v = _mm512_min_epu16(rm2.v, _mm512_min_epu16(a2e, a2o));
  rm3.v = _mm512_min_epu16(rm3.v, _mm512_min_epu16(a3e, a3o));
  rm4.v = _mm512_min_epu16(rm4.v, _mm512_min_epu16(a4e, a4o));
  rm5.v = _mm512_min_epu16(rm5.v, _mm512_min_epu16(a5e, a5o));
}

inline void scan_64_batched(const uint8_t* const* luts, size_t nq,
                            const uint8_t* codes_ptr, uint32_t num_blocks,
                            RunningMinV* rmins) {
  size_t qi = 0;
  for (; qi + 5 < nq; qi += 6)
    scan_64_6q_running_min(luts[qi], luts[qi + 1], luts[qi + 2], luts[qi + 3], luts[qi + 4],
                           luts[qi + 5], codes_ptr, num_blocks, rmins[qi], rmins[qi + 1],
                           rmins[qi + 2], rmins[qi + 3], rmins[qi + 4], rmins[qi + 5]);
  for (; qi < nq; ++qi)
    scan_64_running_min(luts[qi], codes_ptr, num_blocks, rmins[qi]);
}

inline void scan_64_6q_running_min_masked(
    const uint8_t* lut0, const uint8_t* lut1, const uint8_t* lut2,
    const uint8_t* lut3, const uint8_t* lut4, const uint8_t* lut5,
    const uint8_t* codes_ptr, uint32_t num_blocks,
    int lo, int hi,
    RunningMinV& rm0, RunningMinV& rm1, RunningMinV& rm2,
    RunningMinV& rm3, RunningMinV& rm4, RunningMinV& rm5) {

  lo = std::max(lo, 0);
  hi = std::min(hi, 64);
  if (hi <= lo) return;

  __m512i a0e = _mm512_setzero_si512(), a0o = _mm512_setzero_si512();
  __m512i a1e = _mm512_setzero_si512(), a1o = _mm512_setzero_si512();
  __m512i a2e = _mm512_setzero_si512(), a2o = _mm512_setzero_si512();
  __m512i a3e = _mm512_setzero_si512(), a3o = _mm512_setzero_si512();
  __m512i a4e = _mm512_setzero_si512(), a4o = _mm512_setzero_si512();
  __m512i a5e = _mm512_setzero_si512(), a5o = _mm512_setzero_si512();

  const __m512i low_mask = _mm512_set1_epi8(0x0F);
  const __m512i offset_mask = _mm512_set1_epi16(0x1000);
  const __m512i ones = _mm512_set1_epi8(1);

  for (uint32_t b = 0; b < num_blocks; b += 2) {
    const __m512i packed = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(codes_ptr));
    codes_ptr += 64;
    const __m512i ce = _mm512_or_si512(_mm512_and_si512(packed, low_mask), offset_mask);
    const __m512i co =
        _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(packed, 4), low_mask), offset_mask);

#define FSMV_VBMI_QM(Q, lp)                                                                       \
    {                                                                                              \
      const __m256i l256 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&lp[b * 16]));      \
      const __m512i l512 = _mm512_castsi256_si512(l256);                                           \
      a##Q##e = _mm512_add_epi16(                                                                  \
          a##Q##e, _mm512_maddubs_epi16(_mm512_permutexvar_epi8(ce, l512), ones));                 \
      a##Q##o = _mm512_add_epi16(                                                                  \
          a##Q##o, _mm512_maddubs_epi16(_mm512_permutexvar_epi8(co, l512), ones));                 \
    }
    FSMV_VBMI_QM(0, lut0)
    FSMV_VBMI_QM(1, lut1)
    FSMV_VBMI_QM(2, lut2)
    FSMV_VBMI_QM(3, lut3)
    FSMV_VBMI_QM(4, lut4)
    FSMV_VBMI_QM(5, lut5)
#undef FSMV_VBMI_QM
  }

  const __m512i INF = _mm512_set1_epi16(0xFFFF);
  const __mmask32 me = mask_even_lanes(lo, hi);
  const __mmask32 mo = mask_odd_lanes(lo, hi);

#define FSMV_VBMI_MREDUCE(Q, rm)                                                                   \
  rm.v = _mm512_min_epu16(rm.v, _mm512_min_epu16(                                                 \
      _mm512_mask_mov_epi16(INF, me, a##Q##e),                                                    \
      _mm512_mask_mov_epi16(INF, mo, a##Q##o)));
  FSMV_VBMI_MREDUCE(0, rm0)
  FSMV_VBMI_MREDUCE(1, rm1)
  FSMV_VBMI_MREDUCE(2, rm2)
  FSMV_VBMI_MREDUCE(3, rm3)
  FSMV_VBMI_MREDUCE(4, rm4)
  FSMV_VBMI_MREDUCE(5, rm5)
#undef FSMV_VBMI_MREDUCE
}

inline void scan_64_batched_masked(const uint8_t* const* luts, size_t nq,
                                   const uint8_t* codes_ptr, uint32_t num_blocks,
                                   int lo, int hi, RunningMinV* rmins) {
  size_t qi = 0;
  for (; qi + 5 < nq; qi += 6)
    scan_64_6q_running_min_masked(luts[qi], luts[qi + 1], luts[qi + 2], luts[qi + 3], luts[qi + 4],
                                  luts[qi + 5], codes_ptr, num_blocks, lo, hi,
                                  rmins[qi], rmins[qi + 1], rmins[qi + 2],
                                  rmins[qi + 3], rmins[qi + 4], rmins[qi + 5]);
  for (; qi < nq; ++qi) {
    uint16_t t = scan_64_chunk_min_masked(luts[qi], codes_ptr, num_blocks, lo, hi);
    rmins[qi].v = _mm512_min_epu16(rmins[qi].v, _mm512_set1_epi16(static_cast<short>(t)));
  }
}

#elif defined(__AVX512F__)
// =========================================================================
// 2. AVX512-F (1-Block Striped Layout)
// =========================================================================
struct RunningMinV {
  __m512i v;
  void set_max() { v = _mm512_set1_epi16(0xFFFF); }
  static RunningMinV max() {
    RunningMinV r;
    r.v = _mm512_set1_epi16(0xFFFF);
    return r;
  }
};

inline uint16_t hmin_512_epu16(__m512i v) {
  v = _mm512_min_epu16(v, _mm512_shuffle_i32x4(v, v, _MM_SHUFFLE(1, 0, 3, 2)));
  v = _mm512_min_epu16(v, _mm512_shuffle_i32x4(v, v, _MM_SHUFFLE(0, 0, 1, 1)));
  __m128i v128 = _mm512_castsi512_si128(v);
  v128 = _mm_min_epu16(v128, _mm_srli_si128(v128, 8));
  v128 = _mm_min_epu16(v128, _mm_srli_si128(v128, 4));
  v128 = _mm_min_epu16(v128, _mm_srli_si128(v128, 2));
  return static_cast<uint16_t>(_mm_extract_epi16(v128, 0));
}

inline void scan_64_running_min(const uint8_t* lut, const uint8_t* codes_ptr, uint32_t num_blocks,
                                RunningMinV& current_min_v) {
  __m512i acc_even = _mm512_setzero_si512();
  __m512i acc_odd = _mm512_setzero_si512();
  const __m256i low_mask = _mm256_set1_epi8(0x0F);

  for (uint32_t b = 0; b < num_blocks; ++b) {
    const __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(codes_ptr));
    codes_ptr += 32;
    const __m256i codes_even = _mm256_and_si256(packed, low_mask);
    const __m256i codes_odd = _mm256_and_si256(_mm256_srli_epi16(packed, 4), low_mask);
    const __m128i lut128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&lut[b * 16]));
    const __m256i lut256 = _mm256_broadcastsi128_si256(lut128);
    const __m256i scores_even_u8 = _mm256_shuffle_epi8(lut256, codes_even);
    const __m256i scores_odd_u8 = _mm256_shuffle_epi8(lut256, codes_odd);
    acc_even = _mm512_add_epi16(acc_even, _mm512_cvtepu8_epi16(scores_even_u8));
    acc_odd = _mm512_add_epi16(acc_odd, _mm512_cvtepu8_epi16(scores_odd_u8));
  }
  current_min_v.v = _mm512_min_epu16(current_min_v.v, acc_even);
  current_min_v.v = _mm512_min_epu16(current_min_v.v, acc_odd);
}

inline uint16_t reduce_running_min(const RunningMinV& running_min_v) {
  return hmin_512_epu16(running_min_v.v);
}

inline __mmask32 mask_even_lanes(int lo, int hi) {
  __mmask32 m = 0;
  for (int i = 0; i < 32; ++i) {
    if (lo <= 2 * i && 2 * i < hi) m |= (__mmask32(1) << i);
  }
  return m;
}

inline __mmask32 mask_odd_lanes(int lo, int hi) {
  __mmask32 m = 0;
  for (int i = 0; i < 32; ++i) {
    if (lo <= 2 * i + 1 && 2 * i + 1 < hi) m |= (__mmask32(1) << i);
  }
  return m;
}

inline uint16_t scan_64_chunk_min_masked(const uint8_t* lut, const uint8_t* codes_ptr,
                                         uint32_t num_blocks, int lo, int hi) {
  lo = std::max(lo, 0);
  hi = std::min(hi, 64);
  if (hi <= lo) return 0xFFFF;

  __m512i acc_even = _mm512_setzero_si512();
  __m512i acc_odd = _mm512_setzero_si512();
  const __m256i low_mask = _mm256_set1_epi8(0x0F);

  for (uint32_t b = 0; b < num_blocks; ++b) {
    const __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(codes_ptr));
    codes_ptr += 32;
    const __m256i codes_even = _mm256_and_si256(packed, low_mask);
    const __m256i codes_odd = _mm256_and_si256(_mm256_srli_epi16(packed, 4), low_mask);
    const __m128i lut128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&lut[b * 16]));
    const __m256i lut256 = _mm256_broadcastsi128_si256(lut128);
    const __m256i scores_even_u8 = _mm256_shuffle_epi8(lut256, codes_even);
    const __m256i scores_odd_u8 = _mm256_shuffle_epi8(lut256, codes_odd);
    acc_even = _mm512_add_epi16(acc_even, _mm512_cvtepu8_epi16(scores_even_u8));
    acc_odd = _mm512_add_epi16(acc_odd, _mm512_cvtepu8_epi16(scores_odd_u8));
  }

  const __m512i INF = _mm512_set1_epi16(0xFFFF);
  const __mmask32 me = mask_even_lanes(lo, hi);
  const __mmask32 mo = mask_odd_lanes(lo, hi);
  acc_even = _mm512_mask_mov_epi16(INF, me, acc_even);
  acc_odd = _mm512_mask_mov_epi16(INF, mo, acc_odd);
  const __m512i vmin = _mm512_min_epu16(acc_even, acc_odd);
  return hmin_512_epu16(vmin);
}

static constexpr size_t SCAN_Q_BATCH = 6;

inline void scan_64_6q_running_min(
    const uint8_t* lut0, const uint8_t* lut1, const uint8_t* lut2,
    const uint8_t* lut3, const uint8_t* lut4, const uint8_t* lut5,
    const uint8_t* codes_ptr, uint32_t num_blocks,
    RunningMinV& rm0, RunningMinV& rm1, RunningMinV& rm2,
    RunningMinV& rm3, RunningMinV& rm4, RunningMinV& rm5) {

  __m512i a0e = _mm512_setzero_si512(), a0o = _mm512_setzero_si512();
  __m512i a1e = _mm512_setzero_si512(), a1o = _mm512_setzero_si512();
  __m512i a2e = _mm512_setzero_si512(), a2o = _mm512_setzero_si512();
  __m512i a3e = _mm512_setzero_si512(), a3o = _mm512_setzero_si512();
  __m512i a4e = _mm512_setzero_si512(), a4o = _mm512_setzero_si512();
  __m512i a5e = _mm512_setzero_si512(), a5o = _mm512_setzero_si512();

  const __m256i low_mask = _mm256_set1_epi8(0x0F);

  for (uint32_t b = 0; b < num_blocks; ++b) {
    const __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(codes_ptr));
    codes_ptr += 32;
    const __m256i ce = _mm256_and_si256(packed, low_mask);
    const __m256i co = _mm256_and_si256(_mm256_srli_epi16(packed, 4), low_mask);

#define FSMV_AVX512F_Q(Q, lp)                                                                     \
    {                                                                                              \
      const __m128i l128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&lp[b * 16]));         \
      const __m256i l256 = _mm256_broadcastsi128_si256(l128);                                      \
      a##Q##e = _mm512_add_epi16(a##Q##e, _mm512_cvtepu8_epi16(_mm256_shuffle_epi8(l256, ce)));    \
      a##Q##o = _mm512_add_epi16(a##Q##o, _mm512_cvtepu8_epi16(_mm256_shuffle_epi8(l256, co)));    \
    }
    FSMV_AVX512F_Q(0, lut0)
    FSMV_AVX512F_Q(1, lut1)
    FSMV_AVX512F_Q(2, lut2)
    FSMV_AVX512F_Q(3, lut3)
    FSMV_AVX512F_Q(4, lut4)
    FSMV_AVX512F_Q(5, lut5)
#undef FSMV_AVX512F_Q
  }

  rm0.v = _mm512_min_epu16(rm0.v, _mm512_min_epu16(a0e, a0o));
  rm1.v = _mm512_min_epu16(rm1.v, _mm512_min_epu16(a1e, a1o));
  rm2.v = _mm512_min_epu16(rm2.v, _mm512_min_epu16(a2e, a2o));
  rm3.v = _mm512_min_epu16(rm3.v, _mm512_min_epu16(a3e, a3o));
  rm4.v = _mm512_min_epu16(rm4.v, _mm512_min_epu16(a4e, a4o));
  rm5.v = _mm512_min_epu16(rm5.v, _mm512_min_epu16(a5e, a5o));
}

inline void scan_64_batched(const uint8_t* const* luts, size_t nq,
                            const uint8_t* codes_ptr, uint32_t num_blocks,
                            RunningMinV* rmins) {
  size_t qi = 0;
  for (; qi + 5 < nq; qi += 6)
    scan_64_6q_running_min(luts[qi], luts[qi + 1], luts[qi + 2], luts[qi + 3], luts[qi + 4],
                           luts[qi + 5], codes_ptr, num_blocks, rmins[qi], rmins[qi + 1],
                           rmins[qi + 2], rmins[qi + 3], rmins[qi + 4], rmins[qi + 5]);
  for (; qi < nq; ++qi)
    scan_64_running_min(luts[qi], codes_ptr, num_blocks, rmins[qi]);
}

inline void scan_64_6q_running_min_masked(
    const uint8_t* lut0, const uint8_t* lut1, const uint8_t* lut2,
    const uint8_t* lut3, const uint8_t* lut4, const uint8_t* lut5,
    const uint8_t* codes_ptr, uint32_t num_blocks,
    int lo, int hi,
    RunningMinV& rm0, RunningMinV& rm1, RunningMinV& rm2,
    RunningMinV& rm3, RunningMinV& rm4, RunningMinV& rm5) {

  lo = std::max(lo, 0);
  hi = std::min(hi, 64);
  if (hi <= lo) return;

  __m512i a0e = _mm512_setzero_si512(), a0o = _mm512_setzero_si512();
  __m512i a1e = _mm512_setzero_si512(), a1o = _mm512_setzero_si512();
  __m512i a2e = _mm512_setzero_si512(), a2o = _mm512_setzero_si512();
  __m512i a3e = _mm512_setzero_si512(), a3o = _mm512_setzero_si512();
  __m512i a4e = _mm512_setzero_si512(), a4o = _mm512_setzero_si512();
  __m512i a5e = _mm512_setzero_si512(), a5o = _mm512_setzero_si512();

  const __m256i low_mask = _mm256_set1_epi8(0x0F);

  for (uint32_t b = 0; b < num_blocks; ++b) {
    const __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(codes_ptr));
    codes_ptr += 32;
    const __m256i ce = _mm256_and_si256(packed, low_mask);
    const __m256i co = _mm256_and_si256(_mm256_srli_epi16(packed, 4), low_mask);

#define FSMV_AVX512F_QM(Q, lp)                                                                    \
    {                                                                                              \
      const __m128i l128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&lp[b * 16]));         \
      const __m256i l256 = _mm256_broadcastsi128_si256(l128);                                      \
      a##Q##e = _mm512_add_epi16(a##Q##e, _mm512_cvtepu8_epi16(_mm256_shuffle_epi8(l256, ce)));    \
      a##Q##o = _mm512_add_epi16(a##Q##o, _mm512_cvtepu8_epi16(_mm256_shuffle_epi8(l256, co)));    \
    }
    FSMV_AVX512F_QM(0, lut0)
    FSMV_AVX512F_QM(1, lut1)
    FSMV_AVX512F_QM(2, lut2)
    FSMV_AVX512F_QM(3, lut3)
    FSMV_AVX512F_QM(4, lut4)
    FSMV_AVX512F_QM(5, lut5)
#undef FSMV_AVX512F_QM
  }

  const __m512i INF = _mm512_set1_epi16(0xFFFF);
  const __mmask32 me = mask_even_lanes(lo, hi);
  const __mmask32 mo = mask_odd_lanes(lo, hi);

#define FSMV_AVX512F_MREDUCE(Q, rm)                                                                \
  rm.v = _mm512_min_epu16(rm.v, _mm512_min_epu16(                                                 \
      _mm512_mask_mov_epi16(INF, me, a##Q##e),                                                    \
      _mm512_mask_mov_epi16(INF, mo, a##Q##o)));
  FSMV_AVX512F_MREDUCE(0, rm0)
  FSMV_AVX512F_MREDUCE(1, rm1)
  FSMV_AVX512F_MREDUCE(2, rm2)
  FSMV_AVX512F_MREDUCE(3, rm3)
  FSMV_AVX512F_MREDUCE(4, rm4)
  FSMV_AVX512F_MREDUCE(5, rm5)
#undef FSMV_AVX512F_MREDUCE
}

inline void scan_64_batched_masked(const uint8_t* const* luts, size_t nq,
                                   const uint8_t* codes_ptr, uint32_t num_blocks,
                                   int lo, int hi, RunningMinV* rmins) {
  size_t qi = 0;
  for (; qi + 5 < nq; qi += 6)
    scan_64_6q_running_min_masked(luts[qi], luts[qi + 1], luts[qi + 2], luts[qi + 3], luts[qi + 4],
                                  luts[qi + 5], codes_ptr, num_blocks, lo, hi,
                                  rmins[qi], rmins[qi + 1], rmins[qi + 2],
                                  rmins[qi + 3], rmins[qi + 4], rmins[qi + 5]);
  for (; qi < nq; ++qi) {
    uint16_t t = scan_64_chunk_min_masked(luts[qi], codes_ptr, num_blocks, lo, hi);
    rmins[qi].v = _mm512_min_epu16(rmins[qi].v, _mm512_set1_epi16(static_cast<short>(t)));
  }
}

#elif defined(__AVX2__)
// =========================================================================
// 3. AVX2 FALLBACK (1-Block Striped Layout)
// =========================================================================
struct RunningMinV {
  __m256i lo, hi;
  void set_max() {
    lo = _mm256_set1_epi16(0xFFFF);
    hi = _mm256_set1_epi16(0xFFFF);
  }
  static RunningMinV max() {
    RunningMinV r;
    r.lo = _mm256_set1_epi16(0xFFFF);
    r.hi = _mm256_set1_epi16(0xFFFF);
    return r;
  }
};

inline uint16_t hmin_256_epu16(__m256i v) {
  __m128i lo = _mm256_castsi256_si128(v);
  __m128i hi = _mm256_extracti128_si256(v, 1);
  __m128i m = _mm_min_epu16(lo, hi);
  m = _mm_min_epu16(m, _mm_srli_si128(m, 8));
  m = _mm_min_epu16(m, _mm_srli_si128(m, 4));
  m = _mm_min_epu16(m, _mm_srli_si128(m, 2));
  return static_cast<uint16_t>(_mm_extract_epi16(m, 0));
}

inline void scan_64_running_min(const uint8_t* lut, const uint8_t* codes_ptr, uint32_t num_blocks,
                                RunningMinV& current_min_v) {
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
    const __m128i lut128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&lut[b * 16]));
    const __m256i lut256 = _mm256_broadcastsi128_si256(lut128);
    const __m256i se = _mm256_shuffle_epi8(lut256, codes_even);
    const __m256i so = _mm256_shuffle_epi8(lut256, codes_odd);
    acc_even_lo = _mm256_add_epi16(acc_even_lo, _mm256_cvtepu8_epi16(_mm256_castsi256_si128(se)));
    acc_even_hi =
        _mm256_add_epi16(acc_even_hi, _mm256_cvtepu8_epi16(_mm256_extracti128_si256(se, 1)));
    acc_odd_lo = _mm256_add_epi16(acc_odd_lo, _mm256_cvtepu8_epi16(_mm256_castsi256_si128(so)));
    acc_odd_hi =
        _mm256_add_epi16(acc_odd_hi, _mm256_cvtepu8_epi16(_mm256_extracti128_si256(so, 1)));
  }
  current_min_v.lo = _mm256_min_epu16(current_min_v.lo, acc_even_lo);
  current_min_v.lo = _mm256_min_epu16(current_min_v.lo, acc_odd_lo);
  current_min_v.hi = _mm256_min_epu16(current_min_v.hi, acc_even_hi);
  current_min_v.hi = _mm256_min_epu16(current_min_v.hi, acc_odd_hi);
}

inline uint16_t reduce_running_min(const RunningMinV& running_min_v) {
  return std::min(hmin_256_epu16(running_min_v.lo), hmin_256_epu16(running_min_v.hi));
}

inline uint16_t scan_64_chunk_min_masked(const uint8_t* lut, const uint8_t* codes_ptr,
                                         uint32_t num_blocks, int lo, int hi) {
  lo = std::max(lo, 0);
  hi = std::min(hi, 64);
  if (hi <= lo) return 0xFFFF;

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
    const __m128i lut128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&lut[b * 16]));
    const __m256i lut256 = _mm256_broadcastsi128_si256(lut128);
    const __m256i se = _mm256_shuffle_epi8(lut256, codes_even);
    const __m256i so = _mm256_shuffle_epi8(lut256, codes_odd);

    acc_even_lo = _mm256_add_epi16(acc_even_lo, _mm256_cvtepu8_epi16(_mm256_castsi256_si128(se)));
    acc_even_hi =
        _mm256_add_epi16(acc_even_hi, _mm256_cvtepu8_epi16(_mm256_extracti128_si256(se, 1)));
    acc_odd_lo = _mm256_add_epi16(acc_odd_lo, _mm256_cvtepu8_epi16(_mm256_castsi256_si128(so)));
    acc_odd_hi =
        _mm256_add_epi16(acc_odd_hi, _mm256_cvtepu8_epi16(_mm256_extracti128_si256(so, 1)));
  }

  const __m256i v_lo = _mm256_set1_epi16(static_cast<short>(lo));
  const __m256i v_hi = _mm256_set1_epi16(static_cast<short>(hi));
  __m256i idx_even_lo =
      _mm256_set_epi16(30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0);
  __m256i idx_even_hi = _mm256_add_epi16(idx_even_lo, _mm256_set1_epi16(32));
  __m256i idx_odd_lo = _mm256_add_epi16(idx_even_lo, _mm256_set1_epi16(1));
  __m256i idx_odd_hi = _mm256_add_epi16(idx_even_hi, _mm256_set1_epi16(1));

  auto get_mask = [&](__m256i idx) {
    __m256i m_ge = _mm256_cmpgt_epi16(idx, v_lo);
    __m256i m_eq = _mm256_cmpeq_epi16(idx, v_lo);
    __m256i in_lo = _mm256_or_si256(m_ge, m_eq);
    __m256i in_hi = _mm256_cmpgt_epi16(v_hi, idx);
    return _mm256_and_si256(in_lo, in_hi);
  };

  const __m256i INF = _mm256_set1_epi16(static_cast<short>(0xFFFF));

  acc_even_lo = _mm256_blendv_epi8(INF, acc_even_lo, get_mask(idx_even_lo));
  acc_even_hi = _mm256_blendv_epi8(INF, acc_even_hi, get_mask(idx_even_hi));
  acc_odd_lo = _mm256_blendv_epi8(INF, acc_odd_lo, get_mask(idx_odd_lo));
  acc_odd_hi = _mm256_blendv_epi8(INF, acc_odd_hi, get_mask(idx_odd_hi));

  __m256i m1 = _mm256_min_epu16(acc_even_lo, acc_even_hi);
  __m256i m2 = _mm256_min_epu16(acc_odd_lo, acc_odd_hi);
  return hmin_256_epu16(_mm256_min_epu16(m1, m2));
}

static constexpr size_t SCAN_Q_BATCH = 3;

inline void scan_64_3q_running_min(
    const uint8_t* lut0, const uint8_t* lut1, const uint8_t* lut2,
    const uint8_t* codes_ptr, uint32_t num_blocks,
    RunningMinV& rm0, RunningMinV& rm1, RunningMinV& rm2) {

  __m256i a0el = _mm256_setzero_si256(), a0eh = _mm256_setzero_si256();
  __m256i a0ol = _mm256_setzero_si256(), a0oh = _mm256_setzero_si256();
  __m256i a1el = _mm256_setzero_si256(), a1eh = _mm256_setzero_si256();
  __m256i a1ol = _mm256_setzero_si256(), a1oh = _mm256_setzero_si256();
  __m256i a2el = _mm256_setzero_si256(), a2eh = _mm256_setzero_si256();
  __m256i a2ol = _mm256_setzero_si256(), a2oh = _mm256_setzero_si256();
  const __m256i low_mask = _mm256_set1_epi8(0x0F);

  for (uint32_t b = 0; b < num_blocks; ++b) {
    const __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(codes_ptr));
    codes_ptr += 32;
    const __m256i ce = _mm256_and_si256(packed, low_mask);
    const __m256i co = _mm256_and_si256(_mm256_srli_epi16(packed, 4), low_mask);

#define FSMV_AVX2_Q(Q, lp)                                                                        \
    {                                                                                              \
      const __m128i l128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&lp[b * 16]));         \
      const __m256i l256 = _mm256_broadcastsi128_si256(l128);                                      \
      const __m256i se = _mm256_shuffle_epi8(l256, ce);                                            \
      const __m256i so = _mm256_shuffle_epi8(l256, co);                                            \
      a##Q##el = _mm256_add_epi16(a##Q##el, _mm256_cvtepu8_epi16(_mm256_castsi256_si128(se)));     \
      a##Q##eh =                                                                                   \
          _mm256_add_epi16(a##Q##eh, _mm256_cvtepu8_epi16(_mm256_extracti128_si256(se, 1)));       \
      a##Q##ol = _mm256_add_epi16(a##Q##ol, _mm256_cvtepu8_epi16(_mm256_castsi256_si128(so)));     \
      a##Q##oh =                                                                                   \
          _mm256_add_epi16(a##Q##oh, _mm256_cvtepu8_epi16(_mm256_extracti128_si256(so, 1)));       \
    }
    FSMV_AVX2_Q(0, lut0)
    FSMV_AVX2_Q(1, lut1)
    FSMV_AVX2_Q(2, lut2)
#undef FSMV_AVX2_Q
  }

#define FSMV_AVX2_REDUCE(Q, rm)                                              \
  rm.lo = _mm256_min_epu16(rm.lo, _mm256_min_epu16(a##Q##el, a##Q##ol));     \
  rm.hi = _mm256_min_epu16(rm.hi, _mm256_min_epu16(a##Q##eh, a##Q##oh));
  FSMV_AVX2_REDUCE(0, rm0)
  FSMV_AVX2_REDUCE(1, rm1)
  FSMV_AVX2_REDUCE(2, rm2)
#undef FSMV_AVX2_REDUCE
}

inline void scan_64_batched(const uint8_t* const* luts, size_t nq,
                            const uint8_t* codes_ptr, uint32_t num_blocks,
                            RunningMinV* rmins) {
  size_t qi = 0;
  for (; qi + 2 < nq; qi += 3)
    scan_64_3q_running_min(luts[qi], luts[qi + 1], luts[qi + 2], codes_ptr, num_blocks,
                           rmins[qi], rmins[qi + 1], rmins[qi + 2]);
  for (; qi < nq; ++qi)
    scan_64_running_min(luts[qi], codes_ptr, num_blocks, rmins[qi]);
}

inline void scan_64_3q_running_min_masked(
    const uint8_t* lut0, const uint8_t* lut1, const uint8_t* lut2,
    const uint8_t* codes_ptr, uint32_t num_blocks,
    int lo, int hi,
    RunningMinV& rm0, RunningMinV& rm1, RunningMinV& rm2) {

  lo = std::max(lo, 0);
  hi = std::min(hi, 64);
  if (hi <= lo) return;

  __m256i a0el = _mm256_setzero_si256(), a0eh = _mm256_setzero_si256();
  __m256i a0ol = _mm256_setzero_si256(), a0oh = _mm256_setzero_si256();
  __m256i a1el = _mm256_setzero_si256(), a1eh = _mm256_setzero_si256();
  __m256i a1ol = _mm256_setzero_si256(), a1oh = _mm256_setzero_si256();
  __m256i a2el = _mm256_setzero_si256(), a2eh = _mm256_setzero_si256();
  __m256i a2ol = _mm256_setzero_si256(), a2oh = _mm256_setzero_si256();
  const __m256i low_mask = _mm256_set1_epi8(0x0F);

  for (uint32_t b = 0; b < num_blocks; ++b) {
    const __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(codes_ptr));
    codes_ptr += 32;
    const __m256i ce = _mm256_and_si256(packed, low_mask);
    const __m256i co = _mm256_and_si256(_mm256_srli_epi16(packed, 4), low_mask);

#define FSMV_AVX2_QM(Q, lp)                                                                       \
    {                                                                                              \
      const __m128i l128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&lp[b * 16]));         \
      const __m256i l256 = _mm256_broadcastsi128_si256(l128);                                      \
      const __m256i se = _mm256_shuffle_epi8(l256, ce);                                            \
      const __m256i so = _mm256_shuffle_epi8(l256, co);                                            \
      a##Q##el = _mm256_add_epi16(a##Q##el, _mm256_cvtepu8_epi16(_mm256_castsi256_si128(se)));     \
      a##Q##eh =                                                                                   \
          _mm256_add_epi16(a##Q##eh, _mm256_cvtepu8_epi16(_mm256_extracti128_si256(se, 1)));       \
      a##Q##ol = _mm256_add_epi16(a##Q##ol, _mm256_cvtepu8_epi16(_mm256_castsi256_si128(so)));     \
      a##Q##oh =                                                                                   \
          _mm256_add_epi16(a##Q##oh, _mm256_cvtepu8_epi16(_mm256_extracti128_si256(so, 1)));       \
    }
    FSMV_AVX2_QM(0, lut0)
    FSMV_AVX2_QM(1, lut1)
    FSMV_AVX2_QM(2, lut2)
#undef FSMV_AVX2_QM
  }

  const __m256i v_lo = _mm256_set1_epi16(static_cast<short>(lo));
  const __m256i v_hi = _mm256_set1_epi16(static_cast<short>(hi));
  __m256i idx_even_lo =
      _mm256_set_epi16(30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0);
  __m256i idx_even_hi = _mm256_add_epi16(idx_even_lo, _mm256_set1_epi16(32));
  __m256i idx_odd_lo = _mm256_add_epi16(idx_even_lo, _mm256_set1_epi16(1));
  __m256i idx_odd_hi = _mm256_add_epi16(idx_even_hi, _mm256_set1_epi16(1));

  auto get_mask = [&](__m256i idx) {
    __m256i m_ge = _mm256_cmpgt_epi16(idx, v_lo);
    __m256i m_eq = _mm256_cmpeq_epi16(idx, v_lo);
    __m256i in_lo = _mm256_or_si256(m_ge, m_eq);
    __m256i in_hi = _mm256_cmpgt_epi16(v_hi, idx);
    return _mm256_and_si256(in_lo, in_hi);
  };

  const __m256i INF = _mm256_set1_epi16(static_cast<short>(0xFFFF));
  const __m256i mel = get_mask(idx_even_lo);
  const __m256i meh = get_mask(idx_even_hi);
  const __m256i mol = get_mask(idx_odd_lo);
  const __m256i moh = get_mask(idx_odd_hi);

#define FSMV_AVX2_MREDUCE(Q, rm)                                                                   \
  rm.lo = _mm256_min_epu16(rm.lo, _mm256_min_epu16(                                               \
      _mm256_blendv_epi8(INF, a##Q##el, mel),                                                     \
      _mm256_blendv_epi8(INF, a##Q##ol, mol)));                                                   \
  rm.hi = _mm256_min_epu16(rm.hi, _mm256_min_epu16(                                               \
      _mm256_blendv_epi8(INF, a##Q##eh, meh),                                                     \
      _mm256_blendv_epi8(INF, a##Q##oh, moh)));
  FSMV_AVX2_MREDUCE(0, rm0)
  FSMV_AVX2_MREDUCE(1, rm1)
  FSMV_AVX2_MREDUCE(2, rm2)
#undef FSMV_AVX2_MREDUCE
}

inline void scan_64_batched_masked(const uint8_t* const* luts, size_t nq,
                                   const uint8_t* codes_ptr, uint32_t num_blocks,
                                   int lo, int hi, RunningMinV* rmins) {
  size_t qi = 0;
  for (; qi + 2 < nq; qi += 3)
    scan_64_3q_running_min_masked(luts[qi], luts[qi + 1], luts[qi + 2], codes_ptr, num_blocks,
                                  lo, hi, rmins[qi], rmins[qi + 1], rmins[qi + 2]);
  for (; qi < nq; ++qi) {
    uint16_t t = scan_64_chunk_min_masked(luts[qi], codes_ptr, num_blocks, lo, hi);
    __m256i tv = _mm256_set1_epi16(static_cast<short>(t));
    rmins[qi].lo = _mm256_min_epu16(rmins[qi].lo, tv);
    rmins[qi].hi = _mm256_min_epu16(rmins[qi].hi, tv);
  }
}

#else
// =========================================================================
// 4. SCALAR FALLBACK (No SIMD)
// =========================================================================
struct RunningMinV {
  alignas(64) uint16_t data[64];
  void set_max() {
    for (int i = 0; i < 64; ++i)
      data[i] = 0xFFFF;
  }
  static RunningMinV max() {
    RunningMinV r;
    r.set_max();
    return r;
  }
};

inline void scan_64_running_min(const uint8_t* lut, const uint8_t* codes_ptr, uint32_t num_blocks,
                                RunningMinV& current_min_v) {
  for (int lane = 0; lane < 64; ++lane) {
    uint16_t acc = 0;
    for (uint32_t b = 0; b < num_blocks; ++b) {
      uint8_t packed = codes_ptr[b * 32 + (lane / 2)];
      uint8_t code = (lane % 2 == 0) ? (packed & 0x0F) : (packed >> 4);
      acc += static_cast<uint16_t>(lut[b * 16 + code]);
    }
    current_min_v.data[lane] = std::min(current_min_v.data[lane], acc);
  }
}

inline uint16_t reduce_running_min(const RunningMinV& running_min_v) {
  uint16_t best = 0xFFFF;
  for (int i = 0; i < 64; ++i)
    best = std::min(best, running_min_v.data[i]);
  return best;
}

inline uint16_t scan_64_chunk_min_masked(const uint8_t* lut, const uint8_t* codes_ptr,
                                         uint32_t num_blocks, int lo, int hi) {
  lo = std::max(lo, 0);
  hi = std::min(hi, 64);
  if (hi <= lo) return 0xFFFF;
  uint16_t best = 0xFFFF;
  for (int lane = lo; lane < hi; ++lane) {
    uint16_t acc = 0;
    for (uint32_t b = 0; b < num_blocks; ++b) {
      uint8_t packed = codes_ptr[b * 32 + (lane / 2)];
      uint8_t code = (lane % 2 == 0) ? (packed & 0x0F) : (packed >> 4);
      acc += static_cast<uint16_t>(lut[b * 16 + code]);
    }
    if (acc < best) best = acc;
  }
  return best;
}

static constexpr size_t SCAN_Q_BATCH = 1;

inline void scan_64_batched(const uint8_t* const* luts, size_t nq,
                            const uint8_t* codes_ptr, uint32_t num_blocks,
                            RunningMinV* rmins) {
  for (size_t qi = 0; qi < nq; ++qi)
    scan_64_running_min(luts[qi], codes_ptr, num_blocks, rmins[qi]);
}

inline void scan_64_batched_masked(const uint8_t* const* luts, size_t nq,
                                   const uint8_t* codes_ptr, uint32_t num_blocks,
                                   int lo, int hi, RunningMinV* rmins) {
  for (size_t qi = 0; qi < nq; ++qi) {
    uint16_t t = scan_64_chunk_min_masked(luts[qi], codes_ptr, num_blocks, lo, hi);
    rmins[qi].data[0] = std::min(rmins[qi].data[0], t);
  }
}
#endif

// ------------------------------------------------------------------
// Fused FastScan Kernel (Strip-outer, Query-inner, Code-load Amortized)
// Unified 2-way dispatch: all full strips → batched, tail → batched_masked.
// Exploits the invariant that start_vec is always 64-aligned (per-cloud padding).
// ------------------------------------------------------------------
template<bool Metric>
inline void fastscan_mv_chamfer_fused(const uint8_t* fused_luts, const float* q_scales,
                                      const float* q_min_dists, size_t num_fused_embeddings,
                                      uint32_t num_blocks, const uint8_t* strip_data,
                                      size_t strip_stride, size_t start_vec, size_t cloud_size,
                                      float* out_dists,
                                      std::vector<internal::RunningMinV>& scratch_combined,
                                      std::vector<uint16_t>& scratch_min_raw) {
  constexpr size_t PREFETCH_STRIPS = 2;
  const uint32_t lut_blk = lut_padded_blocks(num_blocks);
  const size_t lut_stride = static_cast<size_t>(lut_blk) * 16;

  if (cloud_size == 0) {
    for (size_t qi = 0; qi < num_fused_embeddings; ++qi)
      out_dists[qi] = std::numeric_limits<float>::max();
    return;
  }

  const size_t strip0 = start_vec / 64;
  const size_t num_full_strips = cloud_size / 64;
  const int tail = static_cast<int>(cloud_size % 64);
  const size_t total_strips = num_full_strips + (tail > 0 ? 1 : 0);

  auto strip_ptr = [&](size_t s) -> const uint8_t* { return strip_data + s * strip_stride; };

  for (size_t q_start = 0; q_start < num_fused_embeddings; q_start += NQ_BATCH) {
    const size_t q_end = std::min(q_start + NQ_BATCH, num_fused_embeddings);
    const size_t q_count = q_end - q_start;

    const uint8_t* lut_ptrs[NQ_BATCH];
    for (size_t qi = 0; qi < q_count; ++qi)
      lut_ptrs[qi] = fused_luts + (q_start + qi) * lut_stride;

    scratch_combined.resize(q_count);
    for (size_t qi = 0; qi < q_count; ++qi)
      scratch_combined[qi] = RunningMinV::max();

    for (size_t s = 0; s < num_full_strips; ++s) {
      const uint8_t* sp = strip_ptr(strip0 + s);
      if (s + PREFETCH_STRIPS < total_strips)
        __builtin_prefetch(strip_ptr(strip0 + s + PREFETCH_STRIPS), 0, 3);
      scan_64_batched(lut_ptrs, q_count, sp, num_blocks, scratch_combined.data());
    }

    if (tail > 0) {
      scan_64_batched_masked(lut_ptrs, q_count,
                             strip_ptr(strip0 + num_full_strips), num_blocks,
                             0, tail, scratch_combined.data());
    }

    scratch_min_raw.resize(q_count);
    for (size_t qi = 0; qi < q_count; ++qi)
      scratch_min_raw[qi] = reduce_running_min(scratch_combined[qi]);

    for (size_t qi = 0; qi < q_count; ++qi) {
      size_t gqi = q_start + qi;
      out_dists[gqi] = (q_min_dists[gqi] * static_cast<float>(num_blocks)) +
                       (static_cast<float>(scratch_min_raw[qi]) * q_scales[gqi]);
    }
  }
}

}  // namespace internal

// =========================================================================
// Forward declarations
// =========================================================================
template<bool Metric>
class Quantized_Point_Cloud_Set;
template<bool Metric>
class Quantized_Query_Point_Cloud;

// Lightweight view of one cloud's vector range [start_idx, end_idx) in strip layout.
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
float fastscan_mv_chamfer_distance(const Quantized_Query_Point_Cloud<Metric>& q,
                                   const Quantized_Point_Cloud_Set<Metric>& db, size_t start,
                                   size_t true_end);

// =========================================================================
// Multi-Vector Query (Struct of Arrays / Contiguous Layout)
// =========================================================================
template<bool Metric>
class Quantized_Query_Point_Cloud {
 public:
  using distanceType = float;
  size_t num_queries = 0;
  uint32_t num_blocks = 0;
  static constexpr uint32_t K = 16;

  std::vector<uint8_t> flat_int_luts;
  std::vector<float> min_dists;
  std::vector<float> scales;

  Quantized_Query_Point_Cloud() = default;

  inline const uint8_t* get_lut(size_t qi) const {
    const uint32_t lpb = internal::lut_padded_blocks(num_blocks);
    return flat_int_luts.data() + qi * static_cast<size_t>(lpb) * 16;
  }

  inline float decode(size_t qi, uint16_t int_dist) const {
    return (min_dists[qi] * static_cast<float>(num_blocks)) +
           (static_cast<float>(int_dist) * scales[qi]);
  }

  template<typename CloudView>
  float distance(const CloudView& cloud) const {
    return fastscan_mv_chamfer_distance(*this, *cloud.db, cloud.start_idx, cloud.end_idx);
  }

  template<typename CloudHandle>
  std::pair<float, size_t> distance_w_cmps(const CloudHandle& cloud) const {
    const size_t cloud_size = cloud.end_idx - cloud.start_idx;
    const size_t bytes_per_vec = (static_cast<size_t>(cloud.db->num_blocks) + 1) / 2;
    return {this->distance(cloud), cloud_size * bytes_per_vec};
  }

  static constexpr bool is_metric() { return Metric; }
};

// =========================================================================
// Multi-Vector Set (Manages FastScan Strips & Cloud Offsets)
// =========================================================================
template<bool Metric>
class Quantized_Point_Cloud_Set {
 public:
  parlay::sequence<uint8_t> packed_codes;
  uint32_t num_blocks = 0;

  parlay::sequence<size_t> offsets;
  parlay::sequence<uint32_t> sizes_unpadded;
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

  void distances_all(const Quantized_Query_Point_Cloud<Metric>& q,
                     std::pair<uint32_t, float>* results) const {
    const size_t num_q = q.num_queries;
    const size_t n_clouds = (offsets.size() > 0) ? offsets.size() - 1 : 0;

    if (num_q == 0 || n_clouds == 0) return;

    parlay::parallel_for(0, n_clouds, [&](size_t cid) {
      const uint32_t cloud_id = static_cast<uint32_t>(cid);
      const size_t start = offsets[cloud_id];
      size_t cloud_size = 0;
      if (sizes_unpadded.size() == static_cast<size_t>(n_clouds)) {
        cloud_size = static_cast<size_t>(sizes_unpadded[cloud_id]);
      } else {
        const size_t end_padded = offsets[cloud_id + 1];
        cloud_size = (end_padded > start) ? (end_padded - start) : 0;
      }
      const float d = fastscan_mv_chamfer_distance(q, *this, start, start + cloud_size);
      results[cid] = {get_id(cloud_id), d};
    });
  }

  void save(std::ofstream& out) const {
    out.write(reinterpret_cast<const char*>(&num_blocks), sizeof(num_blocks));
    size_t pc_sz = packed_codes.size();
    out.write(reinterpret_cast<const char*>(&pc_sz), sizeof(pc_sz));
    if (pc_sz)
      out.write(reinterpret_cast<const char*>(packed_codes.data()), pc_sz * sizeof(uint8_t));

    size_t n_clouds = offsets.size() > 0 ? offsets.size() - 1 : 0;
    out.write(reinterpret_cast<const char*>(&n_clouds), sizeof(n_clouds));

    size_t off_size = offsets.size();
    out.write(reinterpret_cast<const char*>(&off_size), sizeof(off_size));
    if (off_size)
      out.write(reinterpret_cast<const char*>(offsets.data()), off_size * sizeof(size_t));

    size_t sz_size = sizes_unpadded.size();
    out.write(reinterpret_cast<const char*>(&sz_size), sizeof(sz_size));
    if (sz_size)
      out.write(reinterpret_cast<const char*>(sizes_unpadded.data()), sz_size * sizeof(uint32_t));
  }

  void load(std::ifstream& in) {
    in.read(reinterpret_cast<char*>(&num_blocks), sizeof(num_blocks));
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
  }
};

// =========================================================================
// Chamfer distance: strip-outer, query-inner with fused code-load amortization.
// Unified 2-way dispatch: full strips → batched, tail → batched_masked.
// =========================================================================
template<bool Metric>
float fastscan_mv_chamfer_distance(const Quantized_Query_Point_Cloud<Metric>& q,
                                   const Quantized_Point_Cloud_Set<Metric>& db, size_t start,
                                   size_t true_end) {
  const size_t num_q = q.num_queries;
  if (num_q == 0) return 0.0f;
  if (true_end <= start) return std::numeric_limits<float>::max();

  const size_t cloud_size = true_end - start;
  const size_t strip_stride = internal::compute_strip_stride(db.num_blocks);
  const size_t strip0 = start / 64;
  const size_t num_full_strips = cloud_size / 64;
  const int tail = static_cast<int>(cloud_size % 64);
  const size_t total_strips = num_full_strips + (tail > 0 ? 1 : 0);

  auto strip_ptr = [&](size_t s) -> const uint8_t* { return &db.packed_codes[s * strip_stride]; };

  float total_chamfer = 0.0f;

  for (size_t q_base = 0; q_base < num_q; q_base += internal::NQ_BATCH) {
    const size_t q_count = std::min(internal::NQ_BATCH, num_q - q_base);

    const uint8_t* lut_ptrs[internal::NQ_BATCH];
    for (size_t qi = 0; qi < q_count; ++qi)
      lut_ptrs[qi] = q.get_lut(q_base + qi);

    internal::RunningMinV rmins[internal::NQ_BATCH];
    for (size_t qi = 0; qi < q_count; ++qi)
      rmins[qi] = internal::RunningMinV::max();

    for (size_t s = 0; s < num_full_strips; ++s) {
      const uint8_t* sp = strip_ptr(strip0 + s);
      if (s + 2 < total_strips) __builtin_prefetch(strip_ptr(strip0 + s + 2), 0, 3);
      internal::scan_64_batched(lut_ptrs, q_count, sp, db.num_blocks, rmins);
    }

    if (tail > 0) {
      internal::scan_64_batched_masked(lut_ptrs, q_count,
                                       strip_ptr(strip0 + num_full_strips), db.num_blocks,
                                       0, tail, rmins);
    }

    for (size_t qi = 0; qi < q_count; ++qi)
      total_chamfer += q.decode(q_base + qi, internal::reduce_running_min(rmins[qi]));
  }

  return total_chamfer / static_cast<float>(num_q);
}

// =========================================================================
// Multi-Vector Model
// =========================================================================
template<bool Metric>
class Model {
 public:
  uint32_t num_blocks = 0;
  size_t dim = 0;
  size_t dim_per_block = 0;
  static constexpr uint32_t K = 16;

  std::vector<Eigen::MatrixXf> codebooks;
  std::vector<Eigen::VectorXf> codebook_norms;

  Model() = default;

  template<typename PCSet>
  void train(const PCSet& pcs, uint32_t block_size = 32) {
    dim = pcs.get_dims();
    dim_per_block = block_size;
    if (dim_per_block == 0 || (dim % dim_per_block) != 0) {
      std::cerr << "Error: FastScan dim=" << dim << " not divisible by block_size=" << dim_per_block
                << "\n";
      abort();
    }
    num_blocks = static_cast<uint32_t>(dim / dim_per_block);
    codebooks.resize(num_blocks);
    codebook_norms.resize(num_blocks);
    const size_t n_points_raw_unpadded = pcs.total_size();
    const size_t sample_size = std::min(static_cast<size_t>(4096), n_points_raw_unpadded);

    parlay::parallel_for(0, num_blocks, [&](size_t b) {
      const size_t offset = b * dim_per_block;
      parlay::sequence<parlay::sequence<float>> sub(sample_size);
      std::mt19937 rng(static_cast<unsigned>(b + 1));
      std::uniform_int_distribution<size_t> distu(0, n_points_raw_unpadded - 1);
      for (size_t i = 0; i < sample_size; ++i) {
        const float* raw = reinterpret_cast<const float*>(pcs.data() + distu(rng) * dim);
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

  // ---------------------------------------------------------
  // Database Encoder (Compile-Time Dispatch)
  // ---------------------------------------------------------
  template<typename PCSet>
  Quantized_Point_Cloud_Set<Metric> encode(const PCSet& pcs) const {
    Quantized_Point_Cloud_Set<Metric> res;
    res.num_blocks = num_blocks;

    auto float_offsets = pcs.get_offsets();
    size_t n_clouds = float_offsets.size() > 0 ? float_offsets.size() - 1 : 0;

    res.offsets.resize(n_clouds + 1);
    res.offsets[0] = 0;
    res.sizes_unpadded = parlay::sequence<uint32_t>(n_clouds);

    size_t cur_padded = 0;
    for (size_t c = 0; c < n_clouds; ++c) {
      res.offsets[c] = cur_padded;
      size_t n_vecs = (float_offsets[c + 1] - float_offsets[c]) / dim;
      res.sizes_unpadded[c] = static_cast<uint32_t>(n_vecs);
      cur_padded += ((n_vecs + 63) / 64) * 64;
    }
    res.offsets[n_clouds] = cur_padded;

    size_t n_strips = cur_padded / 64;
    const size_t strip_stride = internal::compute_strip_stride(num_blocks);

    res.packed_codes.resize(n_strips * strip_stride, 0);

    auto pcs_ids = pcs.get_ids();
    res.ids = parlay::sequence<uint32_t>(pcs_ids.begin(), pcs_ids.end());

    parlay::parallel_for(0, n_clouds, [&](size_t c) {
      size_t src_start = float_offsets[c] / dim;
      size_t n_vecs = res.sizes_unpadded[c];
      size_t dst_start = res.offsets[c];
      size_t padded_sz = res.offsets[c + 1] - res.offsets[c];
      size_t n_strips_c = padded_sz / 64;
      size_t strip0_dst = dst_start / 64;

#if defined(__AVX512VNNI__) && defined(__AVX512VBMI__)
      // --- VNNI 4-BLOCK INTERLEAVED LAYOUT ---
      const uint32_t nb4 = ((num_blocks + 3) / 4) * 4;
      for (size_t s = 0; s < n_strips_c; ++s) {
        uint8_t* strip_base = res.packed_codes.data() + (strip0_dst + s) * strip_stride;
        for (uint32_t g = 0; g < nb4 / 4; ++g) {
          for (size_t lp = 0; lp < 32; ++lp) {
            const size_t v_even_in_cloud = s * 64 + lp * 2;
            const size_t v_odd_in_cloud = v_even_in_cloud + 1;
            const bool has_even = (v_even_in_cloud < n_vecs);
            const bool has_odd = (v_odd_in_cloud < n_vecs);
            const size_t v_even_src = src_start + v_even_in_cloud;
            const size_t v_odd_src = src_start + v_odd_in_cloud;
            for (uint32_t j = 0; j < 4; ++j) {
              const uint32_t b = g * 4 + j;
              const uint8_t c_e =
                  (b < num_blocks && has_even) ? find_best(pcs.data() + v_even_src * dim, b) : 0;
              const uint8_t c_o =
                  (b < num_blocks && has_odd) ? find_best(pcs.data() + v_odd_src * dim, b) : 0;
              strip_base[g * 128 + lp * 4 + j] = (c_e & 0x0F) | ((c_o & 0x0F) << 4);
            }
          }
        }
      }
#elif defined(__AVX512VBMI__)
      // --- VBMI 2-BLOCK LAYOUT ---
      for (size_t s = 0; s < n_strips_c; ++s) {
        uint8_t* strip_base = res.packed_codes.data() + (strip0_dst + s) * strip_stride;
        for (uint32_t b = 0; b < num_blocks; b += 2) {
          uint8_t* block_pair_base = strip_base + (b / 2) * 64;
          for (size_t lane_pair = 0; lane_pair < 32; ++lane_pair) {
            const size_t v_even_in_cloud = s * 64 + lane_pair * 2;
            const size_t v_odd_in_cloud = v_even_in_cloud + 1;
            const bool has_even = (v_even_in_cloud < n_vecs);
            const bool has_odd = (v_odd_in_cloud < n_vecs);
            const size_t v_even_src = src_start + v_even_in_cloud;
            const size_t v_odd_src = src_start + v_odd_in_cloud;

            const uint8_t c_e_b0 = has_even ? find_best(pcs.data() + v_even_src * dim, b) : 0;
            const uint8_t c_o_b0 = has_odd ? find_best(pcs.data() + v_odd_src * dim, b) : 0;
            block_pair_base[lane_pair * 2] = (c_e_b0 & 0x0F) | ((c_o_b0 & 0x0F) << 4);

            if (b + 1 < num_blocks) {
              const uint8_t c_e_b1 = has_even ? find_best(pcs.data() + v_even_src * dim, b + 1) : 0;
              const uint8_t c_o_b1 = has_odd ? find_best(pcs.data() + v_odd_src * dim, b + 1) : 0;
              block_pair_base[lane_pair * 2 + 1] = (c_e_b1 & 0x0F) | ((c_o_b1 & 0x0F) << 4);
            }
          }
        }
      }
#else
      // --- ORIGINAL STRIPED LAYOUT (AVX512F / AVX2 / SCALAR) ---
      for (size_t s = 0; s < n_strips_c; ++s) {
        uint8_t* strip_base = res.packed_codes.data() + (strip0_dst + s) * strip_stride;
        for (size_t lane_pair = 0; lane_pair < 32; ++lane_pair) {
          const size_t v_even_in_cloud = s * 64 + lane_pair * 2;
          const size_t v_odd_in_cloud = v_even_in_cloud + 1;
          const bool has_even = (v_even_in_cloud < n_vecs);
          const bool has_odd = (v_odd_in_cloud < n_vecs);
          const size_t v_even_src = src_start + v_even_in_cloud;
          const size_t v_odd_src = src_start + v_odd_in_cloud;
          for (uint32_t b = 0; b < num_blocks; ++b) {
            const uint8_t c_e = has_even ? find_best(pcs.data() + v_even_src * dim, b) : 0;
            const uint8_t c_o = has_odd ? find_best(pcs.data() + v_odd_src * dim, b) : 0;
            strip_base[static_cast<size_t>(b) * 32 + lane_pair] =
                (c_e & 0x0F) | ((c_o & 0x0F) << 4);
          }
        }
      }
#endif
    });

    return res;
  }

  template<typename ChPoint>
  Quantized_Query_Point_Cloud<Metric> quantize_query(const ChPoint& query_cloud) const {
    Quantized_Query_Point_Cloud<Metric> res;
    res.num_queries = query_cloud.size();
    res.num_blocks = num_blocks;

    if (res.num_queries == 0) return res;

    const uint32_t lpb = internal::lut_padded_blocks(num_blocks);
    res.flat_int_luts.resize(res.num_queries * static_cast<size_t>(lpb) * 16, 0);
    res.min_dists.resize(res.num_queries, 0.0f);
    res.scales.resize(res.num_queries, 0.0f);

    const float* base = query_cloud.data();
    size_t q_dim = query_cloud.get_dims();

    for (size_t qi = 0; qi < res.num_queries; ++qi) {
      const float* qptr = base + qi * q_dim;

      std::vector<float> float_lut(num_blocks * K);
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

      res.min_dists[qi] = g_min;
      res.scales[qi] = std::max(1e-6f, (g_max - g_min) / 255.0f);

      uint8_t* out_lut = res.flat_int_luts.data() + qi * static_cast<size_t>(lpb) * 16;
      for (size_t i = 0; i < float_lut.size(); ++i) {
        out_lut[i] = static_cast<uint8_t>((float_lut[i] - g_min) / res.scales[qi]);
      }
    }

    return res;
  }

  void save(std::ofstream& out) const {
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

  void load(std::ifstream& in) {
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
};

// ------------------------------------------------------------------
// ManyToMany Batch Operator
// ------------------------------------------------------------------
template<typename PCS>
class ManyToMany {
 public:
  struct M2MScratch {
    std::vector<float> emb_min_dists;
    std::vector<internal::RunningMinV> running_mins;
    std::vector<uint16_t> min_dist_raw;
  };

  static void TopKIntoUninitialized(
      const std::vector<const Quantized_Query_Point_Cloud<PCS::is_metric()>*>& A, const PCS& B,
      uint32_t k, std::pair<uint32_t, float>* results, size_t q_block = 4,
      bool parallel_query_blocks = false) {

    const size_t num_q_clouds = A.size();
    const size_t num_db_clouds = (B.offsets.size() > 0) ? B.offsets.size() - 1 : 0;
    if (num_q_clouds == 0 || num_db_clouds == 0 || k == 0) return;

    const uint32_t num_blocks = B.num_blocks;
    const size_t db_strip_stride = internal::compute_strip_stride(num_blocks);

    auto process_chunk = [&](size_t q_start, size_t q_end) {
      thread_local M2MScratch scratch;
      const size_t nq_local = q_end - q_start;
      if (nq_local == 0) return;

      size_t max_embs = 0;
      for (size_t qi = 0; qi < nq_local; ++qi)
        max_embs = std::max(max_embs, static_cast<size_t>(A[q_start + qi]->num_queries));
      if (scratch.emb_min_dists.size() < max_embs)
        scratch.emb_min_dists.resize(max_embs);

      // Initialize results with sentinels for sorted insertion
      for (size_t qi = 0; qi < nq_local; ++qi)
        for (size_t ki = 0; ki < k; ++ki)
          results[(q_start + qi) * k + ki] = {UINT32_MAX, std::numeric_limits<float>::max()};

      for (size_t qi = 0; qi < nq_local; ++qi) {
        const auto* qc = A[q_start + qi];
        const size_t ne = qc->num_queries;
        if (ne == 0) continue;
        const float ne_f = static_cast<float>(ne);

        for (size_t c = 0; c < num_db_clouds; ++c) {
          const size_t start_vec = B.offsets[c];
          const size_t cloud_size =
              (B.sizes_unpadded.size() == num_db_clouds)
                  ? static_cast<size_t>(B.sizes_unpadded[c])
                  : (B.offsets[c + 1] - start_vec);
          if (cloud_size == 0) continue;

          internal::fastscan_mv_chamfer_fused<PCS::is_metric()>(
              qc->flat_int_luts.data(), qc->scales.data(),
              qc->min_dists.data(), ne, num_blocks,
              B.packed_codes.data(), db_strip_stride, start_vec, cloud_size,
              scratch.emb_min_dists.data(),
              scratch.running_mins, scratch.min_dist_raw);

          float chamfer_dist = 0.0f;
          for (size_t e = 0; e < ne; ++e)
            chamfer_dist += scratch.emb_min_dists[e];
          chamfer_dist /= ne_f;

          const size_t idx = (q_start + qi) * k;
          if (chamfer_dist < results[idx + k - 1].second) {
            results[idx + k - 1] = {B.get_id(c), chamfer_dist};
            for (size_t j = k - 1; j > 0 && results[idx + j].second < results[idx + j - 1].second;
                 --j)
              std::swap(results[idx + j], results[idx + j - 1]);
          }
        }
      }

    };

    if (parallel_query_blocks) {
      if (q_block == 0) q_block = 4;
      parlay::blocked_for(
          0, num_q_clouds, q_block,
          [&](size_t /*block_idx*/, size_t start, size_t end) { process_chunk(start, end); });
    } else {
      process_chunk(0, num_q_clouds);
    }
  }
};

}  // namespace fastscan_mv
}  // namespace mvsic
