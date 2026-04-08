#pragma once

// Single-vector PQ-style TurboQuant (PQTQ) for MVSIC.
//
// SYMMETRIC design: queries are PQTQ-encoded exactly like DB points.
// Scoring lookups go through a precomputed 16x16 global symmetric LUT
// (pairwise centroid dot products, int8) expanded into an 8KB `half_lut[256][32]`.
// There is no per-query float→int8 LUT construction; the query is just its
// PQTQ codes. The half_lut (and the codebook / global sym LUT) are computed
// once per block_size the first time a Model of that block_size is trained
// and cached as a process-wide static, matching the prompt's instruction
// that the half_lut be built exactly once at program start time.
//
// Layout:
//   - Flat per-point codes : packed_flat[i]    = num_bytes_per_datapoint bytes
//   - 4-block interleaved  : packed_strips[s]  = ((nb+3)/4)*4 * 32 bytes per 64-point strip
//   - Per-point norms + sqns
//
// Scan kernel (AVX-512 VBMI+VNNI): vnni_scan_strip() loops 4 blocks at a time,
// loads two half_lut rows from the (query) nibble pair, composes a 64-byte
// ZMM LUT (`vinserti64x4`), looks up DB nibbles via `vpermutexvar_epi8`, and
// accumulates with `vpdpbusd`. Bias correction (half_lut is pre-XORed with
// 0x80 so vpdpbusd can consume it as unsigned) is applied in the decode step.
//
// Fallback: a plain scalar scan over the global sym LUT (int8) with the
// same bias-free math for non-AVX512 builds.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

#include "parlay/parallel.h"
#include "parlay/sequence.h"
#include "rabitqlib/utils/rotator.hpp"
#include "mvsic/core/quantization/pqtq_codebooks.h"

namespace mvsic {
namespace pqtq {

// =========================================================================
// Static per-block_size tables (codebook, global sym LUT, half_lut).
// =========================================================================
namespace internal {

static constexpr uint32_t kK = 16;  // centroids per block

struct Tables {
  size_t block_size = 0;
  float codebook[16][16] = {};       // codebook[k][d], up to D=16
  float codebook_sq_norms[16] = {};  // ||c_k||^2
  int8_t codebook_int8[16 * 16] = {};
  alignas(64) int8_t global_sym_lut_int8[256] = {};  // <c_i, c_j> compressed
  float global_sym_lut_scale = 1.0f;
  alignas(64) uint8_t biased_sym_lut[256] = {};  // ^= 0x80 for vpdpbusd
  alignas(64) uint8_t half_lut[256][32] = {};    // two rows per packed-nibble byte
};

inline void build_tables(Tables* t, size_t block_size) {
  t->block_size = block_size;
  auto copy_cb = [&](auto const& cb_f, auto const& cb_i8) {
    for (size_t k = 0; k < 16; ++k)
      for (size_t d = 0; d < block_size; ++d) {
        t->codebook[k][d] = cb_f[k][d];
        t->codebook_int8[k * block_size + d] = cb_i8[k][d];
      }
  };
  using namespace ::mvsic::pqtq_codebooks;
  switch (block_size) {
    case 1: copy_cb(kPQ_CentroidsFloat_D1_K16, kPQ_CentroidsInt8_D1_K16); break;
    case 2: copy_cb(kPQ_CentroidsFloat_D2_K16, kPQ_CentroidsInt8_D2_K16); break;
    case 4: copy_cb(kPQ_CentroidsFloat_D4_K16, kPQ_CentroidsInt8_D4_K16); break;
    case 8: copy_cb(kPQ_CentroidsFloat_D8_K16, kPQ_CentroidsInt8_D8_K16); break;
    case 16: copy_cb(kPQ_CentroidsFloat_D16_K16, kPQ_CentroidsInt8_D16_K16); break;
    default: break;
  }
  for (size_t k = 0; k < 16; ++k) {
    float s = 0.0f;
    for (size_t d = 0; d < block_size; ++d) s += t->codebook[k][d] * t->codebook[k][d];
    t->codebook_sq_norms[k] = s;
  }
  // 16x16 pairwise dot product → int8 compressed global symmetric LUT.
  float sym[256];
  for (int qi = 0; qi < 16; ++qi)
    for (int dj = 0; dj < 16; ++dj) {
      float dot = 0.0f;
      for (size_t d = 0; d < block_size; ++d) dot += t->codebook[qi][d] * t->codebook[dj][d];
      sym[qi * 16 + dj] = dot;
    }
  float max_abs = 0.0f;
  for (int i = 0; i < 256; ++i) max_abs = std::max(max_abs, std::abs(sym[i]));
  if (max_abs > 0.0f) {
    t->global_sym_lut_scale = max_abs / 127.0f;
    const float inv = 1.0f / t->global_sym_lut_scale;
    for (int i = 0; i < 256; ++i) {
      float v = std::round(std::clamp(sym[i] * inv, -127.0f, 127.0f));
      t->global_sym_lut_int8[i] = static_cast<int8_t>(v);
    }
  } else {
    t->global_sym_lut_scale = 1.0f;
    std::memset(t->global_sym_lut_int8, 0, 256);
  }
  // Pre-bias by +128 (XOR 0x80) so vpdpbusd can consume it as unsigned.
  for (int i = 0; i < 256; ++i)
    t->biased_sym_lut[i] = static_cast<uint8_t>(t->global_sym_lut_int8[i] ^ 0x80);
  // Expand to 8KB half_lut[256][32].
  for (int v = 0; v < 256; ++v) {
    std::memcpy(t->half_lut[v], t->biased_sym_lut + (v & 0xF) * 16, 16);
    std::memcpy(t->half_lut[v] + 16, t->biased_sym_lut + (v >> 4) * 16, 16);
  }
}

inline int block_size_to_idx(size_t bs) {
  switch (bs) {
    case 1: return 0;
    case 2: return 1;
    case 4: return 2;
    case 8: return 3;
    case 16: return 4;
    default: return -1;
  }
}

// Process-wide static: built exactly once per block_size on first use.
inline const Tables* get_tables(size_t block_size) {
  static std::array<Tables, 5> kCache{};
  static std::array<std::once_flag, 5> kFlags{};
  const int idx = block_size_to_idx(block_size);
  if (idx < 0) return nullptr;
  std::call_once(kFlags[idx], [idx, block_size] {
    build_tables(&kCache[idx], block_size);
  });
  return &kCache[idx];
}

// -------------------------------------------------------------------------
// Small helpers
// -------------------------------------------------------------------------
inline uint8_t get_packed_nibble(const uint8_t* pc, uint32_t b) {
  return static_cast<uint8_t>((pc[b / 2] >> (4 * (b % 2))) & 0x0F);
}

inline size_t num_strips_for(size_t n_points) {
  return ((n_points + 63) / 64);
}

inline size_t strip_stride_bytes(uint32_t num_blocks) {
  const uint32_t nb4 = ((num_blocks + 3) / 4) * 4;
  return static_cast<size_t>(nb4) * 32;
}

// Build a "per-query scan LUT" of `(num_blocks/2 rounded up) * 32` bytes.
// The scan kernel loads 64 bytes per 4-block iteration (== 2 half_lut rows),
// so laying the rows out consecutively turns a [load, load, vinserti64x4]
// sequence into a single `_mm512_loadu_si512`. Kernel iterates `b += 4`, so
// each iteration consumes `half_lut[q_codes[b/2]] | half_lut[q_codes[b/2+1]]`
// at offset `(b/2) * 16` (bytes). We lay it out as one row per two blocks:
//   scan_lut[(b/2) * 32 .. (b/2+1) * 32] = half_lut[q_codes[b/2]]
// so `_mm512_loadu_si512(scan_lut + (b/2) * 32)` loads the two rows for
// blocks {b, b+1, b+2, b+3} in one go. Size is padded up to `nb4/2 * 32`
// bytes so the final iteration doesn't read out of bounds.
inline size_t query_scan_lut_bytes(uint32_t num_blocks) {
  const uint32_t nb4 = ((num_blocks + 3) / 4) * 4;
  return static_cast<size_t>(nb4 / 2) * 32 + 32;  // +32 so loadu(+nb4/2-1)*32 of 64B is safe
}

inline void build_query_scan_lut(const Tables* t, const uint8_t* q_codes, uint32_t num_blocks,
                                 uint8_t* out) {
  const uint32_t nb4 = ((num_blocks + 3) / 4) * 4;
  const size_t num_bytes_dp = static_cast<size_t>(num_blocks + 1) / 2;
  const size_t total_entries = static_cast<size_t>(nb4 / 2) + 1;  // +1 pad for safe 64B load
  for (size_t e = 0; e < total_entries; ++e) {
    uint8_t byte = 0;
    if (e < num_bytes_dp) byte = q_codes[e];
    std::memcpy(out + e * 32, t->half_lut[byte], 32);
  }
}

// Permute a flat [64] float array (per-point norms / sqns in consecutive
// point order) into the "acc-aligned" lane order the scan kernel uses:
//   out[ 0..15] = in[ 0, 2, 4, ..., 30]   // acc0 lanes
//   out[16..31] = in[ 1, 3, 5, ..., 31]   // acc1 lanes
//   out[32..47] = in[32,34,36, ..., 62]   // acc2 lanes
//   out[48..63] = in[33,35,37, ..., 63]   // acc3 lanes
inline void permute_strip_floats_acc_aligned(const float* in64, float* out64) {
  for (int i = 0; i < 16; ++i) {
    out64[0 + i] = in64[2 * i];
    out64[16 + i] = in64[2 * i + 1];
    out64[32 + i] = in64[32 + 2 * i];
    out64[48 + i] = in64[33 + 2 * i];
  }
}

// Pack flat per-point codes into 4-block interleaved strip layout.
inline void pack_into_strips_interleaved4(const uint8_t* point_codes, size_t n_points,
                                          uint32_t num_blocks, uint8_t* packed_out) {
  const size_t num_bytes_dp = static_cast<size_t>(num_blocks + 1) / 2;
  const size_t n_padded = ((n_points + 63) / 64) * 64;
  const size_t num_strips = n_padded / 64;
  const uint32_t nb4 = ((num_blocks + 3) / 4) * 4;
  const size_t strip_stride = static_cast<size_t>(nb4) * 32;

  for (size_t s = 0; s < num_strips; ++s) {
    uint8_t* strip_base = packed_out + s * strip_stride;
    for (uint32_t g = 0; g < nb4 / 4; ++g) {
      for (size_t lp = 0; lp < 32; ++lp) {
        const size_t v_even = s * 64 + lp * 2;
        const size_t v_odd = v_even + 1;
        const uint8_t* pc_even =
            (v_even < n_points) ? point_codes + v_even * num_bytes_dp : nullptr;
        const uint8_t* pc_odd =
            (v_odd < n_points) ? point_codes + v_odd * num_bytes_dp : nullptr;
        for (uint32_t j = 0; j < 4; ++j) {
          const uint32_t b = g * 4 + j;
          const uint8_t n_e = (b < num_blocks && pc_even) ? get_packed_nibble(pc_even, b) : 0;
          const uint8_t n_o = (b < num_blocks && pc_odd) ? get_packed_nibble(pc_odd, b) : 0;
          strip_base[g * 128 + lp * 4 + j] =
              static_cast<uint8_t>((n_e & 0x0F) | ((n_o & 0x0F) << 4));
        }
      }
    }
  }
}

// Scalar scan of a (possibly partial) 64-point strip. Used on non-AVX512
// builds and as the fallback for the AVX512 tail.
template<bool Metric>
inline void scan_64_scalar(const int8_t* global_sym_lut, float lut_scale, float q_nsf,
                           float q_sqn, uint32_t num_blocks, const uint8_t* qcodes,
                           const uint8_t* flat_points_64, size_t nbytes_per_dp,
                           const float* norms_64, const float* sqnorms_64, size_t valid_count,
                           float* results) {
  const float cs = q_nsf * lut_scale;
  for (size_t lane = 0; lane < 64; ++lane) {
    if (lane >= valid_count) {
      results[lane] = std::numeric_limits<float>::max();
      continue;
    }
    const uint8_t* pc = flat_points_64 + lane * nbytes_per_dp;
    int32_t acc = 0;
    for (uint32_t b = 0; b < num_blocks; ++b) {
      const uint8_t qi = get_packed_nibble(qcodes, b);
      const uint8_t di = get_packed_nibble(pc, b);
      acc += static_cast<int32_t>(global_sym_lut[qi * 16 + di]);
    }
    const float dot = norms_64[lane] * static_cast<float>(acc) * cs;
    if constexpr (Metric) {
      results[lane] = sqnorms_64[lane] + q_sqn - 2.0f * dot;
    } else {
      results[lane] = -dot;
    }
  }
}

#if defined(__AVX512F__) && defined(__AVX512VBMI__) && defined(__AVX512VNNI__)
// Inline VNNI scan for one query against one 64-point 4-block interleaved strip.
// Mirrors reference `vnni_scan_strip` (processleaf_pqtq.cc). Four int32 acc
// registers for {even pts of first 32-pt half, odd of same, even of second
// half, odd of same}; the decode step interleaves them back into consecutive
// point order.
static inline void vnni_scan_strip(const uint8_t* strip_codes, uint32_t nb4,
                                   const uint8_t* qcodes, const uint8_t (*half_lut)[32],
                                   const __m512i low_mask, const __m512i ones_i8,
                                   const __m512i offset_mask, __m512i& acc0, __m512i& acc1,
                                   __m512i& acc2, __m512i& acc3) {
  acc0 = _mm512_setzero_si512();
  acc1 = _mm512_setzero_si512();
  acc2 = _mm512_setzero_si512();
  acc3 = _mm512_setzero_si512();
  const uint8_t* codes_ptr = strip_codes;
  for (uint32_t b = 0; b < nb4; b += 4) {
    const __m512i packed1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(codes_ptr));
    const __m512i packed2 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(codes_ptr + 64));
    codes_ptr += 128;

    const __m512i ce1 = _mm512_or_si512(_mm512_and_si512(packed1, low_mask), offset_mask);
    const __m512i co1 = _mm512_or_si512(
        _mm512_and_si512(_mm512_srli_epi16(packed1, 4), low_mask), offset_mask);
    const __m512i ce2 = _mm512_or_si512(_mm512_and_si512(packed2, low_mask), offset_mask);
    const __m512i co2 = _mm512_or_si512(
        _mm512_and_si512(_mm512_srli_epi16(packed2, 4), low_mask), offset_mask);

    const uint32_t boff = b / 2;
    const __m512i lut64 = _mm512_inserti64x4(
        _mm512_castsi256_si512(_mm256_load_si256(
            reinterpret_cast<const __m256i*>(half_lut[qcodes[boff]]))),
        _mm256_load_si256(reinterpret_cast<const __m256i*>(half_lut[qcodes[boff + 1]])), 1);

    acc0 = _mm512_dpbusd_epi32(acc0, _mm512_permutexvar_epi8(ce1, lut64), ones_i8);
    acc1 = _mm512_dpbusd_epi32(acc1, _mm512_permutexvar_epi8(co1, lut64), ones_i8);
    acc2 = _mm512_dpbusd_epi32(acc2, _mm512_permutexvar_epi8(ce2, lut64), ones_i8);
    acc3 = _mm512_dpbusd_epi32(acc3, _mm512_permutexvar_epi8(co2, lut64), ones_i8);
  }
}

// Decode {acc0..3} (interleaved even/odd halves) into 64 float distances.
// bias_f = num_blocks * 128.0f (the half_lut was pre-biased by +128 per block).
template<bool Metric>
static inline void vnni_decode_strip(__m512i acc0, __m512i acc1, __m512i acc2, __m512i acc3,
                                     float bias_f, float cs, float q_sq, const float* norms_64,
                                     const float* sqnorms_64, float* results) {
  const __m512 v_bias = _mm512_set1_ps(bias_f);
  const __m512 v_cs = _mm512_set1_ps(cs);
  const __m512 v_qsq = _mm512_set1_ps(q_sq);
  const __m512 v_two = _mm512_set1_ps(2.0f);
  const __m512 v_sign = _mm512_set1_ps(-0.0f);

  __m512 f0 = _mm512_sub_ps(_mm512_cvtepi32_ps(acc0), v_bias);
  __m512 f1 = _mm512_sub_ps(_mm512_cvtepi32_ps(acc1), v_bias);
  __m512 f2 = _mm512_sub_ps(_mm512_cvtepi32_ps(acc2), v_bias);
  __m512 f3 = _mm512_sub_ps(_mm512_cvtepi32_ps(acc3), v_bias);

  __m512 ilo01 = _mm512_unpacklo_ps(f0, f1);
  __m512 ihi01 = _mm512_unpackhi_ps(f0, f1);
  __m512 ilo23 = _mm512_unpacklo_ps(f2, f3);
  __m512 ihi23 = _mm512_unpackhi_ps(f2, f3);

  auto decode16 = [&]<int lane_lo, int lane_hi>(__m512 ilo, __m512 ihi, const float* nrm,
                                                const float* sqn, float* out) {
    __m128 a = _mm512_extractf32x4_ps(ilo, lane_lo);
    __m128 b = _mm512_extractf32x4_ps(ihi, lane_lo);
    __m128 c = _mm512_extractf32x4_ps(ilo, lane_hi);
    __m128 d = _mm512_extractf32x4_ps(ihi, lane_hi);
    __m256 lo8 = _mm256_insertf128_ps(_mm256_castps128_ps256(a), b, 1);
    __m256 hi8 = _mm256_insertf128_ps(_mm256_castps128_ps256(c), d, 1);
    __m512 sum16 = _mm512_insertf32x8(_mm512_castps256_ps512(lo8), hi8, 1);
    __m512 nrm_v = _mm512_loadu_ps(nrm);
    __m512 dot = _mm512_mul_ps(_mm512_mul_ps(nrm_v, sum16), v_cs);
    if constexpr (Metric) {
      __m512 sqn_v = _mm512_loadu_ps(sqn);
      _mm512_storeu_ps(out, _mm512_add_ps(sqn_v, _mm512_fnmadd_ps(v_two, dot, v_qsq)));
    } else {
      _mm512_storeu_ps(out, _mm512_xor_ps(dot, v_sign));
    }
  };
  decode16.template operator()<0, 1>(ilo01, ihi01, norms_64, sqnorms_64, results);
  decode16.template operator()<2, 3>(ilo01, ihi01, norms_64 + 16,
                                     Metric ? sqnorms_64 + 16 : nullptr, results + 16);
  decode16.template operator()<0, 1>(ilo23, ihi23, norms_64 + 32,
                                     Metric ? sqnorms_64 + 32 : nullptr, results + 32);
  decode16.template operator()<2, 3>(ilo23, ihi23, norms_64 + 48,
                                     Metric ? sqnorms_64 + 48 : nullptr, results + 48);
}

// -------------------------------------------------------------------------
// Fused scan + decode + min-accumulate, 3 queries per strip, no tmp[] buffer
// -------------------------------------------------------------------------
// The strip scan produces four int32 accumulators whose 16 lanes each hold
// results for a *subset* of the 64 points:
//     acc0 lanes 0..15 -> points {0, 2, 4, ..., 30}   (even, first half)
//     acc1 lanes 0..15 -> points {1, 3, 5, ..., 31}   (odd,  first half)
//     acc2 lanes 0..15 -> points {32, 34, ..., 62}    (even, second half)
//     acc3 lanes 0..15 -> points {33, 35, ..., 63}    (odd,  second half)
//
// The previous kernel reordered these into consecutive-point order (via an
// unpack/extract/insert chain) and stored the result to a `tmp[64]` buffer so
// a follow-up `reduce_min_valid` could re-load and reduce. For min-reduction
// the lane order does not matter, so we skip the reorder entirely: if norms
// and squared norms are stored in the same "acc-aligned" lane layout, each
// acc decodes with just (cvt + sub + mul + fnmadd + add) and min-accumulates
// into a single __m512 running min. No store/reload, no shuffle chain.
//
// `norms_p[0..15]` / `sqns_p[0..15]` must be the norms for acc0's points
// (pts 0,2,...,30), `norms_p[16..31]` for acc1's points, etc. Build these at
// encode time with `permute_strip_floats_acc_aligned`.
//
// Tail masking on the last strip is handled by `mask_blend_ps(mask_i, +inf,
// dist_i)` before the min-accumulate. `masks_acc` should be 4 __mmask16 where
// bit l is set iff that lane is a valid point (for non-last strips, pass
// 0xFFFF × 4).
//
// The per-query "scan LUTs" (`q0_scan_lut`, ...) are built with
// `build_query_scan_lut` at query-encode time.
template<bool Metric>
static inline void vnni_scan_strip_fused3_minacc(
    const uint8_t* strip_codes, uint32_t nb4,
    const uint8_t* q0_scan_lut, const uint8_t* q1_scan_lut, const uint8_t* q2_scan_lut,
    const float* norms_p, const float* sqns_p,
    float cs0, float cs1, float cs2, float qsq0, float qsq1, float qsq2,
    int32_t bias_int, const __mmask16 masks_acc[4], const uint8_t* prefetch_strip,
    __m512& min0, __m512& min1, __m512& min2) {
  const __m512i low_mask = _mm512_set1_epi8(0x0F);
  const __m512i ones_i8 = _mm512_set1_epi8(1);
  const __m512i offset_mask = _mm512_set1_epi32(0x30201000);

  // Init accumulators to -bias so `cvtepi32_ps(final_acc)` already yields the
  // bias-corrected value and the decode path can drop its `sub_ps(v_bias)`.
  const __m512i acc_init = _mm512_set1_epi32(-bias_int);
  __m512i a00 = acc_init;
  __m512i a01 = acc_init;
  __m512i a02 = acc_init;
  __m512i a03 = acc_init;
  __m512i a10 = acc_init;
  __m512i a11 = acc_init;
  __m512i a12 = acc_init;
  __m512i a13 = acc_init;
  __m512i a20 = acc_init;
  __m512i a21 = acc_init;
  __m512i a22 = acc_init;
  __m512i a23 = acc_init;

  // Touch the next strip's first cache line to hide L2 → L1 latency on
  // cloud-internal strip transitions. Caller passes nullptr to skip.
  if (prefetch_strip != nullptr) {
    _mm_prefetch(reinterpret_cast<const char*>(prefetch_strip), _MM_HINT_T0);
    _mm_prefetch(reinterpret_cast<const char*>(prefetch_strip) + 64, _MM_HINT_T0);
  }

  const uint8_t* codes_ptr = strip_codes;
  const uint8_t* q0p = q0_scan_lut;
  const uint8_t* q1p = q1_scan_lut;
  const uint8_t* q2p = q2_scan_lut;
  for (uint32_t b = 0; b < nb4; b += 4) {
    const __m512i packed1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(codes_ptr));
    const __m512i packed2 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(codes_ptr + 64));
    codes_ptr += 128;

    const __m512i ce1 = _mm512_or_si512(_mm512_and_si512(packed1, low_mask), offset_mask);
    const __m512i co1 = _mm512_or_si512(
        _mm512_and_si512(_mm512_srli_epi16(packed1, 4), low_mask), offset_mask);
    const __m512i ce2 = _mm512_or_si512(_mm512_and_si512(packed2, low_mask), offset_mask);
    const __m512i co2 = _mm512_or_si512(
        _mm512_and_si512(_mm512_srli_epi16(packed2, 4), low_mask), offset_mask);

    const __m512i lut0 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(q0p));
    const __m512i lut1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(q1p));
    const __m512i lut2 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(q2p));
    q0p += 64;
    q1p += 64;
    q2p += 64;

    a00 = _mm512_dpbusd_epi32(a00, _mm512_permutexvar_epi8(ce1, lut0), ones_i8);
    a01 = _mm512_dpbusd_epi32(a01, _mm512_permutexvar_epi8(co1, lut0), ones_i8);
    a02 = _mm512_dpbusd_epi32(a02, _mm512_permutexvar_epi8(ce2, lut0), ones_i8);
    a03 = _mm512_dpbusd_epi32(a03, _mm512_permutexvar_epi8(co2, lut0), ones_i8);
    a10 = _mm512_dpbusd_epi32(a10, _mm512_permutexvar_epi8(ce1, lut1), ones_i8);
    a11 = _mm512_dpbusd_epi32(a11, _mm512_permutexvar_epi8(co1, lut1), ones_i8);
    a12 = _mm512_dpbusd_epi32(a12, _mm512_permutexvar_epi8(ce2, lut1), ones_i8);
    a13 = _mm512_dpbusd_epi32(a13, _mm512_permutexvar_epi8(co2, lut1), ones_i8);
    a20 = _mm512_dpbusd_epi32(a20, _mm512_permutexvar_epi8(ce1, lut2), ones_i8);
    a21 = _mm512_dpbusd_epi32(a21, _mm512_permutexvar_epi8(co1, lut2), ones_i8);
    a22 = _mm512_dpbusd_epi32(a22, _mm512_permutexvar_epi8(ce2, lut2), ones_i8);
    a23 = _mm512_dpbusd_epi32(a23, _mm512_permutexvar_epi8(co2, lut2), ones_i8);
  }

  // Decode + min-accumulate. Norms and sqns are shared across all 3 queries.
  // Note: bias is pre-applied via acc_init above, so no sub_ps here.
  //
  // IP case:   dist = -(f * nrm * cs)           = fmadd(f, -nrm*cs, 0)
  // L2 case:   dist = sqn + qsq - 2*f*nrm*cs    = fmadd(f, -2*nrm*cs, sqn + qsq)
  //
  // We precompute per-query-per-acc:
  //   coeff_i = -nrm_i * cs        (IP)
  //   coeff_i = -2 * nrm_i * cs    (L2)
  //   bias_i  = 0                   (IP)
  //   bias_i  = sqn_i + qsq         (L2)
  // Then per-acc decode = fmadd(cvtps(a), coeff, bias) → 1 cvt + 1 fmadd.
  const __m512 v_inf = _mm512_set1_ps(std::numeric_limits<float>::infinity());

  // Load shared norms/sqns once per acc, reuse across queries.
  const __m512 nrm0 = _mm512_loadu_ps(norms_p + 0);
  const __m512 nrm1 = _mm512_loadu_ps(norms_p + 16);
  const __m512 nrm2 = _mm512_loadu_ps(norms_p + 32);
  const __m512 nrm3 = _mm512_loadu_ps(norms_p + 48);
  __m512 sqn0, sqn1, sqn2, sqn3;
  if constexpr (Metric) {
    sqn0 = _mm512_loadu_ps(sqns_p + 0);
    sqn1 = _mm512_loadu_ps(sqns_p + 16);
    sqn2 = _mm512_loadu_ps(sqns_p + 32);
    sqn3 = _mm512_loadu_ps(sqns_p + 48);
  } else {
    sqn0 = _mm512_setzero_ps();
    sqn1 = _mm512_setzero_ps();
    sqn2 = _mm512_setzero_ps();
    sqn3 = _mm512_setzero_ps();
  }

  const float cs_mul = Metric ? -2.0f : -1.0f;

  auto decode_one = [&](__m512i a0, __m512i a1, __m512i a2, __m512i a3, float cs, float qsq,
                        __m512& running_min) {
    const __m512 v_coef = _mm512_set1_ps(cs * cs_mul);  // broadcast -cs or -2*cs
    // Per-acc coefficient: -nrm*cs (IP) or -2*nrm*cs (L2)
    const __m512 c0 = _mm512_mul_ps(nrm0, v_coef);
    const __m512 c1 = _mm512_mul_ps(nrm1, v_coef);
    const __m512 c2 = _mm512_mul_ps(nrm2, v_coef);
    const __m512 c3 = _mm512_mul_ps(nrm3, v_coef);
    __m512 b0, b1, b2, b3;
    if constexpr (Metric) {
      const __m512 v_qsq = _mm512_set1_ps(qsq);
      b0 = _mm512_add_ps(sqn0, v_qsq);
      b1 = _mm512_add_ps(sqn1, v_qsq);
      b2 = _mm512_add_ps(sqn2, v_qsq);
      b3 = _mm512_add_ps(sqn3, v_qsq);
    } else {
      b0 = _mm512_setzero_ps();
      b1 = _mm512_setzero_ps();
      b2 = _mm512_setzero_ps();
      b3 = _mm512_setzero_ps();
    }
    // dist = fmadd(cvtps(a), c, b). Port: 4× p0/p5 cvt + 4× p0/p5 fmadd.
    __m512 dist0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(a0), c0, b0);
    __m512 dist1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(a1), c1, b1);
    __m512 dist2 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(a2), c2, b2);
    __m512 dist3 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(a3), c3, b3);
    // Apply tail mask (blend +inf into invalid lanes on last strip).
    dist0 = _mm512_mask_blend_ps(masks_acc[0], v_inf, dist0);
    dist1 = _mm512_mask_blend_ps(masks_acc[1], v_inf, dist1);
    dist2 = _mm512_mask_blend_ps(masks_acc[2], v_inf, dist2);
    dist3 = _mm512_mask_blend_ps(masks_acc[3], v_inf, dist3);
    // Min-accumulate (4-way tree feeds 2 p0/p1 min units).
    __m512 m01 = _mm512_min_ps(dist0, dist1);
    __m512 m23 = _mm512_min_ps(dist2, dist3);
    __m512 m = _mm512_min_ps(m01, m23);
    running_min = _mm512_min_ps(running_min, m);
  };

  decode_one(a00, a01, a02, a03, cs0, qsq0, min0);
  decode_one(a10, a11, a12, a13, cs1, qsq1, min1);
  decode_one(a20, a21, a22, a23, cs2, qsq2, min2);
}

// Single-query variant of the fused kernel — used for the batch-tail when
// `num_queries % 3 != 0`. Same semantics as the Q=3 version above.
template<bool Metric>
static inline void vnni_scan_strip_fused1_minacc(
    const uint8_t* strip_codes, uint32_t nb4, const uint8_t* q_scan_lut,
    const float* norms_p, const float* sqns_p, float cs, float qsq, int32_t bias_int,
    const __mmask16 masks_acc[4], const uint8_t* prefetch_strip, __m512& running_min) {
  const __m512i low_mask = _mm512_set1_epi8(0x0F);
  const __m512i ones_i8 = _mm512_set1_epi8(1);
  const __m512i offset_mask = _mm512_set1_epi32(0x30201000);

  const __m512i acc_init = _mm512_set1_epi32(-bias_int);
  __m512i a0 = acc_init;
  __m512i a1 = acc_init;
  __m512i a2 = acc_init;
  __m512i a3 = acc_init;

  if (prefetch_strip != nullptr) {
    _mm_prefetch(reinterpret_cast<const char*>(prefetch_strip), _MM_HINT_T0);
    _mm_prefetch(reinterpret_cast<const char*>(prefetch_strip) + 64, _MM_HINT_T0);
  }

  const uint8_t* codes_ptr = strip_codes;
  const uint8_t* qp = q_scan_lut;
  for (uint32_t b = 0; b < nb4; b += 4) {
    const __m512i packed1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(codes_ptr));
    const __m512i packed2 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(codes_ptr + 64));
    codes_ptr += 128;

    const __m512i ce1 = _mm512_or_si512(_mm512_and_si512(packed1, low_mask), offset_mask);
    const __m512i co1 = _mm512_or_si512(
        _mm512_and_si512(_mm512_srli_epi16(packed1, 4), low_mask), offset_mask);
    const __m512i ce2 = _mm512_or_si512(_mm512_and_si512(packed2, low_mask), offset_mask);
    const __m512i co2 = _mm512_or_si512(
        _mm512_and_si512(_mm512_srli_epi16(packed2, 4), low_mask), offset_mask);

    const __m512i lut = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(qp));
    qp += 64;

    a0 = _mm512_dpbusd_epi32(a0, _mm512_permutexvar_epi8(ce1, lut), ones_i8);
    a1 = _mm512_dpbusd_epi32(a1, _mm512_permutexvar_epi8(co1, lut), ones_i8);
    a2 = _mm512_dpbusd_epi32(a2, _mm512_permutexvar_epi8(ce2, lut), ones_i8);
    a3 = _mm512_dpbusd_epi32(a3, _mm512_permutexvar_epi8(co2, lut), ones_i8);
  }

  // Matching fused3's decode formula: dist = fmadd(cvtps(a), coef, bias).
  const __m512 v_inf = _mm512_set1_ps(std::numeric_limits<float>::infinity());
  const float cs_mul = Metric ? -2.0f : -1.0f;
  const __m512 v_coef = _mm512_set1_ps(cs * cs_mul);
  const __m512 nrm0 = _mm512_loadu_ps(norms_p + 0);
  const __m512 nrm1 = _mm512_loadu_ps(norms_p + 16);
  const __m512 nrm2 = _mm512_loadu_ps(norms_p + 32);
  const __m512 nrm3 = _mm512_loadu_ps(norms_p + 48);
  const __m512 c0 = _mm512_mul_ps(nrm0, v_coef);
  const __m512 c1 = _mm512_mul_ps(nrm1, v_coef);
  const __m512 c2 = _mm512_mul_ps(nrm2, v_coef);
  const __m512 c3 = _mm512_mul_ps(nrm3, v_coef);

  __m512 b0, b1, b2, b3;
  if constexpr (Metric) {
    const __m512 v_qsq = _mm512_set1_ps(qsq);
    b0 = _mm512_add_ps(_mm512_loadu_ps(sqns_p + 0), v_qsq);
    b1 = _mm512_add_ps(_mm512_loadu_ps(sqns_p + 16), v_qsq);
    b2 = _mm512_add_ps(_mm512_loadu_ps(sqns_p + 32), v_qsq);
    b3 = _mm512_add_ps(_mm512_loadu_ps(sqns_p + 48), v_qsq);
  } else {
    b0 = _mm512_setzero_ps();
    b1 = _mm512_setzero_ps();
    b2 = _mm512_setzero_ps();
    b3 = _mm512_setzero_ps();
  }

  __m512 dist0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(a0), c0, b0);
  __m512 dist1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(a1), c1, b1);
  __m512 dist2 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(a2), c2, b2);
  __m512 dist3 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(a3), c3, b3);

  dist0 = _mm512_mask_blend_ps(masks_acc[0], v_inf, dist0);
  dist1 = _mm512_mask_blend_ps(masks_acc[1], v_inf, dist1);
  dist2 = _mm512_mask_blend_ps(masks_acc[2], v_inf, dist2);
  dist3 = _mm512_mask_blend_ps(masks_acc[3], v_inf, dist3);
  __m512 m01 = _mm512_min_ps(dist0, dist1);
  __m512 m23 = _mm512_min_ps(dist2, dist3);
  running_min = _mm512_min_ps(running_min, _mm512_min_ps(m01, m23));
}
#endif  // AVX512F && VBMI && VNNI

}  // namespace internal

// =========================================================================
// Encoder: shared core used by single-vector and multi-vector Models.
// =========================================================================
class Encoder {
 public:
  size_t dim = 0;
  size_t padded_dim = 0;
  size_t block_size = 8;
  uint32_t num_blocks = 0;
  size_t num_bytes_per_datapoint = 0;

  std::unique_ptr<rabitqlib::Rotator<float>> rotator;
  const internal::Tables* tables = nullptr;  // non-owning; points into static cache

  Encoder() = default;
  Encoder(Encoder&&) = default;
  Encoder& operator=(Encoder&&) = default;
  Encoder(const Encoder&) = delete;
  Encoder& operator=(const Encoder&) = delete;

  bool is_trained() const noexcept { return rotator != nullptr && num_blocks > 0 && tables; }

  void train(size_t input_dim, size_t bs = 8) {
    dim = input_dim;
    block_size = bs;
    padded_dim = dim;
    num_blocks = 0;
    num_bytes_per_datapoint = 0;
    rotator.reset();
    tables = internal::get_tables(block_size);
    if (dim == 0 || tables == nullptr) return;

    rotator.reset(rabitqlib::choose_rotator<float>(dim, rabitqlib::RotatorType::FhtKacRotator));
    if (!rotator) return;
    padded_dim = rotator->size();
    if (padded_dim % block_size != 0) {
      rotator.reset();
      padded_dim = dim;
      return;
    }
    num_blocks = static_cast<uint32_t>(padded_dim / block_size);
    num_bytes_per_datapoint = static_cast<size_t>(num_blocks + 1) / 2;
  }

  // Encodes one point: rotate → normalize → per-block nearest centroid.
  // ws must have size >= padded_dim. Returns {sqr_norm, norm_scaling_factor}.
  std::pair<float, float> encode_single(const float* p, uint8_t* output_codes,
                                        std::vector<float>& ws) const {
    rotator->rotate(p, ws.data());
    float sqr_norm = 0.0f;
    for (size_t i = 0; i < padded_dim; ++i) sqr_norm += ws[i] * ws[i];
    if (sqr_norm == 0.0f || !std::isfinite(sqr_norm)) {
      std::memset(output_codes, 0, num_bytes_per_datapoint);
      return {0.0f, 0.0f};
    }
    const float norm = std::sqrt(sqr_norm);
    const float inv_norm = 1.0f / norm;
    const float hadamard_scale = std::sqrt(static_cast<float>(padded_dim));
    const float scale = inv_norm * hadamard_scale;
    for (size_t i = 0; i < padded_dim; ++i) ws[i] *= scale;

    std::memset(output_codes, 0, num_bytes_per_datapoint);
    float quant_sq_norm = 0.0f;
    for (uint32_t b = 0; b < num_blocks; ++b) {
      const float* blk = ws.data() + static_cast<size_t>(b) * block_size;
      // d2(blk, c_k) = ||c_k||^2 - 2<blk, c_k> + const; pick smallest.
      int best = 0;
      float best_val = std::numeric_limits<float>::max();
      for (int k = 0; k < 16; ++k) {
        float dot = 0.0f;
        for (size_t d = 0; d < block_size; ++d) dot += blk[d] * tables->codebook[k][d];
        const float v = tables->codebook_sq_norms[k] - 2.0f * dot;
        if (v < best_val) {
          best_val = v;
          best = k;
        }
      }
      quant_sq_norm += tables->codebook_sq_norms[best];
      const uint8_t code = static_cast<uint8_t>(best);
      if (b % 2 == 0)
        output_codes[b / 2] = code;
      else
        output_codes[b / 2] |= static_cast<uint8_t>(code << 4);
    }
    const float nsf = norm / std::sqrt(std::max(1e-20f, quant_sq_norm));
    return {sqr_norm, nsf};
  }

  void save(std::ofstream& out) const {
    out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    out.write(reinterpret_cast<const char*>(&padded_dim), sizeof(padded_dim));
    out.write(reinterpret_cast<const char*>(&block_size), sizeof(block_size));
    if (rotator) rotator->save(out);
  }
  void load(std::ifstream& in) {
    in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    in.read(reinterpret_cast<char*>(&padded_dim), sizeof(padded_dim));
    in.read(reinterpret_cast<char*>(&block_size), sizeof(block_size));
    tables = internal::get_tables(block_size);
    if (dim == 0 || tables == nullptr) return;
    rotator.reset(rabitqlib::choose_rotator<float>(dim, rabitqlib::RotatorType::FhtKacRotator));
    if (rotator) rotator->load(in);
    num_blocks = static_cast<uint32_t>(padded_dim / block_size);
    num_bytes_per_datapoint = static_cast<size_t>(num_blocks + 1) / 2;
  }
};

// =========================================================================
// Forward decls
// =========================================================================
template<bool Metric>
class Quantized_Point;

// =========================================================================
// Single-Vector Query (symmetric: query is itself a PQTQ-encoded point)
// =========================================================================
template<bool Metric>
class Quantized_Query {
 public:
  using distanceType = float;

  std::vector<uint8_t> codes;  // num_bytes_per_datapoint bytes
  uint32_t num_blocks = 0;
  size_t num_bytes_per_datapoint = 0;
  float norm_scaling_factor = 0.0f;
  float unquantized_squared_norm = 0.0f;
  const internal::Tables* tables = nullptr;

  Quantized_Query() = default;

  inline float distance(const Quantized_Point<Metric>& p) const;

  template<typename PointRangeTy>
  void distances_all(const PointRangeTy& db, float* out) const;
};

// =========================================================================
// Single-Vector Point Handle (for scalar single-pair distance in beam search)
// =========================================================================
template<bool Metric>
class Quantized_Point {
 public:
  const uint8_t* code_ptr = nullptr;
  uint32_t num_blocks = 0;
  size_t num_bytes = 0;
  float norm_scaling_factor = 0.0f;
  float unquantized_squared_norm = 0.0f;
  const internal::Tables* tables = nullptr;

  Quantized_Point() = default;
  Quantized_Point(const uint8_t* ptr, uint32_t nb, size_t nbytes, float nsf, float usn,
                  const internal::Tables* t)
      : code_ptr(ptr),
        num_blocks(nb),
        num_bytes(nbytes),
        norm_scaling_factor(nsf),
        unquantized_squared_norm(usn),
        tables(t) {}

  inline float distance(const Quantized_Query<Metric>& qq) const {
    if (!code_ptr || !qq.tables) return 0.0f;
    const int8_t* sym = tables->global_sym_lut_int8;
    int32_t acc = 0;
    for (uint32_t b = 0; b < num_blocks; ++b) {
      const uint8_t qi = internal::get_packed_nibble(qq.codes.data(), b);
      const uint8_t di = internal::get_packed_nibble(code_ptr, b);
      acc += static_cast<int32_t>(sym[qi * 16 + di]);
    }
    const float cs = qq.norm_scaling_factor * tables->global_sym_lut_scale;
    const float dot = norm_scaling_factor * static_cast<float>(acc) * cs;
    if constexpr (Metric) {
      return unquantized_squared_norm + qq.unquantized_squared_norm - 2.0f * dot;
    } else {
      return -dot;
    }
  }

  void prefetch() const {
    if (code_ptr) __builtin_prefetch(code_ptr, 0, 3);
  }

  bool same_as(const Quantized_Point<Metric>&) const { return false; }
  bool same_as(const Quantized_Query<Metric>&) const { return false; }
  bool is_metric() const { return Metric; }
};

// =========================================================================
// Single-Vector Set
// =========================================================================
template<typename PointRangeTy, bool Metric>
class Quantized_Point_Range {
 public:
  static constexpr bool is_fastscan = false;

  size_t n_points = 0;
  uint32_t num_blocks = 0;
  size_t num_bytes_per_datapoint = 0;
  size_t num_strips = 0;
  size_t strip_stride = 0;
  const internal::Tables* tables = nullptr;

  // Flat per-point codes (for beam-search random access via operator[]).
  parlay::sequence<uint8_t> packed_flat;
  // 4-block interleaved strip-packed codes (for strip scan).
  parlay::sequence<uint8_t> packed_strips;
  parlay::sequence<float> norm_scaling_factors;
  parlay::sequence<float> unquantized_squared_norms;  // used only when Metric=true

  Quantized_Point_Range() = default;

  inline size_t num_bytes_per_point() const noexcept {
    return num_bytes_per_datapoint + sizeof(float) + (Metric ? sizeof(float) : 0);
  }

  inline Quantized_Point<Metric> operator[](size_t i) const {
    const uint8_t* ptr = packed_flat.data() + i * num_bytes_per_datapoint;
    float sqn = Metric ? unquantized_squared_norms[i] : 0.0f;
    return Quantized_Point<Metric>(ptr, num_blocks, num_bytes_per_datapoint,
                                   norm_scaling_factors[i], sqn, tables);
  }

  inline uint32_t size() const noexcept { return static_cast<uint32_t>(n_points); }

  void save(std::ofstream& out) const {
    out.write(reinterpret_cast<const char*>(&n_points), sizeof(n_points));
    out.write(reinterpret_cast<const char*>(&num_blocks), sizeof(num_blocks));
    out.write(reinterpret_cast<const char*>(&num_bytes_per_datapoint),
              sizeof(num_bytes_per_datapoint));
    out.write(reinterpret_cast<const char*>(&num_strips), sizeof(num_strips));
    out.write(reinterpret_cast<const char*>(&strip_stride), sizeof(strip_stride));
    auto w = [&](const auto& seq) {
      using T = typename std::remove_reference_t<decltype(seq)>::value_type;
      size_t sz = seq.size();
      out.write(reinterpret_cast<const char*>(&sz), sizeof(sz));
      if (sz) out.write(reinterpret_cast<const char*>(seq.data()), sz * sizeof(T));
    };
    w(packed_flat);
    w(packed_strips);
    w(norm_scaling_factors);
    w(unquantized_squared_norms);
  }

  void load(std::ifstream& in) {
    in.read(reinterpret_cast<char*>(&n_points), sizeof(n_points));
    in.read(reinterpret_cast<char*>(&num_blocks), sizeof(num_blocks));
    in.read(reinterpret_cast<char*>(&num_bytes_per_datapoint), sizeof(num_bytes_per_datapoint));
    in.read(reinterpret_cast<char*>(&num_strips), sizeof(num_strips));
    in.read(reinterpret_cast<char*>(&strip_stride), sizeof(strip_stride));
    auto r = [&](auto& seq) {
      using T = typename std::remove_reference_t<decltype(seq)>::value_type;
      size_t sz = 0;
      in.read(reinterpret_cast<char*>(&sz), sizeof(sz));
      seq.resize(sz);
      if (sz) in.read(reinterpret_cast<char*>(seq.data()), sz * sizeof(T));
    };
    r(packed_flat);
    r(packed_strips);
    r(norm_scaling_factors);
    r(unquantized_squared_norms);
    // tables is populated by Model::load_into() / the enclosing Model
  }
};

// distance(Point, Query) delegates back through the point.
template<bool Metric>
inline float Quantized_Query<Metric>::distance(const Quantized_Point<Metric>& p) const {
  return p.distance(*this);
}

// Scan all points: loop strips, run vnni_scan_strip per strip, decode to float.
template<bool Metric>
template<typename PointRangeTy>
void Quantized_Query<Metric>::distances_all(const PointRangeTy& db, float* out) const {
  const size_t N = db.size();
  if (!tables || num_blocks == 0 || N == 0) return;
  const size_t ns = db.num_strips;
  const size_t ss = db.strip_stride;
  const uint32_t nb = num_blocks;
  const uint32_t nb4 = ((nb + 3) / 4) * 4;
  const float cs = norm_scaling_factor * tables->global_sym_lut_scale;
  const float bias_f = static_cast<float>(nb) * 128.0f;
  const float q_sqn = unquantized_squared_norm;

#if defined(__AVX512F__) && defined(__AVX512VBMI__) && defined(__AVX512VNNI__)
  const __m512i low_mask = _mm512_set1_epi8(0x0F);
  const __m512i ones_i8 = _mm512_set1_epi8(1);
  const __m512i offset_mask = _mm512_set1_epi32(0x30201000);
#endif

  parlay::parallel_for(
      0, ns,
      [&](size_t s) {
        alignas(64) float tmp[64];
        const size_t base = s * 64;
        const size_t count = std::min<size_t>(64, N - base);
        const uint8_t* strip_codes = db.packed_strips.data() + s * ss;
        const float* norms_64 = db.norm_scaling_factors.data() + s * 64;
        const float* sqn_64 =
            Metric ? db.unquantized_squared_norms.data() + s * 64 : nullptr;
#if defined(__AVX512F__) && defined(__AVX512VBMI__) && defined(__AVX512VNNI__)
        __m512i a0, a1, a2, a3;
        internal::vnni_scan_strip(strip_codes, nb4, codes.data(), tables->half_lut, low_mask,
                                  ones_i8, offset_mask, a0, a1, a2, a3);
        internal::vnni_decode_strip<Metric>(a0, a1, a2, a3, bias_f, cs, q_sqn, norms_64, sqn_64,
                                            tmp);
#else
        internal::scan_64_scalar<Metric>(
            tables->global_sym_lut_int8, tables->global_sym_lut_scale, norm_scaling_factor,
            q_sqn, nb, codes.data(),
            db.packed_flat.data() + base * db.num_bytes_per_datapoint,
            db.num_bytes_per_datapoint, norms_64, sqn_64, count, tmp);
#endif
        std::memcpy(out + base, tmp, count * sizeof(float));
      },
      /*granularity=*/64);
}

// =========================================================================
// Single-Vector Model
// =========================================================================
template<bool Metric>
class Model {
 public:
  Encoder encoder;
  size_t block_size = 8;

  Model() = default;
  Model(Model&&) = default;
  Model& operator=(Model&&) = default;
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  template<typename PointRangeTy>
  void train(const PointRangeTy& data, size_t bs = 8) {
    block_size = bs;
    encoder.train(data.get_dims(), bs);
  }

  // Convenience overload used by Index::train_quantizer which passes no block size.
  template<typename PointRangeTy>
  void train(const PointRangeTy& data) {
    train(data, block_size);
  }

  template<typename PointRangeTy>
  Quantized_Point_Range<PointRangeTy, Metric> encode(const PointRangeTy& data) const {
    Quantized_Point_Range<PointRangeTy, Metric> enc;
    enc.n_points = data.size();
    enc.num_blocks = encoder.num_blocks;
    enc.num_bytes_per_datapoint = encoder.num_bytes_per_datapoint;
    enc.tables = encoder.tables;
    enc.num_strips = internal::num_strips_for(enc.n_points);
    enc.strip_stride = internal::strip_stride_bytes(enc.num_blocks);
    const size_t n_padded = enc.num_strips * 64;

    enc.packed_flat.resize(enc.n_points * enc.num_bytes_per_datapoint, 0);
    enc.packed_strips.resize(enc.num_strips * enc.strip_stride, 0);
    enc.norm_scaling_factors.resize(n_padded, 0.0f);
    if constexpr (Metric) {
      enc.unquantized_squared_norms.resize(n_padded, 0.0f);
    }

    parlay::parallel_for(0, enc.n_points, [&](size_t i) {
      std::vector<float> ws(encoder.padded_dim);
      const float* p = reinterpret_cast<const float*>(data.location(i));
      uint8_t* out_ptr = enc.packed_flat.data() + i * enc.num_bytes_per_datapoint;
      auto [sqn, nsf] = encoder.encode_single(p, out_ptr, ws);
      enc.norm_scaling_factors[i] = nsf;
      if constexpr (Metric) {
        enc.unquantized_squared_norms[i] = sqn;
      }
    });

    internal::pack_into_strips_interleaved4(enc.packed_flat.data(), enc.n_points, enc.num_blocks,
                                            enc.packed_strips.data());
    return enc;
  }

  template<typename PointTy>
  Quantized_Query<Metric> quantize_query(const PointTy& query) const {
    std::vector<float> tmp(encoder.dim);
    for (size_t i = 0; i < encoder.dim; ++i) tmp[i] = query[i];
    return quantize_query_from_ptr(tmp.data());
  }

  Quantized_Query<Metric> quantize_query(const float* qptr) const {
    return quantize_query_from_ptr(qptr);
  }

  void save(std::ofstream& out) const {
    encoder.save(out);
    out.write(reinterpret_cast<const char*>(&block_size), sizeof(block_size));
  }
  void load(std::ifstream& in) {
    encoder.load(in);
    in.read(reinterpret_cast<char*>(&block_size), sizeof(block_size));
  }

 private:
  Quantized_Query<Metric> quantize_query_from_ptr(const float* qptr) const {
    Quantized_Query<Metric> qq;
    if (!encoder.is_trained()) return qq;
    qq.num_blocks = encoder.num_blocks;
    qq.num_bytes_per_datapoint = encoder.num_bytes_per_datapoint;
    qq.tables = encoder.tables;
    qq.codes.assign(encoder.num_bytes_per_datapoint, 0);

    std::vector<float> ws(encoder.padded_dim);
    auto [sqn, nsf] = encoder.encode_single(qptr, qq.codes.data(), ws);
    qq.norm_scaling_factor = nsf;
    qq.unquantized_squared_norm = sqn;
    return qq;
  }
};

}  // namespace pqtq
}  // namespace mvsic
