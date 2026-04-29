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
//     // across tiles using VPOPCNTD (per-32-bit-lane popcount), so each
//     // i32 lane of acc[q] is the per-point Hamming count directly.
//     hamming[p] = sum_t popcount(panel[t][p] XOR q_broadcast[t])  for p in panel
//     // ASSUMES INPUT VECTORS ARE UNIT-NORM. Under that assumption the
//     // per-vector norm-scaling factor is the constant 1/(127*sqrt(padded_dim))
//     // and ||q||^2 == ||d||^2 == 1, so the distance reduces to a single
//     // affine function of the integer Hamming count:
//     //   metric (Euclidean):   dist = (4/padded_dim) * hamming
//     //   non-metric (-IP):     dist = (2/padded_dim) * hamming - 1
//     // Because that affine map is monotone in `hamming`, we can keep the
//     // running min in INTEGER space across all panels and convert to
//     // float exactly once at the end of the cloud:
//     //   for p in panels: min_h = _mm512_min_epi32(min_h, hamming[p])
//     //   dist = scale * reduce_min(min_h) + (Metric ? 0 : -1)
//     // Padding lanes (only ever in the LAST panel of a cloud whose size
//     // isn't a multiple of kPanelPoints) are excluded from the running
//     // min via a per-lane mask derived from cloud_size — there is no
//     // longer any per-padded-point sentinel array.
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

// AVX-512 lane counts for the panel layout.
static constexpr size_t kPanelPoints = 16;     // points per panel (matches int32 lanes)
static constexpr size_t kPanelLaneBytes = 4;   // bytes per point per tile
static constexpr size_t kTileBytes = kPanelPoints * kPanelLaneBytes;  // 64 bytes per tile

// uint8 sums of popcount-bytes can hold at most 255. Each popcount byte is in
// [0, 8], so chunking 31 tiles at a time keeps us safely under 256 (31*8=248).
static constexpr size_t kMaxSafeAccumTiles = 31;

// Number of queries to batch together in the inner loop. 4 keeps register
// pressure manageable (4 panels * 4 queries * 1 acc each = 16 zmm) and matches
// the SPR sweet spot used elsewhere. Tried Mq=5 and Mq=6: both regressed
// (~157 vs 161 QPS) — see 1BTQ-Optimization-Ideas.md "Idea C" for analysis.
static constexpr size_t kMq1bit = 4;

// Maximum supported top-k bucket capacity. 64 covers common num_rerank
// values (up to 64) without heap allocation. Bounded by a constant so the
// bucket can live in stack-allocated SoA arrays.
static constexpr size_t kMaxBucketK = 64;

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

// Per-32-bit-lane popcount (VPOPCNTD, requires VPOPCNTDQ). Each lane of `x`
// is treated as a u32; output lane = popcount of that u32. This is the same
// throughput as popcnt_epi8 (1/cycle, port 5 on SPR), but produces per-point
// counts directly — no need for the sum_quad_accum fold afterwards. This
// removes 16 vpdpbusd ops per 4-panel × Mq=4 outer iteration (~13% port-5
// reduction; the rest of the inner loop is unchanged).
inline __m512i popcnt_d32(__m512i x) {
#if defined(__AVX512VPOPCNTDQ__)
  return _mm512_popcnt_epi32(x);
#else
  // Fallback: u8 popcount + horizontal sum within each 32-bit lane.
  const __m512i pc = popcnt_u8(x);
  const __m512i ones16 = _mm512_set1_epi16(1);
  const __m512i ones8 = _mm512_set1_epi8(1);
  __m512i prod16 = _mm512_maddubs_epi16(pc, ones8);
  return _mm512_madd_epi16(prod16, ones16);
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
// Uses VPOPCNTD (per-32-bit-lane popcount): each lane is treated as a u32
// and produces the per-point Hamming distance directly (no u8→i32 fold
// needed). Same throughput as u8 popcount, but eliminates the final
// sum_quad_accum stage entirely. Also removes the kMaxSafeAccumTiles
// chunked fallback because i32 accumulators don't overflow at any
// realistic padded_dim.
template<size_t Mq>
inline void hamming_micro_kernel_1panel(const uint8_t* qbuf, size_t qbuf_tile_stride,
                                        const uint8_t* panel, size_t num_hamming_tiles,
                                        __m512i* acc) {
  for (size_t q = 0; q < Mq; ++q) acc[q] = _mm512_setzero_si512();

  for (size_t t = 0; t < num_hamming_tiles; ++t) {
    const __m512i pv = _mm512_loadu_si512(
        reinterpret_cast<const __m512i*>(panel + t * kTileBytes));
    for (size_t q = 0; q < Mq; ++q) {
      const __m512i qb = _mm512_loadu_si512(
          reinterpret_cast<const __m512i*>(qbuf + q * qbuf_tile_stride + t * kTileBytes));
      acc[q] = _mm512_add_epi32(acc[q], popcnt_d32(_mm512_xor_si512(pv, qb)));
    }
  }
}

// Four-panel Hamming accumulation (best ILP). Uses VPOPCNTD (per-32-bit-lane
// popcount) so the inner loop accumulates directly into the i32 acc; no u8
// staging and no terminal sum_quad_accum (saves 16 vpdpbusd ops per outer
// iteration vs the previous u8-staged kernel). Live state with Mq=4: 16 i32
// accs + 4 panel loads + 1 query load + popcount intermediate ≈ 22 zmm.
template<size_t Mq>
inline void hamming_micro_kernel_4panel(const uint8_t* qbuf, size_t qbuf_tile_stride,
                                        const uint8_t* panel0, const uint8_t* panel1,
                                        const uint8_t* panel2, const uint8_t* panel3,
                                        size_t num_hamming_tiles, __m512i* acc0,
                                        __m512i* acc1, __m512i* acc2, __m512i* acc3) {
  for (size_t q = 0; q < Mq; ++q) {
    acc0[q] = _mm512_setzero_si512();
    acc1[q] = _mm512_setzero_si512();
    acc2[q] = _mm512_setzero_si512();
    acc3[q] = _mm512_setzero_si512();
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
      acc0[q] = _mm512_add_epi32(acc0[q], popcnt_d32(_mm512_xor_si512(pa, qb)));
      acc1[q] = _mm512_add_epi32(acc1[q], popcnt_d32(_mm512_xor_si512(pb, qb)));
      acc2[q] = _mm512_add_epi32(acc2[q], popcnt_d32(_mm512_xor_si512(pc, qb)));
      acc3[q] = _mm512_add_epi32(acc3[q], popcnt_d32(_mm512_xor_si512(pd, qb)));
    }
  }
}

// Convert a per-query int32 cross-panel min hamming into a float distance
// and accumulate into the chamfer running total.
//
// Assuming all input vectors are unit-norm (which holds for every dataset
// in this project), the per-vector norm-scaling factor and squared norm
// are constants and distance reduces to a single affine map of the
// integer Hamming count:
//   dist = scale * hamming + (Metric ? 0 : -1)
// where scale = (Metric ? 4 : 2) / padded_dim. Because the map is
// monotone in `hamming`, we can keep the per-cloud running min in INTEGER
// space and convert exactly once. Padding lanes that were never updated
// (still INT32_MAX) are larger than any real hamming and so are ignored
// by `_mm512_reduce_min_epi32`.
template<bool Metric>
inline float reduce_to_dist(__m512i min_h, float scale_f) {
  const int32_t hmin = _mm512_reduce_min_epi32(min_h);
  constexpr float kValidAddend = Metric ? 0.0f : -1.0f;
  return scale_f * static_cast<float>(hmin) + kValidAddend;
}

// Bulk horizontal-min of 4 __m512i's → sum of the 4 scalar mins (int32).
//
// Instead of 4 independent `_mm512_reduce_min_epi32` calls (each doing its
// own 4-level shuffle+min tree to collapse one 16-lane vector to a scalar),
// we:
//   (1) reduce each 512 → 128-bit chunk (4 lanes per query), which is
//       2 half-shuffles + 2 mins per query (same as step 1+2 of the stock
//       reduce), AND
//   (2) reduce those four 4-lane vectors to four scalars in a single
//       shared shuffle tree: min across lanes [0..3] after a 4x4 transpose
//       uses 8 unpacks + 3 mins for 4 queries (vs 4*2 = 8 shuffles + 4*2
//       = 8 mins for four independent step-3-and-4 tails).
//
// Net vs 4 × reduce_min_epi32: same p5 shuffle count, but 5 fewer p0/1
// mins and (more importantly) a SHORTER critical-path latency — the 8
// lane-4 reduces run in parallel across queries, then the final shuffle
// tree is shared. The tail scalar sum is then a single int add per query.
inline int32_t reduce4_min_sum_epi32(const __m512i (&min_h)[4]) {
  // Step 1: each 512-bit → 256-bit pairwise min.
  const __m256i a0 = _mm256_min_epi32(_mm512_castsi512_si256(min_h[0]),
                                       _mm512_extracti32x8_epi32(min_h[0], 1));
  const __m256i a1 = _mm256_min_epi32(_mm512_castsi512_si256(min_h[1]),
                                       _mm512_extracti32x8_epi32(min_h[1], 1));
  const __m256i a2 = _mm256_min_epi32(_mm512_castsi512_si256(min_h[2]),
                                       _mm512_extracti32x8_epi32(min_h[2], 1));
  const __m256i a3 = _mm256_min_epi32(_mm512_castsi512_si256(min_h[3]),
                                       _mm512_extracti32x8_epi32(min_h[3], 1));

  // Step 2: each 256-bit → 128-bit pairwise min (4 lanes per query).
  const __m128i b0 = _mm_min_epi32(_mm256_castsi256_si128(a0),
                                    _mm256_extracti128_si256(a0, 1));
  const __m128i b1 = _mm_min_epi32(_mm256_castsi256_si128(a1),
                                    _mm256_extracti128_si256(a1, 1));
  const __m128i b2 = _mm_min_epi32(_mm256_castsi256_si128(a2),
                                    _mm256_extracti128_si256(a2, 1));
  const __m128i b3 = _mm_min_epi32(_mm256_castsi256_si128(a3),
                                    _mm256_extracti128_si256(a3, 1));

  // Step 3: 4x4 int32 transpose via unpack{lo,hi}_epi{32,64}.
  //   b0 = [q0a, q0b, q0c, q0d], similarly for b1,b2,b3
  // After transpose, t[k] = [q0_k, q1_k, q2_k, q3_k] for k=0..3.
  const __m128i t01lo = _mm_unpacklo_epi32(b0, b1);  // [q0a, q1a, q0b, q1b]
  const __m128i t01hi = _mm_unpackhi_epi32(b0, b1);  // [q0c, q1c, q0d, q1d]
  const __m128i t23lo = _mm_unpacklo_epi32(b2, b3);  // [q2a, q3a, q2b, q3b]
  const __m128i t23hi = _mm_unpackhi_epi32(b2, b3);  // [q2c, q3c, q2d, q3d]

  const __m128i r_a = _mm_unpacklo_epi64(t01lo, t23lo);  // [q0a, q1a, q2a, q3a]
  const __m128i r_b = _mm_unpackhi_epi64(t01lo, t23lo);  // [q0b, q1b, q2b, q3b]
  const __m128i r_c = _mm_unpacklo_epi64(t01hi, t23hi);  // [q0c, q1c, q2c, q3c]
  const __m128i r_d = _mm_unpackhi_epi64(t01hi, t23hi);  // [q0d, q1d, q2d, q3d]

  // Step 4: 3 element-wise mins collapse 4 sets of 4 lanes into 1 set of
  // 4 lanes holding [q0_min, q1_min, q2_min, q3_min].
  const __m128i m01 = _mm_min_epi32(r_a, r_b);
  const __m128i m23 = _mm_min_epi32(r_c, r_d);
  const __m128i h4  = _mm_min_epi32(m01, m23);

  // Step 5: horizontal sum of 4 int32 lanes.
  //   [q0_min, q1_min, q2_min, q3_min] → scalar sum
  const __m128i h4_hi = _mm_shuffle_epi32(h4, _MM_SHUFFLE(1, 0, 3, 2));
  const __m128i s2 = _mm_add_epi32(h4, h4_hi);           // lanes: [q0+q2, q1+q3, ...]
  const __m128i s2_hi = _mm_shuffle_epi32(s2, _MM_SHUFFLE(2, 3, 0, 1));
  const __m128i s1 = _mm_add_epi32(s2, s2_hi);           // lane 0 holds q0+q1+q2+q3
  return _mm_cvtsi128_si32(s1);
}

// Core Chamfer kernel over PRE-PANELED 1-bit codes, taking a qbuf that has
// already been pre-broadcasted for all num_q query embeddings. Layout is the
// same as `FusedQueryBatch::flat_qbuf`: qbuf + q * qbuf_tile_stride holds the
// `num_hamming_tiles * 64`-byte broadcast form for query embedding `q`.
//
// Returns sum_{q} min_{point} dist(q, point); caller divides by num_q.
template<bool Metric>
inline float chamfer_panels_qbuf(const uint8_t* qbuf, size_t qbuf_tile_stride,
                                 size_t num_q, const uint8_t* panel_data,
                                 size_t panel_bytes, size_t num_hamming_tiles,
                                 size_t cloud_size, size_t padded_dim) {
  if (num_q == 0 || cloud_size == 0) return 0.0f;

  // Affine map from integer Hamming -> float distance under the unit-norm
  // assumption. Used at the very end of the cloud, after the running int
  // min has been collapsed across panels:
  //   Metric:    dist = (4/padded_dim) * hamming
  //   non-Metric: dist = (2/padded_dim) * hamming - 1
  const float scale_f = (Metric ? 4.0f : 2.0f) / static_cast<float>(padded_dim);

  // Panel geometry: full panels are exactly 16 valid lanes; if cloud_size
  // isn't a multiple of kPanelPoints, exactly one trailing partial panel
  // has `tail_valid` < 16 lanes and the rest are padding. We exclude the
  // padding lanes from the running min via _mm512_mask_min_epi32 on that
  // last panel only.
  const size_t full_np = cloud_size / kPanelPoints;
  const size_t tail_valid = cloud_size - full_np * kPanelPoints;
  const __mmask16 tail_mask =
      (tail_valid == 0) ? __mmask16{0}
                        : static_cast<__mmask16>((1u << tail_valid) - 1u);

  // Accumulate the per-query horizontal-min Hamming counts as INT32 across
  // all Mq batches, and defer the single `scale_f * hi + addend` affine map
  // until the very end of the cloud. The map is linear in hi, so
  //   sum_q (scale_f * hi + addend) == scale_f * sum_q(hi) + num_q * addend
  // With num_q <= 32, padded_dim <= 16384, each hi <= padded_dim and
  // sum_q(hi) <= 2^19 — well within int32. Benefits:
  //   - Removes `num_q` scalar cvt+fmadd+add off the critical path and
  //     replaces with one cvt + one fmadd at the end of the cloud.
  //   - Removes the FP dependency chain from the reduce_min_epi32 critical
  //     path: each reduce is now followed by a cheap scalar int add instead
  //     of cvt → fmadd → scalar add (~4 fewer ops on that chain).
  //   - For the common Mq=4 case we sum h0..h3 into an int running total
  //     before accumulating into h_sum, which lets the compiler schedule
  //     the 4 horizontal reduces in parallel.
  int32_t h_sum = 0;
  size_t qi = 0;
  const __m512i kIntMaxV = _mm512_set1_epi32(std::numeric_limits<int32_t>::max());

  for (; qi + kMq1bit <= num_q; qi += kMq1bit) {
    const uint8_t* qbuf_local = qbuf + qi * qbuf_tile_stride;

    // Per-query running int-min hamming across all panels of this cloud.
    __m512i min_h[kMq1bit];
    for (size_t q = 0; q < kMq1bit; ++q) min_h[q] = kIntMaxV;

    // Sweep the FULL panels (4-wide where possible, 1-wide for the tail).
    // Padding never appears here, so the min is unmasked.
    size_t p = 0;
    for (; p + 4 <= full_np; p += 4) {
      __m512i a0[kMq1bit], a1[kMq1bit], a2[kMq1bit], a3[kMq1bit];
      hamming_micro_kernel_4panel<kMq1bit>(
          qbuf_local, qbuf_tile_stride, panel_data + p * panel_bytes,
          panel_data + (p + 1) * panel_bytes, panel_data + (p + 2) * panel_bytes,
          panel_data + (p + 3) * panel_bytes, num_hamming_tiles, a0, a1, a2, a3);
      for (size_t q = 0; q < kMq1bit; ++q) {
        min_h[q] = _mm512_min_epi32(min_h[q], a0[q]);
        min_h[q] = _mm512_min_epi32(min_h[q], a1[q]);
        min_h[q] = _mm512_min_epi32(min_h[q], a2[q]);
        min_h[q] = _mm512_min_epi32(min_h[q], a3[q]);
      }
    }
    for (; p < full_np; ++p) {
      __m512i acc[kMq1bit];
      hamming_micro_kernel_1panel<kMq1bit>(qbuf_local, qbuf_tile_stride,
                                            panel_data + p * panel_bytes, num_hamming_tiles, acc);
      for (size_t q = 0; q < kMq1bit; ++q) {
        min_h[q] = _mm512_min_epi32(min_h[q], acc[q]);
      }
    }

    // Optional partial trailing panel: only `tail_valid` lanes are real.
    if (tail_valid > 0) {
      __m512i acc[kMq1bit];
      hamming_micro_kernel_1panel<kMq1bit>(qbuf_local, qbuf_tile_stride,
                                            panel_data + full_np * panel_bytes,
                                            num_hamming_tiles, acc);
      for (size_t q = 0; q < kMq1bit; ++q) {
        min_h[q] = _mm512_mask_min_epi32(min_h[q], tail_mask, min_h[q], acc[q]);
      }
    }

    // Horizontally reduce Mq independent int mins and accumulate. For the
    // common Mq=4 case use the parallel bulk reducer that shares the final
    // 4x4 transpose+min across all four queries. For other Mq values we
    // fall back to 4 independent reduces.
    if constexpr (kMq1bit == 4) {
      h_sum += reduce4_min_sum_epi32(min_h);
    } else {
      int32_t mq_sum = 0;
      for (size_t q = 0; q < kMq1bit; ++q) {
        mq_sum += _mm512_reduce_min_epi32(min_h[q]);
      }
      h_sum += mq_sum;
    }
  }

  // Tail (< kMq1bit queries). Same int-min structure, Mq=1.
  for (; qi < num_q; ++qi) {
    const uint8_t* qbuf_local = qbuf + qi * qbuf_tile_stride;
    __m512i min_h_one = kIntMaxV;
    for (size_t p = 0; p < full_np; ++p) {
      __m512i acc;
      hamming_micro_kernel_1panel<1>(qbuf_local, qbuf_tile_stride,
                                      panel_data + p * panel_bytes,
                                      num_hamming_tiles, &acc);
      min_h_one = _mm512_min_epi32(min_h_one, acc);
    }
    if (tail_valid > 0) {
      __m512i acc;
      hamming_micro_kernel_1panel<1>(qbuf_local, qbuf_tile_stride,
                                      panel_data + full_np * panel_bytes,
                                      num_hamming_tiles, &acc);
      min_h_one = _mm512_mask_min_epi32(min_h_one, tail_mask, min_h_one, acc);
    }
    h_sum += _mm512_reduce_min_epi32(min_h_one);
  }

  // Single affine map at the very end of the cloud:
  //   total_chamfer = scale_f * h_sum + num_q * kValidAddend
  constexpr float kValidAddend = Metric ? 0.0f : -1.0f;
  return scale_f * static_cast<float>(h_sum) +
         static_cast<float>(num_q) * kValidAddend;
}

// Backwards-compatible wrapper: builds the qbuf once for all num_q queries
// then delegates to `chamfer_panels_qbuf`. Used by the (rare) one-off
// `Quantized_Query_Point_Cloud::distance(cloud)` path; the hot
// `distances_all` path hoists the qbuf build to leaf scope and calls
// `chamfer_panels_qbuf` directly.
template<bool Metric>
inline float chamfer_panels(const uint8_t* q_packed_data, size_t q_byte_stride, size_t num_q,
                            size_t num_bytes, const uint8_t* panel_data,
                            size_t panel_bytes, size_t num_hamming_tiles,
                            size_t cloud_size, size_t padded_dim) {
  if (num_q == 0 || cloud_size == 0) return 0.0f;
  const size_t qbuf_tile_stride = num_hamming_tiles * kTileBytes;
  std::vector<uint8_t> qbuf_storage(num_q * qbuf_tile_stride);
  for (size_t q = 0; q < num_q; ++q) {
    pre_broadcast_query(q_packed_data + q * q_byte_stride, num_bytes,
                        num_hamming_tiles, qbuf_storage.data() + q * qbuf_tile_stride);
  }
  return chamfer_panels_qbuf<Metric>(qbuf_storage.data(), qbuf_tile_stride, num_q,
                                     panel_data, panel_bytes, num_hamming_tiles,
                                     cloud_size, padded_dim);
}

#endif  // __AVX512F__

// =========================================================================
// Scalar fallback (correct, slow). Used when AVX-512 is unavailable.
// Same unit-norm assumption + int-min-throughout structure as the AVX-512
// path: per query, find the min hamming over all REAL points in the cloud
// (skipping padding lanes naturally because we only iterate i < cloud_size),
// then convert that min once via the affine map.
// =========================================================================
template<bool Metric>
inline float chamfer_panels_scalar(const uint8_t* q_packed_data, size_t q_byte_stride,
                                   size_t num_q, size_t num_bytes, const uint8_t* panel_data,
                                   size_t panel_bytes, size_t num_hamming_tiles,
                                   size_t cloud_size, size_t padded_dim) {
  const float scale = (Metric ? 4.0f : 2.0f) / static_cast<float>(padded_dim);
  constexpr float kValidAddend = Metric ? 0.0f : -1.0f;
  float total = 0.0f;
  for (size_t qi = 0; qi < num_q; ++qi) {
    const uint8_t* qp = q_packed_data + qi * q_byte_stride;
    int min_h = std::numeric_limits<int>::max();
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
      if (hamming < min_h) min_h = hamming;
    }
    total += scale * static_cast<float>(min_h) + kValidAddend;
  }
  return total;
}

#ifdef __AVX512F__
// Vectorized sum-reduction over `n` consecutive floats. Used to compute the
// chamfer distance (sum of per-embedding mins) once per (query, db-cloud)
// pair inside the M2M top-k loop. Replaces a scalar `for` loop that the
// compiler was not auto-vectorizing (non-restrict pointer + runtime n).
inline float reduce_sum_ps_n(const float* __restrict p, size_t n) {
  __m512 acc = _mm512_setzero_ps();
  size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    acc = _mm512_add_ps(acc, _mm512_loadu_ps(p + i));
  }
  if (i < n) {
    const __mmask16 tail_mask = static_cast<__mmask16>((1u << (n - i)) - 1u);
    acc = _mm512_mask_add_ps(acc, tail_mask, acc,
                             _mm512_maskz_loadu_ps(tail_mask, p + i));
  }
  return _mm512_reduce_add_ps(acc);
}
#endif

// Bounded-capacity top-k bucket stored as parallel `float dists[]` and
// `uint32_t ids[]` arrays. Replaces `std::priority_queue<std::pair<float,
// uint32_t>>` in the 1BTQ (and PQTQ) M2M top-k loop: no heap allocation,
// inlined hot-path, vectorized max-refresh, and allows the compiler to keep
// the current-max in a register across consecutive db-cloud iterations.
//
// Typical usage (k <= kMaxBucketK, bucket reused across db-clouds):
//   BoundedTopKBucket bucket; bucket.init(k);
//   for each db cloud c:
//     bucket.try_insert(chamfer_dist, id);
//   bucket.finalize_sorted(out);  // writes k entries, padded with sentinel
struct BoundedTopKBucket {
  alignas(64) float dists[kMaxBucketK];
  alignas(64) uint32_t ids[kMaxBucketK];
  size_t k = 0;
  size_t size = 0;
  float current_max = std::numeric_limits<float>::infinity();
  size_t max_lane = 0;

  void init(size_t k_in) {
    k = k_in;
    size = 0;
    current_max = std::numeric_limits<float>::infinity();
    max_lane = 0;
    // Initialize all lanes to sentinel values so padding never wins the
    // max-refresh scan and finalize_sorted produces correct output when the
    // bucket is under-filled (size < k).
    for (size_t i = 0; i < kMaxBucketK; ++i) {
      dists[i] = std::numeric_limits<float>::infinity();
      ids[i] = UINT32_MAX;
    }
  }

  // Refresh current_max and max_lane via a vectorized max-reduce over the
  // `k` real entries (padding lanes stay at +inf, which is fine because
  // we're looking for the LARGEST stored distance and +inf always wins —
  // so we restrict the vectorized reduce to the first k lanes and then
  // find the argmax with a scalar scan (k <= 64, cheap).
  void refresh_max() {
#ifdef __AVX512F__
    __m512 vmax = _mm512_loadu_ps(dists);
    for (size_t off = 16; off < k; off += 16) {
      vmax = _mm512_max_ps(vmax, _mm512_loadu_ps(dists + off));
    }
    // Mask lanes beyond k in the LAST block to -inf so they don't pollute
    // reduce_max. (Earlier blocks are fully within [0,k) because k > off
    // when we entered the loop.)
    const size_t last_block_start = (k >= 16) ? ((k - 1) & ~size_t{15}) : 0;
    const size_t real_in_last = k - last_block_start;
    if (real_in_last < 16) {
      const __mmask16 keep = static_cast<__mmask16>((1u << real_in_last) - 1u);
      __m512 last = _mm512_loadu_ps(dists + last_block_start);
      const __m512 neg_inf = _mm512_set1_ps(-std::numeric_limits<float>::infinity());
      last = _mm512_mask_blend_ps(keep, neg_inf, last);
      // Recompute vmax from scratch using the masked last block to avoid
      // double-counting it.
      if (last_block_start == 0) {
        vmax = last;
      } else {
        vmax = _mm512_loadu_ps(dists);
        for (size_t off = 16; off < last_block_start; off += 16) {
          vmax = _mm512_max_ps(vmax, _mm512_loadu_ps(dists + off));
        }
        vmax = _mm512_max_ps(vmax, last);
      }
    }
    const float mx = _mm512_reduce_max_ps(vmax);
#else
    float mx = dists[0];
    for (size_t i = 1; i < k; ++i) if (dists[i] > mx) mx = dists[i];
#endif
    current_max = mx;
    // Scalar argmax — k <= 64, so this is ~trivial compared to any heap op.
    for (size_t i = 0; i < k; ++i) {
      if (dists[i] == mx) { max_lane = i; break; }
    }
  }

  // Fast-path insert. Common case (d >= current_max after bucket is full)
  // returns on a single compare.
  inline void try_insert(float d, uint32_t id) {
    if (size < k) {
      dists[size] = d;
      ids[size] = id;
      ++size;
      if (size == k) refresh_max();
      return;
    }
    if (d >= current_max) return;
    dists[max_lane] = d;
    ids[max_lane] = id;
    refresh_max();
  }

  // Drain into `out` in ascending-distance order. Unfilled slots become
  // {UINT32_MAX, FLT_MAX} sentinel so downstream (mvivf.h candidate
  // compaction) can strip them the same way it did for the heap path.
  void finalize_sorted(std::pair<uint32_t, float>* out) const {
    std::pair<float, uint32_t> tmp[kMaxBucketK];
    for (size_t i = 0; i < size; ++i) tmp[i] = {dists[i], ids[i]};
    std::sort(tmp, tmp + size,
              [](const std::pair<float, uint32_t>& a,
                 const std::pair<float, uint32_t>& b) { return a.first < b.first; });
    for (size_t i = 0; i < size; ++i) out[i] = {tmp[i].second, tmp[i].first};
    for (size_t i = size; i < k; ++i) {
      out[i] = {UINT32_MAX, std::numeric_limits<float>::max()};
    }
  }
};

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

  // Flat per-query packed bit codes: num_queries * num_bytes_per_datapoint.
  // No norm-scaling factor / squared-norm arrays: under the unit-norm
  // assumption these are constants and folded into the kernel epilogue.
  std::vector<uint8_t> flat_query_codes;

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
    // bytes/vector now: just the packed code bytes. With int-min throughout,
    // padding handling is derived from cloud_size at scoring time, so we no
    // longer touch any per-padded-point sidecar at scoring time.
    const size_t bytes_per_vec = cloud.db->num_bytes_per_datapoint;
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

  // Per-point offsets, padded so each cloud is a multiple of kPanelPoints.
  // Kept for downstream code that needs to address per-padded-point data;
  // the kernel itself only needs cloud_sizes (it derives the partial-panel
  // mask from cloud_size % kPanelPoints).
  parlay::sequence<size_t> point_offsets;  // size: n_clouds + 1, in points

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
    return panel_data.size() * sizeof(uint8_t);
  }

  void distances_all(const Quantized_Query_Point_Cloud<Metric>& q,
                     std::pair<uint32_t, float>* results) const {
    const size_t num_q = q.num_queries;
    const size_t nc = num_clouds();
    if (num_q == 0 || nc == 0) return;

#ifdef __AVX512F__
    // Hoist the per-query broadcast out of the per-cloud loop: every cloud
    // in this leaf re-broadcasts the *same* query bytes into the same qbuf
    // shape, so we build one shared qbuf per distances_all call and pass it
    // into the scoring kernel via `chamfer_panels_qbuf`. On fiqa this was
    // ~14% of t_leaves (0.32s/rep out of 2.27s) — see
    // 1BTQ-Optimization-Ideas.md "Profiling: t_leaves sub-breakdown".
    const size_t qbuf_tile_stride = num_hamming_tiles * internal::kTileBytes;
    const size_t qbuf_total = num_q * qbuf_tile_stride;

    // Inline scratch covers the common fiqa-class case (num_q=32, nht=4 →
    // 8192 bytes). Larger (higher-dim or more query embeddings) falls back
    // to a heap allocation.
    static constexpr size_t kMaxInlineQbufBytes = 8192;
    alignas(64) uint8_t qbuf_storage[kMaxInlineQbufBytes];
    std::vector<uint8_t> qbuf_heap;
    uint8_t* qbuf;
    if (qbuf_total <= kMaxInlineQbufBytes) {
      qbuf = qbuf_storage;
    } else {
      qbuf_heap.resize(qbuf_total);
      qbuf = qbuf_heap.data();
    }
    const uint8_t* q_codes = q.flat_query_codes.data();
    const size_t q_stride = num_bytes_per_datapoint;
    for (size_t qi = 0; qi < num_q; ++qi) {
      internal::pre_broadcast_query(q_codes + qi * q_stride, q_stride,
                                    num_hamming_tiles,
                                    qbuf + qi * qbuf_tile_stride);
    }

    parlay::parallel_for(0, nc, [&](size_t c) {
      if (cloud_sizes[c] == 0) {
        results[c] = {get_id(c), std::numeric_limits<float>::max()};
        return;
      }
      const uint8_t* panel_ptr = panel_data.data() + panel_offsets[c];
      const float dist_sum = internal::chamfer_panels_qbuf<Metric>(
          qbuf, qbuf_tile_stride, num_q, panel_ptr, panel_bytes,
          num_hamming_tiles, cloud_sizes[c], padded_dim);
      results[c] = {get_id(c), dist_sum / static_cast<float>(num_q)};
    });
#else
    parlay::parallel_for(0, nc, [&](size_t c) {
      if (cloud_sizes[c] == 0) {
        results[c] = {get_id(c), std::numeric_limits<float>::max()};
        return;
      }
      const float d = turboquant_1bit_mv_chamfer_distance(q, *this, c);
      results[c] = {get_id(c), d};
    });
#endif
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
  const uint8_t* panel_ptr = db.panel_data.data() + panel_byte_off;

  float dist_sum = 0.0f;
#ifdef __AVX512F__
  dist_sum = internal::chamfer_panels<Metric>(
      q.flat_query_codes.data(), q.num_bytes_per_datapoint, num_q,
      q.num_bytes_per_datapoint, panel_ptr, db.panel_bytes, db.num_hamming_tiles,
      cs, db.padded_dim);
#else
  dist_sum = internal::chamfer_panels_scalar<Metric>(
      q.flat_query_codes.data(), q.num_bytes_per_datapoint, num_q,
      q.num_bytes_per_datapoint, panel_ptr, db.panel_bytes, db.num_hamming_tiles,
      cs, db.padded_dim);
#endif
  return dist_sum / static_cast<float>(num_q);
}

// =========================================================================
// Multi-Vector Model
// =========================================================================
template<bool Metric>
class Model {
 public:
  static constexpr uint32_t kClassId = 6;  // QuantizerTag::kOneBitTQ
  static constexpr uint32_t kBatchAlignment = 4;  // kMq1bit
  static constexpr const char* kName = "1bit_tq";
  using EncodedSet = ::mvsic::turboquant_1bit_mv::Quantized_Point_Cloud_Set<Metric>;
  using EncodedQuery = ::mvsic::turboquant_1bit_mv::Quantized_Query_Point_Cloud<Metric>;
  struct Params {};

  // Reuses the TQ rotator (FhtKacRotator). We don't use BaseEncoder's 4-bit
  // encode_single — only its rotator setup.
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

  // Encode a single rotated vector into 1-bit codes (sign bits).
  //
  // Assumes the input vector is unit-norm (true for every dataset in this
  // project), so the rotated `ws` is also unit-norm and the per-vector
  // norm-scaling-factor / squared-norm are constants. We therefore drop
  // them entirely and store only the sign bits.
  inline void encode_single_bits(const float* p, std::vector<float>& ws,
                                 uint8_t* out_codes) const {
    const size_t pdim = encoder.padded_dim;
    const size_t nbytes = (pdim + 7) / 8;
    std::memset(out_codes, 0, nbytes);
    encoder.rotator->rotate(p, ws.data());

    // Sign bit only. Convention (matches one_to_many_1bit.cc / hamming_1bit.h):
    //   bit set  => coordinate is *negative*  (so XOR Hamming counts mismatches)
    //   bit clear => coordinate is non-negative
    for (size_t i = 0; i < pdim; ++i) {
      if (ws[i] < 0.0f) out_codes[i / 8] |= static_cast<uint8_t>(1u << (i % 8));
    }
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

      for (size_t i = 0; i < n_vecs; ++i) {
        const float* p =
            reinterpret_cast<const float*>(pcs.data() + (src_start + i) * encoder.dim);
        encode_single_bits(p, ws, flat_codes.data() + i * enc.num_bytes_per_datapoint);
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
            // Pad with zeros (panel was already memset). Padding lanes are
            // excluded from the running min at scoring time via the
            // last-panel mask derived from cloud_size.
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

    const float* base_ptr = query_cloud.data();
    std::vector<float> ws(encoder.padded_dim);
    for (size_t qi = 0; qi < res.num_queries; ++qi) {
      encode_single_bits(base_ptr + qi * encoder.dim, ws,
                         res.flat_query_codes.data() + qi * res.num_bytes_per_datapoint);
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
  // No per-query nsf / sqn arrays — under the unit-norm assumption these
  // are constants and folded into the kernel epilogue.
  std::vector<uint8_t> flat_qbuf;
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
  const uint8_t* panel_ptr = db.panel_data.data() + panel_byte_off;
  const size_t panel_bytes = db.panel_bytes;
  const size_t num_hamming_tiles = fq.num_hamming_tiles;
  const size_t qbuf_tile_stride = fq.qbuf_tile_stride;

#ifdef __AVX512F__
  using internal::kMq1bit;
  using internal::kPanelPoints;

  // Affine map from integer Hamming -> float distance under unit-norm
  // assumption. Applied once per (cloud, embedding) at the end.
  const float scale_f = (Metric ? 4.0f : 2.0f) / static_cast<float>(fq.padded_dim);

  // Panel geometry derived from the unpadded cloud size: see chamfer_panels.
  const size_t full_np = cs / kPanelPoints;
  const size_t tail_valid = cs - full_np * kPanelPoints;
  const __mmask16 tail_mask =
      (tail_valid == 0) ? __mmask16{0}
                        : static_cast<__mmask16>((1u << tail_valid) - 1u);
  const __m512i kIntMaxV = _mm512_set1_epi32(std::numeric_limits<int32_t>::max());

  size_t qi = 0;
  for (; qi + kMq1bit <= num_emb; qi += kMq1bit) {
    const uint8_t* qbuf = fq.flat_qbuf.data() + qi * qbuf_tile_stride;

    __m512i min_h[kMq1bit];
    for (size_t q = 0; q < kMq1bit; ++q) min_h[q] = kIntMaxV;

    size_t p = 0;
    for (; p + 4 <= full_np; p += 4) {
      __m512i a0[kMq1bit], a1[kMq1bit], a2[kMq1bit], a3[kMq1bit];
      internal::hamming_micro_kernel_4panel<kMq1bit>(
          qbuf, qbuf_tile_stride,
          panel_ptr + p * panel_bytes, panel_ptr + (p + 1) * panel_bytes,
          panel_ptr + (p + 2) * panel_bytes, panel_ptr + (p + 3) * panel_bytes,
          num_hamming_tiles, a0, a1, a2, a3);
      for (size_t q = 0; q < kMq1bit; ++q) {
        min_h[q] = _mm512_min_epi32(min_h[q], a0[q]);
        min_h[q] = _mm512_min_epi32(min_h[q], a1[q]);
        min_h[q] = _mm512_min_epi32(min_h[q], a2[q]);
        min_h[q] = _mm512_min_epi32(min_h[q], a3[q]);
      }
    }
    for (; p < full_np; ++p) {
      __m512i acc[kMq1bit];
      internal::hamming_micro_kernel_1panel<kMq1bit>(qbuf, qbuf_tile_stride,
                                                       panel_ptr + p * panel_bytes,
                                                       num_hamming_tiles, acc);
      for (size_t q = 0; q < kMq1bit; ++q) {
        min_h[q] = _mm512_min_epi32(min_h[q], acc[q]);
      }
    }
    if (tail_valid > 0) {
      __m512i acc[kMq1bit];
      internal::hamming_micro_kernel_1panel<kMq1bit>(qbuf, qbuf_tile_stride,
                                                       panel_ptr + full_np * panel_bytes,
                                                       num_hamming_tiles, acc);
      for (size_t q = 0; q < kMq1bit; ++q) {
        min_h[q] = _mm512_mask_min_epi32(min_h[q], tail_mask, min_h[q], acc[q]);
      }
    }

    for (size_t q = 0; q < kMq1bit; ++q) {
      emb_min[qi + q] = internal::reduce_to_dist<Metric>(min_h[q], scale_f);
    }
  }

  // Tail: < kMq1bit embeddings remaining.
  for (; qi < num_emb; ++qi) {
    const uint8_t* qbuf = fq.flat_qbuf.data() + qi * qbuf_tile_stride;
    __m512i min_h_one = kIntMaxV;
    for (size_t p = 0; p < full_np; ++p) {
      __m512i acc;
      internal::hamming_micro_kernel_1panel<1>(qbuf, qbuf_tile_stride,
                                                panel_ptr + p * panel_bytes,
                                                num_hamming_tiles, &acc);
      min_h_one = _mm512_min_epi32(min_h_one, acc);
    }
    if (tail_valid > 0) {
      __m512i acc;
      internal::hamming_micro_kernel_1panel<1>(qbuf, qbuf_tile_stride,
                                                panel_ptr + full_np * panel_bytes,
                                                num_hamming_tiles, &acc);
      min_h_one = _mm512_mask_min_epi32(min_h_one, tail_mask, min_h_one, acc);
    }
    emb_min[qi] = internal::reduce_to_dist<Metric>(min_h_one, scale_f);
  }
#else
  // Scalar fallback: leave +inf so callers see worst-case distances.
  (void)panel_ptr;
  (void)panel_bytes;
  (void)num_hamming_tiles;
  (void)qbuf_tile_stride;
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

    // Bounded-capacity bucket replaces std::priority_queue for the per-leaf
    // top-k. Safe because `k` is `num_rerank` from the search params and the
    // indexer caps that at `kMaxBucketK`. If a future caller exceeds it we
    // fall back to the heap path at the bottom of this function.
    if (k > internal::kMaxBucketK) {
      // Fallback: use the legacy heap path for any unusually-large k.
      auto process_query_range_heap = [&](size_t q_start, size_t q_end) {
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
                              process_query_range_heap(q_start, q_end);
                            });
      } else {
        for (size_t q_start = 0; q_start < num_q_clouds; q_start += q_block) {
          const size_t q_end = std::min(q_start + q_block, num_q_clouds);
          process_query_range_heap(q_start, q_end);
        }
      }
      return;
    }

    auto process_query_range = [&](size_t q_start, size_t q_end) {
      const size_t q_count = q_end - q_start;
      if (q_count == 0) return;

      // Stack-allocated bucket array: k <= kMaxBucketK and q_block is
      // typically tiny (4 on the fiqa call site). Heap-free hot path.
      std::vector<internal::BoundedTopKBucket> buckets(q_count);
      for (size_t i = 0; i < q_count; ++i) buckets[i].init(k);

      std::vector<const Quantized_Query_Point_Cloud<Metric>*> slice(q_count);
      for (size_t i = 0; i < q_count; ++i) slice[i] = A[q_start + i];
      FusedQueryBatch<Metric> fq;
      fq.Build(slice);

      std::vector<float> emb_min_dists(fq.total_embeddings);

      for (size_t c = 0; c < num_db_clouds; ++c) {
        const size_t cs = B.cloud_sizes[c];
        if (cs == 0) continue;

        score_one_db_cloud<Metric>(fq, B, c, emb_min_dists.data());
        const uint32_t cid = B.get_id(c);

        for (size_t i = 0; i < q_count; ++i) {
          const size_t e_start = fq.emb_offsets[i];
          const size_t e_count = fq.emb_offsets[i + 1] - e_start;
          if (e_count == 0) continue;
#ifdef __AVX512F__
          const float dist_sum =
              internal::reduce_sum_ps_n(emb_min_dists.data() + e_start, e_count);
#else
          float dist_sum = 0.0f;
          for (size_t e = 0; e < e_count; ++e) dist_sum += emb_min_dists[e_start + e];
#endif
          const float chamfer_dist = dist_sum / static_cast<float>(e_count);
          buckets[i].try_insert(chamfer_dist, cid);
        }
      }

      for (size_t i = 0; i < q_count; ++i) {
        const size_t global_idx = q_start + i;
        buckets[i].finalize_sorted(results + global_idx * k);
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
