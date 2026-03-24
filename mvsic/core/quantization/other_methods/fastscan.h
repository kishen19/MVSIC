#pragma once

// fastscan.h
//
// FastScan with VNNI and AVX2 fallback.
// - When __AVX512VNNI__ is defined: scan_64_chunk uses vpdpbusd (one-hot decode + LUT dot).
// - When __AVX512F__ is defined: all other kernels use AVX-512 (shuffle + int16 accumulate).
// - When only __AVX2__ is defined: all kernels use AVX2 (256-bit shuffle + int16 accumulate).
// - Running-min and dual-query types differ by ISA (__m512i vs struct of 2x __m256i).

#include <immintrin.h>
#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include "parlay/primitives.h"
#include "mvsic/core/utils/kmeans_util.h"

namespace mvsic {
namespace fastscan {

template<bool Metric>
class Quantized_Point;

// ---------------------------------------------------------
// FastScan Quantized Query (8-bit Fixed Point)
// ---------------------------------------------------------
template<bool Metric>
class Quantized_Query {
 public:
  using distanceType = float;

  alignas(64) std::vector<uint8_t> int_lut;  // [num_blocks * K]
  float min_dist = 0.0f;
  float scale = 1.0f;
  uint32_t num_blocks = 0;
  static constexpr uint32_t K = 16;

  Quantized_Query() = default;
  explicit Quantized_Query(uint32_t m) : num_blocks(m) {
    int_lut.resize(static_cast<size_t>(m) * K);
  }

  inline float decode(uint16_t int_dist) const {
    return (min_dist * static_cast<float>(num_blocks)) + (static_cast<float>(int_dist) * scale);
  }

  // For VNNI path: decode from int32 (after bias correction).
  inline float decode_int32(int32_t int_dist) const {
    auto clamped = static_cast<int32_t>(std::max(0, std::min(65535, int_dist)));
    return decode(static_cast<uint16_t>(clamped));
  }

  inline float distance(const Quantized_Point<Metric>& p) const {
    uint16_t acc = 0;
    for (uint32_t b = 0; b < num_blocks; ++b) {
      uint8_t packed = p.code_ptr[b * 32];
      uint8_t code = (p.lane_idx % 2 == 0) ? (packed & 0x0F) : (packed >> 4);
      acc += static_cast<uint16_t>(int_lut[static_cast<size_t>(b) * K + code]);
    }
    return decode(acc);
  }

  template<typename EncRange>
  void distances_all(const EncRange& db, float* out) const {
    const size_t N = (db.n_points_raw_unpadded != 0) ? db.n_points_raw_unpadded : db.n_points_raw;
    if (N == 0) return;

    const size_t strip_stride = static_cast<size_t>(db.num_blocks) * 32;
    const size_t n_full_strips = N / 64;
    const size_t full = n_full_strips * 64;

    parlay::parallel_for(
        0, n_full_strips,
        [&](size_t s) {
          const uint8_t* codes_ptr = db.packed_codes.data() + s * strip_stride;
          db.scan_64_chunk(*this, codes_ptr, out + s * 64);
        },
        /*granularity=*/16);

    if (full < N) {
      alignas(64) float tmp[64];
      const size_t tail_strip = n_full_strips;
      const uint8_t* codes_ptr = db.packed_codes.data() + tail_strip * strip_stride;
      db.scan_64_chunk(*this, codes_ptr, tmp);
      const size_t tail = N - full;
      std::memcpy(out + full, tmp, tail * sizeof(float));
    }
  }
};

// ---------------------------------------------------------
// FastScan Point Handle
// ---------------------------------------------------------
template<bool Metric>
class Quantized_Point {
 public:
  const uint8_t* code_ptr = nullptr;
  uint32_t lane_idx = 0;

  Quantized_Point() = default;
  Quantized_Point(const uint8_t* ptr, uint32_t lane) : code_ptr(ptr), lane_idx(lane) {}

  inline float distance(const Quantized_Query<Metric>& qq) const { return qq.distance(*this); }
  void prefetch() const { __builtin_prefetch(code_ptr, 0, 3); }

  bool same_as(const Quantized_Point<Metric>&) const { return false; }
  bool same_as(const Quantized_Query<Metric>&) const { return false; }
  bool is_metric() const { return Metric; }
};

// ---------------------------------------------------------
// Running-min vector type: 512-bit on AVX-512, 2x256-bit on AVX2
// ---------------------------------------------------------
#ifdef __AVX512F__
struct RunningMinV {
  __m512i v;
  void set_max() { v = _mm512_set1_epi16(0xFFFF); }
  static RunningMinV max() {
    RunningMinV r;
    r.v = _mm512_set1_epi16(0xFFFF);
    return r;
  }
};
#elif defined(__AVX2__)
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
#else
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
#endif

// ---------------------------------------------------------
// Quantized Point Range with SIMD kernels (AVX-512 / AVX2 / scalar)
// ---------------------------------------------------------
template<typename PointRange, bool Metric>
class Quantized_Point_Range {
 public:
  static constexpr bool is_fastscan = true;
  static constexpr uint32_t K = 16;

  // Exposed so wrapper can use RunningMinV without including this header.
  using RunningMinVType = RunningMinV;

  uint32_t num_blocks = 0;
  size_t n_points_raw = 0;
  size_t n_points_raw_unpadded = 0;
  size_t dim = 0;
  size_t dim_per_block = 0;

  parlay::sequence<uint8_t> packed_codes;
  parlay::sequence<size_t> cloud_vec_offsets;

  Quantized_Point_Range() = default;

  Quantized_Point<Metric> operator[](size_t i) const {
    const size_t strip_idx = i / 64;
    const size_t lane_idx = i % 64;
    const size_t strip_stride = static_cast<size_t>(num_blocks) * 32;
    const uint8_t* ptr = &packed_codes[strip_idx * strip_stride + (lane_idx / 2)];
    return Quantized_Point<Metric>(ptr, static_cast<uint32_t>(lane_idx));
  }

  // ---------- SIMD kernels: AVX-512 (with optional VNNI for scan_64_chunk) ----------
#ifdef __AVX512F__

  void scan_64_chunk(const Quantized_Query<Metric>& q, const uint8_t* codes_ptr,
                     float* results) const {
#ifdef __AVX512VNNI__
    scan_64_chunk_vnni(q, codes_ptr, results);
#else
    scan_64_chunk_avx512_shuffle(q, codes_ptr, results);
#endif
  }

  // VNNI path: decode 64 nibbles to one-hot (64 x 16 bytes), then vpdpbusd with LUT (int8).
  void scan_64_chunk_vnni(const Quantized_Query<Metric>& q, const uint8_t* codes_ptr,
                          float* results) const {
    // One-hot table: row c has 1 at byte c, 0 elsewhere (16 bytes per row).
    alignas(16) static const uint8_t kOneHot[16][16] = {
        {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
    };
    constexpr int kPoints = 64;
    constexpr int kOneHotSize = kPoints * 16;  // 1024 bytes per block
    alignas(64) uint8_t onehot_buf[kOneHotSize];

    alignas(64) int32_t acc32[kPoints];
    for (int i = 0; i < kPoints; ++i)
      acc32[i] = 0;

    const int32_t bias = 128 * static_cast<int32_t>(num_blocks);

    for (uint32_t b = 0; b < num_blocks; ++b) {
      // Decode 32 bytes (64 nibbles) to 64 x 16 one-hot bytes.
      for (int i = 0; i < 32; ++i) {
        uint8_t byte = codes_ptr[i];
        uint8_t lo = byte & 0x0Fu;
        uint8_t hi = (byte >> 4) & 0x0Fu;
        std::memcpy(onehot_buf + (2 * i + 0) * 16, kOneHot[lo], 16);
        std::memcpy(onehot_buf + (2 * i + 1) * 16, kOneHot[hi], 16);
      }
      codes_ptr += 32;

      // LUT as int8 (LUT - 128) for vpdpbusd (unsigned one-hot x signed LUT).
      alignas(64) int8_t lut_signed[16];
      const uint8_t* lut_u = &q.int_lut[static_cast<size_t>(b) * K];
      for (int i = 0; i < 16; ++i)
        lut_signed[i] = static_cast<int8_t>(static_cast<int>(lut_u[i]) - 128);

      // Four panels of 16 points; each point has 16 bytes (one-hot). vpdpbusd: 16 x 4 bytes.
      for (int panel = 0; panel < 4; ++panel) {
        const uint8_t* panel_ptr = onehot_buf + panel * (16 * 16);
        __m512i block_acc = _mm512_setzero_si512();
        for (int t = 0; t < 4; ++t) {
          const __m512i a =
              _mm512_loadu_si512(panel_ptr + t * 64);  // 16 points x 4 bytes (unsigned)
          const __m512i b_vec = _mm512_set1_epi32(
              *reinterpret_cast<const int32_t*>(lut_signed + t * 4));  // broadcast 4 signed bytes
          block_acc = _mm512_dpbusd_epi32(block_acc, a, b_vec);
        }
        __m512i running = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(acc32 + panel * 16));
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(acc32 + panel * 16),
                            _mm512_add_epi32(running, block_acc));
      }
    }

    // Bias correction: we computed sum(lut[c]-128), so real sum = acc32 + 128*num_blocks.
    for (int i = 0; i < kPoints; ++i)
      results[i] = q.decode_int32(acc32[i] + bias);
  }

  void scan_64_chunk_avx512_shuffle(const Quantized_Query<Metric>& q, const uint8_t* codes_ptr,
                                    float* results) const {
    __m512i acc_even = _mm512_setzero_si512();
    __m512i acc_odd = _mm512_setzero_si512();
    const __m256i low_mask = _mm256_set1_epi8(0x0F);

    for (uint32_t b = 0; b < num_blocks; ++b) {
      const __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(codes_ptr));
      codes_ptr += 32;

      const __m256i codes_even = _mm256_and_si256(packed, low_mask);
      const __m256i codes_odd = _mm256_and_si256(_mm256_srli_epi16(packed, 4), low_mask);

      const __m128i lut128 =
          _mm_loadu_si128(reinterpret_cast<const __m128i*>(&q.int_lut[static_cast<size_t>(b) * K]));
      const __m256i lut256 = _mm256_broadcastsi128_si256(lut128);

      const __m256i scores_even_u8 = _mm256_shuffle_epi8(lut256, codes_even);
      const __m256i scores_odd_u8 = _mm256_shuffle_epi8(lut256, codes_odd);

      acc_even = _mm512_add_epi16(acc_even, _mm512_cvtepu8_epi16(scores_even_u8));
      acc_odd = _mm512_add_epi16(acc_odd, _mm512_cvtepu8_epi16(scores_odd_u8));
    }

    alignas(64) uint16_t raw_even[32];
    alignas(64) uint16_t raw_odd[32];
    _mm512_store_si512(reinterpret_cast<__m512i*>(raw_even), acc_even);
    _mm512_store_si512(reinterpret_cast<__m512i*>(raw_odd), acc_odd);

    for (int i = 0; i < 32; ++i) {
      results[i * 2] = q.decode(raw_even[i]);
      results[i * 2 + 1] = q.decode(raw_odd[i]);
    }
  }

  inline uint16_t hmin_512_epu16(__m512i v) const {
    v = _mm512_min_epu16(v, _mm512_shuffle_i32x4(v, v, _MM_SHUFFLE(1, 0, 3, 2)));
    v = _mm512_min_epu16(v, _mm512_shuffle_i32x4(v, v, _MM_SHUFFLE(0, 0, 1, 1)));
    __m128i v128 = _mm512_castsi512_si128(v);
    v128 = _mm_min_epu16(v128, _mm_srli_si128(v128, 8));
    v128 = _mm_min_epu16(v128, _mm_srli_si128(v128, 4));
    v128 = _mm_min_epu16(v128, _mm_srli_si128(v128, 2));
    return static_cast<uint16_t>(_mm_extract_epi16(v128, 0));
  }

  float scan_64_chunk_min(const Quantized_Query<Metric>& q, const uint8_t* codes_ptr) const {
    __m512i acc_even = _mm512_setzero_si512();
    __m512i acc_odd = _mm512_setzero_si512();
    const __m256i low_mask = _mm256_set1_epi8(0x0F);

    for (uint32_t b = 0; b < num_blocks; ++b) {
      const __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(codes_ptr));
      codes_ptr += 32;
      const __m256i codes_even = _mm256_and_si256(packed, low_mask);
      const __m256i codes_odd = _mm256_and_si256(_mm256_srli_epi16(packed, 4), low_mask);
      const __m128i lut128 =
          _mm_loadu_si128(reinterpret_cast<const __m128i*>(&q.int_lut[static_cast<size_t>(b) * K]));
      const __m256i lut256 = _mm256_broadcastsi128_si256(lut128);
      const __m256i scores_even_u8 = _mm256_shuffle_epi8(lut256, codes_even);
      const __m256i scores_odd_u8 = _mm256_shuffle_epi8(lut256, codes_odd);
      acc_even = _mm512_add_epi16(acc_even, _mm512_cvtepu8_epi16(scores_even_u8));
      acc_odd = _mm512_add_epi16(acc_odd, _mm512_cvtepu8_epi16(scores_odd_u8));
    }
    const __m512i vmin = _mm512_min_epu16(acc_even, acc_odd);
    return q.decode(hmin_512_epu16(vmin));
  }

  void scan_64_running_min(const Quantized_Query<Metric>& q, const uint8_t* codes_ptr,
                           RunningMinV& current_min_v) const {
    __m512i acc_even = _mm512_setzero_si512();
    __m512i acc_odd = _mm512_setzero_si512();
    const __m256i low_mask = _mm256_set1_epi8(0x0F);

    for (uint32_t b = 0; b < num_blocks; ++b) {
      const __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(codes_ptr));
      codes_ptr += 32;
      const __m256i codes_even = _mm256_and_si256(packed, low_mask);
      const __m256i codes_odd = _mm256_and_si256(_mm256_srli_epi16(packed, 4), low_mask);
      const __m128i lut128 =
          _mm_loadu_si128(reinterpret_cast<const __m128i*>(&q.int_lut[static_cast<size_t>(b) * K]));
      const __m256i lut256 = _mm256_broadcastsi128_si256(lut128);
      const __m256i scores_even_u8 = _mm256_shuffle_epi8(lut256, codes_even);
      const __m256i scores_odd_u8 = _mm256_shuffle_epi8(lut256, codes_odd);
      acc_even = _mm512_add_epi16(acc_even, _mm512_cvtepu8_epi16(scores_even_u8));
      acc_odd = _mm512_add_epi16(acc_odd, _mm512_cvtepu8_epi16(scores_odd_u8));
    }
    current_min_v.v = _mm512_min_epu16(current_min_v.v, acc_even);
    current_min_v.v = _mm512_min_epu16(current_min_v.v, acc_odd);
  }

  float reduce_running_min(const Quantized_Query<Metric>& q,
                           const RunningMinV& running_min_v) const {
    return q.decode(hmin_512_epu16(running_min_v.v));
  }

  void scan_64_dual_query(const Quantized_Query<Metric>& q1, const Quantized_Query<Metric>& q2,
                          const uint8_t* codes_ptr, RunningMinV& min_v1,
                          RunningMinV& min_v2) const {
    __m512i acc1_even = _mm512_setzero_si512();
    __m512i acc1_odd = _mm512_setzero_si512();
    __m512i acc2_even = _mm512_setzero_si512();
    __m512i acc2_odd = _mm512_setzero_si512();
    const __m256i low_mask = _mm256_set1_epi8(0x0F);

    for (uint32_t b = 0; b < num_blocks; ++b) {
      const __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(codes_ptr));
      codes_ptr += 32;
      const __m256i codes_even = _mm256_and_si256(packed, low_mask);
      const __m256i codes_odd = _mm256_and_si256(_mm256_srli_epi16(packed, 4), low_mask);
      const __m128i lut1_128 = _mm_loadu_si128(
          reinterpret_cast<const __m128i*>(&q1.int_lut[static_cast<size_t>(b) * K]));
      const __m128i lut2_128 = _mm_loadu_si128(
          reinterpret_cast<const __m128i*>(&q2.int_lut[static_cast<size_t>(b) * K]));
      const __m256i lut1_256 = _mm256_broadcastsi128_si256(lut1_128);
      const __m256i lut2_256 = _mm256_broadcastsi128_si256(lut2_128);
      const __m256i s1_even_u8 = _mm256_shuffle_epi8(lut1_256, codes_even);
      const __m256i s1_odd_u8 = _mm256_shuffle_epi8(lut1_256, codes_odd);
      const __m256i s2_even_u8 = _mm256_shuffle_epi8(lut2_256, codes_even);
      const __m256i s2_odd_u8 = _mm256_shuffle_epi8(lut2_256, codes_odd);
      acc1_even = _mm512_add_epi16(acc1_even, _mm512_cvtepu8_epi16(s1_even_u8));
      acc1_odd = _mm512_add_epi16(acc1_odd, _mm512_cvtepu8_epi16(s1_odd_u8));
      acc2_even = _mm512_add_epi16(acc2_even, _mm512_cvtepu8_epi16(s2_even_u8));
      acc2_odd = _mm512_add_epi16(acc2_odd, _mm512_cvtepu8_epi16(s2_odd_u8));
    }
    min_v1.v = _mm512_min_epu16(min_v1.v, _mm512_min_epu16(acc1_even, acc1_odd));
    min_v2.v = _mm512_min_epu16(min_v2.v, _mm512_min_epu16(acc2_even, acc2_odd));
  }

  static inline __mmask32 mask_even_lanes(int lo, int hi) {
    __mmask32 m = 0;
    for (int i = 0; i < 32; ++i) {
      if (lo <= 2 * i && 2 * i < hi) m |= (__mmask32(1) << i);
    }
    return m;
  }
  static inline __mmask32 mask_odd_lanes(int lo, int hi) {
    __mmask32 m = 0;
    for (int i = 0; i < 32; ++i) {
      if (lo <= 2 * i + 1 && 2 * i + 1 < hi) m |= (__mmask32(1) << i);
    }
    return m;
  }

  float scan_64_chunk_min_masked(const Quantized_Query<Metric>& q, const uint8_t* codes_ptr, int lo,
                                 int hi) const {
    lo = std::max(lo, 0);
    hi = std::min(hi, 64);
    if (hi <= lo) return std::numeric_limits<float>::infinity();

    __m512i acc_even = _mm512_setzero_si512();
    __m512i acc_odd = _mm512_setzero_si512();
    const __m256i low_mask = _mm256_set1_epi8(0x0F);

    for (uint32_t b = 0; b < num_blocks; ++b) {
      const __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(codes_ptr));
      codes_ptr += 32;
      const __m256i codes_even = _mm256_and_si256(packed, low_mask);
      const __m256i codes_odd = _mm256_and_si256(_mm256_srli_epi16(packed, 4), low_mask);
      const __m128i lut128 =
          _mm_loadu_si128(reinterpret_cast<const __m128i*>(&q.int_lut[static_cast<size_t>(b) * K]));
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
    return q.decode(hmin_512_epu16(vmin));
  }

  // ---------- AVX2 fallback ----------
#elif defined(__AVX2__)

  inline uint16_t hmin_256_epu16(__m256i v) const {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i m = _mm_min_epu16(lo, hi);
    m = _mm_min_epu16(m, _mm_srli_si128(m, 8));
    m = _mm_min_epu16(m, _mm_srli_si128(m, 4));
    m = _mm_min_epu16(m, _mm_srli_si128(m, 2));
    return static_cast<uint16_t>(_mm_extract_epi16(m, 0));
  }

  void scan_64_chunk_avx2_impl(const Quantized_Query<Metric>& q, const uint8_t* codes_ptr,
                               float* results) const {
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
      const __m128i lut128 =
          _mm_loadu_si128(reinterpret_cast<const __m128i*>(&q.int_lut[static_cast<size_t>(b) * K]));
      const __m256i lut256 = _mm256_broadcastsi128_si256(lut128);
      const __m256i scores_even_u8 = _mm256_shuffle_epi8(lut256, codes_even);
      const __m256i scores_odd_u8 = _mm256_shuffle_epi8(lut256, codes_odd);
      acc_even_lo = _mm256_add_epi16(acc_even_lo,
                                     _mm256_cvtepu8_epi16(_mm256_castsi256_si128(scores_even_u8)));
      acc_even_hi = _mm256_add_epi16(
          acc_even_hi, _mm256_cvtepu8_epi16(_mm256_extracti128_si256(scores_even_u8, 1)));
      acc_odd_lo =
          _mm256_add_epi16(acc_odd_lo, _mm256_cvtepu8_epi16(_mm256_castsi256_si128(scores_odd_u8)));
      acc_odd_hi = _mm256_add_epi16(
          acc_odd_hi, _mm256_cvtepu8_epi16(_mm256_extracti128_si256(scores_odd_u8, 1)));
    }

    alignas(32) uint16_t raw_even[32];
    alignas(32) uint16_t raw_odd[32];
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(raw_even), acc_even_lo);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(raw_even + 16), acc_even_hi);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(raw_odd), acc_odd_lo);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(raw_odd + 16), acc_odd_hi);

    for (int i = 0; i < 32; ++i) {
      results[i * 2] = q.decode(raw_even[i]);
      results[i * 2 + 1] = q.decode(raw_odd[i]);
    }
  }

  void scan_64_chunk(const Quantized_Query<Metric>& q, const uint8_t* codes_ptr,
                     float* results) const {
    scan_64_chunk_avx2_impl(q, codes_ptr, results);
  }

  float scan_64_chunk_min(const Quantized_Query<Metric>& q, const uint8_t* codes_ptr) const {
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
      const __m128i lut128 =
          _mm_loadu_si128(reinterpret_cast<const __m128i*>(&q.int_lut[static_cast<size_t>(b) * K]));
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
    __m256i sum_lo = _mm256_add_epi16(acc_even_lo, acc_odd_lo);
    __m256i sum_hi = _mm256_add_epi16(acc_even_hi, acc_odd_hi);
    uint16_t best = std::min(hmin_256_epu16(sum_lo), hmin_256_epu16(sum_hi));
    return q.decode(best);
  }

  void scan_64_running_min(const Quantized_Query<Metric>& q, const uint8_t* codes_ptr,
                           RunningMinV& current_min_v) const {
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
      const __m128i lut128 =
          _mm_loadu_si128(reinterpret_cast<const __m128i*>(&q.int_lut[static_cast<size_t>(b) * K]));
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
    __m256i sum_lo = _mm256_add_epi16(acc_even_lo, acc_odd_lo);
    __m256i sum_hi = _mm256_add_epi16(acc_even_hi, acc_odd_hi);
    current_min_v.lo = _mm256_min_epu16(current_min_v.lo, sum_lo);
    current_min_v.hi = _mm256_min_epu16(current_min_v.hi, sum_hi);
  }

  float reduce_running_min(const Quantized_Query<Metric>& q,
                           const RunningMinV& running_min_v) const {
    uint16_t best = std::min(hmin_256_epu16(running_min_v.lo), hmin_256_epu16(running_min_v.hi));
    return q.decode(best);
  }

  void scan_64_dual_query(const Quantized_Query<Metric>& q1, const Quantized_Query<Metric>& q2,
                          const uint8_t* codes_ptr, RunningMinV& min_v1,
                          RunningMinV& min_v2) const {
    __m256i a1_lo = _mm256_setzero_si256(), a1_hi = _mm256_setzero_si256();
    __m256i a2_lo = _mm256_setzero_si256(), a2_hi = _mm256_setzero_si256();
    const __m256i low_mask = _mm256_set1_epi8(0x0F);

    for (uint32_t b = 0; b < num_blocks; ++b) {
      const __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(codes_ptr));
      codes_ptr += 32;
      const __m256i codes_even = _mm256_and_si256(packed, low_mask);
      const __m256i codes_odd = _mm256_and_si256(_mm256_srli_epi16(packed, 4), low_mask);
      const __m128i lut1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&q1.int_lut[b * K]));
      const __m128i lut2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&q2.int_lut[b * K]));
      const __m256i l1 = _mm256_broadcastsi128_si256(lut1);
      const __m256i l2 = _mm256_broadcastsi128_si256(lut2);
      const __m256i s1e = _mm256_shuffle_epi8(l1, codes_even);
      const __m256i s1o = _mm256_shuffle_epi8(l1, codes_odd);
      const __m256i s2e = _mm256_shuffle_epi8(l2, codes_even);
      const __m256i s2o = _mm256_shuffle_epi8(l2, codes_odd);
      a1_lo = _mm256_add_epi16(a1_lo, _mm256_cvtepu8_epi16(_mm256_castsi256_si128(s1e)));
      a1_hi = _mm256_add_epi16(a1_hi, _mm256_cvtepu8_epi16(_mm256_extracti128_si256(s1e, 1)));
      a1_lo = _mm256_add_epi16(a1_lo, _mm256_cvtepu8_epi16(_mm256_castsi256_si128(s1o)));
      a1_hi = _mm256_add_epi16(a1_hi, _mm256_cvtepu8_epi16(_mm256_extracti128_si256(s1o, 1)));
      a2_lo = _mm256_add_epi16(a2_lo, _mm256_cvtepu8_epi16(_mm256_castsi256_si128(s2e)));
      a2_hi = _mm256_add_epi16(a2_hi, _mm256_cvtepu8_epi16(_mm256_extracti128_si256(s2e, 1)));
      a2_lo = _mm256_add_epi16(a2_lo, _mm256_cvtepu8_epi16(_mm256_castsi256_si128(s2o)));
      a2_hi = _mm256_add_epi16(a2_hi, _mm256_cvtepu8_epi16(_mm256_extracti128_si256(s2o, 1)));
    }
    min_v1.lo = _mm256_min_epu16(min_v1.lo, a1_lo);
    min_v1.hi = _mm256_min_epu16(min_v1.hi, a1_hi);
    min_v2.lo = _mm256_min_epu16(min_v2.lo, a2_lo);
    min_v2.hi = _mm256_min_epu16(min_v2.hi, a2_hi);
  }

  float scan_64_chunk_min_masked(const Quantized_Query<Metric>& q, const uint8_t* codes_ptr, int lo,
                                 int hi) const {
    lo = std::max(lo, 0);
    hi = std::min(hi, 64);
    if (hi <= lo) return std::numeric_limits<float>::infinity();
    alignas(64) float tmp[64];
    scan_64_chunk_avx2_impl(q, codes_ptr, tmp);
    float best = std::numeric_limits<float>::max();
    for (int i = lo; i < hi; ++i)
      best = std::min(best, tmp[i]);
    return best;
  }

  // ---------- Scalar fallback (no SIMD) ----------
#else

  void scan_64_chunk(const Quantized_Query<Metric>& q, const uint8_t* codes_ptr,
                     float* results) const {
    for (int lane = 0; lane < 64; ++lane) {
      uint16_t acc = 0;
      for (uint32_t b = 0; b < num_blocks; ++b) {
        uint8_t packed = codes_ptr[b * 32 + (lane / 2)];
        uint8_t code = (lane % 2 == 0) ? (packed & 0x0F) : (packed >> 4);
        acc += static_cast<uint16_t>(q.int_lut[b * K + code]);
      }
      results[lane] = q.decode(acc);
    }
  }

  float scan_64_chunk_min(const Quantized_Query<Metric>& q, const uint8_t* codes_ptr) const {
    float best = std::numeric_limits<float>::max();
    for (int lane = 0; lane < 64; ++lane) {
      uint16_t acc = 0;
      for (uint32_t b = 0; b < num_blocks; ++b) {
        uint8_t packed = codes_ptr[b * 32 + (lane / 2)];
        uint8_t code = (lane % 2 == 0) ? (packed & 0x0F) : (packed >> 4);
        acc += static_cast<uint16_t>(q.int_lut[b * K + code]);
      }
      best = std::min(best, q.decode(acc));
    }
    return best;
  }

  void scan_64_running_min(const Quantized_Query<Metric>& q, const uint8_t* codes_ptr,
                           RunningMinV& current_min_v) const {
    for (int lane = 0; lane < 64; ++lane) {
      uint16_t acc = 0;
      for (uint32_t b = 0; b < num_blocks; ++b) {
        uint8_t packed = codes_ptr[b * 32 + (lane / 2)];
        uint8_t code = (lane % 2 == 0) ? (packed & 0x0F) : (packed >> 4);
        acc += static_cast<uint16_t>(q.int_lut[b * K + code]);
      }
      current_min_v.data[lane] = std::min(current_min_v.data[lane], acc);
    }
  }

  float reduce_running_min(const Quantized_Query<Metric>& q,
                           const RunningMinV& running_min_v) const {
    uint16_t best = 0xFFFF;
    for (int i = 0; i < 64; ++i)
      best = std::min(best, running_min_v.data[i]);
    return q.decode(best);
  }

  void scan_64_dual_query(const Quantized_Query<Metric>& q1, const Quantized_Query<Metric>& q2,
                          const uint8_t* codes_ptr, RunningMinV& min_v1,
                          RunningMinV& min_v2) const {
    for (int lane = 0; lane < 64; ++lane) {
      uint16_t acc1 = 0, acc2 = 0;
      for (uint32_t b = 0; b < num_blocks; ++b) {
        uint8_t packed = codes_ptr[b * 32 + (lane / 2)];
        uint8_t code = (lane % 2 == 0) ? (packed & 0x0F) : (packed >> 4);
        acc1 += static_cast<uint16_t>(q1.int_lut[b * K + code]);
        acc2 += static_cast<uint16_t>(q2.int_lut[b * K + code]);
      }
      min_v1.data[lane] = std::min(min_v1.data[lane], acc1);
      min_v2.data[lane] = std::min(min_v2.data[lane], acc2);
    }
  }

  float scan_64_chunk_min_masked(const Quantized_Query<Metric>& q, const uint8_t* codes_ptr, int lo,
                                 int hi) const {
    lo = std::max(lo, 0);
    hi = std::min(hi, 64);
    if (hi <= lo) return std::numeric_limits<float>::infinity();
    float best = std::numeric_limits<float>::max();
    for (int lane = lo; lane < hi; ++lane) {
      uint16_t acc = 0;
      for (uint32_t b = 0; b < num_blocks; ++b) {
        uint8_t packed = codes_ptr[b * 32 + (lane / 2)];
        uint8_t code = (lane % 2 == 0) ? (packed & 0x0F) : (packed >> 4);
        acc += static_cast<uint16_t>(q.int_lut[b * K + code]);
      }
      best = std::min(best, q.decode(acc));
    }
    return best;
  }

#endif

  // ---------------------------------------------------------
  // Save/load (same as original)
  // ---------------------------------------------------------
  void save(std::ostream& out) const {
    out.write(reinterpret_cast<const char*>(&num_blocks), sizeof(num_blocks));
    out.write(reinterpret_cast<const char*>(&n_points_raw), sizeof(n_points_raw));
    out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    out.write(reinterpret_cast<const char*>(&dim_per_block), sizeof(dim_per_block));
    out.write(reinterpret_cast<const char*>(&n_points_raw_unpadded), sizeof(n_points_raw_unpadded));
    size_t n_off = cloud_vec_offsets.size();
    out.write(reinterpret_cast<const char*>(&n_off), sizeof(n_off));
    if (n_off)
      out.write(reinterpret_cast<const char*>(cloud_vec_offsets.data()), n_off * sizeof(size_t));
    size_t sz = packed_codes.size();
    out.write(reinterpret_cast<const char*>(&sz), sizeof(sz));
    if (sz) out.write(reinterpret_cast<const char*>(packed_codes.data()), sz * sizeof(uint8_t));
  }

  void load(std::istream& in) {
    in.read(reinterpret_cast<char*>(&num_blocks), sizeof(num_blocks));
    in.read(reinterpret_cast<char*>(&n_points_raw), sizeof(n_points_raw));
    in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    in.read(reinterpret_cast<char*>(&dim_per_block), sizeof(dim_per_block));
    in.read(reinterpret_cast<char*>(&n_points_raw_unpadded), sizeof(n_points_raw_unpadded));
    size_t n_off = 0;
    in.read(reinterpret_cast<char*>(&n_off), sizeof(n_off));
    cloud_vec_offsets.resize(n_off);
    if (n_off) in.read(reinterpret_cast<char*>(cloud_vec_offsets.data()), n_off * sizeof(size_t));
    size_t sz = 0;
    in.read(reinterpret_cast<char*>(&sz), sizeof(sz));
    packed_codes.resize(sz);
    if (sz) in.read(reinterpret_cast<char*>(packed_codes.data()), sz * sizeof(uint8_t));
  }

  inline uint32_t size() const noexcept { return static_cast<uint32_t>(n_points_raw); }
  inline uint32_t get_dims() const noexcept { return static_cast<uint32_t>(dim); }
};

// ---------------------------------------------------------
// Model
// ---------------------------------------------------------
template<bool Metric>
class Model {
 public:
  static constexpr bool is_fastscan = true;
  static constexpr uint32_t K = 16;

  uint32_t num_blocks = 0;
  size_t dim = 0;
  size_t dim_per_block = 0;

  std::vector<Eigen::MatrixXf> codebooks;
  std::vector<Eigen::VectorXf> codebook_norms;

  Model() = default;

  template<typename PointRange>
  Model(const PointRange& train_data, uint32_t block_size = 32) {
    train(train_data, block_size);
  }

  template<typename PointRange>
  void train(const PointRange& data, uint32_t block_size = 32) {
    dim = data.get_dims();
    dim_per_block = block_size;
    if (dim_per_block == 0 || (dim % dim_per_block) != 0) {
      std::cerr << "Error: FastScan dim=" << dim << " not divisible by block_size=" << dim_per_block
                << "\n";
      abort();
    }
    num_blocks = static_cast<uint32_t>(dim / dim_per_block);
    codebooks.resize(num_blocks);
    codebook_norms.resize(num_blocks);
    const size_t n_points_raw_unpadded = data.size();
    const size_t sample_size = std::min(static_cast<size_t>(4096), n_points_raw_unpadded);

    parlay::parallel_for(0, num_blocks, [&](size_t b) {
      const size_t offset = b * dim_per_block;
      parlay::sequence<parlay::sequence<float>> sub(sample_size);
      std::mt19937 rng(static_cast<unsigned>(b + 1));
      std::uniform_int_distribution<size_t> distu(0, n_points_raw_unpadded - 1);
      for (size_t i = 0; i < sample_size; ++i) {
        const float* raw = reinterpret_cast<const float*>(data.location(distu(rng)));
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

  template<typename PointRange, typename SeqOffsetsFloat>
  Quantized_Point_Range<PointRange, Metric> encode(
      const PointRange& data, const SeqOffsetsFloat& cloud_offsets_float) const {
    Quantized_Point_Range<PointRange, Metric> enc;
    enc.num_blocks = num_blocks;
    enc.dim = data.get_dims();
    enc.dim_per_block = dim_per_block;
    enc.n_points_raw_unpadded = data.size();
    const size_t n_clouds = (cloud_offsets_float.size() > 0) ? (cloud_offsets_float.size() - 1) : 0;
    enc.cloud_vec_offsets.resize(n_clouds + 1);
    size_t cur = 0;
    enc.cloud_vec_offsets[0] = 0;
    for (size_t c = 0; c < n_clouds; ++c) {
      const size_t start_f = static_cast<size_t>(cloud_offsets_float[c]);
      const size_t end_f = static_cast<size_t>(cloud_offsets_float[c + 1]);
      const size_t sz_vecs = (end_f - start_f) / enc.dim;
      const size_t padded = ((sz_vecs + 63) / 64) * 64;
      enc.cloud_vec_offsets[c] = cur;
      cur += padded;
    }
    enc.cloud_vec_offsets[n_clouds] = cur;
    enc.n_points_raw = cur;
    const size_t strip_stride = static_cast<size_t>(enc.num_blocks) * 32;
    const size_t num_strips = (enc.n_points_raw + 63) / 64;
    enc.packed_codes.resize(num_strips * strip_stride);
    std::fill(enc.packed_codes.begin(), enc.packed_codes.end(), uint8_t{0});

    parlay::parallel_for(0, n_clouds, [&](size_t c) {
      const size_t start_f = static_cast<size_t>(cloud_offsets_float[c]);
      const size_t end_f = static_cast<size_t>(cloud_offsets_float[c + 1]);
      const size_t start_vec_src = start_f / enc.dim;
      const size_t sz_vecs = (end_f - start_f) / enc.dim;
      const size_t start_vec_dst = enc.cloud_vec_offsets[c];
      const size_t padded_sz = enc.cloud_vec_offsets[c + 1] - enc.cloud_vec_offsets[c];
      const size_t n_strips_c = padded_sz / 64;
      const size_t strip0_dst = start_vec_dst / 64;

      for (size_t s = 0; s < n_strips_c; ++s) {
        uint8_t* strip_base = enc.packed_codes.data() + (strip0_dst + s) * strip_stride;
        for (size_t lane_pair = 0; lane_pair < 32; ++lane_pair) {
          const size_t v_even_in_cloud = s * 64 + lane_pair * 2;
          const size_t v_odd_in_cloud = v_even_in_cloud + 1;
          const bool has_even = (v_even_in_cloud < sz_vecs);
          const bool has_odd = (v_odd_in_cloud < sz_vecs);
          const size_t v_even_src = start_vec_src + v_even_in_cloud;
          const size_t v_odd_src = start_vec_src + v_odd_in_cloud;
          for (uint32_t b = 0; b < enc.num_blocks; ++b) {
            const uint8_t c_e = has_even ? find_best(data.location(v_even_src), b) : 0;
            const uint8_t c_o = has_odd ? find_best(data.location(v_odd_src), b) : 0;
            strip_base[static_cast<size_t>(b) * 32 + lane_pair] =
                (c_e & 0x0F) | ((c_o & 0x0F) << 4);
          }
        }
      }
    });
    return enc;
  }

  template<typename PointRange>
  Quantized_Point_Range<PointRange, Metric> encode(const PointRange& data) const {
    std::vector<size_t> cloud_offsets_float = {0,
                                               data.size() * static_cast<size_t>(data.get_dims())};
    return encode(data, cloud_offsets_float);
  }

  template<typename PointTy>
  Quantized_Query<Metric> quantize_query(const PointTy& query) const {
    Quantized_Query<Metric> qq(num_blocks);
    std::vector<float> float_lut(static_cast<size_t>(num_blocks) * K);
    float g_min = std::numeric_limits<float>::max();
    float g_max = std::numeric_limits<float>::lowest();
    for (uint32_t b = 0; b < num_blocks; ++b) {
      Eigen::VectorXf q_sub(static_cast<Eigen::Index>(dim_per_block));
      for (size_t j = 0; j < dim_per_block; ++j)
        q_sub(static_cast<Eigen::Index>(j)) = query[b * dim_per_block + j];
      Eigen::VectorXf dots = codebooks[b] * q_sub;
      float q_sq = Metric ? q_sub.squaredNorm() : 0.0f;
      for (uint32_t i = 0; i < K; ++i) {
        float val = Metric ? (codebook_norms[b][i] - 2.0f * dots[i] + q_sq) : -dots[i];
        float_lut[b * K + i] = val;
        g_min = std::min(g_min, val);
        g_max = std::max(g_max, val);
      }
    }
    return finish_quantize_query(qq, float_lut, g_min, g_max);
  }

  Quantized_Query<Metric> quantize_query(const float* qptr) const {
    Quantized_Query<Metric> qq(num_blocks);
    std::vector<float> float_lut(static_cast<size_t>(num_blocks) * K);
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
    return finish_quantize_query(qq, float_lut, g_min, g_max);
  }

  template<typename PointCloudTy>
  void quantize_query_batch(const PointCloudTy& query_cloud,
                            parlay::sequence<Quantized_Query<Metric>>& out_luts) const {
    const uint32_t num_q = query_cloud.size();
    const uint32_t dims = query_cloud.get_dims();
    const float* base = query_cloud.data();
    out_luts.clear();
    out_luts.reserve(num_q);
    for (uint32_t i = 0; i < num_q; ++i)
      out_luts.emplace_back(num_blocks);
    using RowMajorMat = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    RowMajorMat Q_sub(static_cast<Eigen::Index>(num_q), static_cast<Eigen::Index>(dim_per_block));
    Eigen::MatrixXf dot_products(static_cast<Eigen::Index>(K), static_cast<Eigen::Index>(num_q));
    std::vector<std::vector<float>> batch_float_luts(num_q, std::vector<float>(num_blocks * K));
    std::vector<float> q_mins(num_q, std::numeric_limits<float>::max());
    std::vector<float> q_maxs(num_q, std::numeric_limits<float>::lowest());
    for (uint32_t b = 0; b < num_blocks; ++b) {
      for (uint32_t i = 0; i < num_q; ++i)
        std::memcpy(&Q_sub(i, 0), base + i * dims + b * dim_per_block,
                    sizeof(float) * dim_per_block);
      Eigen::VectorXf q_sq;
      if constexpr (Metric) q_sq = Q_sub.rowwise().squaredNorm();
      dot_products.noalias() = codebooks[b] * Q_sub.transpose();
      for (uint32_t i = 0; i < num_q; ++i) {
        for (uint32_t k_idx = 0; k_idx < K; ++k_idx) {
          float val = Metric ? (codebook_norms[b][k_idx] - 2.0f * dot_products(k_idx, i) + q_sq[i])
                             : -dot_products(k_idx, i);
          batch_float_luts[i][b * K + k_idx] = val;
          q_mins[i] = std::min(q_mins[i], val);
          q_maxs[i] = std::max(q_maxs[i], val);
        }
      }
    }
    for (uint32_t i = 0; i < num_q; ++i)
      finish_quantize_query(out_luts[i], batch_float_luts[i], q_mins[i], q_maxs[i]);
  }

  void save(std::ostream& out) const {
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

  void load(std::istream& in) {
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

 private:
  Quantized_Query<Metric>& finish_quantize_query(Quantized_Query<Metric>& qq,
                                                 const std::vector<float>& float_lut, float g_min,
                                                 float g_max) const {
    qq.min_dist = g_min;
    qq.scale = std::max(1e-6f, (g_max - g_min) / 255.0f);
    for (size_t i = 0; i < float_lut.size(); ++i)
      qq.int_lut[i] = static_cast<uint8_t>((float_lut[i] - g_min) / qq.scale);
    return qq;
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
};

}  // namespace fastscan
}  // namespace mvsic
