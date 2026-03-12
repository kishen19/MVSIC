// turboquant_pq_4bit.h
//
// PQ-style TurboQuant: block-wise k-means (2/4/8/16 dims per block), 16 centroids
// per block, 4-bit codes. Same preprocessing as TurboQuant (rotate, normalize,
// scale sqrt(d)). Centroids stored as int8; query blocks quantized to int8; LUT
// is uint8 with decode (FastScan-style). Strip layout: num_blocks*32 bytes per
// strip (2 points per byte, nibbles). block_size=1 is not supported.
//
// Interface: Model<Metric, BlockSize>, Quantized_Query<Metric, BlockSize>,
//            Quantized_Point<Metric, BlockSize>, Quantized_Point_Range<...>

#pragma once

#include <algorithm>
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

#include <Eigen/Core>
#include <immintrin.h>

#include "parlay/parallel.h"
#include "parlay/sequence.h"

#include "rabitqlib/utils/rotator.hpp"

#include "mvsic/core/quantization/turboquant_pq_codebooks.h"

namespace mvsic {
namespace turboquant_pq_4bit {

// ---------------------------------------------------------
// Running-min vector type: 64 floats (matches strip size)
// ---------------------------------------------------------
#ifdef __AVX512F__
struct RunningMinV {
  __m512 v[4];
  void set_max() {
    for (int i = 0; i < 4; ++i) v[i] = _mm512_set1_ps(std::numeric_limits<float>::max());
  }
  static RunningMinV max() {
    RunningMinV r;
    r.set_max();
    return r;
  }
  inline float reduce() const {
    __m512 m = _mm512_min_ps(_mm512_min_ps(v[0], v[1]), _mm512_min_ps(v[2], v[3]));
    __m256 lo = _mm512_castps512_ps256(m);
    __m256 hi = _mm512_extractf32x8_ps(m, 1);
    __m256 m2 = _mm256_min_ps(lo, hi);
    __m128 m128_lo = _mm256_castps256_ps128(m2);
    __m128 m128_hi = _mm256_extractf128_ps(m2, 1);
    __m128 m128 = _mm_min_ps(m128_lo, m128_hi);
    m128 = _mm_min_ps(m128, _mm_shuffle_ps(m128, m128, _MM_SHUFFLE(1, 0, 3, 2)));
    m128 = _mm_min_ps(m128, _mm_shuffle_ps(m128, m128, _MM_SHUFFLE(0, 0, 0, 1)));
    return _mm_cvtss_f32(m128);
  }
};
#elif defined(__AVX2__)
struct RunningMinV {
  __m256 v[8];
  void set_max() {
    for (int i = 0; i < 8; ++i) v[i] = _mm256_set1_ps(std::numeric_limits<float>::max());
  }
  static RunningMinV max() {
    RunningMinV r;
    r.set_max();
    return r;
  }
  inline float reduce() const {
    __m256 m = v[0];
    for (int i = 1; i < 8; ++i) m = _mm256_min_ps(m, v[i]);
    __m128 m128_lo = _mm256_castps256_ps128(m);
    __m128 m128_hi = _mm256_extractf128_ps(m, 1);
    __m128 m128 = _mm_min_ps(m128_lo, m128_hi);
    m128 = _mm_min_ps(m128, _mm_shuffle_ps(m128, m128, _MM_SHUFFLE(1, 0, 3, 2)));
    m128 = _mm_min_ps(m128, _mm_shuffle_ps(m128, m128, _MM_SHUFFLE(0, 0, 0, 1)));
    return _mm_cvtss_f32(m128);
  }
};
#else
struct RunningMinV {
  alignas(64) float data[64];
  void set_max() {
    for (int i = 0; i < 64; ++i) data[i] = std::numeric_limits<float>::max();
  }
  static RunningMinV max() {
    RunningMinV r;
    r.set_max();
    return r;
  }
  inline float reduce() const {
    float best = std::numeric_limits<float>::max();
    for (int i = 0; i < 64; ++i)
      if (data[i] < best) best = data[i];
    return best;
  }
};
#endif

template<size_t BlockSize>
struct BlockSizeTraits {
  static_assert(BlockSize == 1 || BlockSize == 2 || BlockSize == 4 || BlockSize == 8 ||
                    BlockSize == 16,
                "BlockSize must be 1, 2, 4, 8, or 16.");
  static constexpr size_t value = BlockSize;
  static constexpr size_t K = 16;
  static inline float get_centroid_scale() {
    if constexpr (BlockSize == 1) return kPQ_Int8Scale_D1_K16;
    if constexpr (BlockSize == 2) return kPQ_Int8Scale_D2_K16;
    if constexpr (BlockSize == 4) return kPQ_Int8Scale_D4_K16;
    if constexpr (BlockSize == 8) return kPQ_Int8Scale_D8_K16;
    if constexpr (BlockSize == 16) return kPQ_Int8Scale_D16_K16;
    return 1.0f;
  }
};

namespace internal {
static constexpr size_t kStripSize = 64;
// Clamp used in TurboQuant (matches turboquant_4bit/byte).
static constexpr float kValueCap = 3.91724f;

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
}  // namespace internal

template<bool Metric, size_t BlockSize>
class Quantized_Point;
template<typename PointRange, bool Metric, size_t BlockSize>
class Quantized_Point_Range;
template<bool Metric, size_t BlockSize>
class Quantized_Query;

template<bool Metric, size_t BlockSize>
class Quantized_Point {
 public:
  using Traits = BlockSizeTraits<BlockSize>;
  // FastScan-style: code_ptr points to strip_base + (lane_idx/2); for block b, code at
  // code_ptr[b*32].
  const uint8_t* code_ptr = nullptr;
  uint32_t lane_idx = 0;
  float norm_scaling_factor = 0.0f;
  float unquantized_squared_norm = 0.0f;

  Quantized_Point() = default;
  Quantized_Point(const uint8_t* ptr, uint32_t lane, float nsf, float usn) :
      code_ptr(ptr), lane_idx(lane), norm_scaling_factor(nsf), unquantized_squared_norm(usn) {}

  inline float distance(const Quantized_Query<Metric, BlockSize>& qq) const;
  void prefetch() const {
    if (code_ptr) __builtin_prefetch(code_ptr, 0, 3);
  }
  bool same_as(const Quantized_Point<Metric, BlockSize>&) const { return false; }
  bool same_as(const Quantized_Query<Metric, BlockSize>&) const { return false; }
  bool is_metric() const { return Metric; }
};

template<bool Metric, size_t BlockSize>
class Quantized_Query {
 public:
  using Traits = BlockSizeTraits<BlockSize>;
  using distanceType = float;
  static constexpr size_t K = 16;
  // Advertise batch distance support so the multi-vector wrapper can use the
  // optimized distances_slice / strip layout paths instead of per-point fallback.
  static constexpr bool has_batch_distances = true;
  size_t num_blocks = 0;
  size_t num_bytes_per_datapoint = 0;
  // Query-side LUT: raw int8×int8 dot products per block/centroid.
  // Entry (b, k) holds dot(q_block_b, centroid_b_k) as 32-bit integer.
  std::vector<int32_t> lut_int32;
  // Optional 8-bit compressed LUT for fast SIMD scoring. Values are scalar-
  // quantized versions of lut_int32 using a single global scale per query:
  //   lut_int32[b,k] ≈ lut_int8[b,k] * lut_int8_scale.
  std::vector<int8_t> lut_int8;
  float lut_int8_scale = 1.0f;
  // Query norm information (for potential rescaling); currently not used in
  // the scoring path, but kept for completeness and future extensions.
  float norm_scaling_factor = 0.0f;
  float unquantized_squared_norm = 0.0f;

  // sum(query_int8) for VNNI bias correction.
  int32_t byte_sum = 0;
  // (norm_scaling_factor * lut_int8_scale) / centroid_scale
  float alpha = 1.0f;
  float centroid_scale = 1.0f;

  Quantized_Query() = default;

  inline float distance(const Quantized_Point<Metric, BlockSize>& p) const {
    return p.distance(*this);
  }

  // Unified distances_all: uses optimized scan_64_chunk if available.
  template<typename EncRange>
  void distances_all(const EncRange& db, float* out) const {
    const size_t N = (db.n_points_raw_unpadded != 0) ? db.n_points_raw_unpadded : db.n_points_raw;
    if (N == 0) return;

    if constexpr (EncRange::is_tqpq_fast) {
      const size_t strip_stride = db.stride;
      const size_t n_strips = (N + 63) / 64;

#ifdef __AVX512F__
      const uint8_t* strip_data = db.packed_codes.data();
      const float* norms = db.norm_scaling_factors.data();
      const float* squared_norms = db.unquantized_squared_norms.data();
      const size_t nb = num_blocks;
      const __m256i low_mask = _mm256_set1_epi8(0x0F);
      const float alpha_val = alpha;
      const float usn_q = unquantized_squared_norm;

      parlay::parallel_for(
          0, n_strips,
          [&](size_t s) {
            const uint8_t* codes_ptr = strip_data + s * strip_stride;
            const float* ns = norms + s * 64;
            const float* sq = squared_norms + s * 64;
            const size_t base = s * 64;
            const size_t count = std::min<size_t>(64, N - base);

            __m512i acc_even = _mm512_setzero_si512();
            __m512i acc_odd = _mm512_setzero_si512();

            for (size_t b = 0; b < nb; ++b) {
              const __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(codes_ptr));
              codes_ptr += 32;
              const __m256i codes_even = _mm256_and_si256(packed, low_mask);
              const __m256i codes_odd = _mm256_and_si256(_mm256_srli_epi16(packed, 4), low_mask);
              const __m256i lut256 = _mm256_broadcastsi128_si256(
                  _mm_loadu_si128(reinterpret_cast<const __m128i*>(&lut_int8[b * 16])));
              const __m256i s_even = _mm256_shuffle_epi8(lut256, codes_even);
              const __m256i s_odd = _mm256_shuffle_epi8(lut256, codes_odd);
              acc_even = _mm512_add_epi16(acc_even, _mm512_cvtepi8_epi16(s_even));
              acc_odd = _mm512_add_epi16(acc_odd, _mm512_cvtepi8_epi16(s_odd));
            }

            alignas(64) int16_t r_even[32], r_odd[32];
            _mm512_store_si512(reinterpret_cast<__m512i*>(r_even), acc_even);
            _mm512_store_si512(reinterpret_cast<__m512i*>(r_odd), acc_odd);

            for (size_t i = 0; i < count; ++i) {
              const size_t pair = i / 2;
              const int16_t s8 = (i % 2 == 0) ? r_even[pair] : r_odd[pair];
              const float dot_est = ns[i] * alpha_val * static_cast<float>(s8);
              out[base + i] = Metric ? (sq[i] + usn_q - 2.0f * dot_est) : -dot_est;
            }
          },
          /*granularity=*/1);
#else
      parlay::parallel_for(
          0, n_strips,
          [&](size_t s) {
            const uint8_t* codes_ptr = db.packed_codes.data() + s * strip_stride;
            const float* ns = db.norm_scaling_factors.data() + s * 64;
            const float* sq = db.unquantized_squared_norms.data() + s * 64;
            const size_t base = s * 64;
            if (base + 64 <= N) {
              db.scan_64_chunk(*this, codes_ptr, out + base, ns, sq);
            } else {
              alignas(64) float tmp[64];
              db.scan_64_chunk(*this, codes_ptr, tmp, ns, sq);
              std::memcpy(out + base, tmp, (N - base) * sizeof(float));
            }
          },
          /*granularity=*/1);
#endif
    } else {
      parlay::parallel_for(
          0, N, [&](size_t i) { out[i] = db[static_cast<size_t>(i)].distance(*this); },
          /*granularity=*/1024);
    }
  }

  // Compute distances for a contiguous slice [start, start+count) of an encoded range.
  // Used by the multi-vector wrapper to compute per-cloud Chamfer efficiently.
  template<typename PointRangeTy>
  inline void distances_slice(const Quantized_Point_Range<PointRangeTy, Metric, BlockSize>& enc,
                              size_t start, size_t count, float* out) const {
    if (count == 0) return;

    const size_t end = start + count;
    const size_t strip0 = start / 64;
    const size_t lane0 = start % 64;
    const size_t strip1 = end / 64;
    const size_t lane1 = end % 64;

    alignas(64) float buf64[64];
    const size_t strip_stride = enc.stride;
    const uint8_t* strip_data = enc.packed_codes.data();
    const float* norms = enc.norm_scaling_factors.data();
    const float* squared_norms = enc.unquantized_squared_norms.data();

    size_t out_off = 0;
    if (strip0 == strip1) {
      enc.scan_64_chunk(*this, strip_data + strip0 * strip_stride, buf64, norms + strip0 * 64,
                        squared_norms + strip0 * 64);
      const size_t hi = (lane1 == 0) ? 64 : lane1;
      for (size_t lane = lane0; lane < hi; ++lane)
        out[out_off++] = buf64[lane];
      return;
    }

    // First partial strip.
    enc.scan_64_chunk(*this, strip_data + strip0 * strip_stride, buf64, norms + strip0 * 64,
                      squared_norms + strip0 * 64);
    for (size_t lane = lane0; lane < 64; ++lane)
      out[out_off++] = buf64[lane];

    // Full strips.
    for (size_t s = strip0 + 1; s < strip1; ++s) {
      enc.scan_64_chunk(*this, strip_data + s * strip_stride, out + out_off, norms + s * 64,
                        squared_norms + s * 64);
      out_off += 64;
    }

    // Last partial strip (if lane1 != 0).
    if (lane1 != 0) {
      enc.scan_64_chunk(*this, strip_data + strip1 * strip_stride, buf64, norms + strip1 * 64,
                        squared_norms + strip1 * 64);
      for (size_t lane = 0; lane < lane1; ++lane)
        out[out_off++] = buf64[lane];
    }
  }
};

template<bool Metric, size_t BlockSize>
inline float Quantized_Point<Metric, BlockSize>::distance(
    const Quantized_Query<Metric, BlockSize>& qq) const {
  int32_t acc = 0;
  for (size_t b = 0; b < qq.num_blocks; ++b) {
    uint8_t packed = code_ptr[b * 32];
    uint8_t code = (lane_idx % 2 == 0) ? (packed & 0x0F) : (packed >> 4);
    acc += qq.lut_int32[static_cast<size_t>(b) * Quantized_Query<Metric, BlockSize>::K +
                        static_cast<size_t>(code)];
  }
  // Convert raw integer dot-sum into an approximate inner product using
  // TurboQuant-style norm scaling factors. We treat acc as proportional to the
  // dot product in the rotated/scaled space; the NSFs map it back to the
  // original norms.
  const float acc_f = static_cast<float>(acc);
  const float alpha_x = norm_scaling_factor;
  const float alpha_q = qq.norm_scaling_factor;
  const float dot_est = (alpha_x * alpha_q * acc_f) / qq.centroid_scale;

  if constexpr (Metric) {
    // Approximate L2 distance: ||x - q||^2 ≈ ||x||^2 + ||q||^2 - 2 * dot_est.
    return qq.unquantized_squared_norm + unquantized_squared_norm - 2.0f * dot_est;
  } else {
    // IP mode: distance = -<x, q>.
    return -dot_est;
  }
}

template<typename PointRange, bool Metric, size_t BlockSize>
class Quantized_Point_Range {
 public:
  static constexpr bool is_fastscan = false;
  static constexpr bool is_tqpq_fast = true;
  size_t n_points_raw = 0, n_points_raw_unpadded = 0, dim = 0;
  size_t num_bytes_per_datapoint = 0, stride = 0;
  parlay::sequence<uint8_t> packed_codes;
  parlay::sequence<float> norm_scaling_factors;
  parlay::sequence<float> unquantized_squared_norms;
  parlay::sequence<size_t> cloud_vec_offsets;

  Quantized_Point<Metric, BlockSize> operator[](size_t i) const {
    const size_t strip_idx = i / 64;
    const size_t lane_idx = i % 64;
    const uint8_t* ptr = &packed_codes[strip_idx * stride + (lane_idx / 2)];
    return Quantized_Point<Metric, BlockSize>(ptr, static_cast<uint32_t>(lane_idx),
                                              norm_scaling_factors[i],
                                              Metric ? unquantized_squared_norms[i] : 0.0f);
  }
  uint32_t size() const noexcept { return static_cast<uint32_t>(n_points_raw_unpadded); }
  uint32_t get_dims() const noexcept { return static_cast<uint32_t>(dim); }

  void distances_all(const Quantized_Query<Metric, BlockSize>& q, float* out) const {
    q.distances_all(*this, out);
  }

  size_t num_blocks_for_scan() const { return stride / 32; }

#ifdef __AVX512F__
  // Optimized AVX-512 scan: dispatches to VNNI if available.
  void scan_64_chunk(const Quantized_Query<Metric, BlockSize>& q, const uint8_t* codes_ptr,
                     float* results, const float* ns_ptr, const float* sq_ptr) const {
#ifdef __AVX512VNNI__
    scan_64_chunk_vnni(q, codes_ptr, results, ns_ptr, sq_ptr);
#else
    scan_64_chunk_avx512_shuffle(q, codes_ptr, results, ns_ptr, sq_ptr);
#endif
  }

  void scan_64_chunk_avx512_shuffle(const Quantized_Query<Metric, BlockSize>& q,
                                   const uint8_t* codes_ptr, float* results, const float* ns_ptr,
                                   const float* sq_ptr) const {
    const size_t nb = num_blocks_for_scan();
    __m512i acc_even = _mm512_setzero_si512(), acc_odd = _mm512_setzero_si512();
    const __m256i low_mask = _mm256_set1_epi8(0x0F);
    for (size_t b = 0; b < nb; ++b) {
      const __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(codes_ptr));
      codes_ptr += 32;
      const __m256i codes_even = _mm256_and_si256(packed, low_mask);
      const __m256i codes_odd = _mm256_and_si256(_mm256_srli_epi16(packed, 4), low_mask);
      const __m128i lut128 = _mm_loadu_si128(
          reinterpret_cast<const __m128i*>(&q.lut_int8[static_cast<size_t>(b) * 16]));
      const __m256i lut256 = _mm256_broadcastsi128_si256(lut128);
      const __m256i scores_even_i8 = _mm256_shuffle_epi8(lut256, codes_even);
      const __m256i scores_odd_i8 = _mm256_shuffle_epi8(lut256, codes_odd);
      acc_even = _mm512_add_epi16(acc_even, _mm512_cvtepi8_epi16(scores_even_i8));
      acc_odd = _mm512_add_epi16(acc_odd, _mm512_cvtepi8_epi16(scores_odd_i8));
    }
    alignas(64) int16_t raw_even[32], raw_odd[32];
    _mm512_store_si512(reinterpret_cast<__m512i*>(raw_even), acc_even);
    _mm512_store_si512(reinterpret_cast<__m512i*>(raw_odd), acc_odd);
    const float alpha = q.alpha;
    for (int i = 0; i < 32; ++i) {
      const float dot0 = ns_ptr[2 * i] * alpha * static_cast<float>(raw_even[i]);
      const float dot1 = ns_ptr[2 * i + 1] * alpha * static_cast<float>(raw_odd[i]);
      if constexpr (Metric) {
        results[2 * i] = sq_ptr[2 * i] + q.unquantized_squared_norm - 2.0f * dot0;
        results[2 * i + 1] = sq_ptr[2 * i + 1] + q.unquantized_squared_norm - 2.0f * dot1;
      } else {
        results[2 * i] = -dot0;
        results[2 * i + 1] = -dot1;
      }
    }
  }

  void scan_64_chunk_vnni(const Quantized_Query<Metric, BlockSize>& q, const uint8_t* codes_ptr,
                         float* results, const float* ns_ptr, const float* sq_ptr) const {
    const size_t nb = num_blocks_for_scan();
    alignas(64) uint8_t onehot[64 * 16];
    alignas(64) int32_t acc32[64];
    for (int i = 0; i < 64; ++i) acc32[i] = 0;
    for (size_t b = 0; b < nb; ++b) {
      for (int i = 0; i < 32; ++i) {
        uint8_t byte = codes_ptr[i], lo = byte & 0x0F, hi = byte >> 4;
        std::memcpy(onehot + (2 * i) * 16, internal::kOneHot[lo], 16);
        std::memcpy(onehot + (2 * i + 1) * 16, internal::kOneHot[hi], 16);
      }
      codes_ptr += 32;
      const int8_t* lut = &q.lut_int8[b * 16];
      for (int p = 0; p < 4; ++p) {
        __m512i b_acc = _mm512_setzero_si512();
        for (int t = 0; t < 4; ++t) {
          const __m512i a = _mm512_loadu_si512(onehot + p * 256 + t * 64);
          const __m512i bv = _mm512_set1_epi32(*reinterpret_cast<const int32_t*>(lut + t * 4));
          b_acc = _mm512_dpbusd_epi32(b_acc, a, bv);
        }
        __m512i run = _mm512_loadu_si512(acc32 + p * 16);
        _mm512_storeu_si512(acc32 + p * 16, _mm512_add_epi32(run, b_acc));
      }
    }
    const float alpha = q.alpha;
    for (int i = 0; i < 64; ++i) {
      const float dot = ns_ptr[i] * alpha * static_cast<float>(acc32[i]);
      results[i] = Metric ? (sq_ptr[i] + q.unquantized_squared_norm - 2.0f * dot) : -dot;
    }
  }

  void scan_64_running_min(const Quantized_Query<Metric, BlockSize>& q, const uint8_t* codes_ptr,
                           const float* ns_ptr, const float* sq_ptr,
                           RunningMinV& cur_min) const {
    alignas(64) float res[64];
    scan_64_chunk(q, codes_ptr, res, ns_ptr, sq_ptr);
    for (int i = 0; i < 4; ++i)
      cur_min.v[i] = _mm512_min_ps(cur_min.v[i], _mm512_load_ps(res + i * 16));
  }

  void scan_64_dual_query(const Quantized_Query<Metric, BlockSize>& q1,
                         const Quantized_Query<Metric, BlockSize>& q2, const uint8_t* codes_ptr,
                         const float* ns_ptr, const float* sq_ptr, RunningMinV& m1,
                         RunningMinV& m2) const {
    alignas(64) float r1[64], r2[64];
    scan_64_chunk(q1, codes_ptr, r1, ns_ptr, sq_ptr);
    scan_64_chunk(q2, codes_ptr, r2, ns_ptr, sq_ptr);
    for (int i = 0; i < 4; ++i) {
      m1.v[i] = _mm512_min_ps(m1.v[i], _mm512_load_ps(r1 + i * 16));
      m2.v[i] = _mm512_min_ps(m2.v[i], _mm512_load_ps(r2 + i * 16));
    }
  }
#elif defined(__AVX2__)
  void scan_64_chunk(const Quantized_Query<Metric, BlockSize>& q, const uint8_t* codes_ptr,
                     float* results, const float* ns_ptr, const float* sq_ptr) const {
    const size_t nb = num_blocks_for_scan();
    __m256i acc_even_lo = _mm256_setzero_si256(), acc_even_hi = _mm256_setzero_si256();
    __m256i acc_odd_lo = _mm256_setzero_si256(), acc_odd_hi = _mm256_setzero_si256();
    const __m256i low_mask = _mm256_set1_epi8(0x0F);
    for (size_t b = 0; b < nb; ++b) {
      const __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(codes_ptr));
      codes_ptr += 32;
      const __m256i c_even = _mm256_and_si256(packed, low_mask);
      const __m256i c_odd = _mm256_and_si256(_mm256_srli_epi16(packed, 4), low_mask);
      const __m256i lut256 = _mm256_broadcastsi128_si256(
          _mm_loadu_si128(reinterpret_cast<const __m128i*>(&q.lut_int8[b * 16])));
      const __m256i s_even = _mm256_shuffle_epi8(lut256, c_even);
      const __m256i s_odd = _mm256_shuffle_epi8(lut256, c_odd);
      acc_even_lo =
          _mm256_add_epi16(acc_even_lo, _mm256_cvtepi8_epi16(_mm256_castsi256_si128(s_even)));
      acc_even_hi =
          _mm256_add_epi16(acc_even_hi, _mm256_cvtepi8_epi16(_mm256_extracti128_si256(s_even, 1)));
      acc_odd_lo = _mm256_add_epi16(acc_odd_lo, _mm256_cvtepi8_epi16(_mm256_castsi256_si128(s_odd)));
      acc_odd_hi =
          _mm256_add_epi16(acc_odd_hi, _mm256_cvtepi8_epi16(_mm256_extracti128_si256(s_odd, 1)));
    }
    alignas(64) int16_t r_even[32], r_odd[32];
    _mm256_store_si256(reinterpret_cast<__m256i*>(r_even), acc_even_lo);
    _mm256_store_si256(reinterpret_cast<__m256i*>(r_even + 16), acc_even_hi);
    _mm256_store_si256(reinterpret_cast<__m256i*>(r_odd), acc_odd_lo);
    _mm256_store_si256(reinterpret_cast<__m256i*>(r_odd + 16), acc_odd_hi);
    const float alpha = q.alpha;
    for (int i = 0; i < 32; ++i) {
      float d0 = ns_ptr[2 * i] * alpha * r_even[i], d1 = ns_ptr[2 * i + 1] * alpha * r_odd[i];
      results[2 * i] = Metric ? (sq_ptr[2 * i] + q.unquantized_squared_norm - 2.0f * d0) : -d0;
      results[2 * i + 1] =
          Metric ? (sq_ptr[2 * i + 1] + q.unquantized_squared_norm - 2.0f * d1) : -d1;
    }
  }

  void scan_64_running_min(const Quantized_Query<Metric, BlockSize>& q, const uint8_t* codes_ptr,
                           const float* ns_ptr, const float* sq_ptr,
                           RunningMinV& cur_min) const {
    alignas(64) float res[64];
    scan_64_chunk(q, codes_ptr, res, ns_ptr, sq_ptr);
    for (int i = 0; i < 8; ++i) cur_min.v[i] = _mm256_min_ps(cur_min.v[i], _mm256_load_ps(res + i * 8));
  }

  void scan_64_dual_query(const Quantized_Query<Metric, BlockSize>& q1,
                         const Quantized_Query<Metric, BlockSize>& q2, const uint8_t* codes_ptr,
                         const float* ns_ptr, const float* sq_ptr, RunningMinV& m1,
                         RunningMinV& m2) const {
    alignas(64) float r1[64], r2[64];
    scan_64_chunk(q1, codes_ptr, r1, ns_ptr, sq_ptr);
    scan_64_chunk(q2, codes_ptr, r2, ns_ptr, sq_ptr);
    for (int i = 0; i < 8; ++i) {
      m1.v[i] = _mm256_min_ps(m1.v[i], _mm256_load_ps(r1 + i * 8));
      m2.v[i] = _mm256_min_ps(m2.v[i], _mm256_load_ps(r2 + i * 8));
    }
  }
#else
  void scan_64_chunk(const Quantized_Query<Metric, BlockSize>& q, const uint8_t* codes_ptr,
                     float* results, const float* ns_ptr, const float* sq_ptr) const {
    const size_t nb = num_blocks_for_scan();
    for (int lane = 0; lane < 64; ++lane) {
      int32_t acc = 0;
      for (size_t b = 0; b < nb; ++b) {
        uint8_t packed = codes_ptr[b * 32 + (lane / 2)];
        uint8_t code = (lane % 2 == 0) ? (packed & 0x0F) : (packed >> 4);
        acc += q.lut_int32[b * 16 + code];
      }
      const float dot =
          (static_cast<float>(acc) * q.norm_scaling_factor * ns_ptr[lane]) / q.centroid_scale;
      results[lane] = Metric ? (sq_ptr[lane] + q.unquantized_squared_norm - 2.0f * dot) : -dot;
    }
  }

  void scan_64_running_min(const Quantized_Query<Metric, BlockSize>& q, const uint8_t* codes_ptr,
                           const float* ns_ptr, const float* sq_ptr,
                           RunningMinV& cur_min) const {
    alignas(64) float res[64];
    scan_64_chunk(q, codes_ptr, res, ns_ptr, sq_ptr);
    for (int i = 0; i < 64; ++i)
      if (res[i] < cur_min.data[i]) cur_min.data[i] = res[i];
  }

  void scan_64_dual_query(const Quantized_Query<Metric, BlockSize>& q1,
                         const Quantized_Query<Metric, BlockSize>& q2, const uint8_t* codes_ptr,
                         const float* ns_ptr, const float* sq_ptr, RunningMinV& m1,
                         RunningMinV& m2) const {
    scan_64_running_min(q1, codes_ptr, ns_ptr, sq_ptr, m1);
    scan_64_running_min(q2, codes_ptr, ns_ptr, sq_ptr, m2);
  }
#endif

  void save(std::ostream& out) const {
    out.write(reinterpret_cast<const char*>(&n_points_raw), sizeof(n_points_raw));
    out.write(reinterpret_cast<const char*>(&n_points_raw_unpadded), sizeof(n_points_raw_unpadded));
    out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
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
    size_t osz = cloud_vec_offsets.size();
    out.write(reinterpret_cast<const char*>(&osz), sizeof(osz));
    if (osz)
      out.write(reinterpret_cast<const char*>(cloud_vec_offsets.data()), osz * sizeof(size_t));
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
    if (ssz)
      in.read(reinterpret_cast<char*>(unquantized_squared_norms.data()), ssz * sizeof(float));
    size_t osz = 0;
    in.read(reinterpret_cast<char*>(&osz), sizeof(osz));
    cloud_vec_offsets.resize(osz);
    if (osz) in.read(reinterpret_cast<char*>(cloud_vec_offsets.data()), osz * sizeof(size_t));
  }
};

template<bool Metric, size_t BlockSize>
class Model {
 public:
  using Traits = BlockSizeTraits<BlockSize>;
  static constexpr bool is_fastscan = false;
  static constexpr size_t block_size = BlockSize;
  static constexpr size_t K = Traits::K;

  size_t dim = 0, padded_dim = 0, num_blocks = 0, num_bytes_per_datapoint = 0, seed_ = 42;
  std::vector<float> signs;
  rabitqlib::RotatorType rotator_type = rabitqlib::RotatorType::FhtKacRotator;
  std::unique_ptr<rabitqlib::Rotator<float>> rotator;
  // Single shared codebook for all blocks (same precomputed centroids everywhere).
  Eigen::MatrixXf codebook_;
  Eigen::VectorXf codebook_norms_;
  std::vector<int8_t> codebook_int8_;  // K * BlockSize, row-major
  float codebook_scale_ = 1.0f;

  Model() = default;

  template<typename PointRange>
  void train(const PointRange& data) {
    dim = data.get_dims();
    if (dim == 0) return;
    rotator.reset(rabitqlib::choose_rotator<float>(dim, rotator_type));
    if (!rotator) return;
    padded_dim = rotator->size();
    if (padded_dim % (2 * BlockSize) != 0) {
      std::cerr << "turboquant_pq_4bit: padded_dim must be divisible by 2*BlockSize\n";
      return;
    }
    num_blocks = padded_dim / BlockSize;
    num_bytes_per_datapoint = num_blocks / 2;
    signs.resize(padded_dim);
    std::mt19937 gen(seed_);
    std::uniform_int_distribution<> dist(0, 1);
    for (size_t i = 0; i < padded_dim; ++i)
      signs[i] = (2.0f * dist(gen) - 1.0f);
    // Single shared codebook (same precomputed centroids for all blocks).
    codebook_ = Eigen::MatrixXf(static_cast<Eigen::Index>(K), static_cast<Eigen::Index>(BlockSize));
    for (size_t k = 0; k < K; ++k) {
      for (size_t d = 0; d < BlockSize; ++d) {
        if constexpr (BlockSize == 1) {
          codebook_(static_cast<Eigen::Index>(k), static_cast<Eigen::Index>(d)) =
              kPQ_CentroidsFloat_D1_K16[k][d];
        } else if constexpr (BlockSize == 2) {
          codebook_(static_cast<Eigen::Index>(k), static_cast<Eigen::Index>(d)) =
              kPQ_CentroidsFloat_D2_K16[k][d];
        } else if constexpr (BlockSize == 4) {
          codebook_(static_cast<Eigen::Index>(k), static_cast<Eigen::Index>(d)) =
              kPQ_CentroidsFloat_D4_K16[k][d];
        } else if constexpr (BlockSize == 8) {
          codebook_(static_cast<Eigen::Index>(k), static_cast<Eigen::Index>(d)) =
              kPQ_CentroidsFloat_D8_K16[k][d];
        } else if constexpr (BlockSize == 16) {
          codebook_(static_cast<Eigen::Index>(k), static_cast<Eigen::Index>(d)) =
              kPQ_CentroidsFloat_D16_K16[k][d];
        } else {
          codebook_(static_cast<Eigen::Index>(k), static_cast<Eigen::Index>(d)) = 0.0f;
        }
      }
    }
    codebook_norms_ = codebook_.rowwise().squaredNorm();

    if constexpr (BlockSize == 1) {
      codebook_scale_ = kPQ_Int8Scale_D1_K16;
    } else if constexpr (BlockSize == 2) {
      codebook_scale_ = kPQ_Int8Scale_D2_K16;
    } else if constexpr (BlockSize == 4) {
      codebook_scale_ = kPQ_Int8Scale_D4_K16;
    } else if constexpr (BlockSize == 8) {
      codebook_scale_ = kPQ_Int8Scale_D8_K16;
    } else if constexpr (BlockSize == 16) {
      codebook_scale_ = kPQ_Int8Scale_D16_K16;
    } else {
      codebook_scale_ = 1.0f;
    }
    codebook_int8_.resize(static_cast<size_t>(K * BlockSize));
    for (size_t k = 0; k < K; ++k) {
      for (size_t d = 0; d < BlockSize; ++d) {
        if constexpr (BlockSize == 1) {
          codebook_int8_[k * BlockSize + d] = kPQ_CentroidsInt8_D1_K16[k][d];
        } else if constexpr (BlockSize == 2) {
          codebook_int8_[k * BlockSize + d] = kPQ_CentroidsInt8_D2_K16[k][d];
        } else if constexpr (BlockSize == 4) {
          codebook_int8_[k * BlockSize + d] = kPQ_CentroidsInt8_D4_K16[k][d];
        } else if constexpr (BlockSize == 8) {
          codebook_int8_[k * BlockSize + d] = kPQ_CentroidsInt8_D8_K16[k][d];
        } else if constexpr (BlockSize == 16) {
          codebook_int8_[k * BlockSize + d] = kPQ_CentroidsInt8_D16_K16[k][d];
        } else {
          codebook_int8_[k * BlockSize + d] = 0;
        }
      }
    }
  }

  std::pair<float, float> encode_single(const float* p, uint8_t* output,
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
    const float hadamard_scale = std::sqrt(static_cast<float>(padded_dim));
    for (size_t i = 0; i < padded_dim; ++i)
      ws[i] *= inv_norm * hadamard_scale * signs[i];

    float quantized_sq_norm = 0.0f;
    for (size_t b = 0; b < num_blocks; ++b) {
      Eigen::Map<const Eigen::VectorXf> block(ws.data() + b * BlockSize,
                                              static_cast<Eigen::Index>(BlockSize));
      Eigen::VectorXf dots = codebook_ * block;
      Eigen::Index best = 0;
      // ALWAYS use L2 assignment to find the closest centroid
      float best_val = codebook_norms_[0] - 2.0f * dots[0];
      for (Eigen::Index c = 1; c < static_cast<Eigen::Index>(K); ++c) {
        float v = codebook_norms_[c] - 2.0f * dots[c];
        if (v < best_val) {
          best_val = v;
          best = c;
        }
      }
      uint8_t code = static_cast<uint8_t>(best);
      quantized_sq_norm += codebook_norms_[best];
      if (b % 2 == 0)
        output[b / 2] = code;
      else
        output[b / 2] |= (code << 4);
    }
    return {sqr_norm, norm / std::sqrt(quantized_sq_norm)};
  }

  template<typename PointRangeTy>
  Quantized_Point_Range<PointRangeTy, Metric, BlockSize> encode(const PointRangeTy& data) const {
    if (!rotator) return Quantized_Point_Range<PointRangeTy, Metric, BlockSize>();
    const size_t N = data.size();
    const size_t N_padded = ((N + 63) / 64) * 64;
    const size_t n_strips = N_padded / 64;
    const size_t strip_stride = num_blocks * 32;  // FastScan layout
    Quantized_Point_Range<PointRangeTy, Metric, BlockSize> enc;
    enc.dim = padded_dim;
    enc.num_bytes_per_datapoint = num_bytes_per_datapoint;
    enc.stride = strip_stride;
    enc.n_points_raw = N_padded;
    enc.n_points_raw_unpadded = N;
    enc.cloud_vec_offsets = {0, N_padded};
    enc.packed_codes.resize(n_strips * strip_stride, 0);
    enc.norm_scaling_factors.resize(N_padded, 0.0f);
    enc.unquantized_squared_norms.resize(N_padded, 0.0f);

    parlay::sequence<uint8_t> point_codes_temp(N * num_bytes_per_datapoint, 0);
    struct Ws {
      std::vector<float> rot;
      std::vector<uint8_t> point_codes;
      void ensure(size_t pdim, size_t nb) {
        if (rot.size() != pdim) rot.resize(pdim);
        if (point_codes.size() != nb) point_codes.resize(nb);
      }
    };
    parlay::parallel_for(0, N, [&](size_t vi) {
      static thread_local Ws ws;
      ws.ensure(padded_dim, num_bytes_per_datapoint);
      const float* p = reinterpret_cast<const float*>(data.location(vi));
      auto [sqn, nsf] = encode_single(p, ws.point_codes.data(), ws.rot);
      enc.norm_scaling_factors[vi] = nsf;
      enc.unquantized_squared_norms[vi] = sqn;
      for (size_t j = 0; j < num_bytes_per_datapoint; ++j)
        point_codes_temp[vi * num_bytes_per_datapoint + j] = ws.point_codes[j];
    });

    auto get_nibble = [](const uint8_t* pc, size_t b) -> uint8_t {
      return static_cast<uint8_t>((pc[b / 2] >> (4 * (b % 2))) & 0x0F);
    };
    parlay::parallel_for(0, n_strips * 32, [&](size_t idx) {
      const size_t s = idx / 32;
      const size_t lane_pair = idx % 32;
      const size_t vi0 = s * 64 + lane_pair * 2;
      const size_t vi1 = vi0 + 1;
      const uint8_t* pc0 = point_codes_temp.data() + vi0 * num_bytes_per_datapoint;
      const uint8_t* pc1 = (vi1 < N) ? point_codes_temp.data() + vi1 * num_bytes_per_datapoint
                                     : point_codes_temp.data();
      uint8_t* strip_base = enc.packed_codes.data() + s * strip_stride;
      for (size_t b = 0; b < num_blocks; ++b)
        strip_base[b * 32 + lane_pair] =
            (get_nibble(pc0, b) & 0x0F) | ((vi1 < N ? get_nibble(pc1, b) : 0u) << 4);
    });
    return enc;
  }

  template<typename PointRangeTy, typename SeqOffsetsFloat>
  Quantized_Point_Range<PointRangeTy, Metric, BlockSize> encode(
      const PointRangeTy& data, const SeqOffsetsFloat& cloud_offsets_float) const {
    if (!rotator) return Quantized_Point_Range<PointRangeTy, Metric, BlockSize>();
    const size_t D = padded_dim;
    const size_t n_clouds = cloud_offsets_float.size() > 0 ? cloud_offsets_float.size() - 1 : 0;
    const size_t strip_stride = num_blocks * 32;
    Quantized_Point_Range<PointRangeTy, Metric, BlockSize> enc;
    enc.dim = padded_dim;
    enc.num_bytes_per_datapoint = num_bytes_per_datapoint;
    enc.stride = strip_stride;
    enc.n_points_raw_unpadded = data.size();
    enc.cloud_vec_offsets.resize(n_clouds + 1);
    size_t cur = 0;
    for (size_t c = 0; c < n_clouds; ++c) {
      enc.cloud_vec_offsets[c] = cur;
      const size_t start_f = static_cast<size_t>(cloud_offsets_float[c]);
      const size_t end_f = static_cast<size_t>(cloud_offsets_float[c + 1]);
      cur += ((end_f - start_f) / D + 63) / 64 * 64;
    }
    enc.cloud_vec_offsets[n_clouds] = cur;
    enc.n_points_raw = cur;
    enc.packed_codes.resize((cur / 64) * strip_stride, 0);
    enc.norm_scaling_factors.resize(cur, 0.0f);
    enc.unquantized_squared_norms.resize(cur, 0.0f);

    parlay::sequence<uint8_t> point_codes_temp(cur * num_bytes_per_datapoint, 0);
    struct Ws {
      std::vector<float> rot;
      std::vector<uint8_t> point_codes;
      void ensure(size_t pdim, size_t nb) {
        if (rot.size() != pdim) rot.resize(pdim);
        if (point_codes.size() != nb) point_codes.resize(nb);
      }
    };
    for (size_t c = 0; c < n_clouds; ++c) {
      const size_t start_f = static_cast<size_t>(cloud_offsets_float[c]);
      const size_t end_f = static_cast<size_t>(cloud_offsets_float[c + 1]);
      const size_t src_start = start_f / D;
      const size_t n_vecs = (end_f - start_f) / D;
      const size_t dst_start = enc.cloud_vec_offsets[c];
      parlay::parallel_for(0, n_vecs, [&](size_t i) {
        static thread_local Ws ws;
        ws.ensure(padded_dim, num_bytes_per_datapoint);
        const float* p = reinterpret_cast<const float*>(data.location(src_start + i));
        auto [sqn, nsf] = encode_single(p, ws.point_codes.data(), ws.rot);
        const size_t dst_idx = dst_start + i;
        enc.norm_scaling_factors[dst_idx] = nsf;
        enc.unquantized_squared_norms[dst_idx] = sqn;
        for (size_t j = 0; j < num_bytes_per_datapoint; ++j)
          point_codes_temp[dst_idx * num_bytes_per_datapoint + j] = ws.point_codes[j];
      });
    }
    auto get_nibble = [](const uint8_t* pc, size_t b) -> uint8_t {
      return static_cast<uint8_t>((pc[b / 2] >> (4 * (b % 2))) & 0x0F);
    };
    const size_t n_strips = cur / 64;
    parlay::parallel_for(0, n_strips * 32, [&](size_t idx) {
      const size_t s = idx / 32;
      const size_t lane_pair = idx % 32;
      const size_t vi0 = s * 64 + lane_pair * 2;
      const size_t vi1 = vi0 + 1;
      const uint8_t* pc0 = point_codes_temp.data() + vi0 * num_bytes_per_datapoint;
      const uint8_t* pc1 = (vi1 < cur) ? point_codes_temp.data() + vi1 * num_bytes_per_datapoint
                                       : point_codes_temp.data();
      uint8_t* strip_base = enc.packed_codes.data() + s * strip_stride;
      for (size_t b = 0; b < num_blocks; ++b)
        strip_base[b * 32 + lane_pair] =
            (get_nibble(pc0, b) & 0x0F) | ((vi1 < cur ? get_nibble(pc1, b) : 0u) << 4);
    });
    return enc;
  }

  Quantized_Query<Metric, BlockSize> quantize_query(const float* qptr) const {
    if (!rotator) return Quantized_Query<Metric, BlockSize>();
    Quantized_Query<Metric, BlockSize> qq;
    qq.num_blocks = num_blocks;
    qq.num_bytes_per_datapoint = num_bytes_per_datapoint;
    qq.centroid_scale = codebook_scale_;
    qq.lut_int32.assign(num_blocks * 16, 0);
    qq.lut_int8.assign(num_blocks * 16, 0);
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
    const float hadamard_scale = std::sqrt(static_cast<float>(padded_dim));
    for (size_t i = 0; i < padded_dim; ++i)
      q_rot[i] *= (1.0f / norm) * hadamard_scale * signs[i];

    // Clamp in TurboQuant space, same as turboquant_4bit.
    for (size_t i = 0; i < padded_dim; ++i)
      q_rot[i] = std::clamp(q_rot[i], -internal::kValueCap, internal::kValueCap);

    qq.unquantized_squared_norm = sqr_norm;

    std::vector<int8_t> q_int8(padded_dim);

    // Use a single global int8 scale for the whole query, like turboquant_4bit.
    float max_value = 1e-6f;
    for (size_t i = 0; i < padded_dim; ++i)
      max_value = std::max(max_value, std::abs(q_rot[i]));
    const float s_q_global = 127.0f / max_value;

    int32_t total_byte_sum = 0;
    const int8_t* c_all = codebook_int8_.data();
    for (size_t b = 0; b < num_blocks; ++b) {
      for (size_t d = 0; d < BlockSize; ++d) {
        const float v = q_rot[b * BlockSize + d] * s_q_global;
        int8_t snapped = static_cast<int8_t>(std::clamp(std::round(v), -127.0f, 127.0f));
        q_int8[b * BlockSize + d] = snapped;
        total_byte_sum += static_cast<int32_t>(snapped);
      }
      const int8_t* q_b = q_int8.data() + b * BlockSize;
      for (size_t k = 0; k < K; ++k) {
        int32_t dot = 0;
        for (size_t d = 0; d < BlockSize; ++d)
          dot += static_cast<int32_t>(q_b[d]) * static_cast<int32_t>(c_all[k * BlockSize + d]);
        qq.lut_int32[b * 16 + k] = dot;
      }
    }
    qq.byte_sum = total_byte_sum;

    // Build an 8-bit compressed LUT from lut_int32 for fast SIMD scoring.
    int32_t max_abs = 0;
    for (size_t i = 0; i < qq.lut_int32.size(); ++i)
      max_abs = std::max<int32_t>(max_abs, std::abs(qq.lut_int32[i]));
    if (max_abs == 0) {
      qq.lut_int8_scale = 1.0f;
      std::fill(qq.lut_int8.begin(), qq.lut_int8.end(), 0);
    } else {
      qq.lut_int8_scale = static_cast<float>(max_abs) / 127.0f;
      const float inv_scale = 1.0f / qq.lut_int8_scale;
      for (size_t i = 0; i < qq.lut_int8.size(); ++i) {
        float v = static_cast<float>(qq.lut_int32[i]) * inv_scale;
        v = std::round(std::clamp(v, -127.0f, 127.0f));
        qq.lut_int8[i] = static_cast<int8_t>(v);
      }
    }

    // Compute a query-side norm scaling factor analogous to TurboQuant's NSF.
    int64_t quant_norm_q = 0;
    for (size_t i = 0; i < padded_dim; ++i) {
      int32_t v = static_cast<int32_t>(q_int8[i]);
      quant_norm_q += static_cast<int64_t>(v) * static_cast<int64_t>(v);
    }
    if (quant_norm_q > 0) {
      qq.norm_scaling_factor = norm / std::sqrt(static_cast<float>(quant_norm_q));
    } else {
      qq.norm_scaling_factor = 0.0f;
    }

    qq.alpha = (qq.norm_scaling_factor * qq.lut_int8_scale) / qq.centroid_scale;
    return qq;
  }

  template<typename PointTy>
  typename std::enable_if<!std::is_pointer<PointTy>::value,
                          Quantized_Query<Metric, BlockSize>>::type
  quantize_query(const PointTy& query) const {
    std::vector<float> tmp(dim);
    for (size_t i = 0; i < dim; ++i)
      tmp[i] = query[i];
    return quantize_query(tmp.data());
  }

  template<typename PointCloudTy>
  void quantize_query_batch(const PointCloudTy& qc,
                            parlay::sequence<Quantized_Query<Metric, BlockSize>>& out) const {
    out.clear();
    out.reserve(qc.size());
    const float* base = qc.data();
    for (uint32_t i = 0; i < qc.size(); ++i)
      out.emplace_back(quantize_query(base + static_cast<size_t>(i) * qc.get_dims()));
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
    size_t nb = 1;
    out.write(reinterpret_cast<const char*>(&nb), sizeof(nb));
    Eigen::Index rows = codebook_.rows(), cols = codebook_.cols();
    out.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    out.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
    out.write(reinterpret_cast<const char*>(codebook_.data()), rows * cols * sizeof(float));
    size_t sz = codebook_int8_.size();
    out.write(reinterpret_cast<const char*>(&sz), sizeof(sz));
    out.write(reinterpret_cast<const char*>(codebook_int8_.data()), sz);
    size_t nsc = 1;
    out.write(reinterpret_cast<const char*>(&nsc), sizeof(nsc));
    out.write(reinterpret_cast<const char*>(&codebook_scale_), sizeof(codebook_scale_));
  }

  void load(std::ifstream& in) {
    in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    in.read(reinterpret_cast<char*>(&padded_dim), sizeof(padded_dim));
    in.read(reinterpret_cast<char*>(&seed_), sizeof(seed_));
    num_blocks = padded_dim / BlockSize;
    num_bytes_per_datapoint = num_blocks / 2;
    int rt = 0;
    in.read(reinterpret_cast<char*>(&rt), sizeof(rt));
    rotator_type = static_cast<rabitqlib::RotatorType>(rt);
    rotator.reset(rabitqlib::choose_rotator<float>(dim, rotator_type));
    if (rotator) rotator->load(in);
    size_t ss = 0;
    in.read(reinterpret_cast<char*>(&ss), sizeof(ss));
    signs.resize(ss);
    in.read(reinterpret_cast<char*>(signs.data()), ss * sizeof(float));
    size_t nb = 0;
    in.read(reinterpret_cast<char*>(&nb), sizeof(nb));
    for (size_t b = 0; b < nb; ++b) {
      Eigen::Index rows, cols;
      in.read(reinterpret_cast<char*>(&rows), sizeof(rows));
      in.read(reinterpret_cast<char*>(&cols), sizeof(cols));
      if (b == 0) {
        codebook_.resize(rows, cols);
        in.read(reinterpret_cast<char*>(codebook_.data()), rows * cols * sizeof(float));
        codebook_norms_ = codebook_.rowwise().squaredNorm();
      } else {
        std::vector<float> discard(static_cast<size_t>(rows * cols));
        in.read(reinterpret_cast<char*>(discard.data()), rows * cols * sizeof(float));
      }
    }
    for (size_t b = 0; b < nb; ++b) {
      size_t sz = 0;
      in.read(reinterpret_cast<char*>(&sz), sizeof(sz));
      if (b == 0) {
        codebook_int8_.resize(sz);
        in.read(reinterpret_cast<char*>(codebook_int8_.data()), sz);
      } else {
        std::vector<int8_t> discard(sz);
        in.read(reinterpret_cast<char*>(discard.data()), sz);
      }
    }
    size_t nsc = 0;
    in.read(reinterpret_cast<char*>(&nsc), sizeof(nsc));
    if (nsc > 0) {
      in.read(reinterpret_cast<char*>(&codebook_scale_), sizeof(codebook_scale_));
      for (size_t i = 1; i < nsc; ++i) {
        float discard;
        in.read(reinterpret_cast<char*>(&discard), sizeof(discard));
      }
    }
  }
};

}  // namespace turboquant_pq_4bit
}  // namespace mvsic