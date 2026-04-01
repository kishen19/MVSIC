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

  // 1. Standard FastScan Accumulation Loop
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

  // 2. Construct Lane-ID Registers
  // We need to check: lo <= (base_idx + 2*i + parity) < hi
  const __m256i v_lo = _mm256_set1_epi16(static_cast<short>(lo));
  const __m256i v_hi = _mm256_set1_epi16(static_cast<short>(hi));
  const __m256i v_step = _mm256_set1_epi16(2);

  // Indices for even lanes: 0, 2, 4 ... 30 and 32, 34 ... 62
  __m256i idx_even_lo = _mm256_set_epi16(30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0);
  __m256i idx_even_hi = _mm256_add_epi16(idx_even_lo, _mm256_set1_epi16(32));

  // Indices for odd lanes: 1, 3, 5 ... 31 and 33, 35 ... 63
  __m256i idx_odd_lo = _mm256_add_epi16(idx_even_lo, _mm256_set1_epi16(1));
  __m256i idx_odd_hi = _mm256_add_epi16(idx_even_hi, _mm256_set1_epi16(1));

  // 3. Create logical masks: (idx >= lo) AND (idx < hi)
  auto get_mask = [&](__m256i idx) {
    __m256i m_ge = _mm256_cmpgt_epi16(idx, v_lo);
    __m256i m_eq = _mm256_cmpeq_epi16(idx, v_lo);
    __m256i in_lo = _mm256_or_si256(m_ge, m_eq);
    __m256i in_hi = _mm256_cmpgt_epi16(v_hi, idx);
    return _mm256_and_si256(in_lo, in_hi);
  };

  const __m256i INF = _mm256_set1_epi16(static_cast<short>(0xFFFF));

  // 4. Apply Masks using Blend
  // _mm256_blendv_epi8(a, b, mask) picks b if MSB of mask is 1, else a.
  acc_even_lo = _mm256_blendv_epi8(INF, acc_even_lo, get_mask(idx_even_lo));
  acc_even_hi = _mm256_blendv_epi8(INF, acc_even_hi, get_mask(idx_even_hi));
  acc_odd_lo = _mm256_blendv_epi8(INF, acc_odd_lo, get_mask(idx_odd_lo));
  acc_odd_hi = _mm256_blendv_epi8(INF, acc_odd_hi, get_mask(idx_odd_hi));

  // 5. Final Horizontal Min Reduction
  __m256i m1 = _mm256_min_epu16(acc_even_lo, acc_even_hi);
  __m256i m2 = _mm256_min_epu16(acc_odd_lo, acc_odd_hi);
  return hmin_256_epu16(_mm256_min_epu16(m1, m2));
}

#else
// Scalar Fallback
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
#endif

// ------------------------------------------------------------------
// Fused FastScan Kernel
// ------------------------------------------------------------------
template<bool Metric>
inline void fastscan_mv_chamfer_fused(const uint8_t* fused_luts, const float* q_scales,
                                      const float* q_min_dists, size_t num_fused_embeddings,
                                      uint32_t num_blocks, const uint8_t* strip_data,
                                      size_t strip_stride, size_t start_vec, size_t cloud_size,
                                      float* out_dists) {

  if (cloud_size == 0) {
    for (size_t qi = 0; qi < num_fused_embeddings; ++qi)
      out_dists[qi] = std::numeric_limits<float>::max();
    return;
  }

  const size_t strip0 = start_vec / 64;
  const int lane0 = static_cast<int>(start_vec % 64);
  const size_t true_end = start_vec + cloud_size;
  const size_t strip1 = true_end / 64;
  const int lane1 = static_cast<int>(true_end % 64);
  const bool fully_aligned_full = (lane0 == 0) && (lane1 == 0) && (strip1 > strip0);

  auto strip_ptr = [&](size_t s) -> const uint8_t* { return strip_data + s * strip_stride; };

  // CACHE BLOCKING: Process queries in blocks of 64.
  // Keeps RunningMinV state (~4KB) and LUTS (~8KB) locked in L1 Cache.
  const size_t Q_CHUNK = 64;
  for (size_t q_start = 0; q_start < num_fused_embeddings; q_start += Q_CHUNK) {
    size_t q_end = std::min(q_start + Q_CHUNK, num_fused_embeddings);
    size_t q_count = q_end - q_start;

    std::vector<internal::RunningMinV> combined(q_count, internal::RunningMinV::max());
    std::vector<uint16_t> min_dist_raw(q_count, 0xFFFF);

    if (strip0 == strip1) {
      const int hi = (lane1 == 0) ? 64 : lane1;
      for (size_t qi = 0; qi < q_count; ++qi) {
        const uint8_t* q_lut = fused_luts + (q_start + qi) * num_blocks * 16;
        min_dist_raw[qi] =
            scan_64_chunk_min_masked(q_lut, strip_ptr(strip0), num_blocks, lane0, hi);
      }
    } else {
      if (fully_aligned_full) {
        // --- THE LOOP SWAP ---
        // DB Strip is Outer Loop (Pinned in L1). Queries are Inner Loop.
        for (size_t s = strip0; s < strip1; ++s) {
          const uint8_t* s_ptr = strip_ptr(s);
          __builtin_prefetch(strip_ptr(s + 1), 0, 3);  // Hide RAM latency
          for (size_t qi = 0; qi < q_count; ++qi) {
            const uint8_t* q_lut = fused_luts + (q_start + qi) * num_blocks * 16;
            scan_64_running_min(q_lut, s_ptr, num_blocks, combined[qi]);
          }
        }
        for (size_t qi = 0; qi < q_count; ++qi) {
          min_dist_raw[qi] = reduce_running_min(combined[qi]);
        }
      } else {
        // Unaligned Head
        for (size_t qi = 0; qi < q_count; ++qi) {
          const uint8_t* q_lut = fused_luts + (q_start + qi) * num_blocks * 16;
          min_dist_raw[qi] =
              scan_64_chunk_min_masked(q_lut, strip_ptr(strip0), num_blocks, lane0, 64);
        }
        // Main Body
        for (size_t s = strip0 + 1; s < strip1; ++s) {
          const uint8_t* s_ptr = strip_ptr(s);
          __builtin_prefetch(strip_ptr(s + 1), 0, 3);
          for (size_t qi = 0; qi < q_count; ++qi) {
            const uint8_t* q_lut = fused_luts + (q_start + qi) * num_blocks * 16;
            scan_64_running_min(q_lut, s_ptr, num_blocks, combined[qi]);
          }
        }
        for (size_t qi = 0; qi < q_count; ++qi) {
          uint16_t d_full = reduce_running_min(combined[qi]);
          if (d_full < min_dist_raw[qi]) min_dist_raw[qi] = d_full;
        }
        // Unaligned Tail
        if (lane1 != 0) {
          for (size_t qi = 0; qi < q_count; ++qi) {
            const uint8_t* q_lut = fused_luts + (q_start + qi) * num_blocks * 16;
            uint16_t tail =
                scan_64_chunk_min_masked(q_lut, strip_ptr(strip1), num_blocks, 0, lane1);
            if (tail < min_dist_raw[qi]) min_dist_raw[qi] = tail;
          }
        }
      }
    }

    // Write back final float distances
    for (size_t qi = 0; qi < q_count; ++qi) {
      size_t global_qi = q_start + qi;
      out_dists[global_qi] = (q_min_dists[global_qi] * static_cast<float>(num_blocks)) +
                             (static_cast<float>(min_dist_raw[qi]) * q_scales[global_qi]);
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
// end_idx is the exclusive logical end (start + unpadded cloud size), matching distances_all.
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

// Chamfer distance for one cloud; same SIMD path as distances_all (AVX-512 / AVX2 / scalar).
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

  // Flat memory arrays for all queries
  std::vector<uint8_t> flat_int_luts;  // Size: num_queries * num_blocks * 16
  std::vector<float> min_dists;        // Size: num_queries
  std::vector<float> scales;           // Size: num_queries

  Quantized_Query_Point_Cloud() = default;

  inline const uint8_t* get_lut(size_t qi) const {
    return flat_int_luts.data() + qi * num_blocks * 16;
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

// Shared Chamfer kernel: same strip / AVX path as historical distances_all inner loop.
template<bool Metric>
float fastscan_mv_chamfer_distance(const Quantized_Query_Point_Cloud<Metric>& q,
                                   const Quantized_Point_Cloud_Set<Metric>& db, size_t start,
                                   size_t true_end) {
  const size_t num_q = q.num_queries;
  if (num_q == 0) return 0.0f;
  if (true_end <= start) return std::numeric_limits<float>::max();

  const size_t strip_stride = static_cast<size_t>(db.num_blocks) * 32;
  const size_t strip0 = start / 64;
  const int lane0 = static_cast<int>(start % 64);
  const size_t strip1 = true_end / 64;
  const int lane1 = static_cast<int>(true_end % 64);

  auto strip_ptr = [&](size_t s) -> const uint8_t* { return &db.packed_codes[s * strip_stride]; };

  float total_chamfer = 0.0f;
  const bool fully_aligned_full = (lane0 == 0) && (lane1 == 0) && (strip1 > strip0);

  for (size_t qi = 0; qi < num_q; ++qi) {
    const uint8_t* q_lut = q.get_lut(qi);
    uint16_t min_dist_raw = 0xFFFF;

    if (strip0 == strip1) {
      const int hi = (lane1 == 0) ? 64 : lane1;
      min_dist_raw =
          internal::scan_64_chunk_min_masked(q_lut, strip_ptr(strip0), db.num_blocks, lane0, hi);
    } else {
      if (fully_aligned_full) {
        const size_t first_full = strip0;
        const size_t last_full = strip1 - 1;
        internal::RunningMinV min0 = internal::RunningMinV::max();
        internal::RunningMinV min1 = internal::RunningMinV::max();
        internal::RunningMinV min2 = internal::RunningMinV::max();
        internal::RunningMinV min3 = internal::RunningMinV::max();
        size_t s = first_full;
        for (; s + 3 <= last_full; s += 4) {
          const uint8_t* p0 = strip_ptr(s);
          internal::scan_64_running_min(q_lut, p0, db.num_blocks, min0);
          internal::scan_64_running_min(q_lut, p0 + strip_stride, db.num_blocks, min1);
          internal::scan_64_running_min(q_lut, p0 + 2 * strip_stride, db.num_blocks, min2);
          internal::scan_64_running_min(q_lut, p0 + 3 * strip_stride, db.num_blocks, min3);
        }
        internal::RunningMinV combined;
#ifdef __AVX512F__
        combined.v =
            _mm512_min_epu16(_mm512_min_epu16(min0.v, min1.v), _mm512_min_epu16(min2.v, min3.v));
#else
        combined.lo = _mm256_min_epu16(_mm256_min_epu16(min0.lo, min1.lo),
                                       _mm256_min_epu16(min2.lo, min3.lo));
        combined.hi = _mm256_min_epu16(_mm256_min_epu16(min0.hi, min1.hi),
                                       _mm256_min_epu16(min2.hi, min3.hi));
#endif
        for (; s <= last_full; ++s) {
          internal::scan_64_running_min(q_lut, strip_ptr(s), db.num_blocks, combined);
        }
        min_dist_raw = internal::reduce_running_min(combined);
      } else {
        min_dist_raw =
            internal::scan_64_chunk_min_masked(q_lut, strip_ptr(strip0), db.num_blocks, lane0, 64);

        const size_t first_full = strip0 + 1;
        const size_t last_full = strip1 - 1;
        if (first_full <= last_full) {
          internal::RunningMinV min0 = internal::RunningMinV::max();
          internal::RunningMinV min1 = internal::RunningMinV::max();
          internal::RunningMinV min2 = internal::RunningMinV::max();
          internal::RunningMinV min3 = internal::RunningMinV::max();
          size_t s = first_full;
          for (; s + 3 <= last_full; s += 4) {
            const uint8_t* p0 = strip_ptr(s);
            internal::scan_64_running_min(q_lut, p0, db.num_blocks, min0);
            internal::scan_64_running_min(q_lut, p0 + strip_stride, db.num_blocks, min1);
            internal::scan_64_running_min(q_lut, p0 + 2 * strip_stride, db.num_blocks, min2);
            internal::scan_64_running_min(q_lut, p0 + 3 * strip_stride, db.num_blocks, min3);
          }
          internal::RunningMinV combined;
#ifdef __AVX512F__
          combined.v =
              _mm512_min_epu16(_mm512_min_epu16(min0.v, min1.v), _mm512_min_epu16(min2.v, min3.v));
#else
          combined.lo = _mm256_min_epu16(_mm256_min_epu16(min0.lo, min1.lo),
                                         _mm256_min_epu16(min2.lo, min3.lo));
          combined.hi = _mm256_min_epu16(_mm256_min_epu16(min0.hi, min1.hi),
                                         _mm256_min_epu16(min2.hi, min3.hi));
#endif
          for (; s <= last_full; ++s) {
            internal::scan_64_running_min(q_lut, strip_ptr(s), db.num_blocks, combined);
          }
          uint16_t d_full = internal::reduce_running_min(combined);
          if (d_full < min_dist_raw) min_dist_raw = d_full;
        }

        if (lane1 != 0) {
          uint16_t tail_dist =
              internal::scan_64_chunk_min_masked(q_lut, strip_ptr(strip1), db.num_blocks, 0, lane1);
          if (tail_dist < min_dist_raw) min_dist_raw = tail_dist;
        }
      }
    }
    total_chamfer += q.decode(qi, min_dist_raw);
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
      // Always apply L2-based kmeans, as sub-vectors are not L2-normalized
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
    const size_t strip_stride = num_blocks * 32;
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
    });

    return res;
  }

  template<typename ChPoint>
  Quantized_Query_Point_Cloud<Metric> quantize_query(const ChPoint& query_cloud) const {
    Quantized_Query_Point_Cloud<Metric> res;
    res.num_queries = query_cloud.size();
    res.num_blocks = num_blocks;

    if (res.num_queries == 0) return res;

    res.flat_int_luts.resize(res.num_queries * num_blocks * 16, 0);
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

      uint8_t* out_lut = res.flat_int_luts.data() + qi * num_blocks * 16;
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
// ManyToMany Batch Operator (FastScan)
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

    const size_t Q_BLOCK = 16;

    parlay::parallel_for(0, (num_q_clouds + Q_BLOCK - 1) / Q_BLOCK, [&](size_t qb) {
      size_t q_start = qb * Q_BLOCK;
      size_t q_end = std::min(q_start + Q_BLOCK, num_q_clouds);
      size_t q_count = q_end - q_start;

      std::vector<std::priority_queue<std::pair<float, uint32_t>>> heaps(q_count);

      // --- 1. FUSE LUTS ---
      size_t total_embeddings = 0;
      std::vector<size_t> emb_offsets(q_count + 1, 0);
      for (size_t i = 0; i < q_count; ++i) {
        total_embeddings += A[q_start + i]->num_queries;
        emb_offsets[i + 1] = total_embeddings;
      }

      if (total_embeddings == 0) return;

      uint32_t num_blocks = B.num_blocks;
      size_t lut_bytes = num_blocks * 16;
      std::vector<uint8_t> fused_luts(total_embeddings * lut_bytes);
      std::vector<float> fused_scales(total_embeddings);
      std::vector<float> fused_mins(total_embeddings);

      for (size_t i = 0; i < q_count; ++i) {
        const auto* qc = A[q_start + i];
        size_t off = emb_offsets[i];
        size_t count = qc->num_queries;
        std::memcpy(fused_luts.data() + off * lut_bytes, qc->flat_int_luts.data(),
                    count * lut_bytes);
        std::memcpy(fused_scales.data() + off, qc->scales.data(), count * sizeof(float));
        std::memcpy(fused_mins.data() + off, qc->min_dists.data(), count * sizeof(float));
      }

      std::vector<float> emb_min_dists(total_embeddings);

      // --- 2. DATABASE PROBING ---
      for (size_t c = 0; c < num_db_clouds; ++c) {
        const size_t start_vec = B.offsets[c];
        const size_t cloud_size = B.offsets[c + 1] - start_vec;
        if (cloud_size == 0) continue;

        // Stream all fused LUTs against the packed database strips
        internal::fastscan_mv_chamfer_fused<PCS::is_metric()>(
            fused_luts.data(), fused_scales.data(), fused_mins.data(), total_embeddings, num_blocks,
            B.packed_codes.data(), B.num_blocks * 32, start_vec, cloud_size, emb_min_dists.data());

        for (size_t i = 0; i < q_count; ++i) {
          float dist_sum = 0.0f;
          size_t e_start = emb_offsets[i];
          size_t e_count = emb_offsets[i + 1] - e_start;
          for (size_t e = 0; e < e_count; ++e)
            dist_sum += emb_min_dists[e_start + e];
          float chamfer_dist = dist_sum / static_cast<float>(e_count);

          if (heaps[i].size() < k)
            heaps[i].push({chamfer_dist, B.get_id(c)});
          else if (chamfer_dist < heaps[i].top().first) {
            heaps[i].pop();
            heaps[i].push({chamfer_dist, B.get_id(c)});
          }
        }
      }

      // --- 3. DRAIN HEAPS ---
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

}  // namespace fastscan_mv
}  // namespace mvsic