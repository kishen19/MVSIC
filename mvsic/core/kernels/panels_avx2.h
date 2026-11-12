#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <immintrin.h>

namespace mvsic::cpu {

#ifdef __AVX2__
// The prepared TQ4/TQ8 panels share a 16-document, four-coordinate tile.
// Keep dispatch outside this kernel. Splitting unsigned bytes avoids the
// signed-int16 saturation of maddubs for full-range uint8 x int8 products.
inline __m256i panel_dot_avx2(__m256i sum, __m256i db, __m256i query) {
  const __m256i lo = _mm256_and_si256(db, _mm256_set1_epi8(0x7f));
  const __m256i hi = _mm256_and_si256(db, _mm256_set1_epi8(char(0x80)));
  const __m256i ones = _mm256_set1_epi16(1);
  const __m256i a = _mm256_madd_epi16(_mm256_maddubs_epi16(lo, query), ones);
  const __m256i b = _mm256_madd_epi16(_mm256_maddubs_epi16(hi, query), ones);
  return _mm256_add_epi32(sum, _mm256_add_epi32(a, b));
}

template<bool Metric>
inline float chamfer_panels_avx2(const int8_t* queries, const float* query_norms,
    const float* query_sqns, const int32_t* query_sums, size_t query_stride,
    size_t query_count, const uint8_t* panels, const float* norms, const float* sqns,
    size_t tiles, size_t panel_bytes, size_t panel_count, size_t cloud_size) {
  const __m256 infinity = _mm256_set1_ps(std::numeric_limits<float>::max());
  const __m256i lanes = _mm256_setr_epi32(0,1,2,3,4,5,6,7);
  float total = 0;
  for (size_t q = 0; q < query_count; ++q) {
    __m256 best = infinity;
    const __m256 scale = _mm256_set1_ps(-query_norms[q]);
    const __m256i bias = _mm256_set1_epi32(128 * query_sums[q]);
    for (size_t panel = 0; panel < panel_count; ++panel) {
      for (size_t half = 0; half < 2; ++half) {
        const size_t point = panel * 16 + half * 8;
        if (point >= cloud_size) break;
        __m256i sum = _mm256_setzero_si256();
        const uint8_t* base = panels + panel * panel_bytes + half * 32;
        for (size_t tile = 0; tile < tiles; ++tile) {
          int32_t word;
          std::memcpy(&word, queries + q * query_stride + tile * 4, sizeof(word));
          const auto db = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(base + tile * 64));
          sum = panel_dot_avx2(sum, db, _mm256_set1_epi32(word));
        }
        __m256 distance = _mm256_mul_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(sum, bias)),
                                                       _mm256_loadu_ps(norms + point)), scale);
        if constexpr (Metric)
          distance = _mm256_add_ps(_mm256_add_ps(_mm256_loadu_ps(sqns + point),
              _mm256_add_ps(distance, distance)), _mm256_set1_ps(query_sqns[q]));
        if (cloud_size - point < 8) {
          const auto valid = _mm256_cmpgt_epi32(_mm256_set1_epi32(int(cloud_size - point)), lanes);
          distance = _mm256_blendv_ps(infinity, distance, _mm256_castsi256_ps(valid));
        }
        best = _mm256_min_ps(best, distance);
      }
    }
    alignas(32) float minima[8];
    _mm256_store_ps(minima, best);
    total += *std::min_element(minima, minima + 8);
  }
  return total;
}

inline __m256i popcount_bytes_avx2(__m256i bytes) {
  const auto lut = _mm256_setr_epi8(0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,
                                   0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4);
  const auto mask = _mm256_set1_epi8(15);
  const auto lo = _mm256_shuffle_epi8(lut, _mm256_and_si256(bytes, mask));
  const auto hi = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(bytes, 4), mask));
  return _mm256_add_epi8(lo, hi);
}

inline __m256i fold_popcount_bytes_avx2(__m256i counts) {
  const auto pairs = _mm256_maddubs_epi16(counts, _mm256_set1_epi8(1));
  return _mm256_madd_epi16(pairs, _mm256_set1_epi16(1));
}

inline __m256i popcount32_avx2(__m256i bytes) {
  return fold_popcount_bytes_avx2(popcount_bytes_avx2(bytes));
}

template<bool Metric, bool ByteAccumulation>
inline float chamfer_hamming_panels_avx2_impl(const uint8_t* queries, size_t query_stride,
    size_t query_count, size_t query_bytes, const uint8_t* panels,
    size_t panel_bytes, size_t tiles, size_t cloud_size, size_t padded_dim,
    size_t query_tile_stride = 4) {
  const auto infinity = _mm256_set1_epi32(INT32_MAX);
  const auto lanes = _mm256_setr_epi32(0,1,2,3,4,5,6,7);
  float total = 0;
  for (size_t q = 0; q < query_count; ++q) {
    __m256i best = infinity;
    for (size_t point = 0; point < cloud_size; point += 8) {
      __m256i sum = _mm256_setzero_si256();
      const uint8_t* base = panels + (point / 16) * panel_bytes + (point % 16) * 4;
      for (size_t tile = 0; tile < tiles; ++tile) {
        uint32_t word = 0;
        std::memcpy(&word, queries + q * query_stride + tile * query_tile_stride,
                    std::min<size_t>(4, query_bytes - tile * 4));
        const auto db = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(base + tile * 64));
        const auto difference = _mm256_xor_si256(db, _mm256_set1_epi32(word));
        if constexpr (ByteAccumulation)
          sum = _mm256_add_epi8(sum, popcount_bytes_avx2(difference));
        else
          sum = _mm256_add_epi32(sum, popcount32_avx2(difference));
      }
      if constexpr (ByteAccumulation) sum = fold_popcount_bytes_avx2(sum);
      if (cloud_size - point < 8) {
        const auto valid = _mm256_cmpgt_epi32(_mm256_set1_epi32(int(cloud_size - point)), lanes);
        sum = _mm256_blendv_epi8(infinity, sum, valid);
      }
      best = _mm256_min_epi32(best, sum);
    }
    alignas(32) int32_t minima[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(minima), best);
    const float hamming = float(*std::min_element(minima, minima + 8));
    total += (Metric ? 4.0f : 2.0f) / float(padded_dim) * hamming - (Metric ? 0.0f : 1.0f);
  }
  return total;
}
template<bool Metric>
inline float chamfer_hamming_panels_avx2(const uint8_t* queries, size_t query_stride,
    size_t query_count, size_t query_bytes, const uint8_t* panels,
    size_t panel_bytes, size_t tiles, size_t cloud_size, size_t padded_dim,
    size_t query_tile_stride = 4) {
  // Each byte contributes at most eight bits per tile; 31*8 fits uint8.
  // One tile has no fold to amortize. Dispatch before query/point/tile loops.
  if (tiles >= 2 && tiles <= 31)
    return chamfer_hamming_panels_avx2_impl<Metric, true>(queries, query_stride,
        query_count, query_bytes, panels, panel_bytes, tiles, cloud_size, padded_dim, query_tile_stride);
  return chamfer_hamming_panels_avx2_impl<Metric, false>(queries, query_stride,
      query_count, query_bytes, panels, panel_bytes, tiles, cloud_size, padded_dim, query_tile_stride);
}
#endif

}  // namespace mvsic::cpu
