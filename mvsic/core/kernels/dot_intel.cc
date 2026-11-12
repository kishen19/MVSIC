#include "mvsic/core/kernels/dispatch.h"

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>

namespace mvsic::kernels::detail {
__attribute__((target("avx2")))
int32_t dot_avx2(const uint8_t* a, const int8_t* b, size_t n) {
  __m256i sum = _mm256_setzero_si256();
  const __m256i ones = _mm256_set1_epi16(1);
  const __m256i low_mask = _mm256_set1_epi8(0x7f);
  size_t i = 0;
  for (; i + 32 <= n; i += 32) {
    const __m256i u = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
    const __m256i s = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
    // Splitting u into 7-bit and high-bit components prevents maddubs'
    // signed-int16 saturation, even for adversarial full-range bytes.
    const __m256i lo = _mm256_and_si256(u, low_mask);
    const __m256i hi = _mm256_andnot_si256(low_mask, u);
    const __m256i x = _mm256_madd_epi16(_mm256_maddubs_epi16(lo, s), ones);
    const __m256i y = _mm256_madd_epi16(_mm256_maddubs_epi16(hi, s), ones);
    sum = _mm256_add_epi32(sum, _mm256_add_epi32(x, y));
  }
  alignas(32) int32_t lanes[8];
  _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), sum);
  int32_t out = 0;
  for (int32_t v : lanes) out += v;
  for (; i < n; ++i) out += static_cast<int32_t>(a[i]) * b[i];
  return out;
}

__attribute__((target("avx512f,avx512bw,avx512dq,avx512vl")))
int32_t dot_avx512(const uint8_t* a, const int8_t* b, size_t n) {
  __m512i sum = _mm512_setzero_si512();
  const __m512i ones = _mm512_set1_epi16(1);
  const __m512i low_mask = _mm512_set1_epi8(0x7f);
  size_t i = 0;
  for (; i + 64 <= n; i += 64) {
    const __m512i u = _mm512_loadu_si512(a + i);
    const __m512i s = _mm512_loadu_si512(b + i);
    const __m512i lo = _mm512_and_si512(u, low_mask);
    const __m512i hi = _mm512_andnot_si512(low_mask, u);
    const __m512i x = _mm512_madd_epi16(_mm512_maddubs_epi16(lo, s), ones);
    const __m512i y = _mm512_madd_epi16(_mm512_maddubs_epi16(hi, s), ones);
    sum = _mm512_add_epi32(sum, _mm512_add_epi32(x, y));
  }
  alignas(64) int32_t lanes[16];
  _mm512_store_si512(lanes, sum);
  int32_t out = 0;
  for (int32_t v : lanes) out += v;
  for (; i < n; ++i) out += static_cast<int32_t>(a[i]) * b[i];
  return out;
}

__attribute__((target("avx512f,avx512bw,avx512dq,avx512vl,avx512vnni")))
int32_t dot_vnni(const uint8_t* a, const int8_t* b, size_t n) {
  __m512i sum = _mm512_setzero_si512();
  size_t i = 0;
  for (; i + 64 <= n; i += 64) {
    sum = _mm512_dpbusd_epi32(sum, _mm512_loadu_si512(a + i), _mm512_loadu_si512(b + i));
  }
  alignas(64) int32_t lanes[16];
  _mm512_store_si512(lanes, sum);
  int32_t out = 0;
  for (int32_t v : lanes) out += v;
  for (; i < n; ++i) out += static_cast<int32_t>(a[i]) * b[i];
  return out;
}
}  // namespace mvsic::kernels::detail
#else
namespace mvsic::kernels::detail {
int32_t dot_avx2(const uint8_t* a, const int8_t* b, size_t n) { return dot_scalar(a,b,n); }
int32_t dot_avx512(const uint8_t* a, const int8_t* b, size_t n) { return dot_scalar(a,b,n); }
int32_t dot_vnni(const uint8_t* a, const int8_t* b, size_t n) { return dot_scalar(a,b,n); }
}  // namespace mvsic::kernels::detail
#endif
