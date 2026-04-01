#pragma once

#include <vector>
#include <cstdint>
#include <immintrin.h>
#include "parlay/parallel.h"
#include "parlay/sequence.h"
#include "mvsic/core/quantization/turboquant_utils.h"

namespace mvsic {
namespace turboquant {

namespace internal {
// Shifted centroids (+128) to allow unsigned * signed AVX2 dot products
static constexpr std::array<uint8_t, 16> kCentroidsUint8 = {134, 146, 159, 172, 186, 203, 224, 255,
                                                            122, 110, 97,  84,  70,  53,  32,  1};
}  // namespace internal

template<bool Metric>
class Quantized_Point;

// =========================================================================
// Single-Vector Query (Pre-deinterleaved)
// =========================================================================
template<bool Metric>
class Quantized_Query {
 public:
  using distanceType = float;
  // Layout: [even_dim_0, even_dim_2...] ... [odd_dim_1, odd_dim_3...]
  // Padded to a multiple of 32 bytes for safe AVX2 loads.
  std::vector<int8_t> query_data;
  size_t dim = 0;
  size_t num_bytes_per_datapoint = 0;
  size_t padded_bytes_half = 0;

  float norm_scaling_factor = 0.0f;
  float unquantized_squared_norm = 0.0f;
  int32_t byte_sum = 0;

  Quantized_Query() = default;

  inline float distance(const Quantized_Point<Metric>& p) const { return p.distance(*this); }

  template<typename PointRangeTy>
  void distances_all(const PointRangeTy& db, float* out) const {
    const size_t N = db.size();
    const size_t nb = num_bytes_per_datapoint;
    const float q_nsf = norm_scaling_factor;
    const float q_sqn = unquantized_squared_norm;
    const int32_t b_sum = byte_sum;

    // Hoist SIMD Constants outside the point loop but inside the parallel_for
    // to ensure they are available per-thread.
    parlay::parallel_for(
        0, N,
        [&](size_t i) {
          const uint8_t* code_ptr = db.packed_codes.data() + i * nb;
          const float p_nsf = db.norm_scaling_factors[i];
          const float p_sqn = Metric ? db.unquantized_squared_norms[i] : 0.0f;
          int32_t dot = 0;

#if defined(__AVX2__) || defined(__AVX512F__)
          const __m256i codebook = _mm256_broadcastsi128_si256(
              _mm_loadu_si128(reinterpret_cast<const __m128i*>(internal::kCentroidsUint8.data())));
          const __m256i mask_0f = _mm256_set1_epi8(0x0F);
          const __m256i ones = _mm256_set1_epi16(1);
          __m256i acc = _mm256_setzero_si256();

          size_t j = 0;
          for (; j + 32 <= nb; j += 32) {
            __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(code_ptr + j));

            // Even/Odd extraction
            __m256i even_u8 = _mm256_shuffle_epi8(codebook, _mm256_and_si256(packed, mask_0f));
            __m256i odd_u8 = _mm256_shuffle_epi8(
                codebook, _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask_0f));

            __m256i q_even =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(query_data.data() + j));
            __m256i q_odd = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(query_data.data() + padded_bytes_half + j));

            // Multiply-Add sequence
            acc = _mm256_add_epi32(acc,
                                   _mm256_madd_epi16(_mm256_maddubs_epi16(even_u8, q_even), ones));
            acc =
                _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(odd_u8, q_odd), ones));
          }

          alignas(32) int32_t dots[8];
          _mm256_store_si256(reinterpret_cast<__m256i*>(dots), acc);
          dot = dots[0] + dots[1] + dots[2] + dots[3] + dots[4] + dots[5] + dots[6] + dots[7];

          // Scalar tail for remainder blocks
          for (; j < nb; ++j) {
            const uint8_t byte = code_ptr[j];
            dot += static_cast<int32_t>(internal::kCentroidsUint8[byte & 0xF]) * query_data[j];
            dot += static_cast<int32_t>(internal::kCentroidsUint8[byte >> 4]) *
                   query_data[padded_bytes_half + j];
          }
#else
          for (size_t j = 0; j < nb; ++j) {
            const uint8_t byte = code_ptr[j];
            dot += static_cast<int32_t>(internal::kCentroidsUint8[byte & 0xF]) * query_data[j];
            dot += static_cast<int32_t>(internal::kCentroidsUint8[byte >> 4]) *
                   query_data[padded_bytes_half + j];
          }
#endif

          // Bias correction & final post-processing
          dot -= 128 * b_sum;
          float neg_dot = -static_cast<float>(dot) * p_nsf * q_nsf;

          if constexpr (Metric) {
            out[i] = p_sqn + 2.0f * neg_dot + q_sqn;
          } else {
            out[i] = neg_dot;
          }
        },
        /*granularity=*/64);  // Amortize scheduler overhead
  }
};

// =========================================================================
// Single-Vector Point Handle (Flat Memory)
// =========================================================================
template<bool Metric>
class Quantized_Point {
 public:
  const uint8_t* code_ptr = nullptr;
  size_t num_bytes = 0;
  float norm_scaling_factor = 0.0f;
  float unquantized_squared_norm = 0.0f;

  Quantized_Point() = default;
  Quantized_Point(const uint8_t* ptr, size_t nb, float nsf, float usn) :
      code_ptr(ptr), num_bytes(nb), norm_scaling_factor(nsf), unquantized_squared_norm(usn) {}
  inline float distance(const Quantized_Query<Metric>& qq) const {
    int32_t dot = 0;

#if defined(__AVX2__) || defined(__AVX512F__)
    // High-performance AVX2 Vector-Vector Dot Product
    const __m256i codebook = _mm256_broadcastsi128_si256(
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(internal::kCentroidsUint8.data())));
    const __m256i mask_0f = _mm256_set1_epi8(0x0F);
    const __m256i ones = _mm256_set1_epi16(1);

    __m256i acc = _mm256_setzero_si256();
    size_t j = 0;

    // Process 32 bytes (64 dimensions) per iteration
    for (; j + 32 <= num_bytes; j += 32) {
      __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(code_ptr + j));

      // Decode nibbles to uint8 centroids
      __m256i even_idx = _mm256_and_si256(packed, mask_0f);
      __m256i odd_idx = _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask_0f);
      __m256i even_u8 = _mm256_shuffle_epi8(codebook, even_idx);
      __m256i odd_u8 = _mm256_shuffle_epi8(codebook, odd_idx);

      // Load pre-deinterleaved query components
      __m256i q_even =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qq.query_data.data() + j));
      __m256i q_odd = _mm256_loadu_si256(
          reinterpret_cast<const __m256i*>(qq.query_data.data() + qq.padded_bytes_half + j));

      // uint8 * int8 -> int16
      __m256i prod_even16 = _mm256_maddubs_epi16(even_u8, q_even);
      __m256i prod_odd16 = _mm256_maddubs_epi16(odd_u8, q_odd);

      // Adjacent int16 sum -> int32
      __m256i sum_even32 = _mm256_madd_epi16(prod_even16, ones);
      __m256i sum_odd32 = _mm256_madd_epi16(prod_odd16, ones);

      acc = _mm256_add_epi32(acc, _mm256_add_epi32(sum_even32, sum_odd32));
    }

    // Horizontal sum of the AVX2 accumulator
    alignas(32) int32_t dots[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(dots), acc);
    dot = dots[0] + dots[1] + dots[2] + dots[3] + dots[4] + dots[5] + dots[6] + dots[7];

    // Scalar tail
    for (; j < num_bytes; ++j) {
      const uint8_t byte = code_ptr[j];
      dot += static_cast<int32_t>(internal::kCentroidsUint8[byte & 0xF]) * qq.query_data[j];
      dot += static_cast<int32_t>(internal::kCentroidsUint8[byte >> 4]) *
             qq.query_data[qq.padded_bytes_half + j];
    }
#else
    // Pure Scalar Fallback
    for (size_t j = 0; j < num_bytes; ++j) {
      const uint8_t byte = code_ptr[j];
      dot += static_cast<int32_t>(internal::kCentroidsUint8[byte & 0xF]) * qq.query_data[j];
      dot += static_cast<int32_t>(internal::kCentroidsUint8[byte >> 4]) *
             qq.query_data[qq.padded_bytes_half + j];
    }
#endif

    // Bias correction: we multiplied unsigned (C + 128) * Q = C*Q + 128*Q
    dot -= 128 * qq.byte_sum;

    float neg_dot = -static_cast<float>(dot) * norm_scaling_factor * qq.norm_scaling_factor;
    if constexpr (Metric) {
      return unquantized_squared_norm + 2.0f * neg_dot + qq.unquantized_squared_norm;
    } else {
      return neg_dot;
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
// Single-Vector Set (Flat Layout)
// =========================================================================
template<typename PointRangeTy, bool Metric>
class Quantized_Point_Range {
 public:
  static constexpr bool is_fastscan = false;

  size_t n_points = 0;
  size_t num_bytes_per_datapoint = 0;

  parlay::sequence<uint8_t> packed_codes;  // Flat layout: N * num_bytes
  parlay::sequence<float> norm_scaling_factors;
  parlay::sequence<float> unquantized_squared_norms;

  Quantized_Point_Range() = default;

  inline size_t num_bytes_per_point() const noexcept {
    return num_bytes_per_datapoint + sizeof(float) + (Metric ? sizeof(float) : 0);
  }

  Quantized_Point<Metric> operator[](size_t i) const {
    const uint8_t* ptr = packed_codes.data() + i * num_bytes_per_datapoint;
    float sqn = Metric ? unquantized_squared_norms[i] : 0.0f;
    return Quantized_Point<Metric>(ptr, num_bytes_per_datapoint, norm_scaling_factors[i], sqn);
  }

  inline uint32_t size() const noexcept { return static_cast<uint32_t>(n_points); }

  void save(std::ofstream& out) const {
    out.write(reinterpret_cast<const char*>(&n_points), sizeof(n_points));
    out.write(reinterpret_cast<const char*>(&num_bytes_per_datapoint),
              sizeof(num_bytes_per_datapoint));
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
  }

  void load(std::ifstream& in) {
    in.read(reinterpret_cast<char*>(&n_points), sizeof(n_points));
    in.read(reinterpret_cast<char*>(&num_bytes_per_datapoint), sizeof(num_bytes_per_datapoint));
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
  }
};

// =========================================================================
// Single-Vector Model
// =========================================================================
template<bool Metric>
class Model {
 public:
  BaseEncoder encoder;

  Model() = default;

  template<typename PointRangeTy>
  void train(const PointRangeTy& data) {
    encoder.train(data.get_dims());
  }

  template<typename PointRangeTy>
  Quantized_Point_Range<PointRangeTy, Metric> encode(const PointRangeTy& data) const {
    Quantized_Point_Range<PointRangeTy, Metric> enc;
    enc.n_points = data.size();
    enc.num_bytes_per_datapoint = encoder.num_bytes_per_datapoint;
    enc.packed_codes.resize(enc.n_points * enc.num_bytes_per_datapoint);
    enc.norm_scaling_factors.resize(enc.n_points);
    enc.unquantized_squared_norms.resize(enc.n_points);

    parlay::parallel_for(0, enc.n_points, [&](size_t i) {
      std::vector<float> ws(encoder.padded_dim);
      const float* p = reinterpret_cast<const float*>(data.location(i));
      uint8_t* out_ptr = enc.packed_codes.data() + i * enc.num_bytes_per_datapoint;
      auto [sqn, nsf] = encoder.encode_single(p, out_ptr, ws);
      enc.norm_scaling_factors[i] = nsf;
      enc.unquantized_squared_norms[i] = sqn;
    });
    return enc;
  }

  // PointTy must be indexable: query[i] for i in [0, encoder.dim).
  template<typename PointTy>
  Quantized_Query<Metric> quantize_query(const PointTy& query) const {
    std::vector<float> tmp(encoder.dim);
    for (size_t i = 0; i < encoder.dim; ++i)
      tmp[i] = query[i];
    // Must not call quantize_query(tmp.data()): for float* the template overload beats
    // quantize_query(const float*), causing infinite recursion and stack/heap corruption.
    return quantize_query_from_ptr(tmp.data());
  }

  Quantized_Query<Metric> quantize_query(const float* qptr) const {
    return quantize_query_from_ptr(qptr);
  }

  void save(std::ofstream& out) const { encoder.save(out); }
  void load(std::ifstream& in) { encoder.load(in); }

 private:
  Quantized_Query<Metric> quantize_query_from_ptr(const float* qptr) const {
    Quantized_Query<Metric> qq;
    qq.dim = encoder.padded_dim;
    qq.num_bytes_per_datapoint = encoder.num_bytes_per_datapoint;

    // Pad to multiple of 32 for AVX2 bounds safety
    qq.padded_bytes_half = (qq.num_bytes_per_datapoint + 31) & ~31;
    qq.query_data.resize(qq.padded_bytes_half * 2, 0);

    std::vector<float> q_rot(encoder.padded_dim);
    encoder.rotator->rotate(qptr, q_rot.data());

    float sqr_norm = 0.0f;
    for (size_t i = 0; i < encoder.padded_dim; ++i)
      sqr_norm += q_rot[i] * q_rot[i];

    if (sqr_norm == 0.0f || !std::isfinite(sqr_norm)) return qq;

    const float norm = std::sqrt(sqr_norm);
    const float scale = std::sqrt(static_cast<float>(encoder.padded_dim)) / norm;

    float max_value = 0.0f;
    for (size_t i = 0; i < encoder.padded_dim; ++i) {
      q_rot[i] = std::clamp(q_rot[i] * scale, -internal::kValueCap, internal::kValueCap);
      max_value = std::max(max_value, std::abs(q_rot[i]));
    }

    const float sf = max_value > 0.0f ? 127.0f / max_value : 0.0f;

    int quant_norm = 0;
    for (size_t i = 0; i < encoder.padded_dim; ++i) {
      int8_t snapped = static_cast<int8_t>(std::clamp(std::round(q_rot[i] * sf), -127.0f, 127.0f));

      // Pre-deinterleave
      // Even dims go to first half, odd dims go to second half.
      if (i % 2 == 0) {
        qq.query_data[i / 2] = snapped;
      } else {
        qq.query_data[qq.padded_bytes_half + (i / 2)] = snapped;
      }

      quant_norm += snapped * snapped;
      qq.byte_sum += snapped;
    }

    qq.norm_scaling_factor = norm / std::sqrt(static_cast<float>(quant_norm));
    qq.unquantized_squared_norm = sqr_norm;
    return qq;
  }
};

}  // namespace turboquant
}  // namespace mvsic