#pragma once

// Single-vector 8-bit TurboQuant.
//
// Mirrors `turboquant.h` (4-bit), but stores 1 byte per (rotated, padded)
// dimension instead of a packed nibble. Uses per-vector adaptive max-abs
// scaling so each vector fills the full int8 range — same encoding scheme
// as `turboquant_8bit_mv` but for a flat single-vector layout (no panels).
//
// Storage:
//   * num_bytes_per_datapoint = padded_dim
//   * Codes stored as uint8 = (int8 + 128), i.e. bias-form so the AVX2 dot
//     kernel can use _mm256_maddubs_epi16 (uint8 * int8 -> int16) directly,
//     subtracting 128 * byte_sum_of_query at the end to recover the
//     signed-signed inner product.
//
// Distance kernel: same maddubs+madd accumulation pattern as `turboquant.h`
// but without the nibble unpack and even/odd deinterleave (each dim is one
// byte, so the kernel is a straight uint8 * int8 dot).

#include <vector>
#include <cstdint>
#include <immintrin.h>
#include <cmath>
#include <cstring>

#include "parlay/parallel.h"
#include "parlay/sequence.h"
#include "mvsic/core/quantization/turboquant_utils.h"

namespace mvsic {
namespace turboquant_8bit {

template<bool Metric>
class Quantized_Point;

// =========================================================================
// Single-Vector Query
// =========================================================================
template<bool Metric>
class Quantized_Query {
 public:
  using distanceType = float;

  // int8 query, padded to multiple of 32 for safe AVX2 loads.
  std::vector<int8_t> query_data;
  size_t num_bytes_per_datapoint = 0;
  size_t padded_query_bytes = 0;

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

    parlay::parallel_for(
        0, N,
        [&](size_t i) {
          const uint8_t* code_ptr = db.packed_codes.data() + i * nb;
          const float p_nsf = db.norm_scaling_factors[i];
          const float p_sqn = Metric ? db.unquantized_squared_norms[i] : 0.0f;
          int32_t dot = 0;

#if defined(__AVX2__) || defined(__AVX512F__)
          const __m256i ones = _mm256_set1_epi16(1);
          __m256i acc = _mm256_setzero_si256();

          size_t j = 0;
          for (; j + 32 <= nb; j += 32) {
            __m256i b_u8 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(code_ptr + j));
            __m256i q_i8 =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(query_data.data() + j));
            __m256i prod16 = _mm256_maddubs_epi16(b_u8, q_i8);
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(prod16, ones));
          }

          alignas(32) int32_t dots[8];
          _mm256_store_si256(reinterpret_cast<__m256i*>(dots), acc);
          dot = dots[0] + dots[1] + dots[2] + dots[3] + dots[4] + dots[5] + dots[6] + dots[7];

          for (; j < nb; ++j) {
            dot += static_cast<int32_t>(code_ptr[j]) * static_cast<int32_t>(query_data[j]);
          }
#else
          for (size_t j = 0; j < nb; ++j) {
            dot += static_cast<int32_t>(code_ptr[j]) * static_cast<int32_t>(query_data[j]);
          }
#endif

          dot -= 128 * b_sum;
          float neg_dot = -static_cast<float>(dot) * p_nsf * q_nsf;

          if constexpr (Metric) {
            out[i] = p_sqn + 2.0f * neg_dot + q_sqn;
          } else {
            out[i] = neg_dot;
          }
        },
        /*granularity=*/64);
  }
};

// =========================================================================
// Single-Vector Point Handle
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
    const __m256i ones = _mm256_set1_epi16(1);
    __m256i acc = _mm256_setzero_si256();
    size_t j = 0;

    for (; j + 32 <= num_bytes; j += 32) {
      __m256i b_u8 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(code_ptr + j));
      __m256i q_i8 =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qq.query_data.data() + j));
      __m256i prod16 = _mm256_maddubs_epi16(b_u8, q_i8);
      acc = _mm256_add_epi32(acc, _mm256_madd_epi16(prod16, ones));
    }

    alignas(32) int32_t dots[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(dots), acc);
    dot = dots[0] + dots[1] + dots[2] + dots[3] + dots[4] + dots[5] + dots[6] + dots[7];

    for (; j < num_bytes; ++j) {
      dot += static_cast<int32_t>(code_ptr[j]) * static_cast<int32_t>(qq.query_data[j]);
    }
#else
    for (size_t j = 0; j < num_bytes; ++j) {
      dot += static_cast<int32_t>(code_ptr[j]) * static_cast<int32_t>(qq.query_data[j]);
    }
#endif

    // Bias correction: stored uint8 = int8 + 128, so dot has an extra
    // 128 * sum(int8_query) term that we subtract here.
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

  parlay::sequence<uint8_t> packed_codes;  // Flat layout: N * padded_dim
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
  // Reuses turboquant::BaseEncoder solely for the rotator setup; the 4-bit
  // bucket encoding inside BaseEncoder is unused — we re-encode per-vector
  // with adaptive int8 scaling below.
  turboquant::BaseEncoder encoder;

  Model() = default;

  template<typename PointRangeTy>
  void train(const PointRangeTy& data) {
    encoder.train(data.get_dims());
  }

  template<typename PointRangeTy>
  Quantized_Point_Range<PointRangeTy, Metric> encode(const PointRangeTy& data) const {
    Quantized_Point_Range<PointRangeTy, Metric> enc;
    enc.n_points = data.size();
    enc.num_bytes_per_datapoint = encoder.padded_dim;
    enc.packed_codes.resize(enc.n_points * enc.num_bytes_per_datapoint);
    enc.norm_scaling_factors.resize(enc.n_points);
    enc.unquantized_squared_norms.resize(enc.n_points);

    parlay::parallel_for(0, enc.n_points, [&](size_t i) {
      std::vector<float> ws(encoder.padded_dim);
      const float* p = reinterpret_cast<const float*>(data.location(i));
      uint8_t* out_ptr = enc.packed_codes.data() + i * enc.num_bytes_per_datapoint;
      auto [sqn, nsf] = encode_single(p, out_ptr, ws);
      enc.norm_scaling_factors[i] = nsf;
      enc.unquantized_squared_norms[i] = sqn;
    });
    return enc;
  }

  template<typename PointTy>
  Quantized_Query<Metric> quantize_query(const PointTy& query) const {
    std::vector<float> tmp(encoder.dim);
    for (size_t i = 0; i < encoder.dim; ++i)
      tmp[i] = query[i];
    return quantize_query_from_ptr(tmp.data());
  }

  Quantized_Query<Metric> quantize_query(const float* qptr) const {
    return quantize_query_from_ptr(qptr);
  }

  void save(std::ofstream& out) const { encoder.save(out); }
  void load(std::ifstream& in) { encoder.load(in); }

 private:
  // Rotates and encodes a single vector into `padded_dim` bias-form bytes
  // (uint8 = int8 + 128). Returns (unquantized_sqr_norm, norm_scaling_factor).
  std::pair<float, float> encode_single(const float* p, uint8_t* output,
                                        std::vector<float>& ws) const {
    const size_t pdim = encoder.padded_dim;
    encoder.rotator->rotate(p, ws.data());

    float sqr_norm = 0.0f;
    float max_abs = 0.0f;
    for (size_t i = 0; i < pdim; ++i) {
      sqr_norm += ws[i] * ws[i];
      const float a = std::abs(ws[i]);
      if (a > max_abs) max_abs = a;
    }
    if (sqr_norm == 0.0f || max_abs == 0.0f) {
      // 0x80 == bias-form of int8 0, neutral for the dot.
      std::memset(output, 0x80, pdim);
      return {0.0f, 0.0f};
    }

    const float norm = std::sqrt(sqr_norm);
    const float scale = 127.0f / max_abs;

    int64_t q_sqr_norm = 0;
    for (size_t i = 0; i < pdim; ++i) {
      const int snapped = static_cast<int>(std::lround(ws[i] * scale));
      const int8_t iv =
          static_cast<int8_t>(snapped < -127 ? -127 : (snapped > 127 ? 127 : snapped));
      output[i] = static_cast<uint8_t>(static_cast<int>(iv) + 128);
      q_sqr_norm += static_cast<int64_t>(iv) * static_cast<int64_t>(iv);
    }

    if (q_sqr_norm == 0) return {sqr_norm, 0.0f};
    const float nsf = norm / std::sqrt(static_cast<float>(q_sqr_norm));
    return {sqr_norm, nsf};
  }

  Quantized_Query<Metric> quantize_query_from_ptr(const float* qptr) const {
    Quantized_Query<Metric> qq;
    qq.num_bytes_per_datapoint = encoder.padded_dim;
    qq.padded_query_bytes = (encoder.padded_dim + 31) & ~size_t(31);
    qq.query_data.resize(qq.padded_query_bytes, 0);

    std::vector<float> q_rot(encoder.padded_dim);
    encoder.rotator->rotate(qptr, q_rot.data());

    float sqr_norm = 0.0f;
    float max_value = 0.0f;
    for (size_t i = 0; i < encoder.padded_dim; ++i) {
      sqr_norm += q_rot[i] * q_rot[i];
      const float a = std::abs(q_rot[i]);
      if (a > max_value) max_value = a;
    }

    if (sqr_norm == 0.0f || !std::isfinite(sqr_norm) || max_value == 0.0f) return qq;

    const float norm = std::sqrt(sqr_norm);
    const float sf = 127.0f / max_value;

    int64_t quant_norm = 0;
    int32_t byte_sum = 0;
    for (size_t i = 0; i < encoder.padded_dim; ++i) {
      const int snapped = static_cast<int>(std::lround(q_rot[i] * sf));
      const int8_t iv =
          static_cast<int8_t>(snapped < -127 ? -127 : (snapped > 127 ? 127 : snapped));
      qq.query_data[i] = iv;
      quant_norm += static_cast<int64_t>(iv) * static_cast<int64_t>(iv);
      byte_sum += iv;
    }

    qq.norm_scaling_factor =
        quant_norm > 0 ? norm / std::sqrt(static_cast<float>(quant_norm)) : 0.0f;
    qq.unquantized_squared_norm = sqr_norm;
    qq.byte_sum = byte_sum;
    return qq;
  }
};

}  // namespace turboquant_8bit
}  // namespace mvsic
