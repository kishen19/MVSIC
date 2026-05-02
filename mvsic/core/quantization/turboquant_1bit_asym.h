#pragma once

// Single-vector asymmetric 1-bit TurboQuant: 1-bit DB sign codes scored
// against an int4-quantized query (stored as int8 for SIMD ease, values
// clamped to [-7, +7]).
//
// Closes the recall gap between symmetric 1-bit (Ref1BTQSym / production
// 1BTQ) and the asymmetric upper bound (Ref1BTQAsym, full-float query).
// The estimator is:
//
//   <q, x> ≈ s_q · ||x||/sqrt(D) · ||q||/sqrt(sum q_int8^2)
//
// where s_q = sum_d sign(x_d) * q_int8[d]. Under the unit-norm convention
// (true for every dataset in this project) the per-DB-point scaling is
// effectively constant, so the int s_q ranking matches the float distance
// ranking — useful for the chamfer-min path where we keep an int32 running
// argmax and apply the affine map once per (query, cloud).
//
// Hot path (one-to-many): 64 dims/iter through one
// `_mm512_dpbusd_epi32(ones, maskz_mov(db_bits, q))` — same critical-path
// throughput as VPDPBUSD on Sapphire Rapids. The DB streams 8 bytes/point
// per iter (vs 64 in turboquant.h's 4-bit kernel) and q is read at 64
// bytes per iter from L1 (size = padded_dim, typically 128B). Net result
// at padded_dim=128: 2 inner iters = ~2 ops + reduce per point — within
// 2-3x of the symmetric VPOPCNTD kernel and ~10x faster than RaBitQ-1bit.
//
// Layout:
//   Quantized_Query::query_int8       — padded_dim bytes, signed [-7,+7]
//   Quantized_Query::Q_sum            — sum of q_int8[d] (int32)
//   Quantized_Query::norm_scaling_factor — ||q|| / sqrt(sum q_int8^2)
//   Quantized_Point_Range::packed_codes  — flat sign bits, 1 bit/dim
//   Quantized_Point_Range::norm_scaling_factors — ||x|| / sqrt(D), one per pt
//   Quantized_Point_Range::unquantized_squared_norms — ||x||^2 (L2 only)
//
// Tail safety: SIMD body only runs while `d + 64 <= padded_dim`. Buffer
// reads for the 8-byte mask + 64-byte q stay within the per-point /
// per-query buffers. The scalar tail handles padded_dim < 64 cases (e.g.
// padded_dim=32) without any buffer padding.

#include <vector>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <fstream>
#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif
#include "parlay/parallel.h"
#include "parlay/sequence.h"
#include "rabitqlib/utils/rotator.hpp"

namespace mvsic {
namespace turboquant_1bit_asym {

namespace internal {

// Query is quantized to a 4-bit signed range, stored as int8 to allow
// VPDPBUSD's signed-int8 second operand. Range [-7,+7] preserves symmetry
// and avoids the asymmetric -8 bin (consistent with the 4-bit TurboQuant
// query encoding in turboquant.h).
static constexpr int8_t kQMax = 7;
static constexpr int8_t kQMin = -7;

inline size_t round_up_64(size_t x) { return (x + 63) & ~size_t{63}; }

// Scalar masked sum: sum_{d : db_bits[d]=1} q_int8[d].
inline int32_t masked_sum_scalar(const uint8_t* db_bits, const int8_t* q_int8,
                                 size_t start_dim, size_t end_dim) {
  int32_t s = 0;
  for (size_t d = start_dim; d < end_dim; ++d) {
    if ((db_bits[d >> 3] >> (d & 7)) & 1u) s += q_int8[d];
  }
  return s;
}

// SIMD masked sum over `padded_dim` dims. db_bits is 1 bit/dim packed,
// q_int8 is 1 byte/dim signed. Returns sum_{d : bit_d=1} q_int8[d].
inline int32_t masked_sum_q(const uint8_t* db_bits, const int8_t* q_int8, size_t padded_dim) {
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
  __m512i acc = _mm512_setzero_si512();
  const __m512i ones = _mm512_set1_epi8(1);
  size_t d = 0;
  for (; d + 64 <= padded_dim; d += 64) {
    uint64_t mbits;
    std::memcpy(&mbits, db_bits + (d >> 3), 8);
    const __mmask64 m = static_cast<__mmask64>(mbits);
    const __m512i q = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(q_int8 + d));
    const __m512i mq = _mm512_maskz_mov_epi8(m, q);
    acc = _mm512_dpbusd_epi32(acc, ones, mq);
  }
  int32_t sum = _mm512_reduce_add_epi32(acc);
  if (d < padded_dim) sum += masked_sum_scalar(db_bits, q_int8, d, padded_dim);
  return sum;
#else
  return masked_sum_scalar(db_bits, q_int8, 0, padded_dim);
#endif
}

}  // namespace internal

template<bool Metric>
class Quantized_Point;

// =========================================================================
// Single-Vector Query (int8-stored, int4-quantized; range [-7,+7])
// =========================================================================
template<bool Metric>
class Quantized_Query {
 public:
  using distanceType = float;

  // Padded to a multiple of 64 bytes so AVX-512 loads at offset
  // `padded_dim - 64` are safe; bytes past padded_dim are zero.
  std::vector<int8_t> query_int8;
  size_t dim = 0;
  uint32_t padded_dim = 0;

  int32_t Q_sum = 0;                    // sum of q_int8[d]
  float norm_scaling_factor = 0.0f;     // ||q|| / sqrt(sum q_int8^2)
  float unquantized_squared_norm = 0.0f;  // ||q||^2 (L2 only)

  Quantized_Query() = default;

  inline float distance(const Quantized_Point<Metric>& p) const {
    return p.distance(*this);
  }

  template<typename PointRangeTy>
  void distances_all(const PointRangeTy& db, float* out) const {
    const size_t N = db.size();
    const float q_nsf = norm_scaling_factor;
    const float q_sqn = unquantized_squared_norm;
    const uint32_t pdim = padded_dim;
    const int32_t qsum = Q_sum;

    // neg_dot = -s_q * p_nsf * q_nsf
    //         = -(Q_sum - 2*masked_sum) * p_nsf * q_nsf
    //         = (2*masked_sum - Q_sum) * p_nsf * q_nsf
    // Hoist the q-side scaling: neg_dot = (k_a*ms + k_b) * p_nsf, where
    //   k_a = 2 * q_nsf
    //   k_b = -Q_sum * q_nsf
    const float k_a = 2.0f * q_nsf;
    const float k_b = -static_cast<float>(qsum) * q_nsf;
    const int8_t* qd = query_int8.data();

    parlay::parallel_for(
        0, N,
        [&](size_t i) {
          const uint8_t* code_ptr = db.packed_codes.data() + i * db.num_bytes_per_datapoint;
          const float p_nsf = db.norm_scaling_factors[i];
          const float p_sqn = Metric ? db.unquantized_squared_norms[i] : 0.0f;
          const int32_t ms = internal::masked_sum_q(code_ptr, qd, pdim);
          const float ms_f = static_cast<float>(ms);
          const float neg_dot = std::fma(k_a, ms_f, k_b) * p_nsf;
          if constexpr (Metric) {
            out[i] = p_sqn + 2.0f * neg_dot + q_sqn;
          } else {
            out[i] = neg_dot;
          }
        },
        /*granularity=*/256);
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
  uint32_t padded_dim = 0;
  float norm_scaling_factor = 0.0f;        // ||x|| / sqrt(D)
  float unquantized_squared_norm = 0.0f;   // ||x||^2 (L2)

  Quantized_Point() = default;
  Quantized_Point(const uint8_t* ptr, size_t nb, uint32_t pdim, float nsf, float usn) :
      code_ptr(ptr), num_bytes(nb), padded_dim(pdim),
      norm_scaling_factor(nsf), unquantized_squared_norm(usn) {}

  inline float distance(const Quantized_Query<Metric>& qq) const {
    const int32_t ms = internal::masked_sum_q(code_ptr, qq.query_int8.data(), padded_dim);
    const int32_t s_q = qq.Q_sum - 2 * ms;
    const float neg_dot =
        -static_cast<float>(s_q) * norm_scaling_factor * qq.norm_scaling_factor;
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
  uint32_t padded_dim = 0;

  parlay::sequence<uint8_t> packed_codes;  // Flat: N * num_bytes_per_datapoint
  parlay::sequence<float> norm_scaling_factors;       // ||x|| / sqrt(D)
  parlay::sequence<float> unquantized_squared_norms;  // ||x||^2

  Quantized_Point_Range() = default;

  inline size_t num_bytes_per_point() const noexcept {
    return num_bytes_per_datapoint + sizeof(float) + (Metric ? sizeof(float) : 0);
  }

  Quantized_Point<Metric> operator[](size_t i) const {
    const uint8_t* ptr = packed_codes.data() + i * num_bytes_per_datapoint;
    float sqn = Metric ? unquantized_squared_norms[i] : 0.0f;
    return Quantized_Point<Metric>(ptr, num_bytes_per_datapoint, padded_dim,
                                   norm_scaling_factors[i], sqn);
  }

  inline uint32_t size() const noexcept { return static_cast<uint32_t>(n_points); }

  void save(std::ofstream& out) const {
    out.write(reinterpret_cast<const char*>(&n_points), sizeof(n_points));
    out.write(reinterpret_cast<const char*>(&num_bytes_per_datapoint),
              sizeof(num_bytes_per_datapoint));
    out.write(reinterpret_cast<const char*>(&padded_dim), sizeof(padded_dim));
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
    in.read(reinterpret_cast<char*>(&padded_dim), sizeof(padded_dim));
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
// Encoder: shared rotator + sign-bit DB packer + int4 query quantizer.
// =========================================================================
class BaseEncoder1BitAsym {
 public:
  size_t dim = 0;
  size_t padded_dim = 0;
  size_t num_bytes_per_datapoint = 0;
  size_t seed_ = 42;

  rabitqlib::RotatorType rotator_type = rabitqlib::RotatorType::FhtKacRotator;
  std::unique_ptr<rabitqlib::Rotator<float>> rotator;

  void train(size_t input_dim) {
    dim = input_dim;
    if (dim == 0) return;
    rotator.reset(rabitqlib::choose_rotator<float>(dim, rotator_type));
    if (!rotator) return;
    padded_dim = rotator->size();
    num_bytes_per_datapoint = (padded_dim + 7) / 8;
  }

  // Encode one DB vector: rotate, sign-pack, return {sqr_norm, p_nsf}.
  // p_nsf = ||x|| / sqrt(D)  (DB centroid is ±1; absorb 1/sqrt(D) into nsf
  // so the per-pair float math is one fma per point).
  std::pair<float, float> encode_single(const float* p, uint8_t* output,
                                        std::vector<float>& ws) const {
    rotator->rotate(p, ws.data());

    float sqr_norm = 0.0f;
    for (size_t i = 0; i < padded_dim; ++i) sqr_norm += ws[i] * ws[i];

    std::memset(output, 0, num_bytes_per_datapoint);
    if (sqr_norm == 0.0f) return {0.0f, 0.0f};

    for (size_t i = 0; i < padded_dim; ++i) {
      if (ws[i] < 0.0f) output[i / 8] |= static_cast<uint8_t>(1u << (i % 8));
    }
    const float p_nsf = std::sqrt(sqr_norm) / std::sqrt(static_cast<float>(padded_dim));
    return {sqr_norm, p_nsf};
  }

  void save(std::ofstream& out) const {
    out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    out.write(reinterpret_cast<const char*>(&padded_dim), sizeof(padded_dim));
    out.write(reinterpret_cast<const char*>(&seed_), sizeof(seed_));
    int rt = static_cast<int>(rotator_type);
    out.write(reinterpret_cast<const char*>(&rt), sizeof(rt));
    if (rotator) rotator->save(out);
  }

  void load(std::ifstream& in) {
    in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    in.read(reinterpret_cast<char*>(&padded_dim), sizeof(padded_dim));
    in.read(reinterpret_cast<char*>(&seed_), sizeof(seed_));
    num_bytes_per_datapoint = (padded_dim + 7) / 8;
    int rt = 0;
    in.read(reinterpret_cast<char*>(&rt), sizeof(rt));
    rotator_type = static_cast<rabitqlib::RotatorType>(rt);
    rotator.reset(rabitqlib::choose_rotator<float>(dim, rotator_type));
    if (rotator) rotator->load(in);
  }
};

// =========================================================================
// Single-Vector Model
// =========================================================================
template<bool Metric>
class Model {
 public:
  BaseEncoder1BitAsym encoder;

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
    enc.padded_dim = static_cast<uint32_t>(encoder.padded_dim);
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

  template<typename PointTy>
  Quantized_Query<Metric> quantize_query(const PointTy& query) const {
    std::vector<float> tmp(encoder.dim);
    for (size_t i = 0; i < encoder.dim; ++i) tmp[i] = query[i];
    return quantize_query_from_ptr(tmp.data());
  }

  Quantized_Query<Metric> quantize_query(const float* qptr) const {
    return quantize_query_from_ptr(qptr);
  }

  void save(std::ofstream& out) const { encoder.save(out); }
  void load(std::ifstream& in) { encoder.load(in); }

 private:
  // Rotate, find max-abs, scale to int4 range [-7,+7], store sign-extended
  // as int8 (so VPDPBUSD can consume it directly). Precompute Q_sum and
  // q_nsf for the per-pair epilogue.
  Quantized_Query<Metric> quantize_query_from_ptr(const float* qptr) const {
    Quantized_Query<Metric> qq;
    qq.dim = encoder.padded_dim;
    qq.padded_dim = static_cast<uint32_t>(encoder.padded_dim);
    qq.query_int8.assign(internal::round_up_64(encoder.padded_dim), 0);

    if (encoder.padded_dim == 0) return qq;

    std::vector<float> ws(encoder.padded_dim);
    encoder.rotator->rotate(qptr, ws.data());

    float sqr_norm = 0.0f;
    float max_abs = 0.0f;
    for (size_t i = 0; i < encoder.padded_dim; ++i) {
      const float v = ws[i];
      sqr_norm += v * v;
      const float a = std::abs(v);
      if (a > max_abs) max_abs = a;
    }
    if (sqr_norm == 0.0f || !std::isfinite(sqr_norm) || max_abs == 0.0f) return qq;

    const float sf = static_cast<float>(internal::kQMax) / max_abs;
    int32_t qsum = 0;
    int64_t q_quant_sqr = 0;
    int8_t* qbuf = qq.query_int8.data();
    for (size_t i = 0; i < encoder.padded_dim; ++i) {
      int v = static_cast<int>(std::lround(ws[i] * sf));
      if (v > internal::kQMax) v = internal::kQMax;
      else if (v < internal::kQMin) v = internal::kQMin;
      const int8_t iv = static_cast<int8_t>(v);
      qbuf[i] = iv;
      qsum += iv;
      q_quant_sqr += static_cast<int64_t>(iv) * static_cast<int64_t>(iv);
    }

    qq.Q_sum = qsum;
    qq.unquantized_squared_norm = sqr_norm;
    if (q_quant_sqr > 0) {
      qq.norm_scaling_factor =
          std::sqrt(sqr_norm) / std::sqrt(static_cast<float>(q_quant_sqr));
    }
    return qq;
  }
};

}  // namespace turboquant_1bit_asym
}  // namespace mvsic
