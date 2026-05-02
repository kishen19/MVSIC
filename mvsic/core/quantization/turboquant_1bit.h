#pragma once

// Single-vector 1-bit TurboQuant: drop-in equivalent of turboquant::Model<bool>
// (4-bit, see turboquant.h) but with 1-bit / dim sign-bit codes.
//
// Hot path (one-to-many): VPDPBUSD-style nibble decode in the 4-bit version is
// replaced by AVX-512 XOR + VPOPCNTDQ. Each iteration of the inner loop
// consumes 64 bytes (= 512 dims) of packed signs vs 32 bytes (= 64 dims) for
// the 4-bit version, and the popcount itself is one instruction with the same
// reciprocal throughput as VPDPBUSD on Sapphire Rapids.
//
// Encoding pipeline mirrors the 4-bit one:
//   floats --(rotator->rotate)--> rotated coords --(sign bit)--> packed bytes
// We keep the per-vector norm_scaling_factor / unquantized_squared_norm fields
// so the interface matches turboquant.h exactly. (For unit-norm input, both
// reduce to constants, but keeping them per-vector lets us drop the unit-norm
// assumption transparently.)
//
// Distance:
//   centroid value at sign bit 0 = +127, at sign bit 1 = -127.
//   integer dot   = sum_d centroid_q[d] * centroid_db[d]
//                 = 16129 * (D - 2 * H)
//   where D = padded_dim, H = popcount(query XOR db).
//   neg_dot      = -dot * p_nsf * q_nsf
//   IP score     = neg_dot
//   L2 distance  = ||p||^2 + 2*neg_dot + ||q||^2
//
// Layout of Quantized_Query::query_data:
//   single packed-bit buffer, padded to a multiple of 64 bytes for safe
//   AVX-512 unaligned loads. Tail bytes past num_bytes_per_datapoint are zero
//   so the XOR contribution from those lanes is bounded by the popcount of
//   the corresponding DB tail bits (which is at most padded_dim mod 8 = 0..7).
//   Since DB is also zero-padded inside its allocated stride, the SIMD body
//   can run all the way to num_bytes_per_datapoint with a scalar tail for
//   the last <64 bytes — same pattern as turboquant.h.

#include <vector>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>
#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif
#include "parlay/parallel.h"
#include "parlay/sequence.h"
#include "rabitqlib/utils/rotator.hpp"

namespace mvsic {
namespace turboquant_1bit {

namespace internal {

// Squared norm of one quantized centroid (±127): 127^2 = 16129.
static constexpr float kCentroidSqr = 16129.0f;
static constexpr float kCentroidAbs = 127.0f;

// Round up to a multiple of 64 (AVX-512 lane width).
inline size_t round_up_64(size_t x) { return (x + 63) & ~size_t{63}; }

// Scalar popcount of (a XOR b) for `nbytes` bytes.
inline int32_t popcount_xor_scalar(const uint8_t* a, const uint8_t* b, size_t nbytes) {
  int32_t h = 0;
  size_t j = 0;
  // Process 8 bytes at a time when possible.
  for (; j + 8 <= nbytes; j += 8) {
    uint64_t aw, bw;
    std::memcpy(&aw, a + j, 8);
    std::memcpy(&bw, b + j, 8);
    h += __builtin_popcountll(aw ^ bw);
  }
  for (; j < nbytes; ++j) h += __builtin_popcount(a[j] ^ b[j]);
  return h;
}

// Hamming distance over `nbytes` of two packed-bit vectors. Uses the widest
// available SIMD popcount path; falls through to scalar for AVX-only or no-SIMD
// builds.
inline int32_t hamming_distance(const uint8_t* a, const uint8_t* b, size_t nbytes) {
#if defined(__AVX512F__) && defined(__AVX512VPOPCNTDQ__)
  __m512i acc = _mm512_setzero_si512();
  size_t j = 0;
  for (; j + 64 <= nbytes; j += 64) {
    const __m512i av = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(a + j));
    const __m512i bv = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(b + j));
    acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(_mm512_xor_si512(av, bv)));
  }
  int32_t h = static_cast<int32_t>(_mm512_reduce_add_epi64(acc));
  h += popcount_xor_scalar(a + j, b + j, nbytes - j);
  return h;
#elif defined(__AVX512F__) && defined(__AVX512BITALG__)
  // VPOPCNTB path: per-byte popcount, then horizontal sum into i64.
  __m512i acc = _mm512_setzero_si512();
  size_t j = 0;
  for (; j + 64 <= nbytes; j += 64) {
    const __m512i av = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(a + j));
    const __m512i bv = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(b + j));
    const __m512i pc = _mm512_popcnt_epi8(_mm512_xor_si512(av, bv));
    // Sum of 64 u8 in [0,8] fits in u16. Use sad_epu8 against 0 for hsum.
    acc = _mm512_add_epi64(acc, _mm512_sad_epu8(pc, _mm512_setzero_si512()));
  }
  int32_t h = static_cast<int32_t>(_mm512_reduce_add_epi64(acc));
  h += popcount_xor_scalar(a + j, b + j, nbytes - j);
  return h;
#elif defined(__AVX2__)
  // AVX2 nibble-LUT popcount. 32 bytes/iter.
  static const __m256i lut =
      _mm256_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
                       0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
  const __m256i mask = _mm256_set1_epi8(0x0F);
  __m256i acc = _mm256_setzero_si256();
  size_t j = 0;
  for (; j + 32 <= nbytes; j += 32) {
    const __m256i av = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + j));
    const __m256i bv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + j));
    const __m256i x = _mm256_xor_si256(av, bv);
    const __m256i lo = _mm256_and_si256(x, mask);
    const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(x, 4), mask);
    const __m256i pc = _mm256_add_epi8(_mm256_shuffle_epi8(lut, lo),
                                       _mm256_shuffle_epi8(lut, hi));
    acc = _mm256_add_epi64(acc, _mm256_sad_epu8(pc, _mm256_setzero_si256()));
  }
  alignas(32) int64_t buf[4];
  _mm256_store_si256(reinterpret_cast<__m256i*>(buf), acc);
  int32_t h = static_cast<int32_t>(buf[0] + buf[1] + buf[2] + buf[3]);
  h += popcount_xor_scalar(a + j, b + j, nbytes - j);
  return h;
#else
  return popcount_xor_scalar(a, b, nbytes);
#endif
}

}  // namespace internal

template<bool Metric>
class Quantized_Point;

// =========================================================================
// Single-Vector Query (packed sign bits, padded for SIMD safety)
// =========================================================================
template<bool Metric>
class Quantized_Query {
 public:
  using distanceType = float;

  // Layout: packed sign bits, 1 bit / padded dim. Padded to multiple of 64
  // bytes so AVX-512 loads at offset `num_bytes_per_datapoint - k` are safe.
  std::vector<uint8_t> query_data;
  size_t dim = 0;
  size_t num_bytes_per_datapoint = 0;
  uint32_t padded_dim = 0;

  float norm_scaling_factor = 0.0f;
  float unquantized_squared_norm = 0.0f;

  Quantized_Query() = default;

  inline float distance(const Quantized_Point<Metric>& p) const {
    return p.distance(*this);
  }

  template<typename PointRangeTy>
  void distances_all(const PointRangeTy& db, float* out) const {
    const size_t N = db.size();
    const size_t nb = num_bytes_per_datapoint;
    const float q_nsf = norm_scaling_factor;
    const float q_sqn = unquantized_squared_norm;
    const float D = static_cast<float>(padded_dim);
    // Hoist the affine map  neg_dot = -16129 * (D - 2H) * p_nsf * q_nsf
    //                              = (32258 * H - 16129 * D) * p_nsf * q_nsf.
    // We compute neg_dot via two FMAs in the epilogue (one per point).
    const float k_a = 2.0f * internal::kCentroidSqr * q_nsf;       // 32258 * q_nsf
    const float k_b = -internal::kCentroidSqr * D * q_nsf;         // -16129*D * q_nsf
    const uint8_t* qd = query_data.data();

    parlay::parallel_for(
        0, N,
        [&](size_t i) {
          const uint8_t* code_ptr = db.packed_codes.data() + i * nb;
          const float p_nsf = db.norm_scaling_factors[i];
          const float p_sqn = Metric ? db.unquantized_squared_norms[i] : 0.0f;

          const int32_t hamming = internal::hamming_distance(code_ptr, qd, nb);
          const float h_f = static_cast<float>(hamming);
          // neg_dot = (k_a * H + k_b) * p_nsf
          const float neg_dot = std::fma(k_a, h_f, k_b) * p_nsf;
          if constexpr (Metric) {
            out[i] = p_sqn + 2.0f * neg_dot + q_sqn;
          } else {
            out[i] = neg_dot;
          }
        },
        /*granularity=*/256);  // 1-bit kernel is much cheaper per point than 4-bit
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
  float norm_scaling_factor = 0.0f;
  float unquantized_squared_norm = 0.0f;

  Quantized_Point() = default;
  Quantized_Point(const uint8_t* ptr, size_t nb, uint32_t pdim, float nsf, float usn) :
      code_ptr(ptr), num_bytes(nb), padded_dim(pdim),
      norm_scaling_factor(nsf), unquantized_squared_norm(usn) {}

  inline float distance(const Quantized_Query<Metric>& qq) const {
    const int32_t hamming =
        internal::hamming_distance(code_ptr, qq.query_data.data(), num_bytes);
    const float D = static_cast<float>(padded_dim);
    const float dot_f = internal::kCentroidSqr * (D - 2.0f * static_cast<float>(hamming));
    const float neg_dot = -dot_f * norm_scaling_factor * qq.norm_scaling_factor;
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

  parlay::sequence<uint8_t> packed_codes;  // Flat layout: N * num_bytes_per_datapoint
  parlay::sequence<float> norm_scaling_factors;
  parlay::sequence<float> unquantized_squared_norms;

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
// Encoder: rotator + 1-bit sign-bit packer.
// Mirrors turboquant::BaseEncoder but without the 4-bit nibble pipeline.
// =========================================================================
class BaseEncoder1Bit {
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

  // Rotate, normalize, sign-pack. Returns {sqr_norm, norm_scaling_factor}.
  // norm_scaling_factor = ||x|| / sqrt(quantized_sqr_norm) so that the
  // hot-path arithmetic matches the 4-bit interface exactly.
  std::pair<float, float> encode_single(const float* p, uint8_t* output,
                                        std::vector<float>& ws) const {
    rotator->rotate(p, ws.data());

    float sqr_norm = 0.0f;
    for (size_t i = 0; i < padded_dim; ++i) sqr_norm += ws[i] * ws[i];

    std::memset(output, 0, num_bytes_per_datapoint);
    if (sqr_norm == 0.0f) return {0.0f, 0.0f};

    // Pack sign bits: bit set => coordinate is negative (matches the
    // convention in turboquant_1bit_mv.h and 1-bit-code/).
    for (size_t i = 0; i < padded_dim; ++i) {
      if (ws[i] < 0.0f) output[i / 8] |= static_cast<uint8_t>(1u << (i % 8));
    }

    const float norm = std::sqrt(sqr_norm);
    // Quantized squared norm is the constant D * 127^2.
    const float q_sqr = static_cast<float>(padded_dim) * internal::kCentroidSqr;
    return {sqr_norm, norm / std::sqrt(q_sqr)};
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
  BaseEncoder1Bit encoder;

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
  Quantized_Query<Metric> quantize_query_from_ptr(const float* qptr) const {
    Quantized_Query<Metric> qq;
    qq.dim = encoder.padded_dim;
    qq.num_bytes_per_datapoint = encoder.num_bytes_per_datapoint;
    qq.padded_dim = static_cast<uint32_t>(encoder.padded_dim);

    // Pad the buffer to a multiple of 64 so the AVX-512 SIMD body never
    // reads past the end. (The actual hot loop reads only up to nb with a
    // scalar tail, but the padding gives the AVX-512 tail-free variant
    // headroom if we ever switch to it.)
    const size_t padded = internal::round_up_64(qq.num_bytes_per_datapoint);
    qq.query_data.assign(padded, 0);

    if (encoder.padded_dim == 0) return qq;

    std::vector<float> ws(encoder.padded_dim);
    encoder.rotator->rotate(qptr, ws.data());

    float sqr_norm = 0.0f;
    for (size_t i = 0; i < encoder.padded_dim; ++i) sqr_norm += ws[i] * ws[i];
    if (sqr_norm == 0.0f || !std::isfinite(sqr_norm)) return qq;

    for (size_t i = 0; i < encoder.padded_dim; ++i) {
      if (ws[i] < 0.0f)
        qq.query_data[i / 8] |= static_cast<uint8_t>(1u << (i % 8));
    }

    const float norm = std::sqrt(sqr_norm);
    const float q_sqr = static_cast<float>(encoder.padded_dim) * internal::kCentroidSqr;
    qq.norm_scaling_factor = norm / std::sqrt(q_sqr);
    qq.unquantized_squared_norm = sqr_norm;
    return qq;
  }
};

}  // namespace turboquant_1bit
}  // namespace mvsic
