#pragma once

// Reference (slow, scalar) single-vector 1-bit TurboQuant.
//
// Two variants, intended purely for ablation against the production
// `turboquant_1bit.h` and against RaBitQ-1bit:
//
//   asym::Model — 1-bit DB sign codes scored against a full-float
//     rotated query. Estimator:
//         <q, x> ≈ <q_rot, sign(x_rot)> · ||x|| / sqrt(D)
//     This is the "best the 1-bit DB can do" — keeps all of the
//     query's information. RaBitQ-1bit takes this same path but
//     compresses the query to 4 bits (and uses a learned centroid).
//     The legacy 1BTQ in `old/1-bit-code/` also took this path with
//     an 8-bit query × ±127 DB centroids.
//
//   sym::Model — symmetric 1-bit × 1-bit. Both DB and query are
//     sign-packed; scoring is Hamming-based. Estimator:
//         <q, x> ≈ ||q||·||x| · (D − 2·H(sign(R q), sign(R x))) / D
//     This matches what the production `turboquant_1bit.h` actually
//     computes (it's just spelled out as a scalar loop here so the
//     math is auditable).
//
// Both variants share the same DB encoding (rotate, sign-pack, store
// per-vector norms). Only the Quantized_Query layout and the per-pair
// score function differ. They live in sibling namespaces rather than
// behind a template so that each one reads top-to-bottom as a small
// self-contained reference.

#include <vector>
#include <cstdint>
#include <cstring>
#include <cmath>
#include "parlay/parallel.h"
#include "parlay/sequence.h"
#include "rabitqlib/utils/rotator.hpp"

namespace mvsic {
namespace turboquant_1bit_ref {

// =========================================================================
// Shared bits: rotator setup + the sign-pack DB encoder.
// =========================================================================
namespace internal {

inline std::unique_ptr<rabitqlib::Rotator<float>> make_rotator(size_t dim) {
  return std::unique_ptr<rabitqlib::Rotator<float>>(
      rabitqlib::choose_rotator<float>(dim, rabitqlib::RotatorType::FhtKacRotator));
}

// Sign-pack `padded_dim` floats into `(padded_dim+7)/8` bytes; bit set
// iff the corresponding coordinate is negative. Returns squared norm.
inline float sign_pack(const float* rotated, size_t padded_dim, uint8_t* out) {
  const size_t nb = (padded_dim + 7) / 8;
  std::memset(out, 0, nb);
  float sqr = 0.0f;
  for (size_t d = 0; d < padded_dim; ++d) {
    sqr += rotated[d] * rotated[d];
    if (rotated[d] < 0.0f) out[d / 8] |= static_cast<uint8_t>(1u << (d % 8));
  }
  return sqr;
}

}  // namespace internal


// =========================================================================
// Asymmetric reference: 1-bit DB × full-float query.
// =========================================================================
namespace asym {

template<bool Metric>
class Quantized_Point;

template<bool Metric>
class Quantized_Query {
 public:
  using distanceType = float;
  // Full-precision rotated query, length = padded_dim.
  std::vector<float> q_rot;
  uint32_t padded_dim = 0;
  float unquantized_squared_norm = 0.0f;

  Quantized_Query() = default;

  inline float distance(const Quantized_Point<Metric>& p) const {
    return p.distance(*this);
  }
};

template<bool Metric>
class Quantized_Point {
 public:
  const uint8_t* code_ptr = nullptr;
  size_t num_bytes = 0;
  uint32_t padded_dim = 0;
  float norm = 0.0f;                       // ||x||
  float unquantized_squared_norm = 0.0f;   // ||x||^2 (only used when Metric==true)

  Quantized_Point() = default;
  Quantized_Point(const uint8_t* ptr, size_t nb, uint32_t pdim, float n, float usn) :
      code_ptr(ptr), num_bytes(nb), padded_dim(pdim), norm(n), unquantized_squared_norm(usn) {}

  inline float distance(const Quantized_Query<Metric>& qq) const {
    // s = sum_d (bit==0 ? +q_rot[d] : -q_rot[d])
    // est <q,x> ≈ s · ||x|| / sqrt(D).
    const float* q = qq.q_rot.data();
    const uint32_t D = padded_dim;
    float s = 0.0f;
    for (uint32_t d = 0; d < D; ++d) {
      const uint8_t bit = (code_ptr[d >> 3] >> (d & 7)) & 1u;
      s += bit ? -q[d] : q[d];
    }
    const float ip_est = s * norm / std::sqrt(static_cast<float>(D));
    if constexpr (Metric) {
      return unquantized_squared_norm + qq.unquantized_squared_norm - 2.0f * ip_est;
    } else {
      return -ip_est;
    }
  }

  void prefetch() const { if (code_ptr) __builtin_prefetch(code_ptr, 0, 3); }
  bool same_as(const Quantized_Point<Metric>&) const { return false; }
  bool same_as(const Quantized_Query<Metric>&) const { return false; }
  bool is_metric() const { return Metric; }
};

template<typename PointRangeTy, bool Metric>
class Quantized_Point_Range {
 public:
  static constexpr bool is_fastscan = false;
  size_t n_points = 0;
  size_t num_bytes_per_datapoint = 0;
  uint32_t padded_dim = 0;
  parlay::sequence<uint8_t> packed_codes;
  parlay::sequence<float> norms;
  parlay::sequence<float> unquantized_squared_norms;

  Quantized_Point<Metric> operator[](size_t i) const {
    const uint8_t* ptr = packed_codes.data() + i * num_bytes_per_datapoint;
    float sqn = Metric ? unquantized_squared_norms[i] : 0.0f;
    return Quantized_Point<Metric>(ptr, num_bytes_per_datapoint, padded_dim, norms[i], sqn);
  }
  inline uint32_t size() const noexcept { return static_cast<uint32_t>(n_points); }
};

template<bool Metric>
class Model {
 public:
  size_t dim = 0;
  size_t padded_dim = 0;
  size_t num_bytes_per_datapoint = 0;
  std::unique_ptr<rabitqlib::Rotator<float>> rotator;

  Model() = default;

  template<typename PointRangeTy>
  void train(const PointRangeTy& data) {
    dim = data.get_dims();
    if (dim == 0) return;
    rotator = internal::make_rotator(dim);
    if (!rotator) return;
    padded_dim = rotator->size();
    num_bytes_per_datapoint = (padded_dim + 7) / 8;
  }

  template<typename PointRangeTy>
  Quantized_Point_Range<PointRangeTy, Metric> encode(const PointRangeTy& data) const {
    Quantized_Point_Range<PointRangeTy, Metric> enc;
    enc.n_points = data.size();
    enc.num_bytes_per_datapoint = num_bytes_per_datapoint;
    enc.padded_dim = static_cast<uint32_t>(padded_dim);
    enc.packed_codes.resize(enc.n_points * enc.num_bytes_per_datapoint);
    enc.norms.resize(enc.n_points);
    enc.unquantized_squared_norms.resize(enc.n_points);

    parlay::parallel_for(0, enc.n_points, [&](size_t i) {
      std::vector<float> ws(padded_dim);
      const float* p = reinterpret_cast<const float*>(data.location(i));
      rotator->rotate(p, ws.data());
      uint8_t* out = enc.packed_codes.data() + i * num_bytes_per_datapoint;
      const float sqr = internal::sign_pack(ws.data(), padded_dim, out);
      enc.unquantized_squared_norms[i] = sqr;
      enc.norms[i] = std::sqrt(sqr);
    });
    return enc;
  }

  template<typename PointTy>
  Quantized_Query<Metric> quantize_query(const PointTy& query) const {
    std::vector<float> tmp(dim);
    for (size_t i = 0; i < dim; ++i) tmp[i] = query[i];
    return quantize_query_from_ptr(tmp.data());
  }
  Quantized_Query<Metric> quantize_query(const float* qptr) const {
    return quantize_query_from_ptr(qptr);
  }

 private:
  Quantized_Query<Metric> quantize_query_from_ptr(const float* qptr) const {
    Quantized_Query<Metric> qq;
    qq.padded_dim = static_cast<uint32_t>(padded_dim);
    qq.q_rot.assign(padded_dim, 0.0f);
    if (padded_dim == 0) return qq;
    rotator->rotate(qptr, qq.q_rot.data());
    float sqr = 0.0f;
    for (size_t i = 0; i < padded_dim; ++i) sqr += qq.q_rot[i] * qq.q_rot[i];
    qq.unquantized_squared_norm = sqr;
    return qq;
  }
};

}  // namespace asym


// =========================================================================
// Symmetric reference: 1-bit DB × 1-bit query (Hamming).
// Mirrors what the production `turboquant_1bit.h` does, but as a plain
// scalar loop so the math is easy to read and modify.
// =========================================================================
namespace sym {

template<bool Metric>
class Quantized_Point;

template<bool Metric>
class Quantized_Query {
 public:
  using distanceType = float;
  // Sign-packed rotated query, (padded_dim+7)/8 bytes.
  std::vector<uint8_t> q_signs;
  uint32_t padded_dim = 0;
  float norm = 0.0f;                       // ||q||
  float unquantized_squared_norm = 0.0f;   // ||q||^2

  Quantized_Query() = default;

  inline float distance(const Quantized_Point<Metric>& p) const {
    return p.distance(*this);
  }
};

template<bool Metric>
class Quantized_Point {
 public:
  const uint8_t* code_ptr = nullptr;
  size_t num_bytes = 0;
  uint32_t padded_dim = 0;
  float norm = 0.0f;                       // ||x||
  float unquantized_squared_norm = 0.0f;   // ||x||^2 (only used when Metric==true)

  Quantized_Point() = default;
  Quantized_Point(const uint8_t* ptr, size_t nb, uint32_t pdim, float n, float usn) :
      code_ptr(ptr), num_bytes(nb), padded_dim(pdim), norm(n), unquantized_squared_norm(usn) {}

  inline float distance(const Quantized_Query<Metric>& qq) const {
    // H = popcount(q_signs XOR x_signs) over `num_bytes` bytes.
    // est <q,x> ≈ ||q||·||x|| · (D − 2H) / D.
    int32_t H = 0;
    const uint8_t* qs = qq.q_signs.data();
    for (size_t b = 0; b < num_bytes; ++b) {
      H += __builtin_popcount(static_cast<unsigned int>(code_ptr[b] ^ qs[b]));
    }
    const float D = static_cast<float>(padded_dim);
    const float ip_est = qq.norm * norm * (D - 2.0f * static_cast<float>(H)) / D;
    if constexpr (Metric) {
      return unquantized_squared_norm + qq.unquantized_squared_norm - 2.0f * ip_est;
    } else {
      return -ip_est;
    }
  }

  void prefetch() const { if (code_ptr) __builtin_prefetch(code_ptr, 0, 3); }
  bool same_as(const Quantized_Point<Metric>&) const { return false; }
  bool same_as(const Quantized_Query<Metric>&) const { return false; }
  bool is_metric() const { return Metric; }
};

template<typename PointRangeTy, bool Metric>
class Quantized_Point_Range {
 public:
  static constexpr bool is_fastscan = false;
  size_t n_points = 0;
  size_t num_bytes_per_datapoint = 0;
  uint32_t padded_dim = 0;
  parlay::sequence<uint8_t> packed_codes;
  parlay::sequence<float> norms;
  parlay::sequence<float> unquantized_squared_norms;

  Quantized_Point<Metric> operator[](size_t i) const {
    const uint8_t* ptr = packed_codes.data() + i * num_bytes_per_datapoint;
    float sqn = Metric ? unquantized_squared_norms[i] : 0.0f;
    return Quantized_Point<Metric>(ptr, num_bytes_per_datapoint, padded_dim, norms[i], sqn);
  }
  inline uint32_t size() const noexcept { return static_cast<uint32_t>(n_points); }
};

template<bool Metric>
class Model {
 public:
  size_t dim = 0;
  size_t padded_dim = 0;
  size_t num_bytes_per_datapoint = 0;
  std::unique_ptr<rabitqlib::Rotator<float>> rotator;

  Model() = default;

  template<typename PointRangeTy>
  void train(const PointRangeTy& data) {
    dim = data.get_dims();
    if (dim == 0) return;
    rotator = internal::make_rotator(dim);
    if (!rotator) return;
    padded_dim = rotator->size();
    num_bytes_per_datapoint = (padded_dim + 7) / 8;
  }

  template<typename PointRangeTy>
  Quantized_Point_Range<PointRangeTy, Metric> encode(const PointRangeTy& data) const {
    Quantized_Point_Range<PointRangeTy, Metric> enc;
    enc.n_points = data.size();
    enc.num_bytes_per_datapoint = num_bytes_per_datapoint;
    enc.padded_dim = static_cast<uint32_t>(padded_dim);
    enc.packed_codes.resize(enc.n_points * enc.num_bytes_per_datapoint);
    enc.norms.resize(enc.n_points);
    enc.unquantized_squared_norms.resize(enc.n_points);

    parlay::parallel_for(0, enc.n_points, [&](size_t i) {
      std::vector<float> ws(padded_dim);
      const float* p = reinterpret_cast<const float*>(data.location(i));
      rotator->rotate(p, ws.data());
      uint8_t* out = enc.packed_codes.data() + i * num_bytes_per_datapoint;
      const float sqr = internal::sign_pack(ws.data(), padded_dim, out);
      enc.unquantized_squared_norms[i] = sqr;
      enc.norms[i] = std::sqrt(sqr);
    });
    return enc;
  }

  template<typename PointTy>
  Quantized_Query<Metric> quantize_query(const PointTy& query) const {
    std::vector<float> tmp(dim);
    for (size_t i = 0; i < dim; ++i) tmp[i] = query[i];
    return quantize_query_from_ptr(tmp.data());
  }
  Quantized_Query<Metric> quantize_query(const float* qptr) const {
    return quantize_query_from_ptr(qptr);
  }

 private:
  Quantized_Query<Metric> quantize_query_from_ptr(const float* qptr) const {
    Quantized_Query<Metric> qq;
    qq.padded_dim = static_cast<uint32_t>(padded_dim);
    qq.q_signs.assign(num_bytes_per_datapoint, 0u);
    if (padded_dim == 0) return qq;
    std::vector<float> ws(padded_dim);
    rotator->rotate(qptr, ws.data());
    const float sqr = internal::sign_pack(ws.data(), padded_dim, qq.q_signs.data());
    qq.unquantized_squared_norm = sqr;
    qq.norm = std::sqrt(sqr);
    return qq;
  }
};

}  // namespace sym

}  // namespace turboquant_1bit_ref
}  // namespace mvsic
