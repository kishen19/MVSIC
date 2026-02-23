#pragma once

// low_bit_turboquant.h
//
// 1-bit and 2-bit TurboQuant quantization for quality investigation.
// Reuses the same Hadamard rotation + normalization as 4-bit TQ, but uses
// optimal Lloyd-Max codebooks for N(0,1) at fewer bits per dimension.
//
// Storage per vector (D=128, padded_dim=128):
//   1-bit: 16 bytes   (sign only)
//   2-bit: 32 bytes   (sign + 2 magnitude levels)
//
// These are scalar/float implementations intended for quality benchmarking,
// not for production throughput.

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace mvsic {
namespace low_bit_turboquant {

// =========================================================================
// Constants: optimal Lloyd-Max codebooks for N(0,1)
// =========================================================================

// 1-bit (2 levels): centroids = ±E[|X|] for standard normal.
static constexpr float k1BitCentroid = 0.7978845608f;

// 2-bit (4 levels): optimal Lloyd-Max for N(0,1).
//   Centroids: ±0.4528 (inner), ±1.5104 (outer)
//   Boundary between inner/outer: |x| = 0.9816
static constexpr float k2BitBoundary = 0.9816f;
static constexpr std::array<float, 4> k2BitCentroids = {
    0.4528f, 1.5104f, -0.4528f, -1.5104f};
// Code layout: 0 = +small, 1 = +large, 2 = -small, 3 = -large

// =========================================================================
// Data structures
// =========================================================================

// Encoded vector (DB point).
struct EncodedVec {
  std::vector<uint8_t> packed_codes;  // bit-packed codes
  float norm_scaling_factor = 0.0f;
  float unquantized_squared_norm = 0.0f;
};

// Prepared query (float, no quantization error).
struct PreparedQuery {
  std::vector<float> rotated;  // rotated + normalized + scaled float coordinates
  float squared_norm = 0.0f;
};

// =========================================================================
// Encoding functions (templated on TQ Model type)
// =========================================================================

// Encode a single float-vector into 1-bit codes.
template<typename TQModel>
inline EncodedVec encode_1bit(
    const TQModel& model,
    const float* p, std::vector<float>& ws) {
  EncodedVec out;
  const size_t pdim = model.padded_dim;
  ws.resize(pdim);

  // Step 1: Hadamard rotate.
  model.rotator->rotate(p, ws.data());

  // Step 2: Compute norm.
  float sqr_norm = 0.0f;
  for (size_t i = 0; i < pdim; ++i) sqr_norm += ws[i] * ws[i];
  out.unquantized_squared_norm = sqr_norm;
  if (sqr_norm == 0.0f) {
    out.packed_codes.resize((pdim + 7) / 8, 0);
    out.norm_scaling_factor = 0.0f;
    return out;
  }

  // Step 3: Normalize and scale by √padded_dim.
  const float norm = std::sqrt(sqr_norm);
  const float inv_norm = 1.0f / norm;
  const float had_scale = std::sqrt(static_cast<float>(pdim));
  for (size_t i = 0; i < pdim; ++i) ws[i] *= inv_norm * had_scale;

  // Step 4: Quantize — sign only.
  const size_t nbytes = (pdim + 7) / 8;
  out.packed_codes.resize(nbytes, 0);
  float q_sqr = 0.0f;
  for (size_t i = 0; i < pdim; ++i) {
    if (ws[i] >= 0.0f)
      out.packed_codes[i / 8] |= (1u << (i % 8));
    q_sqr += k1BitCentroid * k1BitCentroid;
  }

  out.norm_scaling_factor = norm / std::sqrt(q_sqr);
  return out;
}

// Encode a single float-vector into 2-bit codes.
template<typename TQModel>
inline EncodedVec encode_2bit(
    const TQModel& model,
    const float* p, std::vector<float>& ws) {
  EncodedVec out;
  const size_t pdim = model.padded_dim;
  ws.resize(pdim);

  model.rotator->rotate(p, ws.data());

  float sqr_norm = 0.0f;
  for (size_t i = 0; i < pdim; ++i) sqr_norm += ws[i] * ws[i];
  out.unquantized_squared_norm = sqr_norm;
  if (sqr_norm == 0.0f) {
    out.packed_codes.resize((pdim + 3) / 4, 0);
    out.norm_scaling_factor = 0.0f;
    return out;
  }

  const float norm = std::sqrt(sqr_norm);
  const float inv_norm = 1.0f / norm;
  const float had_scale = std::sqrt(static_cast<float>(pdim));
  for (size_t i = 0; i < pdim; ++i) ws[i] *= inv_norm * had_scale;

  // Quantize: 2-bit (4 levels).
  const size_t nbytes = (pdim + 3) / 4;
  out.packed_codes.resize(nbytes, 0);
  float q_sqr = 0.0f;
  for (size_t i = 0; i < pdim; ++i) {
    float x = ws[i];
    float ax = std::abs(x);
    uint8_t code;
    if (x >= 0.0f) {
      code = (ax < k2BitBoundary) ? 0 : 1;
    } else {
      code = (ax < k2BitBoundary) ? 2 : 3;
    }
    q_sqr += k2BitCentroids[code] * k2BitCentroids[code];
    out.packed_codes[i / 4] |= (code << (2 * (i % 4)));
  }

  out.norm_scaling_factor = norm / std::sqrt(q_sqr);
  return out;
}

// =========================================================================
// Query preparation
// =========================================================================

// Prepare a query vector: rotate + normalize + scale (kept as float).
template<typename TQModel>
inline PreparedQuery prepare_query(
    const TQModel& model,
    const float* qptr) {
  PreparedQuery qq;
  const size_t pdim = model.padded_dim;
  qq.rotated.resize(pdim);

  model.rotator->rotate(qptr, qq.rotated.data());

  float sqr_norm = 0.0f;
  for (size_t i = 0; i < pdim; ++i) sqr_norm += qq.rotated[i] * qq.rotated[i];
  qq.squared_norm = sqr_norm;

  if (sqr_norm == 0.0f) return qq;

  const float norm = std::sqrt(sqr_norm);
  const float inv_norm = 1.0f / norm;
  const float had_scale = std::sqrt(static_cast<float>(pdim));
  for (size_t i = 0; i < pdim; ++i) qq.rotated[i] *= inv_norm * had_scale;

  return qq;
}

// =========================================================================
// Distance functions
// =========================================================================

// Scalar distance: 1-bit encoded point vs float query.
inline float distance_1bit(
    const EncodedVec& db_pt, const PreparedQuery& qq, size_t pdim, bool metric) {
  float dot = 0.0f;
  for (size_t i = 0; i < pdim; ++i) {
    bool positive = (db_pt.packed_codes[i / 8] >> (i % 8)) & 1;
    float c = positive ? k1BitCentroid : -k1BitCentroid;
    dot += c * qq.rotated[i];
  }

  // Post-transform: nsf_query = query_norm / sqrt(pdim) since
  // the rotated query has squared norm = pdim after normalize+scale.
  float query_norm = std::sqrt(qq.squared_norm);
  float nsf_query = (query_norm > 0.0f) ? query_norm / std::sqrt(static_cast<float>(pdim)) : 0.0f;
  float neg_ip = -(dot * db_pt.norm_scaling_factor * nsf_query);

  if (metric) {
    return db_pt.unquantized_squared_norm + 2.0f * neg_ip + qq.squared_norm;
  }
  return neg_ip;
}

// Scalar distance: 2-bit encoded point vs float query.
inline float distance_2bit(
    const EncodedVec& db_pt, const PreparedQuery& qq, size_t pdim, bool metric) {
  float dot = 0.0f;
  for (size_t i = 0; i < pdim; ++i) {
    uint8_t code = (db_pt.packed_codes[i / 4] >> (2 * (i % 4))) & 0x3;
    dot += k2BitCentroids[code] * qq.rotated[i];
  }

  float query_norm = std::sqrt(qq.squared_norm);
  float nsf_query = (query_norm > 0.0f) ? query_norm / std::sqrt(static_cast<float>(pdim)) : 0.0f;

  float neg_ip = -(dot * db_pt.norm_scaling_factor * nsf_query);

  if (metric) {
    return db_pt.unquantized_squared_norm + 2.0f * neg_ip + qq.squared_norm;
  }
  return neg_ip;
}

}  // namespace low_bit_turboquant
}  // namespace mvsic
