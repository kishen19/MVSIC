#pragma once

// centered_turboquant.h
//
// Centered TurboQuant: combines TQ's optimal Lloyd-Max codebook with
// RaBitQ-style mean subtraction for improved quality on structured datasets.
//
// Pipeline:
//   1. Rotate + normalize + scale by sqrt(d) (same as standard TQ)
//   2. Subtract the per-dimension mean μ (learned from training data)
//   3. Quantize the residual with 1-, 2-, or 4-bit codebook
//   4. At distance time, add back the mean contribution:
//        dot_full = dot(centroids, q_scaled) + dot(μ, q_scaled)
//        IP ≈ dot_full × nsf_db × nsf_q
//
// The key insight: for datasets where vectors cluster around a shared
// direction, centering makes the residuals smaller and more isotropic,
// so even coarse quantization captures the distance structure well.

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "mvsic/core/quantization/turboquant_4bit.h"

namespace mvsic {
namespace turboquant_centered {

// =========================================================================
// Codebook constants (same as turboquant_low_bit.h + TQ 4-bit)
// =========================================================================

// 1-bit: ±E[|X|] for N(0,1).
static constexpr float k1BitCentroid = 0.7978845608f;

// 2-bit: optimal 4-level Lloyd-Max for N(0,1).
static constexpr float k2BitBoundary = 0.9816f;
static constexpr std::array<float, 4> k2BitCentroids = {0.4528f, 1.5104f, -0.4528f, -1.5104f};

// 4-bit: use TQ's FourBitEncoding for bucket assignment (same boundaries),
// but with proper Lloyd-Max reconstruction centroids E[X | X ∈ bucket]
// for N(0,1). These differ from kTurboQuantCentroidsFloat, which are
// co-optimized with the int8 scoring kernel and are ~2× too small for
// standalone float-domain use.
//
// Buckets defined by kBoundaries = {0.258, 0.527, 0.807, 1.097, 1.431, 1.840, 2.399}
// Centroids computed as E[X | a ≤ X < b] = (φ(a) − φ(b)) / (Φ(b) − Φ(a))
// for the standard normal distribution.
static constexpr std::array<float, 16> k4BitCentroidsFloat = {
    0.1309f,  0.3900f,  0.6583f,  0.9583f,  1.2488f,  1.5256f,  2.1894f,  2.7494f,
    -0.1309f, -0.3900f, -0.6583f, -0.9583f, -1.2488f, -1.5256f, -2.1894f, -2.7494f};

// =========================================================================
// Centered model: per-dimension mean in scaled rotated space
// =========================================================================

struct CenteredModel {
  std::vector<float> mean_scaled;  // μ_i, length = padded_dim
  size_t padded_dim = 0;
  float mean_sq_norm = 0.0f;  // ||μ||², precomputed
};

// Train: compute per-dimension mean of scaled rotated coordinates.
// Samples up to 50k points for efficiency.
template<typename TQModel, typename PointRange>
inline CenteredModel train_centered(const TQModel& tq_model, const PointRange& data) {
  CenteredModel cm;
  cm.padded_dim = tq_model.padded_dim;
  const size_t pdim = cm.padded_dim;
  cm.mean_scaled.resize(pdim, 0.0f);

  const size_t n = data.size();
  const size_t sample_size = std::min(n, size_t(50000));

  // Accumulate per-dimension sum (double for precision).
  std::vector<double> sum(pdim, 0.0);

  for (size_t i = 0; i < sample_size; ++i) {
    std::vector<float> ws(pdim);
    const float* p = reinterpret_cast<const float*>(data.location(i));
    tq_model.rotator->rotate(p, ws.data());

    // Normalize + scale.
    float sqr_norm = 0.0f;
    for (size_t j = 0; j < pdim; ++j)
      sqr_norm += ws[j] * ws[j];
    if (sqr_norm == 0.0f) continue;
    float inv_norm = 1.0f / std::sqrt(sqr_norm);
    float had_scale = std::sqrt(static_cast<float>(pdim));
    for (size_t j = 0; j < pdim; ++j)
      sum[j] += static_cast<double>(ws[j] * inv_norm * had_scale);
  }

  double inv = 1.0 / static_cast<double>(sample_size);
  cm.mean_sq_norm = 0.0f;
  for (size_t j = 0; j < pdim; ++j) {
    cm.mean_scaled[j] = static_cast<float>(sum[j] * inv);
    cm.mean_sq_norm += cm.mean_scaled[j] * cm.mean_scaled[j];
  }

  return cm;
}

// =========================================================================
// Data structures
// =========================================================================

struct EncodedVec {
  std::vector<uint8_t> packed_codes;  // bit-packed codes for residual
  float norm_scaling_factor = 0.0f;   // ||x_orig|| / ||recon_scaled||
  float unquantized_squared_norm = 0.0f;
};

struct PreparedQuery {
  std::vector<float> scaled;    // q_scaled (NOT centered)
  float squared_norm = 0.0f;    // ||q_orig||²
  float dot_mean_query = 0.0f;  // <μ, q_scaled>, precomputed
};

// =========================================================================
// Encoding functions
// =========================================================================

// Encode a single vector into 1-bit centered codes.
template<typename TQModel>
inline EncodedVec encode_1bit_centered(const TQModel& model, const CenteredModel& cm,
                                       const float* p, std::vector<float>& ws) {
  EncodedVec out;
  const size_t pdim = cm.padded_dim;
  ws.resize(pdim);

  // Rotate.
  model.rotator->rotate(p, ws.data());

  // Compute norm.
  float sqr_norm = 0.0f;
  for (size_t i = 0; i < pdim; ++i)
    sqr_norm += ws[i] * ws[i];
  out.unquantized_squared_norm = sqr_norm;
  if (sqr_norm == 0.0f) {
    out.packed_codes.resize((pdim + 7) / 8, 0);
    return out;
  }

  // Normalize + scale.
  const float norm = std::sqrt(sqr_norm);
  const float inv_norm = 1.0f / norm;
  const float had_scale = std::sqrt(static_cast<float>(pdim));
  for (size_t i = 0; i < pdim; ++i)
    ws[i] *= inv_norm * had_scale;

  // Subtract mean → residual.
  for (size_t i = 0; i < pdim; ++i)
    ws[i] -= cm.mean_scaled[i];

  // Quantize residual: sign only.
  const size_t nbytes = (pdim + 7) / 8;
  out.packed_codes.resize(nbytes, 0);
  float recon_sq = 0.0f;
  for (size_t i = 0; i < pdim; ++i) {
    float centroid = (ws[i] >= 0.0f) ? k1BitCentroid : -k1BitCentroid;
    if (ws[i] >= 0.0f) out.packed_codes[i / 8] |= (1u << (i % 8));
    // Reconstruction = centroid + μ_i.
    float recon_i = centroid + cm.mean_scaled[i];
    recon_sq += recon_i * recon_i;
  }

  out.norm_scaling_factor = (recon_sq > 0.0f) ? norm / std::sqrt(recon_sq) : 0.0f;
  return out;
}

// Encode a single vector into 2-bit centered codes.
template<typename TQModel>
inline EncodedVec encode_2bit_centered(const TQModel& model, const CenteredModel& cm,
                                       const float* p, std::vector<float>& ws) {
  EncodedVec out;
  const size_t pdim = cm.padded_dim;
  ws.resize(pdim);

  model.rotator->rotate(p, ws.data());

  float sqr_norm = 0.0f;
  for (size_t i = 0; i < pdim; ++i)
    sqr_norm += ws[i] * ws[i];
  out.unquantized_squared_norm = sqr_norm;
  if (sqr_norm == 0.0f) {
    out.packed_codes.resize((pdim + 3) / 4, 0);
    return out;
  }

  const float norm = std::sqrt(sqr_norm);
  const float inv_norm = 1.0f / norm;
  const float had_scale = std::sqrt(static_cast<float>(pdim));
  for (size_t i = 0; i < pdim; ++i)
    ws[i] *= inv_norm * had_scale;

  // Subtract mean → residual.
  for (size_t i = 0; i < pdim; ++i)
    ws[i] -= cm.mean_scaled[i];

  // Quantize residual: 2-bit (4 levels).
  const size_t nbytes = (pdim + 3) / 4;
  out.packed_codes.resize(nbytes, 0);
  float recon_sq = 0.0f;
  for (size_t i = 0; i < pdim; ++i) {
    float x = ws[i];
    float ax = std::abs(x);
    uint8_t code;
    if (x >= 0.0f) {
      code = (ax < k2BitBoundary) ? 0 : 1;
    } else {
      code = (ax < k2BitBoundary) ? 2 : 3;
    }
    float centroid = k2BitCentroids[code];
    float recon_i = centroid + cm.mean_scaled[i];
    recon_sq += recon_i * recon_i;
    out.packed_codes[i / 4] |= (code << (2 * (i % 4)));
  }

  out.norm_scaling_factor = (recon_sq > 0.0f) ? norm / std::sqrt(recon_sq) : 0.0f;
  return out;
}

// Encode a single vector into 4-bit centered codes.
template<typename TQModel>
inline EncodedVec encode_4bit_centered(const TQModel& model, const CenteredModel& cm,
                                       const float* p, std::vector<float>& ws) {
  EncodedVec out;
  const size_t pdim = cm.padded_dim;
  ws.resize(pdim);

  model.rotator->rotate(p, ws.data());

  float sqr_norm = 0.0f;
  for (size_t i = 0; i < pdim; ++i)
    sqr_norm += ws[i] * ws[i];
  out.unquantized_squared_norm = sqr_norm;
  if (sqr_norm == 0.0f) {
    out.packed_codes.resize(pdim / 2, 0);
    return out;
  }

  const float norm = std::sqrt(sqr_norm);
  const float inv_norm = 1.0f / norm;
  const float had_scale = std::sqrt(static_cast<float>(pdim));
  for (size_t i = 0; i < pdim; ++i)
    ws[i] *= inv_norm * had_scale;

  // Subtract mean → residual.
  for (size_t i = 0; i < pdim; ++i)
    ws[i] -= cm.mean_scaled[i];

  // Quantize residual: 4-bit using TQ's codebook.
  const size_t nbytes = pdim / 2;
  out.packed_codes.resize(nbytes, 0);
  float recon_sq = 0.0f;
  for (size_t i = 0; i < pdim; ++i) {
    uint8_t code = turboquant_4bit::internal::FourBitEncoding(ws[i]);
    float centroid = k4BitCentroidsFloat[code];
    float recon_i = centroid + cm.mean_scaled[i];
    recon_sq += recon_i * recon_i;
    if (i % 2 == 0)
      out.packed_codes[i / 2] = code;
    else
      out.packed_codes[i / 2] |= (code << 4);
  }

  out.norm_scaling_factor = (recon_sq > 0.0f) ? norm / std::sqrt(recon_sq) : 0.0f;
  return out;
}

// =========================================================================
// Query preparation (shared across all bit depths)
// =========================================================================

template<typename TQModel>
inline PreparedQuery prepare_query_centered(const TQModel& model, const CenteredModel& cm,
                                            const float* qptr) {
  PreparedQuery qq;
  const size_t pdim = cm.padded_dim;
  qq.scaled.resize(pdim);

  model.rotator->rotate(qptr, qq.scaled.data());

  float sqr_norm = 0.0f;
  for (size_t i = 0; i < pdim; ++i)
    sqr_norm += qq.scaled[i] * qq.scaled[i];
  qq.squared_norm = sqr_norm;

  if (sqr_norm == 0.0f) {
    qq.dot_mean_query = 0.0f;
    return qq;
  }

  // Normalize + scale (query keeps original scaled values, NOT centered).
  const float norm = std::sqrt(sqr_norm);
  const float inv_norm = 1.0f / norm;
  const float had_scale = std::sqrt(static_cast<float>(pdim));
  for (size_t i = 0; i < pdim; ++i)
    qq.scaled[i] *= inv_norm * had_scale;

  // Precompute <μ, q_scaled> — same for all DB points.
  float dot_mu_q = 0.0f;
  for (size_t i = 0; i < pdim; ++i)
    dot_mu_q += cm.mean_scaled[i] * qq.scaled[i];
  qq.dot_mean_query = dot_mu_q;

  return qq;
}

// =========================================================================
// Distance functions
// =========================================================================

// 1-bit centered distance.
inline float distance_1bit_centered(const EncodedVec& db_pt, const PreparedQuery& qq, size_t pdim,
                                    bool metric) {
  // dot(centroid_residual, q_scaled):
  float dot_codes = 0.0f;
  for (size_t i = 0; i < pdim; ++i) {
    bool positive = (db_pt.packed_codes[i / 8] >> (i % 8)) & 1;
    float c = positive ? k1BitCentroid : -k1BitCentroid;
    dot_codes += c * qq.scaled[i];
  }

  // Full reconstruction dot product: dot(centroid + μ, q_scaled).
  float dot_full = dot_codes + qq.dot_mean_query;

  // NSFs.
  float query_norm = std::sqrt(qq.squared_norm);
  float nsf_q = (query_norm > 0.0f) ? query_norm / std::sqrt(static_cast<float>(pdim)) : 0.0f;

  float neg_ip = -(dot_full * db_pt.norm_scaling_factor * nsf_q);

  if (metric) {
    return db_pt.unquantized_squared_norm + 2.0f * neg_ip + qq.squared_norm;
  }
  return neg_ip;
}

// 2-bit centered distance.
inline float distance_2bit_centered(const EncodedVec& db_pt, const PreparedQuery& qq, size_t pdim,
                                    bool metric) {
  float dot_codes = 0.0f;
  for (size_t i = 0; i < pdim; ++i) {
    uint8_t code = (db_pt.packed_codes[i / 4] >> (2 * (i % 4))) & 0x3;
    dot_codes += k2BitCentroids[code] * qq.scaled[i];
  }

  float dot_full = dot_codes + qq.dot_mean_query;

  float query_norm = std::sqrt(qq.squared_norm);
  float nsf_q = (query_norm > 0.0f) ? query_norm / std::sqrt(static_cast<float>(pdim)) : 0.0f;

  float neg_ip = -(dot_full * db_pt.norm_scaling_factor * nsf_q);

  if (metric) {
    return db_pt.unquantized_squared_norm + 2.0f * neg_ip + qq.squared_norm;
  }
  return neg_ip;
}

// 4-bit centered distance.
inline float distance_4bit_centered(const EncodedVec& db_pt, const PreparedQuery& qq, size_t pdim,
                                    bool metric) {
  float dot_codes = 0.0f;
  const size_t nbytes = pdim / 2;
  for (size_t j = 0; j < nbytes; ++j) {
    const uint8_t byte = db_pt.packed_codes[j];
    const uint8_t b_even = byte & 0xF;
    const uint8_t b_odd = byte >> 4;
    dot_codes += k4BitCentroidsFloat[b_even] * qq.scaled[2 * j];
    dot_codes += k4BitCentroidsFloat[b_odd] * qq.scaled[2 * j + 1];
  }

  float dot_full = dot_codes + qq.dot_mean_query;

  float query_norm = std::sqrt(qq.squared_norm);
  float nsf_q = (query_norm > 0.0f) ? query_norm / std::sqrt(static_cast<float>(pdim)) : 0.0f;

  float neg_ip = -(dot_full * db_pt.norm_scaling_factor * nsf_q);

  if (metric) {
    return db_pt.unquantized_squared_norm + 2.0f * neg_ip + qq.squared_norm;
  }
  return neg_ip;
}

}  // namespace turboquant_centered
}  // namespace mvsic
