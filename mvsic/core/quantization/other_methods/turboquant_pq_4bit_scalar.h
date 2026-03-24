#pragma once

// turboquant_pq_4bit_scalar.h
//
// Scalar reference implementation of TQ-PQ (4-bit, 16 centroids per block).
// Same preprocessing as TQ-PQ: rotate, normalize, scale sqrt(d). Block-wise
// nearest-centroid assignment using float codebooks from turboquant_pq_codebooks.h.
// No FastScan, no int8 LUTs — float query vs float centroids for clean distance.
// Use this for quality debugging and to compare against turboquant_pq_4bit.h.
//
// Independent of turboquant_pq_4bit.h. Only depends on turboquant_pq_codebooks.h
// and a model type with rotator + padded_dim (e.g. rabitqlib::Rotator).

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "rabitqlib/utils/rotator.hpp"
#include "mvsic/core/quantization/other_methods/turboquant_pq_codebooks.h"

namespace mvsic {
namespace turboquant_pq_scalar {

static constexpr size_t K = 16;

// Encoded DB point: 4-bit codes per block, packed; NSF and original squared norm.
struct EncodedVec {
  std::vector<uint8_t> packed_codes;  // (num_blocks + 1) / 2 bytes, 2 nibbles per byte
  float norm_scaling_factor = 0.0f;
  float unquantized_squared_norm = 0.0f;
};

// Prepared query: rotated + normalized + scaled float coords; norm info for distance.
struct PreparedQuery {
  std::vector<float> rotated;
  float unquantized_squared_norm = 0.0f;
  float norm_scaling_factor = 0.0f;  // norm_q / sqrt(padded_dim) for TQ space
};

// Model concept: any type with .padded_dim and .rotator->rotate(in, out).
// Example: turboquant_pq_4bit::Model<Metric, BlockSize>, or a minimal struct.

// ---- Helpers: centroid float and squared norm (from codebooks) ----
template<size_t BlockSize>
inline float get_centroid_component(size_t k, size_t d) {
  if constexpr (BlockSize == 1) return turboquant_pq_4bit::kPQ_CentroidsFloat_D1_K16[k][d];
  if constexpr (BlockSize == 2) return turboquant_pq_4bit::kPQ_CentroidsFloat_D2_K16[k][d];
  if constexpr (BlockSize == 4) return turboquant_pq_4bit::kPQ_CentroidsFloat_D4_K16[k][d];
  if constexpr (BlockSize == 8) return turboquant_pq_4bit::kPQ_CentroidsFloat_D8_K16[k][d];
  if constexpr (BlockSize == 16) return turboquant_pq_4bit::kPQ_CentroidsFloat_D16_K16[k][d];
  return 0.0f;
}

template<size_t BlockSize>
inline float centroid_sq_norm(size_t k) {
  float s = 0.0f;
  for (size_t d = 0; d < BlockSize; ++d) {
    float c = get_centroid_component<BlockSize>(k, d);
    s += c * c;
  }
  return s;
}

// ---- Encode ----
template<size_t BlockSize, typename TQPQModel>
inline EncodedVec encode_single(const TQPQModel& model, const float* p,
                                std::vector<float>& ws) {
  EncodedVec out;
  const size_t pdim = model.padded_dim;
  const size_t num_blocks = pdim / BlockSize;
  const size_t num_bytes = (num_blocks + 1) / 2;

  ws.resize(pdim);
  model.rotator->rotate(p, ws.data());

  float sqr_norm = 0.0f;
  for (size_t i = 0; i < pdim; ++i)
    sqr_norm += ws[i] * ws[i];
  out.unquantized_squared_norm = sqr_norm;

  if (sqr_norm == 0.0f) {
    out.packed_codes.resize(num_bytes, 0);
    out.norm_scaling_factor = 0.0f;
    return out;
  }

  const float norm = std::sqrt(sqr_norm);
  const float inv_norm = 1.0f / norm;
  const float had_scale = std::sqrt(static_cast<float>(pdim));
  for (size_t i = 0; i < pdim; ++i)
    ws[i] *= inv_norm * had_scale;

  out.packed_codes.resize(num_bytes, 0);
  float quantized_sq_norm = 0.0f;

  for (size_t b = 0; b < num_blocks; ++b) {
    const float* block = ws.data() + b * BlockSize;

    float best_val = 1e30f;
    size_t best_k = 0;
    for (size_t k = 0; k < K; ++k) {
      float dot = 0.0f;
      for (size_t d = 0; d < BlockSize; ++d)
        dot += block[d] * get_centroid_component<BlockSize>(k, d);
      float cn = centroid_sq_norm<BlockSize>(k);
      float val = cn - 2.0f * dot;
      if (val < best_val) {
        best_val = val;
        best_k = k;
      }
    }
    quantized_sq_norm += centroid_sq_norm<BlockSize>(best_k);

    uint8_t code = static_cast<uint8_t>(best_k & 0x0Fu);
    if (b % 2 == 0)
      out.packed_codes[b / 2] = code;
    else
      out.packed_codes[b / 2] |= (code << 4);
  }

  out.norm_scaling_factor = (quantized_sq_norm > 0.0f)
      ? (norm / std::sqrt(quantized_sq_norm))
      : 0.0f;
  return out;
}

// ---- Prepare query ----
template<size_t BlockSize, typename TQPQModel>
inline PreparedQuery prepare_query(const TQPQModel& model, const float* qptr) {
  PreparedQuery qq;
  const size_t pdim = model.padded_dim;
  qq.rotated.resize(pdim);

  model.rotator->rotate(qptr, qq.rotated.data());

  float sqr_norm = 0.0f;
  for (size_t i = 0; i < pdim; ++i)
    sqr_norm += qq.rotated[i] * qq.rotated[i];
  qq.unquantized_squared_norm = sqr_norm;

  if (sqr_norm == 0.0f) {
    qq.norm_scaling_factor = 0.0f;
    return qq;
  }

  const float norm = std::sqrt(sqr_norm);
  const float inv_norm = 1.0f / norm;
  const float had_scale = std::sqrt(static_cast<float>(pdim));
  for (size_t i = 0; i < pdim; ++i)
    qq.rotated[i] *= inv_norm * had_scale;

  // Query in TQ space has squared norm = pdim; NSF restores to original norm.
  qq.norm_scaling_factor = norm / std::sqrt(static_cast<float>(pdim));
  return qq;
}

// ---- Scalar distance ----
template<size_t BlockSize>
inline float distance(const EncodedVec& db_pt, const PreparedQuery& qq,
                      size_t padded_dim, bool metric) {
  const size_t num_blocks = padded_dim / BlockSize;

  float dot_float = 0.0f;
  for (size_t b = 0; b < num_blocks; ++b) {
    uint8_t packed = db_pt.packed_codes[b / 2];
    uint8_t code = (b % 2 == 0) ? (packed & 0x0Fu) : (packed >> 4);
    const float* q_block = qq.rotated.data() + b * BlockSize;
    for (size_t d = 0; d < BlockSize; ++d)
      dot_float += q_block[d] * get_centroid_component<BlockSize>(code, d);
  }

  float dot_est = dot_float * db_pt.norm_scaling_factor * qq.norm_scaling_factor;

  if (metric) {
    return db_pt.unquantized_squared_norm + qq.unquantized_squared_norm - 2.0f * dot_est;
  }
  return -dot_est;
}

}  // namespace turboquant_pq_scalar
}  // namespace mvsic
