#pragma once

// Symmetric RaBitQ implementation inspired by PipNN's approach.
// Key differences from standard RaBitQ:
//   - Symmetric scoring: both query and database points are quantized identically
//   - Per-point normalization with floor quantization + sign-magnitude encoding
//   - Simpler distance formula using code inner products
//
// Uses the same rabitqlib rotator as the existing rabitq.h.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"

#include "rabitqlib/utils/rotator.hpp"

namespace mvsic {
namespace rabitq_sym {

// Per-point metadata computed during quantization.
struct PointMetadata {
  float delta = 0.0f;
  float vl = 0.0f;
  float code_sum = 0.0f;
  float code_sqr_norm = 0.0f;
};

template<bool Metric, int Bits = 4>
class Quantized_Query;

template<bool Metric, int Bits = 4>
class Quantized_Point {
 public:
  const uint8_t* data = nullptr;
  PointMetadata meta;

  Quantized_Point() = default;
  Quantized_Point(const uint8_t* d, PointMetadata m) : data(d), meta(m) {}

  inline float distance(const Quantized_Query<Metric, Bits>& qq) const;

  bool same_as(const Quantized_Point<Metric, Bits>&) const { return false; }
  bool same_as(const Quantized_Query<Metric, Bits>&) const { return false; }

  void prefetch() const { __builtin_prefetch(data, 0, 3); }

  bool is_metric() const { return Metric; }
};

template<bool Metric, int Bits>
class Quantized_Query {
 public:
  using distanceType = float;

  std::vector<uint8_t> packed_codes;
  PointMetadata meta;
  size_t dim = 0;
  size_t padded_dim = 0;
  size_t stride = 0;
  float inv_dim = 0.0f;

  Quantized_Query() = default;

  inline float distance(const Quantized_Point<Metric, Bits>& p) const {
    return distance_raw(p.data, p.meta);
  }

  inline float distance_raw(const uint8_t* p_data, const PointMetadata& p_meta) const {
    constexpr int bias = 1 << (Bits - 1);
    int32_t ip_codes = 0;

    if constexpr (Bits == 4) {
      const size_t packed_dim = (padded_dim + 1) / 2;
      for (size_t i = 0; i < packed_dim; ++i) {
        uint8_t val_p = p_data[i];
        uint8_t val_q = packed_codes[i];

        int32_t p_lo = static_cast<int32_t>(val_p & 0x0F) - bias;
        int32_t p_hi = static_cast<int32_t>((val_p >> 4) & 0x0F) - bias;

        int32_t q_lo = static_cast<int32_t>(val_q & 0x0F) - bias;
        int32_t q_hi = static_cast<int32_t>((val_q >> 4) & 0x0F) - bias;

        ip_codes += p_lo * q_lo;
        if (2 * i + 1 < padded_dim) {
          ip_codes += p_hi * q_hi;
        }
      }
    } else {
      for (size_t i = 0; i < padded_dim; ++i) {
        int32_t pv = static_cast<int32_t>(p_data[i]) - bias;
        int32_t qv = static_cast<int32_t>(packed_codes[i]) - bias;
        ip_codes += pv * qv;
      }
    }

    float term1 = p_meta.delta * meta.delta * static_cast<float>(ip_codes);
    float term2 = p_meta.delta * meta.vl * p_meta.code_sum;
    float term3 = meta.delta * p_meta.vl * meta.code_sum;
    float term4 = p_meta.vl * meta.vl * static_cast<float>(padded_dim);

    float score = (term1 + term2 + term3 + term4) * inv_dim;

    if constexpr (Metric) {
      return -score;
    } else {
      return -score;
    }
  }

  template<typename EncRange>
  void distances_all(const EncRange& db, float* out) const {
    const size_t n = static_cast<size_t>(db.size());
    if (n == 0) return;

    const uint8_t* storage_base = db.storage.data();
    const PointMetadata* metas_base = db.metas.data();
    const size_t db_stride = db.stride;

    parlay::parallel_for(0, n, [&](size_t i) {
      const uint8_t* p_data = storage_base + i * db_stride;
      out[i] = distance_raw(p_data, metas_base[i]);
    });
  }
};

template<bool Metric, int Bits>
inline float Quantized_Point<Metric, Bits>::distance(
    const Quantized_Query<Metric, Bits>& qq) const {
  return qq.distance_raw(data, meta);
}

template<typename PointRange, bool Metric, int Bits = 4>
class Quantized_Point_Range {
 public:
  static constexpr bool is_fastscan = false;

  size_t n = 0;
  size_t dim = 0;
  size_t padded_dim = 0;
  size_t stride = 0;

  parlay::sequence<uint8_t> storage;
  parlay::sequence<PointMetadata> metas;

  Quantized_Point_Range() = default;

  Quantized_Point<Metric, Bits> operator[](size_t i) const {
    return Quantized_Point<Metric, Bits>(&storage[i * stride], metas[i]);
  }

  inline uint32_t size() const noexcept { return static_cast<uint32_t>(n); }
  inline uint32_t get_dims() const noexcept { return static_cast<uint32_t>(dim); }

  void save(std::ostream& out) const {
    out.write(reinterpret_cast<const char*>(&n), sizeof(n));
    out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    out.write(reinterpret_cast<const char*>(&padded_dim), sizeof(padded_dim));
    out.write(reinterpret_cast<const char*>(&stride), sizeof(stride));

    size_t storage_sz = storage.size();
    out.write(reinterpret_cast<const char*>(&storage_sz), sizeof(storage_sz));
    if (storage_sz) out.write(reinterpret_cast<const char*>(storage.data()), storage_sz);

    size_t metas_sz = metas.size();
    out.write(reinterpret_cast<const char*>(&metas_sz), sizeof(metas_sz));
    if (metas_sz)
      out.write(reinterpret_cast<const char*>(metas.data()), metas_sz * sizeof(PointMetadata));
  }

  void load(std::istream& in) {
    in.read(reinterpret_cast<char*>(&n), sizeof(n));
    in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    in.read(reinterpret_cast<char*>(&padded_dim), sizeof(padded_dim));
    in.read(reinterpret_cast<char*>(&stride), sizeof(stride));

    size_t storage_sz = 0;
    in.read(reinterpret_cast<char*>(&storage_sz), sizeof(storage_sz));
    storage.resize(storage_sz);
    if (storage_sz) in.read(reinterpret_cast<char*>(storage.data()), storage_sz);

    size_t metas_sz = 0;
    in.read(reinterpret_cast<char*>(&metas_sz), sizeof(metas_sz));
    metas.resize(metas_sz);
    if (metas_sz) in.read(reinterpret_cast<char*>(metas.data()), metas_sz * sizeof(PointMetadata));
  }
};

template<bool Metric, int Bits = 4>
class Model {
 public:
  static constexpr bool is_fastscan = false;

  size_t dim = 0;
  size_t padded_dim = 0;
  size_t stride = 0;

  rabitqlib::RotatorType rotator_type = rabitqlib::RotatorType::FhtKacRotator;
  std::unique_ptr<rabitqlib::Rotator<float>> rotator;
  std::vector<float> random_signs;

  Model() = default;

  template<typename PointRange>
  Model(const PointRange& train_data) {
    train(train_data);
  }

  template<typename PointRange>
  void train(const PointRange& data) {
    static_assert(Bits == 4 || Bits == 8, "Symmetric RaBitQ supports 4-bit and 8-bit");

    dim = data.get_dims();

    rotator.reset(rabitqlib::choose_rotator<float>(dim, rotator_type));
    padded_dim = rotator->size();

    if constexpr (Bits == 4) {
      stride = (padded_dim + 1) / 2;
    } else {
      stride = padded_dim;
    }

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 1);
    random_signs.resize(padded_dim);
    for (size_t i = 0; i < padded_dim; ++i) {
      random_signs[i] = (dist(rng) == 0) ? -1.0f : 1.0f;
    }

    std::cout << "RaBitQ-PipNN: " << Bits << "-bit Symmetric. Dim=" << dim
              << " Padded=" << padded_dim << std::endl;
  }

  template<typename PointRange>
  Quantized_Point_Range<PointRange, Metric, Bits> encode(const PointRange& data) const {
    if (!rotator) {
      std::cerr << "RaBitQ-PipNN::encode called before train.\n";
      std::abort();
    }

    Quantized_Point_Range<PointRange, Metric, Bits> enc;
    enc.n = data.size();
    enc.dim = data.get_dims();
    enc.padded_dim = padded_dim;
    enc.stride = stride;

    if (enc.dim != dim) {
      std::cerr << "RaBitQ-PipNN::encode dim mismatch: data dim=" << enc.dim << " model dim=" << dim
                << "\n";
      std::abort();
    }

    enc.storage.resize(enc.n * stride);
    enc.metas.resize(enc.n);

    // Local copies for lambda capture
    const size_t local_dim = dim;
    const size_t local_padded_dim = padded_dim;
    const size_t local_stride = stride;
    const float* signs_ptr = random_signs.data();
    rabitqlib::Rotator<float>* rot_ptr = rotator.get();

    struct EncodingWorkspace {
      std::vector<float> pre_rot;
      std::vector<float> rot_buf;
      std::vector<uint8_t> raw_codes;
      void ensure_size(size_t pdim) {
        if (rot_buf.size() != pdim) {
          pre_rot.resize(pdim, 0.0f);
          rot_buf.resize(pdim);
          raw_codes.resize(pdim);
        }
      }
    };

    parlay::parallel_for(
        0, enc.n, [&, local_dim, local_padded_dim, local_stride, signs_ptr, rot_ptr](size_t i) {
          static thread_local EncodingWorkspace ws;
          ws.ensure_size(local_padded_dim);

          std::vector<float>& pre_rot = ws.pre_rot;
          std::vector<float>& rot_buf = ws.rot_buf;
          std::vector<uint8_t>& raw_codes = ws.raw_codes;

          const float* x = reinterpret_cast<const float*>(data.location(i));
          uint8_t* out_packed = &enc.storage[i * local_stride];
          PointMetadata& out_meta = enc.metas[i];

          // Compute L2 norm
          float norm_x = 0.0f;
          for (size_t j = 0; j < local_dim; ++j) {
            norm_x += x[j] * x[j];
          }
          norm_x = std::sqrt(norm_x);
          if (norm_x < 1e-9f) norm_x = 1.0f;

          // Normalize, apply random signs, then rotate
          std::fill(pre_rot.begin(), pre_rot.end(), 0.0f);
          for (size_t j = 0; j < local_dim; ++j) {
            pre_rot[j] = (x[j] / norm_x) * signs_ptr[j];
          }
          rot_ptr->rotate(pre_rot.data(), rot_buf.data());

          // Find max absolute value for scaling
          float max_abs = 0.0f;
          for (size_t j = 0; j < local_padded_dim; ++j) {
            max_abs = std::max(max_abs, std::abs(rot_buf[j]));
          }
          if (max_abs < 1e-9f) max_abs = 1.0f;

          constexpr int limit = (1 << (Bits - 1)) - 1;
          constexpr int bias = 1 << (Bits - 1);
          constexpr float cb = -static_cast<float>(bias) + 0.5f;

          float t = static_cast<float>(limit) / max_abs;

          // Quantize with floor + sign-magnitude encoding
          for (size_t j = 0; j < local_padded_dim; ++j) {
            float val = rot_buf[j];
            int sign_bit = (val >= 0) ? 1 : 0;
            float abs_val = std::abs(val);

            int magnitude = static_cast<int>(std::floor(t * abs_val));
            if (magnitude > limit) magnitude = limit;

            if (sign_bit == 0) {
              magnitude = magnitude ^ limit;
            }

            raw_codes[j] = static_cast<uint8_t>(magnitude + (sign_bit << (Bits - 1)));
          }

          // Pack codes
          std::memset(out_packed, 0, local_stride);
          if constexpr (Bits == 4) {
            for (size_t j = 0; j < local_padded_dim; ++j) {
              uint8_t code = raw_codes[j];
              size_t byte_idx = j / 2;
              if (j % 2 == 0)
                out_packed[byte_idx] |= (code & 0x0F);
              else
                out_packed[byte_idx] |= ((code << 4) & 0xF0);
            }
          } else {
            for (size_t j = 0; j < local_padded_dim; ++j) {
              out_packed[j] = raw_codes[j];
            }
          }

          // Compute metadata
          double norm_quan_sqr = 0.0;
          double ip_data_quan = 0.0;
          double code_sum_accum = 0.0;
          double code_sqr_norm_accum = 0.0;

          for (size_t j = 0; j < local_padded_dim; ++j) {
            float u_cb = static_cast<float>(raw_codes[j]) + cb;
            float sc = static_cast<float>(raw_codes[j]) - static_cast<float>(bias);

            norm_quan_sqr += u_cb * u_cb;
            ip_data_quan += rot_buf[j] * u_cb;
            code_sum_accum += sc;
            code_sqr_norm_accum += sc * sc;
          }

          float norm_quan = std::sqrt(static_cast<float>(norm_quan_sqr));
          float cos_sim = static_cast<float>(ip_data_quan) / norm_quan;

          out_meta.delta = norm_x * cos_sim / norm_quan;
          out_meta.vl = out_meta.delta * 0.5f;
          out_meta.code_sum = static_cast<float>(code_sum_accum);
          out_meta.code_sqr_norm = static_cast<float>(code_sqr_norm_accum);
        });

    return enc;
  }

  template<typename PointTy>
  Quantized_Query<Metric, Bits> quantize_query(const PointTy& query) const {
    std::vector<float> tmp(dim);
    for (size_t i = 0; i < dim; ++i)
      tmp[i] = query[i];
    return quantize_query_impl(tmp.data());
  }

  Quantized_Query<Metric, Bits> quantize_query(const float* qptr) const {
    return quantize_query_impl(qptr);
  }

  Quantized_Query<Metric, Bits> quantize_query_impl(const float* qptr) const {
    if (!rotator) {
      std::cerr << "RaBitQ-PipNN::quantize_query called before train.\n";
      std::abort();
    }

    Quantized_Query<Metric, Bits> qq;
    qq.dim = dim;
    qq.padded_dim = padded_dim;
    qq.stride = stride;
    qq.inv_dim = 1.0f / static_cast<float>(padded_dim);
    qq.packed_codes.resize(stride);

    // Compute L2 norm
    float norm_x = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
      norm_x += qptr[i] * qptr[i];
    }
    norm_x = std::sqrt(norm_x);
    if (norm_x < 1e-9f) norm_x = 1.0f;

    // Normalize, apply random signs, then rotate
    std::vector<float> pre_rot(padded_dim, 0.0f);
    for (size_t i = 0; i < dim; ++i) {
      pre_rot[i] = (qptr[i] / norm_x) * random_signs[i];
    }

    std::vector<float> rot_buf(padded_dim);
    rotator->rotate(pre_rot.data(), rot_buf.data());

    // Find max absolute value for scaling
    float max_abs = 0.0f;
    for (size_t i = 0; i < padded_dim; ++i) {
      max_abs = std::max(max_abs, std::abs(rot_buf[i]));
    }
    if (max_abs < 1e-9f) max_abs = 1.0f;

    constexpr int limit = (1 << (Bits - 1)) - 1;
    constexpr int bias = 1 << (Bits - 1);
    constexpr float cb = -static_cast<float>(bias) + 0.5f;

    float t = static_cast<float>(limit) / max_abs;

    // Quantize with floor + sign-magnitude encoding
    std::vector<uint8_t> raw_codes(padded_dim);
    for (size_t i = 0; i < padded_dim; ++i) {
      float val = rot_buf[i];
      int sign_bit = (val >= 0) ? 1 : 0;
      float abs_val = std::abs(val);

      int magnitude = static_cast<int>(std::floor(t * abs_val));
      if (magnitude > limit) magnitude = limit;

      if (sign_bit == 0) {
        magnitude = magnitude ^ limit;
      }

      raw_codes[i] = static_cast<uint8_t>(magnitude + (sign_bit << (Bits - 1)));
    }

    // Pack codes
    std::memset(qq.packed_codes.data(), 0, stride);
    if constexpr (Bits == 4) {
      for (size_t i = 0; i < padded_dim; ++i) {
        uint8_t code = raw_codes[i];
        size_t byte_idx = i / 2;
        if (i % 2 == 0)
          qq.packed_codes[byte_idx] |= (code & 0x0F);
        else
          qq.packed_codes[byte_idx] |= ((code << 4) & 0xF0);
      }
    } else {
      for (size_t i = 0; i < padded_dim; ++i) {
        qq.packed_codes[i] = raw_codes[i];
      }
    }

    // Compute metadata
    double norm_quan_sqr = 0.0;
    double ip_data_quan = 0.0;
    double code_sum_accum = 0.0;
    double code_sqr_norm_accum = 0.0;

    for (size_t i = 0; i < padded_dim; ++i) {
      float u_cb = static_cast<float>(raw_codes[i]) + cb;
      float sc = static_cast<float>(raw_codes[i]) - static_cast<float>(bias);

      norm_quan_sqr += u_cb * u_cb;
      ip_data_quan += rot_buf[i] * u_cb;
      code_sum_accum += sc;
      code_sqr_norm_accum += sc * sc;
    }

    float norm_quan = std::sqrt(static_cast<float>(norm_quan_sqr));
    float cos_sim = static_cast<float>(ip_data_quan) / norm_quan;

    qq.meta.delta = norm_x * cos_sim / norm_quan;
    qq.meta.vl = qq.meta.delta * 0.5f;
    qq.meta.code_sum = static_cast<float>(code_sum_accum);
    qq.meta.code_sqr_norm = static_cast<float>(code_sqr_norm_accum);

    return qq;
  }

  void save(std::ofstream& out) const {
    out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    out.write(reinterpret_cast<const char*>(&padded_dim), sizeof(padded_dim));
    out.write(reinterpret_cast<const char*>(&stride), sizeof(stride));

    int rot_type_int = static_cast<int>(rotator_type);
    out.write(reinterpret_cast<const char*>(&rot_type_int), sizeof(rot_type_int));

    if (!rotator) {
      std::cerr << "RaBitQ-PipNN::save: rotator is null.\n";
      std::abort();
    }
    rotator->save(out);

    size_t signs_size = random_signs.size();
    out.write(reinterpret_cast<const char*>(&signs_size), sizeof(signs_size));
    out.write(reinterpret_cast<const char*>(random_signs.data()), signs_size * sizeof(float));
  }

  void load(std::ifstream& in) {
    in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    in.read(reinterpret_cast<char*>(&padded_dim), sizeof(padded_dim));
    in.read(reinterpret_cast<char*>(&stride), sizeof(stride));

    int rot_type_int = 0;
    in.read(reinterpret_cast<char*>(&rot_type_int), sizeof(rot_type_int));
    rotator_type = static_cast<rabitqlib::RotatorType>(rot_type_int);

    rotator.reset(rabitqlib::choose_rotator<float>(dim, rotator_type));
    const size_t runtime_padded_dim = rotator->size();
    if (runtime_padded_dim != padded_dim) {
      std::cerr << "RaBitQ-PipNN Error: Saved padded_dim (" << padded_dim
                << ") != Runtime padded_dim (" << runtime_padded_dim << ")\n";
      std::abort();
    }

    rotator->load(in);

    size_t signs_size = 0;
    in.read(reinterpret_cast<char*>(&signs_size), sizeof(signs_size));
    random_signs.resize(signs_size);
    in.read(reinterpret_cast<char*>(random_signs.data()), signs_size * sizeof(float));
  }
};

}  // namespace rabitq_sym
}  // namespace mvsic
