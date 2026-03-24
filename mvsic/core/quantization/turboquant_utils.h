#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <vector>
#include <algorithm>

#include "rabitqlib/utils/rotator.hpp"

namespace mvsic {
namespace turboquant {

namespace internal {
static constexpr std::array<int8_t, 16> kTurboQuantCentroidsInt8 = {
    6, 18, 31, 44, 58, 75, 96, 127, -6, -18, -31, -44, -58, -75, -96, -127};

static constexpr std::array<float, 8> kSquaredCentroidsInt8 = {
    35.66f, 320.92f, 951.87f, 1917.61f, 3332.04f, 5571.56f, 9128.44f, 15975.76f};

static constexpr std::array<float, 7> kBoundaries = {0.2581972f, 0.5271527f, 0.806866f, 1.097338f,
                                                     1.430843f,  1.839655f,  2.399083f};

static constexpr float kValueCap = 3.91724f;

inline uint8_t FindBucket(float x) {
  uint8_t idx = static_cast<uint8_t>(kBoundaries.size());
  while (idx > 0 && x < kBoundaries[idx - 1])
    --idx;
  return idx;
}

inline uint8_t FourBitEncoding(float x) {
  return FindBucket(std::abs(x)) | (x > 0 ? 0 : 8);
}
}  // namespace internal

// Shared Base Encoder to handle rotation, seeding, and single-point math
class BaseEncoder {
 public:
  size_t dim = 0;
  size_t padded_dim = 0;
  size_t num_bytes_per_datapoint = 0;
  size_t seed_ = 42;

  std::vector<float> signs;
  rabitqlib::RotatorType rotator_type = rabitqlib::RotatorType::FhtKacRotator;
  std::unique_ptr<rabitqlib::Rotator<float>> rotator;

  void train(size_t input_dim) {
    dim = input_dim;
    if (dim == 0) return;
    rotator.reset(rabitqlib::choose_rotator<float>(dim, rotator_type));
    if (!rotator) return;

    padded_dim = rotator->size();
    num_bytes_per_datapoint = (padded_dim + 1) / 2;

    signs.resize(padded_dim);
    std::mt19937 gen(seed_);
    std::uniform_int_distribution<> dist(0, 1);
    for (size_t i = 0; i < padded_dim; ++i) {
      signs[i] = (2.0f * dist(gen) - 1.0f);
    }
  }

  // Rotates, normalizes, scales, and packs into continuous bytes
  std::pair<float, float> encode_single(const float* p, uint8_t* output,
                                        std::vector<float>& ws) const {
    rotator->rotate(p, ws.data());

    float sqr_norm = 0.0f;
    for (size_t i = 0; i < padded_dim; ++i)
      sqr_norm += ws[i] * ws[i];

    if (sqr_norm == 0.0f) {
      std::memset(output, 0, num_bytes_per_datapoint);
      return {0.0f, 0.0f};
    }

    const float norm = std::sqrt(sqr_norm);
    const float inv_norm = 1.0f / norm;
    const float hadamard_scale = std::sqrt(static_cast<float>(padded_dim));

    for (size_t i = 0; i < padded_dim; ++i) {
      ws[i] = ws[i] * inv_norm * hadamard_scale;
    }

    float q_sqr_norm = 0.0f;
    for (size_t i = 0; i < padded_dim; ++i) {
      uint8_t code = internal::FourBitEncoding(ws[i]);
      q_sqr_norm += internal::kSquaredCentroidsInt8[code & 7];
      if (i % 2 == 0)
        output[i / 2] = code;
      else
        output[i / 2] |= (code << 4);
    }

    return {sqr_norm, norm / std::sqrt(q_sqr_norm)};
  }

  void save(std::ofstream& out) const {
    out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    out.write(reinterpret_cast<const char*>(&padded_dim), sizeof(padded_dim));
    out.write(reinterpret_cast<const char*>(&seed_), sizeof(seed_));
    int rt = static_cast<int>(rotator_type);
    out.write(reinterpret_cast<const char*>(&rt), sizeof(rt));
    if (rotator) rotator->save(out);
    size_t ss = signs.size();
    out.write(reinterpret_cast<const char*>(&ss), sizeof(ss));
    out.write(reinterpret_cast<const char*>(signs.data()), ss * sizeof(float));
  }

  void load(std::ifstream& in) {
    in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    in.read(reinterpret_cast<char*>(&padded_dim), sizeof(padded_dim));
    in.read(reinterpret_cast<char*>(&seed_), sizeof(seed_));
    num_bytes_per_datapoint = (padded_dim + 1) / 2;
    int rt = 0;
    in.read(reinterpret_cast<char*>(&rt), sizeof(rt));
    rotator_type = static_cast<rabitqlib::RotatorType>(rt);
    rotator.reset(rabitqlib::choose_rotator<float>(dim, rotator_type));
    if (rotator) rotator->load(in);
    size_t ss = 0;
    in.read(reinterpret_cast<char*>(&ss), sizeof(ss));
    signs.resize(ss);
    in.read(reinterpret_cast<char*>(signs.data()), ss * sizeof(float));
  }
};

}  // namespace turboquant
}  // namespace mvsic