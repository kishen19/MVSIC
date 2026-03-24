// Unit test for TurboQuant encoding quality fix.
//
// Verifies that turboquant_4bit matches the tq_reference implementation
// exactly: rotate → normalize → scale(√d) → quantize. No sign-flipping.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

#include "gtest/gtest.h"
#include "parlay/parallel.h"
#include "mvsic/core/quantization/other_methods/turboquant_4bit.h"

using namespace mvsic;
using namespace mvsic::turboquant_4bit;

// ============================================================================
// tq_reference encoder (faithful copy: rotate → normalize → √d → quantize)
// ============================================================================
namespace ref {

constexpr std::array<int8_t, 16> kCentroidsInt8 = {6,  18,  31,  44,  58,  75,  96,  127,
                                                   -6, -18, -31, -44, -58, -75, -96, -127};
constexpr std::array<float, 8> kSqCentroids = {35.66f,   320.92f,  951.87f,  1917.61f,
                                               3332.04f, 5571.56f, 9128.44f, 15975.76f};
constexpr std::array<float, 7> kBounds = {0.2581972f, 0.5271527f, 0.806866f, 1.097338f,
                                          1.430843f,  1.839655f,  2.399083f};
constexpr float kValCap = 3.91724f;

uint8_t FindBucket(float abs_x) {
  uint8_t idx = kBounds.size();
  while (idx > 0 && abs_x < kBounds[idx - 1])
    --idx;
  return idx;
}
uint8_t FourBitEncode(float x) {
  return FindBucket(std::abs(x)) | (x > 0 ? 0 : 8);
}

struct EncResult {
  std::vector<uint8_t> codes;
  float nsf, sqn;
};

EncResult encode_point(const float* pt, size_t dim, const Model<false>& model) {
  size_t pdim = model.padded_dim;
  size_t stride = pdim / 2;
  float hs = std::sqrt(float(pdim));
  EncResult r;
  r.codes.resize(stride, 0);

  std::vector<float> rot(pdim);
  model.rotator->rotate(pt, rot.data());

  float nsq = 0;
  for (size_t i = 0; i < pdim; ++i)
    nsq += rot[i] * rot[i];
  r.sqn = nsq;
  float norm = std::sqrt(nsq);
  if (norm > 1e-9f) {
    float inv = 1.0f / norm;
    for (size_t i = 0; i < pdim; ++i)
      rot[i] *= inv;
  }

  float qsn = 0;
  for (size_t b = 0; b < stride; ++b) {
    uint8_t c0 = FourBitEncode(rot[2 * b] * hs);
    uint8_t c1 = FourBitEncode(rot[2 * b + 1] * hs);
    r.codes[b] = (c0 & 0x0F) | ((c1 << 4) & 0xF0);
    qsn += kSqCentroids[c0 & 7] + kSqCentroids[c1 & 7];
  }
  float qn = std::sqrt(qsn);
  r.nsf = (qn > 1e-9f) ? (norm / qn) : 0.0f;
  return r;
}

struct QueryResult {
  std::vector<int8_t> data;
  float nsf, sqn;
};

// Non-deinterleaved query (matches our impl's layout, not tq_reference's
// even/odd deinterleaved layout — both produce the same int8 values).
QueryResult encode_query(const float* q, size_t dim, const Model<false>& model) {
  size_t pdim = model.padded_dim;
  float hs = std::sqrt(float(pdim));
  QueryResult qr;
  qr.data.resize(pdim, 0);

  std::vector<float> rot(pdim);
  model.rotator->rotate(q, rot.data());

  float nsq = 0;
  for (size_t i = 0; i < pdim; ++i)
    nsq += rot[i] * rot[i];
  qr.sqn = nsq;
  float norm = std::sqrt(nsq);
  if (norm > 1e-9f) {
    float inv = 1.0f / norm;
    for (size_t i = 0; i < pdim; ++i)
      rot[i] *= inv;
  }
  for (size_t i = 0; i < pdim; ++i)
    rot[i] *= hs;
  float max_abs = 0;
  for (size_t i = 0; i < pdim; ++i) {
    rot[i] = std::max(-kValCap, std::min(kValCap, rot[i]));
    max_abs = std::max(max_abs, std::abs(rot[i]));
  }
  float sf = (max_abs > 1e-9f) ? (127.0f / max_abs) : 0.0f;
  float qn_sq = 0;
  for (size_t i = 0; i < pdim; ++i) {
    int8_t s = int8_t(std::max(-127.0f, std::min(127.0f, std::round(rot[i] * sf))));
    qr.data[i] = s;
    qn_sq += float(s) * s;
  }
  float qnorm = std::sqrt(qn_sq);
  qr.nsf = (qnorm > 1e-9f) ? (norm / qnorm) : 0.0f;
  return qr;
}

float score_ip(const EncResult& pt, const QueryResult& q, size_t stride, size_t pdim) {
  int32_t dot = 0;
  for (size_t b = 0; b < stride; ++b) {
    uint8_t byte = pt.codes[b];
    dot += int32_t(kCentroidsInt8[byte & 0x0F]) * int32_t(q.data[2 * b]);
    dot += int32_t(kCentroidsInt8[(byte >> 4) & 0x0F]) * int32_t(q.data[2 * b + 1]);
  }
  return -float(dot) * pt.nsf * q.nsf;
}

}  // namespace ref

// ============================================================================
// Fake PointRange for Model::train
// ============================================================================
struct FakePointRange {
  std::vector<float> data;
  size_t n, d;
  size_t size() const { return n; }
  size_t get_dims() const { return d; }
  size_t get_aligned_dims() const { return d; }
  const uint8_t* location(size_t i) const {
    return reinterpret_cast<const uint8_t*>(data.data() + i * d);
  }
  struct FakePoint {
    const float* ptr;
    size_t d;
    float operator[](size_t i) const { return ptr[i]; }
    using T = float;
  };
  FakePoint operator[](long i) const { return {data.data() + i * d, d}; }
};

FakePointRange make_random(size_t n, size_t d, unsigned seed = 42) {
  FakePointRange pr;
  pr.n = n;
  pr.d = d;
  pr.data.resize(n * d);
  std::mt19937 rng(seed);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  for (auto& v : pr.data)
    v = dist(rng);
  return pr;
}

// ============================================================================
// Tests
// ============================================================================

class TurboQuantTest : public ::testing::Test {
 protected:
  static constexpr size_t kDim = 128;
  static constexpr size_t kN = 500;
  void SetUp() override {
    data_ = make_random(kN, kDim);
    model_.train(data_);
  }
  FakePointRange data_;
  Model<false> model_;
};

// Test 1: encode_single produces identical codes and NSF as tq_reference.
TEST_F(TurboQuantTest, EncodingMatchesReference) {
  size_t stride = model_.padded_dim / 2;
  size_t nbytes = model_.num_bytes_per_datapoint;
  ASSERT_EQ(nbytes, stride);

  int code_mismatches = 0;
  float max_nsf_diff = 0;

  for (size_t i = 0; i < kN; ++i) {
    const float* pt = data_.data.data() + i * kDim;

    std::vector<uint8_t> our_codes(nbytes, 0);
    std::vector<float> ws(model_.padded_dim);
    auto [our_sqn, our_nsf] = model_.encode_single(pt, our_codes.data(), ws);

    auto ref_r = ref::encode_point(pt, kDim, model_);

    for (size_t b = 0; b < stride; ++b) {
      if (our_codes[b] != ref_r.codes[b]) ++code_mismatches;
    }
    max_nsf_diff = std::max(max_nsf_diff, std::abs(our_nsf - ref_r.nsf));
  }

  EXPECT_EQ(code_mismatches, 0) << "Nibble code mismatches";
  EXPECT_LT(max_nsf_diff, 1e-3f) << "NSF divergence";
}

// Test 2: quantize_query produces identical int8 values and NSF as tq_reference.
TEST_F(TurboQuantTest, QueryMatchesReference) {
  float max_nsf_diff = 0;
  int data_mismatches = 0;

  for (size_t qi = 0; qi < std::min(kN, (size_t)50); ++qi) {
    const float* qptr = data_.data.data() + qi * kDim;

    auto our_qq = model_.quantize_query(qptr);
    auto ref_qq = ref::encode_query(qptr, kDim, model_);

    for (size_t i = 0; i < model_.padded_dim; ++i) {
      if (our_qq.query_data[i] != ref_qq.data[i]) ++data_mismatches;
    }
    max_nsf_diff = std::max(max_nsf_diff, std::abs(our_qq.norm_scaling_factor - ref_qq.nsf));
  }

  EXPECT_EQ(data_mismatches, 0) << "Query int8 mismatches";
  EXPECT_LT(max_nsf_diff, 1e-3f) << "Query NSF divergence";
}

// Test 3: scoring matches tq_reference scalar scoring.
TEST_F(TurboQuantTest, ScoringMatchesReference) {
  auto enc = model_.encode(data_);
  size_t stride = model_.padded_dim / 2;
  int bad = 0;

  for (size_t qi = 0; qi < 10; ++qi) {
    const float* qptr = data_.data.data() + qi * kDim;
    auto our_qq = model_.quantize_query(qptr);
    auto ref_qq = ref::encode_query(qptr, kDim, model_);

    for (size_t j = 0; j < std::min(kN, (size_t)50); ++j) {
      auto pt = enc[j];
      float our_score = our_qq.distance(pt);

      auto ref_pt = ref::encode_point(data_.data.data() + j * kDim, kDim, model_);
      float ref_score = ref::score_ip(ref_pt, ref_qq, stride, model_.padded_dim);

      float rel = (std::abs(ref_score) > 1e-6f)
                      ? std::abs(our_score - ref_score) / std::abs(ref_score)
                      : std::abs(our_score - ref_score);
      if (rel > 0.05f) ++bad;
    }
  }
  EXPECT_EQ(bad, 0) << "Score mismatches > 5%";
}

// Test 4: recall@10 sanity >= 0.70.
TEST_F(TurboQuantTest, RecallSanity) {
  auto enc = model_.encode(data_);
  size_t k = 10, nq = std::min(kN, (size_t)100);

  std::vector<std::vector<uint32_t>> gt(nq);
  for (size_t qi = 0; qi < nq; ++qi) {
    const float* qptr = data_.data.data() + qi * kDim;
    std::vector<std::pair<float, uint32_t>> dists(kN);
    for (size_t j = 0; j < kN; ++j) {
      float d = 0;
      for (size_t d2 = 0; d2 < kDim; ++d2)
        d += qptr[d2] * data_.data[j * kDim + d2];
      dists[j] = {-d, (uint32_t)j};
    }
    std::sort(dists.begin(), dists.end());
    gt[qi].resize(k);
    for (size_t i = 0; i < k; ++i)
      gt[qi][i] = dists[i].second;
  }

  size_t correct = 0;
  for (size_t qi = 0; qi < nq; ++qi) {
    auto qq = model_.quantize_query(data_.data.data() + qi * kDim);
    std::vector<std::pair<float, uint32_t>> dists(kN);
    for (size_t j = 0; j < kN; ++j) {
      auto pt = enc[j];
      dists[j] = {qq.distance(pt), (uint32_t)j};
    }
    std::sort(dists.begin(), dists.end());
    for (size_t i = 0; i < k; ++i)
      for (size_t g = 0; g < k; ++g)
        if (dists[i].second == gt[qi][g]) {
          ++correct;
          break;
        }
  }

  double recall = double(correct) / double(nq * k);
  EXPECT_GE(recall, 0.70) << "Recall too low";
  std::cout << "  Recall@10 = " << recall << std::endl;
}

// Test 5: D=768 encoding matches tq_reference (the critical wikipedia_cohere case).
TEST(TurboQuantHighDimTest, EncodingMatchesReference768) {
  auto data = make_random(100, 768, 123);
  Model<false> model;
  model.train(data);

  size_t stride = model.padded_dim / 2;
  int mismatches = 0;
  float max_nsf_diff = 0;

  for (size_t i = 0; i < 100; ++i) {
    const float* pt = data.data.data() + i * 768;
    std::vector<uint8_t> codes(stride, 0);
    std::vector<float> ws(model.padded_dim);
    auto [sqn, nsf] = model.encode_single(pt, codes.data(), ws);
    auto ref_r = ref::encode_point(pt, 768, model);

    for (size_t b = 0; b < stride; ++b)
      if (codes[b] != ref_r.codes[b]) ++mismatches;
    max_nsf_diff = std::max(max_nsf_diff, std::abs(nsf - ref_r.nsf));
  }

  EXPECT_EQ(mismatches, 0) << "D=768 code mismatches";
  EXPECT_LT(max_nsf_diff, 1e-3f) << "D=768 NSF divergence";
}
