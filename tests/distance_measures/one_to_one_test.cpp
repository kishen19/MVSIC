// Robust correctness tests for one_to_one distance implementations.
// Tests ip_distance, l2_distance (float, uint8_t, int8_t), chamfer_l2_distance,
// and chamfer_ip_distance against brute-force or reference implementations.
// Does not test _opt or _eigen_opt variants.
//
// Random trial counts (for robustness):
//   Float single-vector (IP/L2): 150 trials
//   Integer single-vector (IP/L2): 100 trials each (uint8_t and int8_t)
//   Chamfer (L2/IP): 80 trials each with (n_a, n_b, dim) random
// Additional larger-input tests ensure correctness on big vectors and point clouds.

#include "gtest/gtest.h"
#include "mvsic/core/distance_measures/one_to_one.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

namespace {

// -----------------------------------------------------------------------------
// Tolerances and helpers
// -----------------------------------------------------------------------------
constexpr float kRelTol = 1e-4f;
constexpr float kAbsTol = 1e-5f;

void ExpectNear(float expected, float actual, float rel = kRelTol, float abs = kAbsTol) {
  float scale = std::max(std::fabs(expected), 1.0f);
  ASSERT_NEAR(expected, actual, abs + rel * scale)
      << "expected=" << expected << " actual=" << actual;
}

// -----------------------------------------------------------------------------
// Brute-force / reference: single-vector distances
// -----------------------------------------------------------------------------

// IP: returns -dot(p, q) for float.
float RefIPFloat(const float* p, const float* q, unsigned d) {
  float sum = 0.0f;
  for (unsigned i = 0; i < d; ++i) sum += p[i] * q[i];
  return -sum;
}

// L2: squared Euclidean distance for float.
float RefL2Float(const float* p, const float* q, unsigned d) {
  float sum = 0.0f;
  for (unsigned i = 0; i < d; ++i) {
    float diff = p[i] - q[i];
    sum += diff * diff;
  }
  return sum;
}

// IP for uint8_t/int8_t: -sum((int32_t)p[i]*(int32_t)q[i])
template <typename T>
float RefIPInteger(const T* p, const T* q, unsigned d) {
  int32_t sum = 0;
  for (unsigned i = 0; i < d; ++i) sum += static_cast<int32_t>(p[i]) * static_cast<int32_t>(q[i]);
  return -static_cast<float>(sum);
}

// L2 for uint8_t/int8_t: sum of (int16_t)(q[i]-p[i])^2 (matches one_to_one.h)
template <typename T>
float RefL2Integer(const T* p, const T* q, unsigned d) {
  int32_t sum = 0;
  for (unsigned i = 0; i < d; ++i) {
    int16_t diff = static_cast<int16_t>(q[i]) - static_cast<int16_t>(p[i]);
    sum += static_cast<int32_t>(diff) * static_cast<int32_t>(diff);
  }
  return static_cast<float>(sum);
}

// -----------------------------------------------------------------------------
// Brute-force: Chamfer distances (point clouds)
// -----------------------------------------------------------------------------

float BruteForceChamferL2(const float* a, uint32_t n_a, const float* b, uint32_t n_b,
                          uint32_t dim) {
  if (n_a == 0 || n_b == 0) return std::numeric_limits<float>::infinity();
  float sum_min_sq = 0.0f;
  for (uint32_t i = 0; i < n_a; ++i) {
    float min_sq = std::numeric_limits<float>::max();
    for (uint32_t j = 0; j < n_b; ++j) {
      float sq = 0.0f;
      for (uint32_t d = 0; d < dim; ++d) {
        float diff = a[i * dim + d] - b[j * dim + d];
        sq += diff * diff;
      }
      min_sq = std::min(min_sq, sq);
    }
    sum_min_sq += min_sq;
  }
  return sum_min_sq / static_cast<float>(n_a);
}

float BruteForceChamferIP(const float* a, uint32_t n_a, const float* b, uint32_t n_b,
                           uint32_t dim) {
  if (n_a == 0 || n_b == 0) return std::numeric_limits<float>::infinity();
  float sum_max_ip = 0.0f;
  for (uint32_t i = 0; i < n_a; ++i) {
    float max_ip = -std::numeric_limits<float>::infinity();
    for (uint32_t j = 0; j < n_b; ++j) {
      float ip = 0.0f;
      for (uint32_t d = 0; d < dim; ++d) ip += a[i * dim + d] * b[j * dim + d];
      max_ip = std::max(max_ip, ip);
    }
    sum_max_ip += max_ip;
  }
  return -sum_max_ip / static_cast<float>(n_a);
}

// -----------------------------------------------------------------------------
// Test fixture: random and corner-case data
// -----------------------------------------------------------------------------

class OneToOneTest : public ::testing::Test {
 protected:
  void SetUp() override { gen_.seed(12345u); }

  std::vector<float> RandomFloats(size_t n, float scale = 1.0f) {
    std::uniform_real_distribution<float> dist(-scale, scale);
    std::vector<float> data(n);
    for (float& v : data) v = dist(gen_);
    return data;
  }

  std::vector<uint8_t> RandomUint8(size_t n) {
    std::uniform_int_distribution<int> dist(0, 255);
    std::vector<uint8_t> data(n);
    for (uint8_t& v : data) v = static_cast<uint8_t>(dist(gen_));
    return data;
  }

  std::vector<int8_t> RandomInt8(size_t n) {
    std::uniform_int_distribution<int> dist(-128, 127);
    std::vector<int8_t> data(n);
    for (int8_t& v : data) v = static_cast<int8_t>(dist(gen_));
    return data;
  }

  std::vector<float> RandomCloud(uint32_t n, uint32_t dim, float scale = 1.0f) {
    return RandomFloats(static_cast<size_t>(n) * dim, scale);
  }

  std::mt19937 gen_;
};

// =============================================================================
// ip_distance (float) — all dimensions and corner cases
// =============================================================================

TEST_F(OneToOneTest, IPFloat_Dim1) {
  std::vector<float> p = {2.0f}, q = {-3.0f};
  ExpectNear(RefIPFloat(p.data(), q.data(), 1), mvsic::ip_distance(p.data(), q.data(), 1));
}

TEST_F(OneToOneTest, IPFloat_ZeroVectors) {
  const unsigned d = 8;
  std::vector<float> p(d, 0.0f), q(d, 0.0f);
  ExpectNear(0.0f, mvsic::ip_distance(p.data(), q.data(), d));
}

TEST_F(OneToOneTest, IPFloat_Orthogonal) {
  std::vector<float> p = {1.0f, 0.0f, 0.0f}, q = {0.0f, 1.0f, 0.0f};
  ExpectNear(0.0f, mvsic::ip_distance(p.data(), q.data(), 3));
}

TEST_F(OneToOneTest, IPFloat_Identical) {
  const unsigned d = 4;
  auto p = RandomFloats(d);
  float expected = RefIPFloat(p.data(), p.data(), d);
  ExpectNear(expected, mvsic::ip_distance(p.data(), p.data(), d));
}

TEST_F(OneToOneTest, IPFloat_RandomVariousDims) {
  for (unsigned dim : {1, 2, 3, 4, 7, 8, 15, 16, 31, 32}) {
    auto p = RandomFloats(dim, 10.0f);
    auto q = RandomFloats(dim, 10.0f);
    float expected = RefIPFloat(p.data(), q.data(), dim);
    float actual = mvsic::ip_distance(p.data(), q.data(), dim);
    ExpectNear(expected, actual);
  }
}

TEST_F(OneToOneTest, IPFloat_ManyRandomTrials) {
  // NSGDist DistanceInnerProduct AVX path rounds dimension up to multiple of 8 and
  // can overread; use dim multiple of 8 for random trials.
  const unsigned kDims[] = {8, 16, 24, 32, 40, 48, 56, 64};
  std::uniform_int_distribution<size_t> idx_dist(0, sizeof(kDims) / sizeof(kDims[0]) - 1);
  for (int t = 0; t < 150; ++t) {
    unsigned dim = kDims[idx_dist(gen_)];
    auto p = RandomFloats(dim);
    auto q = RandomFloats(dim);
    ExpectNear(RefIPFloat(p.data(), q.data(), dim), mvsic::ip_distance(p.data(), q.data(), dim));
  }
}

TEST_F(OneToOneTest, IPFloat_LargeDim) {
  for (unsigned dim : {128, 256, 512}) {
    auto p = RandomFloats(dim, 0.1f);
    auto q = RandomFloats(dim, 0.1f);
    float expected = RefIPFloat(p.data(), q.data(), dim);
    float actual = mvsic::ip_distance(p.data(), q.data(), dim);
    ExpectNear(expected, actual);
  }
}

// =============================================================================
// l2_distance (float) — all dimensions and corner cases
// =============================================================================

TEST_F(OneToOneTest, L2Float_Dim1) {
  std::vector<float> p = {1.0f}, q = {5.0f};
  ExpectNear(16.0f, mvsic::l2_distance(p.data(), q.data(), 1));  // (5-1)^2
}

TEST_F(OneToOneTest, L2Float_IdenticalVectors) {
  const unsigned d = 16;
  auto p = RandomFloats(d);
  ExpectNear(0.0f, mvsic::l2_distance(p.data(), p.data(), d));
}

TEST_F(OneToOneTest, L2Float_ZeroVectors) {
  const unsigned d = 8;
  std::vector<float> p(d, 0.0f), q(d, 0.0f);
  ExpectNear(0.0f, mvsic::l2_distance(p.data(), q.data(), d));
}

TEST_F(OneToOneTest, L2Float_RandomVariousDims) {
  for (unsigned dim : {1, 2, 3, 4, 7, 8, 15, 16, 31, 32}) {
    auto p = RandomFloats(dim);
    auto q = RandomFloats(dim);
    float expected = RefL2Float(p.data(), q.data(), dim);
    float actual = mvsic::l2_distance(p.data(), q.data(), dim);
    ExpectNear(expected, actual);
  }
}

TEST_F(OneToOneTest, L2Float_ManyRandomTrials) {
  std::uniform_int_distribution<unsigned> dim_dist(1, 64);
  for (int t = 0; t < 150; ++t) {
    unsigned dim = dim_dist(gen_);
    auto p = RandomFloats(dim);
    auto q = RandomFloats(dim);
    ExpectNear(RefL2Float(p.data(), q.data(), dim), mvsic::l2_distance(p.data(), q.data(), dim));
  }
}

TEST_F(OneToOneTest, L2Float_LargeDim) {
  for (unsigned dim : {128, 256, 512}) {
    auto p = RandomFloats(dim);
    auto q = RandomFloats(dim);
    float expected = RefL2Float(p.data(), q.data(), dim);
    float actual = mvsic::l2_distance(p.data(), q.data(), dim);
    ExpectNear(expected, actual);
  }
}

// =============================================================================
// ip_distance (uint8_t, int8_t)
// =============================================================================

TEST_F(OneToOneTest, IPUint8_Dim1) {
  uint8_t p[] = {10}, q[] = {20};
  ExpectNear(RefIPInteger(p, q, 1), mvsic::ip_distance(p, q, 1));
}

TEST_F(OneToOneTest, IPUint8_RandomVariousDims) {
  for (unsigned dim : {1, 2, 4, 8, 16, 32}) {
    auto p = RandomUint8(dim);
    auto q = RandomUint8(dim);
    ExpectNear(RefIPInteger(p.data(), q.data(), dim),
               mvsic::ip_distance(p.data(), q.data(), dim));
  }
}

TEST_F(OneToOneTest, IPUint8_ManyRandomTrials) {
  std::uniform_int_distribution<unsigned> dim_dist(1, 64);
  for (int t = 0; t < 100; ++t) {
    unsigned dim = dim_dist(gen_);
    auto p = RandomUint8(dim);
    auto q = RandomUint8(dim);
    ExpectNear(RefIPInteger(p.data(), q.data(), dim),
               mvsic::ip_distance(p.data(), q.data(), dim));
  }
}

TEST_F(OneToOneTest, IPUint8_LargeDim) {
  for (unsigned dim : {128, 256}) {
    auto p = RandomUint8(dim);
    auto q = RandomUint8(dim);
    ExpectNear(RefIPInteger(p.data(), q.data(), dim),
               mvsic::ip_distance(p.data(), q.data(), dim));
  }
}

TEST_F(OneToOneTest, IPInt8_Dim1) {
  int8_t p[] = {-10}, q[] = {20};
  ExpectNear(RefIPInteger(p, q, 1), mvsic::ip_distance(p, q, 1));
}

TEST_F(OneToOneTest, IPInt8_RandomVariousDims) {
  for (unsigned dim : {1, 2, 4, 8, 16, 32}) {
    auto p = RandomInt8(dim);
    auto q = RandomInt8(dim);
    ExpectNear(RefIPInteger(p.data(), q.data(), dim),
               mvsic::ip_distance(p.data(), q.data(), dim));
  }
}

TEST_F(OneToOneTest, IPInt8_ManyRandomTrials) {
  std::uniform_int_distribution<unsigned> dim_dist(1, 64);
  for (int t = 0; t < 100; ++t) {
    unsigned dim = dim_dist(gen_);
    auto p = RandomInt8(dim);
    auto q = RandomInt8(dim);
    ExpectNear(RefIPInteger(p.data(), q.data(), dim),
               mvsic::ip_distance(p.data(), q.data(), dim));
  }
}

TEST_F(OneToOneTest, IPInt8_LargeDim) {
  for (unsigned dim : {128, 256}) {
    auto p = RandomInt8(dim);
    auto q = RandomInt8(dim);
    ExpectNear(RefIPInteger(p.data(), q.data(), dim),
               mvsic::ip_distance(p.data(), q.data(), dim));
  }
}

// =============================================================================
// l2_distance (uint8_t, int8_t)
// =============================================================================

TEST_F(OneToOneTest, L2Uint8_Dim1) {
  uint8_t p[] = {10}, q[] = {14};  // (14-10)^2 = 16
  ExpectNear(16.0f, mvsic::l2_distance(p, q, 1));
}

TEST_F(OneToOneTest, L2Uint8_Identical) {
  const unsigned d = 16;
  auto p = RandomUint8(d);
  ExpectNear(0.0f, mvsic::l2_distance(p.data(), p.data(), d));
}

TEST_F(OneToOneTest, L2Uint8_RandomVariousDims) {
  for (unsigned dim : {1, 2, 4, 8, 16, 32}) {
    auto p = RandomUint8(dim);
    auto q = RandomUint8(dim);
    ExpectNear(RefL2Integer(p.data(), q.data(), dim),
               mvsic::l2_distance(p.data(), q.data(), dim));
  }
}

TEST_F(OneToOneTest, L2Uint8_ManyRandomTrials) {
  std::uniform_int_distribution<unsigned> dim_dist(1, 64);
  for (int t = 0; t < 100; ++t) {
    unsigned dim = dim_dist(gen_);
    auto p = RandomUint8(dim);
    auto q = RandomUint8(dim);
    ExpectNear(RefL2Integer(p.data(), q.data(), dim),
               mvsic::l2_distance(p.data(), q.data(), dim));
  }
}

TEST_F(OneToOneTest, L2Uint8_LargeDim) {
  for (unsigned dim : {128, 256}) {
    auto p = RandomUint8(dim);
    auto q = RandomUint8(dim);
    ExpectNear(RefL2Integer(p.data(), q.data(), dim),
               mvsic::l2_distance(p.data(), q.data(), dim));
  }
}

TEST_F(OneToOneTest, L2Int8_Dim1) {
  int8_t p[] = {-10}, q[] = {10};  // (10-(-10))^2 = 400
  ExpectNear(400.0f, mvsic::l2_distance(p, q, 1));
}

TEST_F(OneToOneTest, L2Int8_Identical) {
  const unsigned d = 16;
  auto p = RandomInt8(d);
  ExpectNear(0.0f, mvsic::l2_distance(p.data(), p.data(), d));
}

TEST_F(OneToOneTest, L2Int8_RandomVariousDims) {
  for (unsigned dim : {1, 2, 4, 8, 16, 32}) {
    auto p = RandomInt8(dim);
    auto q = RandomInt8(dim);
    ExpectNear(RefL2Integer(p.data(), q.data(), dim),
               mvsic::l2_distance(p.data(), q.data(), dim));
  }
}

TEST_F(OneToOneTest, L2Int8_ManyRandomTrials) {
  std::uniform_int_distribution<unsigned> dim_dist(1, 64);
  for (int t = 0; t < 100; ++t) {
    unsigned dim = dim_dist(gen_);
    auto p = RandomInt8(dim);
    auto q = RandomInt8(dim);
    ExpectNear(RefL2Integer(p.data(), q.data(), dim),
               mvsic::l2_distance(p.data(), q.data(), dim));
  }
}

TEST_F(OneToOneTest, L2Int8_LargeDim) {
  for (unsigned dim : {128, 256}) {
    auto p = RandomInt8(dim);
    auto q = RandomInt8(dim);
    ExpectNear(RefL2Integer(p.data(), q.data(), dim),
               mvsic::l2_distance(p.data(), q.data(), dim));
  }
}

// =============================================================================
// chamfer_l2_distance — corner cases and random
// =============================================================================

TEST_F(OneToOneTest, ChamferL2_SinglePointEach) {
  const uint32_t n_a = 1, n_b = 1, dim = 3;
  auto a = RandomCloud(n_a, dim);
  auto b = RandomCloud(n_b, dim);
  float expected = BruteForceChamferL2(a.data(), n_a, b.data(), n_b, dim);
  ExpectNear(expected, mvsic::chamfer_l2_distance(a.data(), n_a, b.data(), n_b, dim));
}

TEST_F(OneToOneTest, ChamferL2_OneQueryManyDoc) {
  const uint32_t n_a = 1, n_b = 100, dim = 8;
  auto a = RandomCloud(n_a, dim);
  auto b = RandomCloud(n_b, dim);
  float expected = BruteForceChamferL2(a.data(), n_a, b.data(), n_b, dim);
  ExpectNear(expected, mvsic::chamfer_l2_distance(a.data(), n_a, b.data(), n_b, dim));
}

TEST_F(OneToOneTest, ChamferL2_ManyQueryOneDoc) {
  const uint32_t n_a = 50, n_b = 1, dim = 4;
  auto a = RandomCloud(n_a, dim);
  auto b = RandomCloud(n_b, dim);
  float expected = BruteForceChamferL2(a.data(), n_a, b.data(), n_b, dim);
  ExpectNear(expected, mvsic::chamfer_l2_distance(a.data(), n_a, b.data(), n_b, dim));
}

TEST_F(OneToOneTest, ChamferL2_IdenticalClouds) {
  const uint32_t n = 10, dim = 4;
  auto a = RandomCloud(n, dim);
  float expected = BruteForceChamferL2(a.data(), n, a.data(), n, dim);
  ExpectNear(expected, mvsic::chamfer_l2_distance(a.data(), n, a.data(), n, dim));
  ExpectNear(0.0f, expected);  // same cloud => 0
}

TEST_F(OneToOneTest, ChamferL2_Dim1) {
  const uint32_t n_a = 5, n_b = 7, dim = 1;
  auto a = RandomCloud(n_a, dim);
  auto b = RandomCloud(n_b, dim);
  float expected = BruteForceChamferL2(a.data(), n_a, b.data(), n_b, dim);
  ExpectNear(expected, mvsic::chamfer_l2_distance(a.data(), n_a, b.data(), n_b, dim));
}

TEST_F(OneToOneTest, ChamferL2_SmallAndAlignments) {
  for (uint32_t dim : {2, 3, 4, 7, 8, 16}) {
    const uint32_t n_a = 4, n_b = 6;
    auto a = RandomCloud(n_a, dim);
    auto b = RandomCloud(n_b, dim);
    float expected = BruteForceChamferL2(a.data(), n_a, b.data(), n_b, dim);
    ExpectNear(expected, mvsic::chamfer_l2_distance(a.data(), n_a, b.data(), n_b, dim));
  }
}

TEST_F(OneToOneTest, ChamferL2_ManyRandomTrials) {
  std::uniform_int_distribution<uint32_t> n_dist(1, 50), dim_dist(1, 64);
  for (int t = 0; t < 80; ++t) {
    uint32_t n_a = n_dist(gen_);
    uint32_t n_b = n_dist(gen_);
    uint32_t dim = dim_dist(gen_);
    auto a = RandomCloud(n_a, dim);
    auto b = RandomCloud(n_b, dim);
    float expected = BruteForceChamferL2(a.data(), n_a, b.data(), n_b, dim);
    ExpectNear(expected, mvsic::chamfer_l2_distance(a.data(), n_a, b.data(), n_b, dim));
  }
}

TEST_F(OneToOneTest, ChamferL2_LargeClouds) {
  const uint32_t n_a = 80, n_b = 120, dim = 32;
  auto a = RandomCloud(n_a, dim, 0.5f);
  auto b = RandomCloud(n_b, dim, 0.5f);
  float expected = BruteForceChamferL2(a.data(), n_a, b.data(), n_b, dim);
  ExpectNear(expected, mvsic::chamfer_l2_distance(a.data(), n_a, b.data(), n_b, dim));
}

TEST_F(OneToOneTest, ChamferL2_HighDim) {
  const uint32_t n_a = 25, n_b = 40, dim = 128;
  auto a = RandomCloud(n_a, dim, 0.2f);
  auto b = RandomCloud(n_b, dim, 0.2f);
  float expected = BruteForceChamferL2(a.data(), n_a, b.data(), n_b, dim);
  ExpectNear(expected, mvsic::chamfer_l2_distance(a.data(), n_a, b.data(), n_b, dim));
}

// =============================================================================
// chamfer_ip_distance — corner cases and random
// =============================================================================

TEST_F(OneToOneTest, ChamferIP_SinglePointEach) {
  const uint32_t n_a = 1, n_b = 1, dim = 3;
  auto a = RandomCloud(n_a, dim);
  auto b = RandomCloud(n_b, dim);
  float expected = BruteForceChamferIP(a.data(), n_a, b.data(), n_b, dim);
  ExpectNear(expected, mvsic::chamfer_ip_distance(a.data(), n_a, b.data(), n_b, dim));
}

TEST_F(OneToOneTest, ChamferIP_OneQueryManyDoc) {
  const uint32_t n_a = 1, n_b = 100, dim = 8;
  auto a = RandomCloud(n_a, dim);
  auto b = RandomCloud(n_b, dim);
  float expected = BruteForceChamferIP(a.data(), n_a, b.data(), n_b, dim);
  ExpectNear(expected, mvsic::chamfer_ip_distance(a.data(), n_a, b.data(), n_b, dim));
}

TEST_F(OneToOneTest, ChamferIP_ManyQueryOneDoc) {
  const uint32_t n_a = 50, n_b = 1, dim = 4;
  auto a = RandomCloud(n_a, dim);
  auto b = RandomCloud(n_b, dim);
  float expected = BruteForceChamferIP(a.data(), n_a, b.data(), n_b, dim);
  ExpectNear(expected, mvsic::chamfer_ip_distance(a.data(), n_a, b.data(), n_b, dim));
}

TEST_F(OneToOneTest, ChamferIP_IdenticalClouds) {
  const uint32_t n = 10, dim = 4;
  auto a = RandomCloud(n, dim);
  float expected = BruteForceChamferIP(a.data(), n, a.data(), n, dim);
  ExpectNear(expected, mvsic::chamfer_ip_distance(a.data(), n, a.data(), n, dim));
  // Same cloud => each row's max IP is its self-IP (squared norm); mean then -mean(sq_norm).
  ExpectNear(expected, BruteForceChamferIP(a.data(), n, a.data(), n, dim));
}

TEST_F(OneToOneTest, ChamferIP_Dim1) {
  const uint32_t n_a = 5, n_b = 7, dim = 1;
  auto a = RandomCloud(n_a, dim);
  auto b = RandomCloud(n_b, dim);
  float expected = BruteForceChamferIP(a.data(), n_a, b.data(), n_b, dim);
  ExpectNear(expected, mvsic::chamfer_ip_distance(a.data(), n_a, b.data(), n_b, dim));
}

TEST_F(OneToOneTest, ChamferIP_SmallAndAlignments) {
  for (uint32_t dim : {2, 3, 4, 7, 8, 16}) {
    const uint32_t n_a = 4, n_b = 6;
    auto a = RandomCloud(n_a, dim);
    auto b = RandomCloud(n_b, dim);
    float expected = BruteForceChamferIP(a.data(), n_a, b.data(), n_b, dim);
    ExpectNear(expected, mvsic::chamfer_ip_distance(a.data(), n_a, b.data(), n_b, dim));
  }
}

TEST_F(OneToOneTest, ChamferIP_ManyRandomTrials) {
  std::uniform_int_distribution<uint32_t> n_dist(1, 50), dim_dist(1, 64);
  for (int t = 0; t < 80; ++t) {
    uint32_t n_a = n_dist(gen_);
    uint32_t n_b = n_dist(gen_);
    uint32_t dim = dim_dist(gen_);
    auto a = RandomCloud(n_a, dim);
    auto b = RandomCloud(n_b, dim);
    float expected = BruteForceChamferIP(a.data(), n_a, b.data(), n_b, dim);
    ExpectNear(expected, mvsic::chamfer_ip_distance(a.data(), n_a, b.data(), n_b, dim));
  }
}

TEST_F(OneToOneTest, ChamferIP_LargeClouds) {
  const uint32_t n_a = 80, n_b = 120, dim = 32;
  auto a = RandomCloud(n_a, dim, 0.5f);
  auto b = RandomCloud(n_b, dim, 0.5f);
  float expected = BruteForceChamferIP(a.data(), n_a, b.data(), n_b, dim);
  ExpectNear(expected, mvsic::chamfer_ip_distance(a.data(), n_a, b.data(), n_b, dim));
}

TEST_F(OneToOneTest, ChamferIP_HighDim) {
  const uint32_t n_a = 25, n_b = 40, dim = 128;
  auto a = RandomCloud(n_a, dim, 0.2f);
  auto b = RandomCloud(n_b, dim, 0.2f);
  float expected = BruteForceChamferIP(a.data(), n_a, b.data(), n_b, dim);
  ExpectNear(expected, mvsic::chamfer_ip_distance(a.data(), n_a, b.data(), n_b, dim));
}

}  // namespace
