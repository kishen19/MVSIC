#include <array>
#include <random>
#include "gtest/gtest.h"
#include "parlay/primitives.h"
#include "mvsic/core/quantization/turboquant_8bit_mv.h"
#include "mvsic/core/quantization/turboquant_mv.h"
#include "mvsic/core/types/chamfer_ip_point.h"
#include "mvsic/core/types/chamfer_l2_point.h"
#include "mvsic/core/types/point_cloud_set.h"

TEST(TurboQuantDot, FullRangeBytesMatchInt32Reference) {
#ifdef __AVX512F__
  std::mt19937 rng(42);
  alignas(64) std::array<uint8_t, 64> a;
  alignas(64) std::array<int8_t, 64> b;
  alignas(64) std::array<int32_t, 16> got8, got4;
  for (int trial = 0; trial < 1000; ++trial) {
    for (int i = 0; i < 64; ++i) {
      a[i] = trial < 2 ? 255 : rng() % 256;
      b[i] = trial == 0 ? 127 : trial == 1 ? -128 : int(rng() % 256) - 128;
    }
    auto va = _mm512_load_si512(a.data());
    auto vb = _mm512_load_si512(b.data());
    auto start = _mm512_set1_epi32(123);
    _mm512_store_si512(got8.data(), mvsic::turboquant_8bit_mv::internal::tq8_dpbusd(start, va, vb));
    _mm512_store_si512(got4.data(), mvsic::turboquant_mv::internal::tq_dpbusd(start, va, vb));
    for (int lane = 0; lane < 16; ++lane) {
      int32_t expected = 123;
      for (int j = 0; j < 4; ++j) expected += int(a[4 * lane + j]) * int(b[4 * lane + j]);
      ASSERT_EQ(got8[lane], expected) << "TQ8 trial=" << trial;
      ASSERT_EQ(got4[lane], expected) << "TQ4 trial=" << trial;
    }
  }
#else
  GTEST_SKIP() << "AVX512 fallback test requires AVX512";
#endif
}

template<bool Metric>
void CheckPanels() {
  using Point = std::conditional_t<Metric, mvsic::ChamferL2_Point, mvsic::ChamferIP_Point>;
  using Set = mvsic::PointCloudSet<Point>;
  std::mt19937 rng(17);
  std::normal_distribution<float> normal;
  for (size_t cs : {1, 15, 16, 17, 63, 64, 65, 127, 298}) {
    Set docs(2, cs, 128);
    for (size_t i = 0; i < 2 * cs * 128; ++i) docs.data()[i] = normal(rng);
    mvsic::turboquant_8bit_mv::Model<Metric> model;
    model.train(docs);
    auto db = model.encode(docs);
    for (size_t nq : {1, 5, 6, 7, 32}) {
      Set queries(1, nq, 128);
      for (size_t i = 0; i < nq * 128; ++i) queries.data()[i] = normal(rng);
      auto q = model.quantize_query(queries[0]);
      for (size_t c = 0; c < 2; ++c) {
        const size_t po = db.point_offsets[c], bo = db.panel_offsets[c];
        const size_t np = (db.panel_offsets[c + 1] - bo) / db.panel_bytes;
        float scalar = mvsic::turboquant_8bit_mv::internal::chamfer_panels_scalar<Metric>(
            q.flat_query_data.data(), q.norm_scaling_factors.data(),
            q.unquantized_squared_norms.data(), q.byte_sums.data(), q.q_stride, nq,
            db.panel_data.data() + bo, db.norm_scaling_factors.data() + po,
            db.unquantized_squared_norms.data() + po, db.total_tiles, db.panel_bytes, np, cs) / nq;
        float simd = mvsic::turboquant_8bit_mv::turboquant_8bit_mv_chamfer_distance<Metric>(q, db, c);
        ASSERT_NEAR(simd, scalar, 1e-4f * std::max(1.0f, std::abs(scalar)))
            << "cloud_size=" << cs << " query_size=" << nq;
      }
    }
  }
}

TEST(TurboQuantDot, TQ8PanelsMatchScalarIP) { CheckPanels<false>(); }
TEST(TurboQuantDot, TQ8PanelsMatchScalarL2) { CheckPanels<true>(); }

TEST(TurboQuantDot, RotationSeedIsReproducible) {
  mvsic::turboquant::BaseEncoder a, b;
  a.train(128);
  b.train(128);
  std::array<float, 128> input{}, x{}, y{};
  input[0] = 1.0f;
  a.rotator->rotate(input.data(), x.data());
  b.rotator->rotate(input.data(), y.data());
  EXPECT_EQ(x, y);
}
