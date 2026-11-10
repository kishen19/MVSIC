#include "mvsic/emdivf/emd.h"

#include <algorithm>
#include <utility>
#include <vector>
#include <random>
#include <limits>

#include "gtest/gtest.h"

TEST(Emd, EqualCardinalityUsesOptimalAssignment) {
  const float a[] = {0, 2};
  const float b[] = {2, 0};
  EXPECT_NEAR(emdivf::Emd(a, 2, b, 2, 1), 0, 1e-6);
  const float c[] = {1, 3};
  EXPECT_NEAR(emdivf::Emd(a, 2, c, 2, 1), 1, 1e-6);
}

TEST(Emd, CappedMatchesExactForEqualAndUnequalClouds) {
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> draw(-1, 1);
  for (uint32_t na : {1u, 3u, 8u, 32u}) {
    for (uint32_t nb : {1u, 3u, 8u, 32u}) {
      std::vector<float> a(na*8), b(nb*8);
      for (auto& x : a) x=draw(rng);
      for (auto& x : b) x=draw(rng);
      const double exact=emdivf::Emd(a.data(),na,b.data(),nb,8);
      for (double cap : {0., exact*.25, exact*.9, exact, exact*1.1,
                         std::numeric_limits<double>::infinity()})
        EXPECT_NEAR(emdivf::EmdCapped(a.data(),na,b.data(),nb,8,cap),
                    std::min(exact,cap),1e-10);
    }
  }
}

TEST(Emd, UnequalCardinalityUsesBalancedMass) {
  const float a[] = {0, 2};
  const float b[] = {0};
  EXPECT_NEAR(emdivf::Emd(a, 2, b, 1, 1), 1, 1e-6);
  const float c[] = {0, 2, 2};
  EXPECT_NEAR(emdivf::Emd(a, 2, c, 3, 1), 1.0 / 3.0, 1e-6);
}

TEST(Emd, LowerBoundTopKMatchesExhaustiveEmd) {
  const float query[] = {0, 2, 4};
  const float a[] = {0, 2, 4};
  const float b[] = {1, 3, 5};
  const float c[] = {10, 11, 12};
  const float d[] = {-1, 2, 4};
  std::vector<const float*> clouds{c, b, a, d};
  auto result = emdivf::EmdTopKEqual(query, 3, clouds, 1, 2);
  std::vector<std::pair<double, uint32_t>> exhaustive;
  for (uint32_t i = 0; i < clouds.size(); ++i)
    exhaustive.push_back({emdivf::Emd(query, 3, clouds[i], 3, 1), i});
  std::sort(exhaustive.begin(), exhaustive.end());
  ASSERT_EQ(result.indices.size(), 2);
  for (uint32_t rank = 0; rank < 2; ++rank) {
    EXPECT_EQ(result.indices[rank], exhaustive[rank].second);
    EXPECT_NEAR(result.distances[rank], exhaustive[rank].first, 1e-9);
  }
  EXPECT_LT(result.assignments, clouds.size());
}
