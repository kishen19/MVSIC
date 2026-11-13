#include "mvsic/algorithms/mvivf/mvclustering/mvclustering.h"
#include "mvsic/algorithms/mvivf/mvclustering/mvclustering_internal.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>
#include "gtest/gtest.h"
#include "mvsic/algorithms/mvivf/mvclustering/mvclustering_impl.h"
#include "mvsic/core/quantization/turboquant_1bit.h"

namespace mvsic {
namespace {

const ClusteringQuantizer codecs[] = {ClusteringQuantizer::None, ClusteringQuantizer::TQ8,
                                      ClusteringQuantizer::TQ4, ClusteringQuantizer::TQ1};

PointCloudSetView Rays(const std::vector<float>& scales, uint32_t d = 16) {
  std::vector<float> values(scales.size() * d);
  std::vector<size_t> offsets(scales.size() + 1);
  std::vector<uint32_t> ids(scales.size(), 1001);  // IDs are irrelevant to assignments.
  for (size_t i = 0; i < scales.size(); ++i) {
    offsets[i] = i * d;
    for (size_t j = 0; j < d; ++j)
      values[i * d + j] = scales[i] / std::sqrt(float(d));
  }
  offsets.back() = values.size();
  return own_point_clouds(std::move(values), std::move(offsets), d, std::move(ids));
}

float Distance(PointCloud a, PointCloud b) {
  float score = 0;
  for (size_t i = 0; i < a.n; ++i) {
    float nearest = std::numeric_limits<float>::infinity();
    for (size_t j = 0; j < b.n; ++j) {
      float distance = 0;
      for (size_t t = 0; t < a.d; ++t) {
        const float x = a.data(i)[t], y = b.data(j)[t];
        distance -= x * y;
      }
      nearest = std::min(nearest, distance);
    }
    score += nearest;
  }
  return score / a.n;
}

// Decode bytes directly so the oracle shares neither the assignment panels
// nor the single-vector distance helper with the compiled implementation.
std::vector<uint32_t> OneBitOracle(const PointCloudSetView& points,
                                   const PointCloudSetView& centers, uint32_t seed) {
  struct Vectors {
    const PointCloudSetView& points;
    size_t size() const { return points.total_size(); }
    uint32_t get_dims() const { return points.d; }
    const float* location(size_t i) const { return points.values + i * points.d; }
  };
  turboquant_1bit::Model<false> model;
  model.encoder.seed_ = seed;
  model.train(Vectors{centers});
  const auto db = model.encode(Vectors{centers});
  std::vector<turboquant_1bit::Quantized_Query<false>> queries(points.total_size());
  for (size_t q = 0; q < queries.size(); ++q)
    queries[q] = model.quantize_query(points.values + q * points.d);
  std::vector<uint32_t> result(points.n);
  for (size_t row = 0; row < points.n; ++row) {
    float best = std::numeric_limits<float>::infinity();
    for (size_t c = 0; c < centers.n; ++c) {
      float sum = 0;
      for (size_t q = points.offsets[row] / points.d; q < points.offsets[row + 1] / points.d; ++q) {
        float nearest = std::numeric_limits<float>::infinity();
        for (size_t v = centers.offsets[c] / centers.d; v < centers.offsets[c + 1] / centers.d;
             ++v) {
          uint32_t hamming = 0;
          for (size_t byte = 0; byte < db.num_bytes_per_datapoint; ++byte)
            hamming +=
                __builtin_popcount(unsigned(db.packed_codes[v * db.num_bytes_per_datapoint + byte] ^
                                            queries[q].query_data[byte]));
          const float dot = 16129.0f * (float(db.padded_dim) - 2.0f * float(hamming));
          float distance = -dot * db.norm_scaling_factors[v] * queries[q].norm_scaling_factor;
          nearest = std::min(nearest, distance);
        }
        sum += nearest;
      }
      if (sum < best) {
        best = sum;
        result[row] = static_cast<uint32_t>(c);
      }
    }
  }
  return result;
}

TEST(MVClustering, BorrowedInnerRowsKeepCoordinatesOrderAndSourceLifetime) {
  using Set = PointCloudSet<ChamferIP_Point>;
  for (uint32_t d : {7u, 31u, 128u}) {
    const std::vector<size_t> lengths{1, 3, 2, 5};
    const std::vector<uint32_t> public_ids{400, 20, 900, 1};
    std::vector<size_t> offsets{0};
    for (size_t length : lengths) offsets.push_back(offsets.back() + length * d);
    std::vector<float> values(offsets.back());
    const float pattern[]{0.0f, -0.0f, 1.25f, -3.5f, 1e-20f};
    for (size_t i = 0; i < values.size(); ++i) values[i] = pattern[i % 5];
    Set::FlatRows rows;
    {
      auto points = Set::borrow(lengths.size(), d, values.data(), offsets.data(), public_ids.data());
      const parlay::sequence<uint32_t> selection{3, 0, 3, 2};
      rows = points.filter_flattened_view(selection);
    }  // Temporary selection and borrowed cloud wrapper are gone; coordinates still live.
    static_assert(std::is_const_v<std::remove_reference_t<decltype(rows[0][0])>>);
    ASSERT_EQ(rows.size(), 13);
    size_t cursor = 0;
    for (uint32_t doc : {3u, 0u, 3u, 2u})
      for (size_t vector = 0; vector < lengths[doc]; ++vector) {
        const float* expected = values.data() + offsets[doc] + vector * d;
        ASSERT_EQ(rows[cursor].size(), d);
        EXPECT_EQ(rows[cursor].data(), expected);
        EXPECT_EQ(std::memcmp(rows[cursor].data(), expected, d * sizeof(float)), 0);
        ++cursor;
      }
  }
}

TEST(MVClustering, TQ1PreparedInnerCachePreservesRaggedWeightedSamplesAndOwnership) {
  constexpr uint32_t d = 31;
  for (bool all_zero : {false, true}) {
    for (uint32_t inner_limit : {1, 20}) {
      std::vector<float> values;
      std::vector<size_t> offsets{0};
      std::vector<uint32_t> public_ids;
      for (size_t doc = 0; doc < 6; ++doc) {
        const size_t vectors = 3 + doc % 4;
        for (size_t v = 0; v < vectors; ++v)
          for (size_t j = 0; j < d; ++j) {
            const float x = all_zero || (doc == 0 && v == 0) ? 0.0f :
                float(int((doc * 13 + v * 7 + j * 3) % 19) - 9) *
                    (0.125f * float(1 + doc + v));
            values.push_back(x);
          }
        offsets.push_back(values.size());
        public_ids.push_back(uint32_t(9001 - doc * 17));
      }
      auto root = own_point_clouds(std::move(values), std::move(offsets), d, std::move(public_ids));
      // Root positions differ from public IDs; document5 appears twice, out of order.
      const std::vector<uint32_t> rows{5, 1, 5, 0, 3};
      auto local = gather_point_clouds(root, rows, false);
      MVClusteringConfig config;
      config.assignment_quantizer = ClusteringQuantizer::TQ1;
      config.seed = 11;  // Initial and empty-cluster reseeding mutate only trainer.params.seed.
      config.niters = 3;
      config.use_weighted_inner_kmeans = true;
      config.max_points_per_centroid_inner_kmeans = inner_limit;
      auto direct = cluster_point_clouds(local, 2, 2, config);
      ClusteringResult cached;
      {
        ClusteringContext context(root, config);
        cached = context.cluster(local, rows, 2, 2);
      }  // Every inner backend and its temporary index selection have finished.
      root = {};
      local = {};
      ASSERT_TRUE(cached.centers.owner);
      EXPECT_EQ(cached.assignments, direct.assignments);
      ASSERT_EQ(cached.centers.n, direct.centers.n);
      ASSERT_EQ(cached.centers.d, direct.centers.d);
      const size_t count = cached.centers.offsets[cached.centers.n];
      ASSERT_EQ(count, direct.centers.offsets[direct.centers.n]);
      EXPECT_EQ(std::memcmp(cached.centers.values, direct.centers.values, count * sizeof(float)), 0);
    }
  }
}

TEST(MVClustering, OneBitPanelsApplyNormsBeforeCloudMinima) {
  const uint32_t d = 31;
  const auto queries = Rays({1, 0}, d);
  // The second collinear vector wins despite identical sign codes.
  const auto vectors = Rays({0.1f, 10, 5, 5}, d);
  auto centers = own_point_clouds(
      std::vector<float>(vectors.values, vectors.values + vectors.offsets[vectors.n]),
      {0, 2 * d, 4 * d}, d);
  const auto actual = assign_point_clouds(queries, centers, ClusteringQuantizer::TQ1, 11);
  EXPECT_EQ(actual, (std::vector<uint32_t>{0, 0}));
  EXPECT_EQ(actual, OneBitOracle(queries, centers, 11));
}

TEST(MVClustering, OneBitPanelsMatchDecodedRaggedAndSingletonOracles) {
  std::mt19937 rng(317);
  std::normal_distribution<float> random;
  for (uint32_t d : {7, 31, 128, 257})
    for (size_t length : {1, 7, 8, 9, 15, 16, 17, 31, 32, 33}) {
      std::vector<float> query_values(9 * d), center_values((3 * length + 1) * d);
      for (size_t i = 0; i < query_values.size(); ++i)
        query_values[i] = random(rng) * float(1 + (i / d) % 7) / 4;
      for (size_t i = 0; i < center_values.size(); ++i)
        center_values[i] = random(rng) * float(1 + (i / d) % 5) / 3;
      std::fill_n(query_values.begin(), d, 0);
      std::fill_n(center_values.begin(), d, 0);
      auto queries = own_point_clouds(std::move(query_values), {0, d, 4 * d, 9 * d}, d);
      auto centers = own_point_clouds(
          std::move(center_values), {0, length * d, (2 * length + 1) * d, (3 * length + 1) * d}, d);
      const auto actual = assign_point_clouds(queries, centers, ClusteringQuantizer::TQ1, 11);
      EXPECT_EQ(actual, OneBitOracle(queries, centers, 11));
      // The public assignment boundary selects the same packed singleton
      // layout used by inner Lloyd assignment when all clouds contain one row.
      std::vector<size_t> offsets(length + 1);
      for (size_t i = 0; i <= length; ++i)
        offsets[i] = i * d;
      auto single_queries = own_point_clouds(
          std::vector<float>(queries.values, queries.values + 3 * d), {0, d, 2 * d, 3 * d}, d);
      auto single_centers = own_point_clouds(
          std::vector<float>(centers.values, centers.values + length * d), std::move(offsets), d);
      const auto singleton =
          assign_point_clouds(single_queries, single_centers, ClusteringQuantizer::TQ1, 11);
      EXPECT_EQ(singleton, OneBitOracle(single_queries, single_centers, 11));
    }
}

TEST(MVClustering, OneBitSingletonAssignmentKeepsLowestCenterIdAcrossPanels) {
  auto queries = Rays({1, 0, -1}, 31);
  std::vector<float> scales(17, -1);
  scales[1] = scales[8] = scales[16] = 1;
  for (uint32_t expected : {1, 8}) {
    auto centers = Rays(scales, 31);
    const auto actual = assign_point_clouds(queries, centers, ClusteringQuantizer::TQ1, 11);
    EXPECT_EQ(actual, (std::vector<uint32_t>{expected, 0, 0}));
    EXPECT_EQ(actual, OneBitOracle(queries, centers, 11));
    scales[1] = -1;
  }
}

TEST(MVClustering, FloatAssignmentMatchesIndependentRaggedOracle) {
  std::mt19937 rng(31);
  std::uniform_real_distribution<float> random(-3, 3);
  std::vector<float> values(28 * 16);
  for (float& x : values)
    x = random(rng);
  auto points = own_point_clouds(std::move(values), {0, 32, 96, 112, 176, 224, 320, 448}, 16);
  const std::vector<uint32_t> rows{1, 3, 5};
  auto centers = gather_point_clouds(points, rows);
  auto actual = assign_point_clouds(points, centers, ClusteringQuantizer::None);
  for (size_t i = 0; i < points.n; ++i) {
    uint32_t expected = 0;
    float best = std::numeric_limits<float>::infinity();
    for (size_t j = 0; j < centers.n; ++j) {
      const float distance = Distance(points[i], centers[j]);
      if (distance < best) {
        best = distance;
        expected = j;
      }
    }
    EXPECT_EQ(actual[i], expected);
  }
}

TEST(MVClustering, QuantizedAssignmentRetainsNormsAndZeroVectors) {
  auto centers = Rays({0, 1, 10, -10});
  auto queries = Rays({0, 0.8f, 9, -9});
  for (auto codec : codecs) {
    auto ip = assign_point_clouds(queries, centers, codec, 17);
    EXPECT_EQ(ip[1], 2);
    EXPECT_EQ(ip[2], 2);
    EXPECT_EQ(ip[3], 3);
    EXPECT_EQ(ip, assign_point_clouds(queries, centers, codec, 17));
  }
}

TEST(MVClustering, QuantizedAssignmentKeepsFloatWeightedCentroidUpdates) {
  // Documents [1,0] and [0,3] repeated three times have different spherical
  // means when each document receives equal total mass.
  auto points = own_point_clouds({1, 0, 0, 3, 0, 3, 0, 3}, {0, 2, 8}, 2);
  for (auto codec : codecs)
    for (bool weighted : {false, true}) {
      MVClusteringConfig config;
      config.niters = 1;
      config.assignment_quantizer = codec;
      config.use_weighted_inner_kmeans = weighted;
      const auto result = cluster_point_clouds(points, 1, 1, config);
      const float y = weighted ? 3.0f : 9.0f;
      const float norm = std::sqrt(1 + y * y);
      EXPECT_NEAR(result.centers.values[0], 1 / norm, 1e-6f);
      EXPECT_NEAR(result.centers.values[1], y / norm, 1e-6f);
      EXPECT_EQ(result.assignments, (std::vector<uint32_t>{0, 0}));
    }
}

TEST(MVClustering, RepeatedTrainingAndUnderfilledCentroidsAreDeterministic) {
  auto points = Rays({1, 1, 1, 1, -1, -1, -1, -1});
  for (auto codec : codecs) {
    MVClusteringConfig config;
    config.assignment_quantizer = codec;
    config.niters = 2;
    config.seed = 4;
    const auto a = cluster_point_clouds(points, 4, 5, config);
    const auto b = cluster_point_clouds(points, 4, 5, config);
    ASSERT_EQ(a.centers.num_bytes(), b.centers.num_bytes());
    for (size_t i = 0; i < a.centers.offsets[a.centers.n]; ++i) {
      EXPECT_TRUE(std::isfinite(a.centers.values[i]));
      EXPECT_FLOAT_EQ(a.centers.values[i], b.centers.values[i]);
    }
    EXPECT_EQ(a.assignments, b.assignments);
    for (auto id : a.assignments)
      EXPECT_LT(id, 4);
  }
}

TEST(MVClustering, SubsampledTrainingAssignsEveryInputAndOwnsCenters) {
  auto points = Rays(std::vector<float>(2050, 1.0f));
  MVClusteringConfig config;
  config.max_point_clouds_per_cluster = 5;
  config.niters = 1;
  config.assignment_quantizer = ClusteringQuantizer::TQ8;
  auto result = cluster_point_clouds(points, 2, 1, config);
  points = {};
  ASSERT_EQ(result.assignments.size(), 2050);
  EXPECT_TRUE(result.centers.owner);
  EXPECT_TRUE(std::isfinite(result.centers.values[0]));
  for (auto id : result.assignments)
    EXPECT_LT(id, 2);
}

TEST(MVClustering, RecursivePreparedCacheMatchesDirectTraining) {
  auto root = Rays({-8, 7, -5, 4, 3, -2, 1, 9});
  const std::vector<uint32_t> rows{7, 0, 4, 2, 6, 1};
  auto local = gather_point_clouds(root, rows);
  for (auto codec : codecs) {
    MVClusteringConfig config;
    config.assignment_quantizer = codec;
    config.niters = 2;
    config.seed = 5;
    ClusteringContext context(root, config);
    auto cached = context.cluster(local, rows, 2, 1);
    auto direct = cluster_point_clouds(local, 2, 1, config);
    ASSERT_EQ(cached.centers.num_bytes(), direct.centers.num_bytes());
    EXPECT_EQ(cached.assignments, direct.assignments);
    for (size_t i = 0; i < cached.centers.offsets[cached.centers.n]; ++i)
      EXPECT_NEAR(cached.centers.values[i], direct.centers.values[i], 1e-6f);
  }
}

TEST(MVClustering, FloatFacadePreservesLegacyTrainingOnControlledData) {
  auto points = Rays({-4, -3, -2, 2, 3, 4});
  auto borrowed =
      PointCloudSet<ChamferIP_Point>::borrow(points.n, points.d, points.values, points.offsets);
  MVClusteringConfig config;
  config.seed = 3;
  config.niters = 2;
  MVClustering original(points.d, 2, 1, config);
  original.train(borrowed);
  auto actual = cluster_point_clouds(points, 2, 1, config);
  auto expected = original.get_clustering(borrowed);
  EXPECT_EQ(actual.assignments, (std::vector<uint32_t>(expected.begin(), expected.end())));
  for (size_t i = 0; i < actual.centers.offsets[actual.centers.n]; ++i)
    EXPECT_FLOAT_EQ(actual.centers.values[i], original.centers.data()[i]);
}

TEST(MVClustering, SphericalUpdateHandlesSubnormalMeans) {
  auto points = own_point_clouds({1e-40f, 0, 1e-40f, 0}, {0, 4}, 2);
  for (const std::string backend : {"legacy", "adaptive"}) {
    MVClusteringConfig config;
    config.niters = 1;
    config.inner_backend = backend;
    auto result = cluster_point_clouds(points, 1, 1, config);
    EXPECT_FLOAT_EQ(result.centers.values[0], 1.0f);
    EXPECT_FLOAT_EQ(result.centers.values[1], 0.0f);
  }
}

TEST(MVClustering, ExperimentalInnerTrainingRetainsSphericalWeightedUpdates) {
  auto points = own_point_clouds({1, 0, 0, 3, 0, 3, 0, 3}, {0, 2, 8}, 2);
  for (auto codec : codecs)
    for (bool weighted : {false, true}) {
      MVClusteringConfig config;
      config.niters = 1;
      config.assignment_quantizer = codec;
      config.inner_init = "kmeans++";
      config.inner_backend = "adaptive";
      config.use_weighted_inner_kmeans = weighted;
      auto result = cluster_point_clouds(points, 1, 1, config);
      const float y = weighted ? 3.0f : 9.0f;
      const float norm = std::sqrt(1 + y * y);
      EXPECT_NEAR(result.centers.values[0], 1 / norm, 1e-6f);
      EXPECT_NEAR(result.centers.values[1], y / norm, 1e-6f);
      EXPECT_EQ(result.assignments, (std::vector<uint32_t>{0, 0}));
    }
}

template<bool metric>
void CheckSeedAssignments() {
  const parlay::sequence<parlay::sequence<float>> points{
      {0, 0, 0}, {8, 1, 0}, {1, 3, 0}, {-4, 0, 1}, {0, -2, 0}, {1, 0, 7}, {0, 0, 0}, {-1, -1, -1}};
  const parlay::sequence<float> weights{0, 1, 2, 1, 3, 1, 0, 2};
  for (const std::string init : {"kmeans++", "prefix_doubling"}) {
    auto seeded = lloyds::initialize_inner<metric>(points, 4, init, 13, &weights, true);
    auto encoded = lloyds::initialize_inner<metric>(points, 4, init, 13, &weights, false);
    EXPECT_TRUE(encoded.assignments.empty());
    ASSERT_EQ(seeded.assignments.size(), points.size());
    for (size_t c = 0; c < 4; ++c) {
      double norm = 0;
      for (size_t j = 0; j < 3; ++j) {
        EXPECT_FLOAT_EQ(seeded.centers[c][j], encoded.centers[c][j]);
        norm += double(seeded.centers[c][j]) * seeded.centers[c][j];
      }
      if constexpr (!metric) EXPECT_NEAR(norm, 1.0, 1e-6);
    }
    for (size_t i = 0; i < points.size(); ++i) {
      auto score = [&](size_t c) {
        double value = 0;
        for (size_t j = 0; j < 3; ++j) {
          const double x = points[i][j], y = seeded.centers[c][j];
          value += metric ? (x - y) * (x - y) : -x * y;
        }
        return value;
      };
      const double actual = score(seeded.assignments[i]);
      for (size_t c = 0; c < 4; ++c)
        EXPECT_LE(actual, score(c) + 1e-5);
      // Zero input has equal IP scores; the first selected center wins ties.
      if constexpr (!metric)
        if (i == 0 || i == 6) EXPECT_EQ(seeded.assignments[i], 0);
    }
  }
}

TEST(MVClustering, PositiveDistanceSeedsReuseOnlyMatchingFloatAssignments) {
  CheckSeedAssignments<true>();
  CheckSeedAssignments<false>();
  const parlay::sequence<parlay::sequence<float>> repeated(6, parlay::sequence<float>{0, 0});
  const parlay::sequence<float> repeated_weights{1, 2, 3, 1, 2, 3};
  auto zero =
      lloyds::initialize_inner<false>(repeated, 4, "prefix_doubling", 9, &repeated_weights, true);
  for (auto id : zero.assignments)
    EXPECT_EQ(id, 0);
  // A zero center selected after a unit center must not win zero-query ties
  // merely because its chord sampling distance is smaller.
  const parlay::sequence<parlay::sequence<float>> mixed{{0, 0}, {2, 0}, {-3, 0}};
  const parlay::sequence<float> seed_weights{0, 1, 0};
  auto mixed_seed = lloyds::initialize_inner<false>(mixed, 2, "kmeans++", 0, &seed_weights, true);
  EXPECT_EQ(mixed_seed.assignments, (parlay::sequence<uint32_t>{0, 0, 1}));
  for (const std::string init : {"uniform", "kmeans++", "prefix_doubling"}) {
    auto result = lloyds::initialize_inner<true>(repeated, 4, init, 7, nullptr, true);
    ASSERT_EQ(result.centers.size(), 4);
    for (size_t c = 0; c < 4; ++c)
      for (size_t j = 0; j < 2; ++j)
        EXPECT_FLOAT_EQ(result.centers[c][j], 0);
  }
}

TEST(MVClustering, AdaptiveFloatLayoutMatchesLegacyOnUnpaddedDimensions) {
  const parlay::sequence<parlay::sequence<float>> points{
      {0, 0, 1}, {0, 0, 3}, {5, 3, 0}, {7, 5, 0}, {-3, -2, 0}};
  lloyds::CenterSet centers(2, 3);
  centers[0][2] = 2;
  centers[1][0] = 6;
  centers[1][1] = 4;
  lloyds::FloatLloydsBackend<true> legacy(points, 3), adaptive(points, 3, true);
  parlay::sequence<uint32_t> a, b;
  legacy.assign(centers, a);
  adaptive.assign(centers, b);
  ASSERT_EQ(a.size(), b.size());
  for (size_t i = 0; i < a.size(); ++i)
    EXPECT_EQ(a[i], b[i]);
  const parlay::sequence<float> weights{1, 2, 3, 4, 2};
  auto old_means = legacy.compute_means_weighted(a, weights, 2);
  auto new_means = adaptive.compute_means_weighted(b, weights, 2);
  for (size_t c = 0; c < 2; ++c)
    for (size_t j = 0; j < 3; ++j)
      EXPECT_NEAR(old_means[c][j], new_means[c][j], 1e-6f);
}

TEST(MVClustering, TQ8CacheMatchesDirectTrainingWithExperimentalSeeds) {
  auto root = Rays({-8, 7, -5, 4, 3, -2, 1, 9});
  const std::vector<uint32_t> rows{7, 0, 4, 2, 6, 1};
  auto local = gather_point_clouds(root, rows);
  for (const std::string init : {"uniform", "kmeans++", "prefix_doubling"}) {
    MVClusteringConfig config;
    config.assignment_quantizer = ClusteringQuantizer::TQ8;
    config.inner_init = init;
    config.inner_backend = "adaptive";
    config.niters = 2;
    config.seed = 19;
    ClusteringContext context(root, config);
    auto cached = context.cluster(local, rows, 2, 1);
    auto direct = cluster_point_clouds(local, 2, 1, config);
    EXPECT_EQ(cached.assignments, direct.assignments);
    ASSERT_EQ(cached.centers.num_bytes(), direct.centers.num_bytes());
    for (size_t i = 0; i < cached.centers.offsets[cached.centers.n]; ++i)
      EXPECT_NEAR(cached.centers.values[i], direct.centers.values[i], 1e-6f);
  }
}

class RecordedLloydsBackend : public lloyds::FloatLloydsBackend<true> {
  using Base = lloyds::FloatLloydsBackend<true>;

 public:
  using Base::Base;
  std::vector<std::vector<float>> assigned_centers;
  std::vector<std::vector<uint32_t>> update_labels;
  void assign(const lloyds::CenterSet& centers, parlay::sequence<uint32_t>& ids) {
    assigned_centers.emplace_back(centers.data(),
                                  centers.data() + centers.size() * centers.get_dims());
    Base::assign(centers, ids);
  }
  lloyds::CenterSet compute_means(parlay::sequence<uint32_t>& ids, uint32_t k) {
    update_labels.emplace_back(ids.begin(), ids.end());
    return Base::compute_means(ids, k);
  }
  lloyds::CenterSet compute_means_weighted(parlay::sequence<uint32_t>& ids,
                                           const parlay::sequence<float>& weights, uint32_t k) {
    update_labels.emplace_back(ids.begin(), ids.end());
    return Base::compute_means_weighted(ids, weights, k);
  }
};

TEST(MVClustering, QuietLloydsOmitsOnlyTheUnusedFinalAssignment) {
  const parlay::sequence<parlay::sequence<float>> points{{-4, 1}, {-2, 3}, {0, 1}, {2, -1},
                                                         {6, 2},  {7, -3}, {9, 0}};
  const parlay::sequence<float> weights{1, 2, 1, 3, 1, 2, 1};
  constexpr uint32_t k = 3;
  constexpr size_t rounds = 4;
  for (bool cached : {false, true})
    for (bool weighted : {false, true}) {
      const auto* w = weighted ? &weights : nullptr;
      auto seed = lloyds::initialize_inner<true>(points, k, "kmeans++", 5, w, cached);
      RecordedLloydsBackend optimized(points, 2, true), full_schedule(points, 2, true);
      auto actual = lloyds::lloyds_kmeans(optimized, k, "PrefixDoubling", rounds, false, w, &seed);
      auto expected = seed.centers;
      parlay::sequence<uint32_t> ids = seed.assignments;
      if (ids.empty()) full_schedule.assign(expected, ids);
      // Reference complete schedule: every update followed by assignment,
      // including the final one whose labels no subsequent update consumes.
      for (size_t i = 0; i < rounds; ++i) {
        expected = weighted ? full_schedule.compute_means_weighted(ids, weights, k)
                            : full_schedule.compute_means(ids, k);
        full_schedule.assign(expected, ids);
      }
      EXPECT_EQ(optimized.update_labels, full_schedule.update_labels);
      ASSERT_EQ(optimized.assigned_centers.size() + 1, full_schedule.assigned_centers.size());
      for (size_t i = 0; i < optimized.assigned_centers.size(); ++i)
        EXPECT_EQ(optimized.assigned_centers[i], full_schedule.assigned_centers[i]);
      for (size_t c = 0; c < k; ++c)
        for (size_t j = 0; j < 2; ++j)
          EXPECT_FLOAT_EQ(actual[c][j], expected[c][j]);
    }
}

TEST(MVClustering, RejectsNormsThatCanOverflowLongCloudScores) {
  auto zero_queries = own_point_clouds(std::vector<float>(32, 0), {0, 32}, 1);
  auto large_center = own_point_clouds({4e18f}, {0, 1}, 1);
  // Bound both operands uniformly at the public boundary, including zero
  // queries against a center whose norm exceeds the long-cloud scoring cap.
  for (auto codec : codecs)
    EXPECT_THROW(assign_point_clouds(zero_queries, large_center, codec), std::invalid_argument);
  EXPECT_THROW(cluster_point_clouds(large_center, 1, 1), std::invalid_argument);
}

TEST(MVClustering, IterationTimersDoNotCountAssignmentTwice) {
  auto input = Rays({1, 1});
  auto points = PointCloudSet<ChamferIP_Point>::borrow(
      input.n, input.d, input.values, input.offsets);
  MVClusteringConfig config;
  config.niters = 1;
  parlay::execute_with_scheduler(1, [&] {
    MVClustering trainer(input.d, 1, 1, config);
    trainer.assignment_function = [](const auto& points, const auto&, auto& ids) {
      // Make assignment dominate setup so double counting is observable even
      // on this tiny fixture. Representative updates retain the real trainer.
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      ids = parlay::sequence<uint32_t>(points.size(), 0);
    };
    const auto start = std::chrono::steady_clock::now();
    trainer.train(points);
    const double wall = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    ASSERT_EQ(trainer._iteration_stats.size(), 2u);
    const auto& iteration = trainer._iteration_stats[1];
    EXPECT_GE(iteration.assignment_time, .015);
    EXPECT_GE(iteration.centroid_update_time, 0);
    // The two measured phases are disjoint subintervals of train().
    EXPECT_LE(iteration.assignment_time + iteration.centroid_update_time, wall + .001);
  });
}

TEST(MVClustering, RejectsInvalidBoundaryInputs) {
  auto points = Rays({1, 2});
  EXPECT_THROW(cluster_point_clouds(points, 0, 1), std::invalid_argument);
  EXPECT_THROW(cluster_point_clouds(points, 3, 1), std::invalid_argument);
  MVClusteringConfig config;
  config.inner_init = "invalid";
  EXPECT_THROW(cluster_point_clouds(points, 1, 1, config), std::invalid_argument);
  config.inner_init = "default";
  config.inner_backend = "invalid";
  EXPECT_THROW(cluster_point_clouds(points, 1, 1, config), std::invalid_argument);
  config.inner_backend = "legacy";
  config.assignment_quantizer = static_cast<ClusteringQuantizer>(99);
  EXPECT_THROW(cluster_point_clouds(points, 1, 1, config), std::invalid_argument);
  auto invalid = own_point_clouds({std::numeric_limits<float>::quiet_NaN()}, {0, 1}, 1);
  EXPECT_THROW(cluster_point_clouds(invalid, 1, 1), std::invalid_argument);
  auto wrong_dimension = Rays({1}, 8);
  EXPECT_THROW(assign_point_clouds(points, wrong_dimension, ClusteringQuantizer::None),
               std::invalid_argument);
}

}  // namespace
}  // namespace mvsic
