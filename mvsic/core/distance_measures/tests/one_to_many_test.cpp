#include "gtest/gtest.h"
#include "mvsic/core/distance_measures/one_to_many.h"
#include "mvsic/core/types/point_cloud_set.h"
#include "mvsic/core/types/chamfer_ip_point.h"
#include "mvsic/core/types/chamfer_l2_point.h"
#include <random>
#include <algorithm>
#include <limits>
#include <vector>
#include <chrono>

namespace mvsic {

// Brute-force Chamfer distance implementations
float brute_force_chamfer_ip_distance(const float* a, uint32_t n_a, const float* b, uint32_t n_b,
                                      uint32_t dim) {
  if (n_a == 0 || n_b == 0) return std::numeric_limits<float>::infinity();
  float sum_min_dists = 0.0f;
  for (uint32_t i = 0; i < n_a; ++i) {
    float min_dist = std::numeric_limits<float>::max();
    for (uint32_t j = 0; j < n_b; ++j) {
      float current_ip = 0.0f;
      for (uint32_t l = 0; l < dim; ++l) {
        current_ip += a[i * dim + l] * b[j * dim + l];
      }
      min_dist = std::min(min_dist, -current_ip);  // Chamfer IP is -max_ip
    }
    sum_min_dists += min_dist;
  }
  return sum_min_dists / n_a;
}

float brute_force_chamfer_l2_distance(const float* a, uint32_t n_a, const float* b, uint32_t n_b,
                                      uint32_t dim) {
  if (n_a == 0 || n_b == 0) return std::numeric_limits<float>::infinity();
  float sum_min_dists = 0.0f;
  for (uint32_t i = 0; i < n_a; ++i) {
    float min_dist = std::numeric_limits<float>::max();
    for (uint32_t j = 0; j < n_b; ++j) {
      float current_l2_sq = 0.0f;
      for (uint32_t l = 0; l < dim; ++l) {
        float diff = a[i * dim + l] - b[j * dim + l];
        current_l2_sq += diff * diff;
      }
      min_dist = std::min(min_dist, current_l2_sq);  // Chamfer L2 is min_l2_sq
    }
    sum_min_dists += min_dist;
  }
  return sum_min_dists / n_a;
}

// Fixture for the tests, now parameterized by dimension
class OneToManyTest : public ::testing::TestWithParam<uint32_t> {
 protected:
  using PointCloudSetIP = PointCloudSet<ChamferIP_Point>;
  using PointCloudSetL2 = PointCloudSet<ChamferL2_Point>;
  using ChPointIP = ChamferIP_Point;
  using ChPointL2 = ChamferL2_Point;

  // Helper to create a PointCloudSet from raw data
  template<typename ChPoint>
  PointCloudSet<ChPoint> createPointCloudSet(
      const std::vector<std::vector<std::vector<float>>>& data, uint32_t dim) {

    std::vector<float> all_values;
    std::vector<size_t> offsets;
    offsets.push_back(0);
    size_t current_offset = 0;

    for (const auto& pc : data) {
      for (const auto& vec : pc) {
        all_values.insert(all_values.end(), vec.begin(), vec.end());
      }
      current_offset += pc.size() * dim;
      offsets.push_back(current_offset);
    }

    return PointCloudSet<ChPoint>(data.size(), dim, all_values.data(), offsets.data(), nullptr);
  }

  // Brute-force implementations for OneToMany functions
  template<typename ChPoint>
  std::vector<float> brute_force_all_distances(const ChPoint& q, const PointCloudSet<ChPoint>& B) {
    std::vector<float> results(B.size());
    uint32_t dim = B.get_dims();
    for (size_t j = 0; j < B.size(); ++j) {
      if constexpr (std::is_same_v<ChPoint, ChamferIP_Point>) {
        results[j] =
            brute_force_chamfer_ip_distance(q.data(), q.size(), B.data(j), B.get_size(j), dim);
      } else {
        results[j] =
            brute_force_chamfer_l2_distance(q.data(), q.size(), B.data(j), B.get_size(j), dim);
      }
    }
    return results;
  }

  template<typename ChPoint>
  std::pair<uint32_t, float> brute_force_top(const ChPoint& q, const PointCloudSet<ChPoint>& B) {
    float min_dist = std::numeric_limits<float>::max();
    uint32_t best_id = 0;
    uint32_t dim = B.get_dims();
    for (size_t j = 0; j < B.size(); ++j) {
      float dist;
      if constexpr (std::is_same_v<ChPoint, ChamferIP_Point>) {
        dist = brute_force_chamfer_ip_distance(q.data(), q.size(), B.data(j), B.get_size(j), dim);
      } else {
        dist = brute_force_chamfer_l2_distance(q.data(), q.size(), B.data(j), B.get_size(j), dim);
      }
      if (dist < min_dist) {
        min_dist = dist;
        best_id = j;
      }
    }
    return {best_id, min_dist};
  }

  template<typename ChPoint>
  std::vector<std::pair<uint32_t, float>> brute_force_top_k(const ChPoint& q,
                                                            const PointCloudSet<ChPoint>& B,
                                                            uint32_t k) {
    std::vector<std::pair<uint32_t, float>> results(k);
    uint32_t dim = B.get_dims();
    std::vector<std::pair<float, uint32_t>> current_dists;
    for (size_t j = 0; j < B.size(); ++j) {
      float dist;
      if constexpr (std::is_same_v<ChPoint, ChamferIP_Point>) {
        dist = brute_force_chamfer_ip_distance(q.data(), q.size(), B.data(j), B.get_size(j), dim);
      } else {
        dist = brute_force_chamfer_l2_distance(q.data(), q.size(), B.data(j), B.get_size(j), dim);
      }
      current_dists.push_back({dist, (uint32_t)j});
    }
    std::sort(current_dists.begin(), current_dists.end());
    for (uint32_t ki = 0; ki < k; ++ki) {
      if (ki < current_dists.size()) {
        results[ki] = {current_dists[ki].second, current_dists[ki].first};
      } else {
        results[ki] = {0, std::numeric_limits<float>::max()};
      }
    }
    return results;
  }

  void SetUp() override {
    const uint32_t DIM = GetParam();
    const uint32_t NUM_PCS_B = 10;
    const uint32_t MAX_VECTORS_PER_PC = 5;

    std::mt19937 gen(0);  // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::uniform_int_distribution<uint32_t> num_vectors_dist(1, MAX_VECTORS_PER_PC);

    auto generate_random_data = [&](uint32_t num_pcs, uint32_t dim) {
      std::vector<std::vector<std::vector<float>>> data;
      for (uint32_t i = 0; i < num_pcs; ++i) {
        uint32_t num_vectors = num_vectors_dist(gen);
        std::vector<std::vector<float>> pc;
        for (uint32_t v = 0; v < num_vectors; ++v) {
          std::vector<float> vec(dim);
          for (uint32_t d = 0; d < dim; ++d) {
            vec[d] = dist(gen);
          }
          pc.push_back(vec);
        }
        data.push_back(pc);
      }
      return data;
    };

    auto q_data_tmp = generate_random_data(1, DIM);
    q_data = q_data_tmp[0];
    pcs_B_data = generate_random_data(NUM_PCS_B, DIM);

    for (const auto& vec : q_data) {
      q_flat_data.insert(q_flat_data.end(), vec.begin(), vec.end());
    }

    q_ip = ChPointIP(q_data.size(), DIM, q_flat_data.data(), 0);
    q_l2 = ChPointL2(q_data.size(), DIM, q_flat_data.data(), 0);

    pcs_B_ip = createPointCloudSet<ChamferIP_Point>(pcs_B_data, DIM);
    pcs_B_l2 = createPointCloudSet<ChamferL2_Point>(pcs_B_data, DIM);
  }

  std::vector<std::vector<float>> q_data;
  std::vector<float> q_flat_data;
  std::vector<std::vector<std::vector<float>>> pcs_B_data;
  ChPointIP q_ip;
  ChPointL2 q_l2;
  PointCloudSetIP pcs_B_ip;
  PointCloudSetL2 pcs_B_l2;
};

TEST_P(OneToManyTest, AllDistancesIP_Random) {
  auto optimized_results = OneToMany<ChPointIP, PointCloudSetIP>::AllDistances(q_ip, pcs_B_ip);
  auto brute_force_results = brute_force_all_distances<ChamferIP_Point>(q_ip, pcs_B_ip);

  ASSERT_EQ(optimized_results.size(), brute_force_results.size());
  for (size_t i = 0; i < optimized_results.size(); ++i) {
    EXPECT_NEAR(optimized_results[i], brute_force_results[i], 1e-5);
  }
}

TEST_P(OneToManyTest, AllDistancesL2_Random) {
  auto optimized_results = OneToMany<ChPointL2, PointCloudSetL2>::AllDistances(q_l2, pcs_B_l2);
  auto brute_force_results = brute_force_all_distances<ChamferL2_Point>(q_l2, pcs_B_l2);

  ASSERT_EQ(optimized_results.size(), brute_force_results.size());
  for (size_t i = 0; i < optimized_results.size(); ++i) {
    EXPECT_FLOAT_EQ(optimized_results[i], brute_force_results[i]);
  }
}

TEST_P(OneToManyTest, TopIP_Random) {
  auto optimized_result = OneToMany<ChPointIP, PointCloudSetIP>::Top(q_ip, pcs_B_ip);
  auto brute_force_result = brute_force_top<ChamferIP_Point>(q_ip, pcs_B_ip);

  EXPECT_EQ(optimized_result.first, brute_force_result.first);
  EXPECT_NEAR(optimized_result.second, brute_force_result.second, 1e-5);
}

TEST_P(OneToManyTest, TopL2_Random) {
  auto optimized_result = OneToMany<ChPointL2, PointCloudSetL2>::Top(q_l2, pcs_B_l2);
  auto brute_force_result = brute_force_top<ChamferL2_Point>(q_l2, pcs_B_l2);

  EXPECT_EQ(optimized_result.first, brute_force_result.first);
  EXPECT_FLOAT_EQ(optimized_result.second, brute_force_result.second);
}

TEST_P(OneToManyTest, TopKIP_Random) {
  const uint32_t k = 3;
  auto optimized_results = OneToMany<ChPointIP, PointCloudSetIP>::TopK(q_ip, pcs_B_ip, k);
  auto brute_force_results = brute_force_top_k<ChamferIP_Point>(q_ip, pcs_B_ip, k);

  ASSERT_EQ(optimized_results.size(), brute_force_results.size());
  for (size_t i = 0; i < optimized_results.size(); ++i) {
    EXPECT_EQ(optimized_results[i].first, brute_force_results[i].first);
    EXPECT_NEAR(optimized_results[i].second, brute_force_results[i].second, 1e-4);
  }
}

TEST_P(OneToManyTest, TopKL2_Random) {
  const uint32_t k = 3;
  auto optimized_results = OneToMany<ChPointL2, PointCloudSetL2>::TopK(q_l2, pcs_B_l2, k);
  auto brute_force_results = brute_force_top_k<ChamferL2_Point>(q_l2, pcs_B_l2, k);

  ASSERT_EQ(optimized_results.size(), brute_force_results.size());
  for (size_t i = 0; i < optimized_results.size(); ++i) {
    EXPECT_EQ(optimized_results[i].first, brute_force_results[i].first);
    EXPECT_FLOAT_EQ(optimized_results[i].second, brute_force_results[i].second);
  }
}

TEST_P(OneToManyTest, TopK_K_greater_than_B_size) {
  const uint32_t k = pcs_B_ip.size() + 5;
  auto optimized_results = OneToMany<ChPointIP, PointCloudSetIP>::TopK(q_ip, pcs_B_ip, k);
  auto brute_force_results = brute_force_top_k<ChamferIP_Point>(q_ip, pcs_B_ip, k);

  ASSERT_EQ(optimized_results.size(), brute_force_results.size());
  for (size_t i = 0; i < k; ++i) {
    if (i < pcs_B_ip.size()) {
      EXPECT_EQ(optimized_results[i].first, brute_force_results[i].first);
      EXPECT_NEAR(optimized_results[i].second, brute_force_results[i].second, 1e-4);
    } else {
      EXPECT_EQ(optimized_results[i].second, std::numeric_limits<float>::max());
    }
  }
}

INSTANTIATE_TEST_SUITE_P(Dimensions, OneToManyTest, ::testing::Values(15, 16, 128, 255));

TEST_P(OneToManyTest, AllDistancesL2_Comparison) {
  auto start_optimized = std::chrono::high_resolution_clock::now();
  auto optimized_results = OneToMany<ChPointL2, PointCloudSetL2>::AllDistances(q_l2, pcs_B_l2);
  auto end_optimized = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> optimized_duration = end_optimized - start_optimized;
  std::cout << "Optimized L2 duration: " << optimized_duration.count() << "s" << std::endl;

  auto start_unoptimized = std::chrono::high_resolution_clock::now();
  auto unoptimized_results_pair = pcs_B_l2.distances(q_l2);
  auto end_unoptimized = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> unoptimized_duration = end_unoptimized - start_unoptimized;
  std::cout << "Unoptimized L2 duration: " << unoptimized_duration.count() << "s" << std::endl;

  auto& unoptimized_results = unoptimized_results_pair.first;

  ASSERT_EQ(optimized_results.size(), unoptimized_results.size());
  for (size_t i = 0; i < optimized_results.size(); ++i) {
    EXPECT_FLOAT_EQ(optimized_results[i], unoptimized_results[i].second);
  }
}

TEST_P(OneToManyTest, AllDistancesIP_Comparison) {
  auto start_optimized = std::chrono::high_resolution_clock::now();
  auto optimized_results = OneToMany<ChPointIP, PointCloudSetIP>::AllDistances(q_ip, pcs_B_ip);
  auto end_optimized = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> optimized_duration = end_optimized - start_optimized;
  std::cout << "Optimized IP duration: " << optimized_duration.count() << "s" << std::endl;

  auto start_unoptimized = std::chrono::high_resolution_clock::now();
  auto unoptimized_results_pair = pcs_B_ip.distances(q_ip);
  auto end_unoptimized = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> unoptimized_duration = end_unoptimized - start_unoptimized;
  std::cout << "Unoptimized IP duration: " << unoptimized_duration.count() << "s" << std::endl;

  auto& unoptimized_results = unoptimized_results_pair.first;

  ASSERT_EQ(optimized_results.size(), unoptimized_results.size());
  for (size_t i = 0; i < optimized_results.size(); ++i) {
    EXPECT_NEAR(optimized_results[i], unoptimized_results[i].second, 1e-5);
  }
}

}  // namespace mvsic
