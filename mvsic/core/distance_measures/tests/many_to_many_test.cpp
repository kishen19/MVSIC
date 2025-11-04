#include "gtest/gtest.h"
#include "mvsic/core/distance_measures/many_to_many.h"
#include "mvsic/core/types/point_cloud_set.h"
#include "mvsic/core/types/chamfer_ip_point.h"
#include "mvsic/core/types/chamfer_l2_point.h"
#include <random>
#include <algorithm>
#include <limits>

namespace mvsic {

// Brute-force Chamfer distance implementations
float brute_force_chamfer_ip_distance(const float* a, uint32_t n_a, const float* b, uint32_t n_b, uint32_t dim) {
    if (n_a == 0 || n_b == 0) return std::numeric_limits<float>::max();
    float sum_min_dists = 0.0f;
    for (uint32_t i = 0; i < n_a; ++i) {
        float min_dist = std::numeric_limits<float>::max();
        for (uint32_t j = 0; j < n_b; ++j) {
            float current_ip = 0.0f;
            for (uint32_t l = 0; l < dim; ++l) {
                current_ip += a[i * dim + l] * b[j * dim + l];
            }
            min_dist = std::min(min_dist, -current_ip); // Chamfer IP is -max_ip
        }
        sum_min_dists += min_dist;
    }
    return sum_min_dists / n_a;
}

float brute_force_chamfer_l2_distance(const float* a, uint32_t n_a, const float* b, uint32_t n_b, uint32_t dim) {
    if (n_a == 0 || n_b == 0) return std::numeric_limits<float>::max();
    float sum_min_dists = 0.0f;
    for (uint32_t i = 0; i < n_a; ++i) {
        float min_dist = std::numeric_limits<float>::max();
        for (uint32_t j = 0; j < n_b; ++j) {
            float current_l2_sq = 0.0f;
            for (uint32_t l = 0; l < dim; ++l) {
                float diff = a[i * dim + l] - b[j * dim + l];
                current_l2_sq += diff * diff;
            }
            min_dist = std::min(min_dist, current_l2_sq); // Chamfer L2 is min_l2_sq
        }
        sum_min_dists += min_dist;
    }
    return sum_min_dists / n_a;
}

// Fixture for the tests
class ManyToManyTest : public ::testing::Test {
protected:
    using PointCloudSetIP = PointCloudSet<ChamferIP_Point>;
    using PointCloudSetL2 = PointCloudSet<ChamferL2_Point>;

    // Helper to create a PointCloudSet from raw data
    template<typename ChPoint>
    PointCloudSet<ChPoint> createPointCloudSet(
        const std::vector<std::vector<std::vector<float>>>& data, uint32_t dim) {
        
        std::vector<float> all_values;
        std::vector<size_t> offsets;
        offsets.push_back(0);
        uint32_t current_offset = 0;

        for (const auto& pc : data) {
            for (const auto& vec : pc) {
                for (float val : vec) {
                    all_values.push_back(val);
                }
            }
            current_offset += pc.size() * dim;
            offsets.push_back(current_offset);
        }
        
        return PointCloudSet<ChPoint>(data.size(), dim, all_values.data(), offsets.data(), nullptr);
    }

    // Brute-force implementations for ManyToMany functions
    template<typename ChPoint>
    std::vector<float> brute_force_all_pairs(const PointCloudSet<ChPoint>& A, const PointCloudSet<ChPoint>& B) {
        std::vector<float> results(A.size() * B.size());
        uint32_t dim = A.get_dims();
        for (size_t i = 0; i < A.size(); ++i) {
            for (size_t j = 0; j < B.size(); ++j) {
                if constexpr (std::is_same_v<ChPoint, ChamferIP_Point>) {
                    results[i * B.size() + j] = brute_force_chamfer_ip_distance(
                        A.data(i), A.get_size(i), B.data(j), B.get_size(j), dim);
                } else {
                    results[i * B.size() + j] = brute_force_chamfer_l2_distance(
                        A.data(i), A.get_size(i), B.data(j), B.get_size(j), dim);
                }
            }
        }
        return results;
    }

    template<typename ChPoint>
    std::vector<std::pair<uint32_t, float>> brute_force_top(const PointCloudSet<ChPoint>& A, const PointCloudSet<ChPoint>& B) {
        std::vector<std::pair<uint32_t, float>> results(A.size());
        uint32_t dim = A.get_dims();
        for (size_t i = 0; i < A.size(); ++i) {
            float min_dist = std::numeric_limits<float>::max();
            uint32_t best_id = 0;
            for (size_t j = 0; j < B.size(); ++j) {
                float dist;
                if constexpr (std::is_same_v<ChPoint, ChamferIP_Point>) {
                    dist = brute_force_chamfer_ip_distance(
                        A.data(i), A.get_size(i), B.data(j), B.get_size(j), dim);
                } else {
                    dist = brute_force_chamfer_l2_distance(
                        A.data(i), A.get_size(i), B.data(j), B.get_size(j), dim);
                }
                if (dist < min_dist) {
                    min_dist = dist;
                    best_id = j;
                }
            }
            results[i] = {best_id, min_dist};
        }
        return results;
    }

    template<typename ChPoint>
    std::vector<std::pair<uint32_t, float>> brute_force_top_k(const PointCloudSet<ChPoint>& A, const PointCloudSet<ChPoint>& B, uint32_t k) {
        std::vector<std::pair<uint32_t, float>> results(A.size() * k);
        uint32_t dim = A.get_dims();
        for (size_t i = 0; i < A.size(); ++i) {
            std::vector<std::pair<float, uint32_t>> current_dists;
            for (size_t j = 0; j < B.size(); ++j) {
                float dist;
                if constexpr (std::is_same_v<ChPoint, ChamferIP_Point>) {
                    dist = brute_force_chamfer_ip_distance(
                        A.data(i), A.get_size(i), B.data(j), B.get_size(j), dim);
                } else {
                    dist = brute_force_chamfer_l2_distance(
                        A.data(i), A.get_size(i), B.data(j), B.get_size(j), dim);
                }
                current_dists.push_back({dist, j});
            }
            std::sort(current_dists.begin(), current_dists.end());
            for (uint32_t ki = 0; ki < k; ++ki) {
                if (ki < current_dists.size()) {
                    results[i * k + ki] = {current_dists[ki].second, current_dists[ki].first};
                } else {
                    results[i * k + ki] = {0, std::numeric_limits<float>::max()};
                }
            }
        }
        return results;
    }

    void SetUp() override {
        const uint32_t DIM = 128;
        const uint32_t NUM_PCS_A = 5;
        const uint32_t NUM_PCS_B = 10;
        const uint32_t MAX_VECTORS_PER_PC = 5;

        std::mt19937 gen(0); // Fixed seed for reproducibility
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::uniform_int_distribution<uint32_t> num_vectors_dist(1, MAX_VECTORS_PER_PC);

        auto generate_random_data = [&](uint32_t num_pcs) {
            std::vector<std::vector<std::vector<float>>> data;
            for (uint32_t i = 0; i < num_pcs; ++i) {
                uint32_t num_vectors = num_vectors_dist(gen);
                std::vector<std::vector<float>> pc;
                for (uint32_t v = 0; v < num_vectors; ++v) {
                    std::vector<float> vec(DIM);
                    for (uint32_t d = 0; d < DIM; ++d) {
                        vec[d] = dist(gen);
                    }
                    pc.push_back(vec);
                }
                data.push_back(pc);
            }
            return data;
        };

        pcs_A_data = generate_random_data(NUM_PCS_A);
        pcs_B_data = generate_random_data(NUM_PCS_B);

        pcs_A_ip = createPointCloudSet<ChamferIP_Point>(pcs_A_data, DIM);
        pcs_B_ip = createPointCloudSet<ChamferIP_Point>(pcs_B_data, DIM);
        pcs_A_l2 = createPointCloudSet<ChamferL2_Point>(pcs_A_data, DIM);
        pcs_B_l2 = createPointCloudSet<ChamferL2_Point>(pcs_B_data, DIM);
    }

    std::vector<std::vector<std::vector<float>>> pcs_A_data;
    std::vector<std::vector<std::vector<float>>> pcs_B_data;
    PointCloudSetIP pcs_A_ip;
    PointCloudSetIP pcs_B_ip;
    PointCloudSetL2 pcs_A_l2;
    PointCloudSetL2 pcs_B_l2;
};

TEST_F(ManyToManyTest, AllPairsIP_Random) {
    auto optimized_results = ManyToMany<PointCloudSetIP>::AllPairs(pcs_A_ip, pcs_B_ip);
    auto brute_force_results = brute_force_all_pairs<ChamferIP_Point>(pcs_A_ip, pcs_B_ip);

    ASSERT_EQ(optimized_results.size(), brute_force_results.size());
    for (size_t i = 0; i < optimized_results.size(); ++i) {
        EXPECT_FLOAT_EQ(optimized_results[i], brute_force_results[i]);
    }
}

TEST_F(ManyToManyTest, AllPairsL2_Random) {
    auto optimized_results = ManyToMany<PointCloudSetL2>::AllPairs(pcs_A_l2, pcs_B_l2);
    auto brute_force_results = brute_force_all_pairs<ChamferL2_Point>(pcs_A_l2, pcs_B_l2);

    ASSERT_EQ(optimized_results.size(), brute_force_results.size());
    for (size_t i = 0; i < optimized_results.size(); ++i) {
        EXPECT_FLOAT_EQ(optimized_results[i], brute_force_results[i]);
    }
}

TEST_F(ManyToManyTest, TopIP_Random) {
    auto optimized_results = ManyToMany<PointCloudSetIP>::Top(pcs_A_ip, pcs_B_ip);
    auto brute_force_results = brute_force_top<ChamferIP_Point>(pcs_A_ip, pcs_B_ip);

    ASSERT_EQ(optimized_results.size(), brute_force_results.size());
    for (size_t i = 0; i < optimized_results.size(); ++i) {
        EXPECT_EQ(optimized_results[i].first, brute_force_results[i].first);
        EXPECT_FLOAT_EQ(optimized_results[i].second, brute_force_results[i].second);
    }
}

TEST_F(ManyToManyTest, TopL2_Random) {
    auto optimized_results = ManyToMany<PointCloudSetL2>::Top(pcs_A_l2, pcs_B_l2);
    auto brute_force_results = brute_force_top<ChamferL2_Point>(pcs_A_l2, pcs_B_l2);

    ASSERT_EQ(optimized_results.size(), brute_force_results.size());
    for (size_t i = 0; i < optimized_results.size(); ++i) {
        EXPECT_EQ(optimized_results[i].first, brute_force_results[i].first);
        EXPECT_FLOAT_EQ(optimized_results[i].second, brute_force_results[i].second);
    }
}

TEST_F(ManyToManyTest, TopKIP_Random) {
    const uint32_t k = 3;
    auto optimized_results = ManyToMany<PointCloudSetIP>::TopK(pcs_A_ip, pcs_B_ip, k);
    auto brute_force_results = brute_force_top_k<ChamferIP_Point>(pcs_A_ip, pcs_B_ip, k);

    ASSERT_EQ(optimized_results.size(), brute_force_results.size());
    for (size_t i = 0; i < optimized_results.size(); ++i) {
        EXPECT_EQ(optimized_results[i].first, brute_force_results[i].first);
        EXPECT_FLOAT_EQ(optimized_results[i].second, brute_force_results[i].second);
    }
}

TEST_F(ManyToManyTest, TopKL2_Random) {
    const uint32_t k = 3;
    auto optimized_results = ManyToMany<PointCloudSetL2>::TopK(pcs_A_l2, pcs_B_l2, k);
    auto brute_force_results = brute_force_top_k<ChamferL2_Point>(pcs_A_l2, pcs_B_l2, k);

    ASSERT_EQ(optimized_results.size(), brute_force_results.size());
    for (size_t i = 0; i < optimized_results.size(); ++i) {
        EXPECT_EQ(optimized_results[i].first, brute_force_results[i].first);
        EXPECT_FLOAT_EQ(optimized_results[i].second, brute_force_results[i].second);
    }
}

} // namespace mvsic
