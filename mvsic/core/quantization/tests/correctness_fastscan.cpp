#include <gtest/gtest.h>
#include <random>
#include <vector>

#include "mvsic/core/quantization/fastscan.h"

using namespace mvsic::fastscan;

// Simple point range wrapper for testing
struct MockPointRange {
  std::vector<float> data_;
  size_t n_points_;
  size_t dim_;

  MockPointRange(size_t n, size_t d) : data_(n * d, 0.0f), n_points_(n), dim_(d) {}

  size_t size() const { return n_points_; }
  size_t get_dims() const { return dim_; }
  const float* location(size_t i) const { return data_.data() + i * dim_; }
  float* data() { return data_.data(); }
  const float* data() const { return data_.data(); }
};

// Slow, exact evaluation of FastScan distance to verify SIMD kernels and packing logic
template<bool Metric>
float slow_fastscan_distance(const Model<Metric>& quantizer, 
                             const float* query, 
                             const float* point) {
    // 1. Find exact quantized codes for point
    std::vector<uint8_t> codes(quantizer.num_blocks);
    for (uint32_t b = 0; b < quantizer.num_blocks; ++b) {
        Eigen::Map<const Eigen::VectorXf> p_sub(point + b * quantizer.dim_per_block, quantizer.dim_per_block);
        Eigen::VectorXf dots = quantizer.codebooks[b] * p_sub;
        float min_val = std::numeric_limits<float>::max();
        uint8_t best = 0;
        for (uint32_t i = 0; i < Model<Metric>::K; ++i) {
            float val = Metric ? (quantizer.codebook_norms[b][i] - 2.0f * dots[i]) : -dots[i];
            if (val < min_val) {
                min_val = val;
                best = i;
            }
        }
        codes[b] = best;
    }

    // 2. Compute float LUT for query
    std::vector<std::vector<float>> float_lut(quantizer.num_blocks, std::vector<float>(Model<Metric>::K));
    float g_min = std::numeric_limits<float>::max();
    float g_max = std::numeric_limits<float>::lowest();
    for (uint32_t b = 0; b < quantizer.num_blocks; ++b) {
        Eigen::Map<const Eigen::VectorXf> q_sub(query + b * quantizer.dim_per_block, quantizer.dim_per_block);
        Eigen::VectorXf dots = quantizer.codebooks[b] * q_sub;
        float q_sq = Metric ? q_sub.squaredNorm() : 0.0f;
        for (uint32_t i = 0; i < Model<Metric>::K; ++i) {
            float val = Metric ? (quantizer.codebook_norms[b][i] - 2.0f * dots[i] + q_sq) : -dots[i];
            float_lut[b][i] = val;
            g_min = std::min(g_min, val);
            g_max = std::max(g_max, val);
        }
    }

    // 3. Quantize the LUT into 8-bit integers
    float scale = std::max(1e-6f, (g_max - g_min) / 255.0f);
    uint16_t acc = 0;
    for (uint32_t b = 0; b < quantizer.num_blocks; ++b) {
        acc += static_cast<uint8_t>((float_lut[b][codes[b]] - g_min) / scale);
    }

    return g_min * static_cast<float>(quantizer.num_blocks) + static_cast<float>(acc) * scale;
}

template<bool Metric>
void RunFastScanTests(size_t n_points, size_t dim_per_block, size_t num_blocks) {
    size_t dim = num_blocks * dim_per_block;
    MockPointRange data(n_points, dim);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (size_t i = 0; i < n_points * dim; ++i) {
        data.data_[i] = dist(rng);
    }

    Model<Metric> fsq;
    fsq.num_blocks = num_blocks;
    fsq.dim = dim;
    fsq.dim_per_block = dim_per_block;
    fsq.codebooks.resize(num_blocks);
    fsq.codebook_norms.resize(num_blocks);
    
    for (size_t b = 0; b < num_blocks; ++b) {
        fsq.codebooks[b] = Eigen::MatrixXf(Model<Metric>::K, dim_per_block);
        for (size_t i = 0; i < Model<Metric>::K; ++i) {
            for (size_t j = 0; j < dim_per_block; ++j) {
                fsq.codebooks[b](i, j) = dist(rng);
            }
        }
        fsq.codebook_norms[b] = fsq.codebooks[b].rowwise().squaredNorm();
    }

    auto encoded = fsq.encode(data);
    
    // Check single point distances
    for (size_t i = 0; i < n_points; ++i) {
        MockPointRange query(1, dim);
        for (size_t d = 0; d < dim; ++d) {
            query.data_[d] = dist(rng);
        }

        Quantized_Query<Metric> qq = fsq.quantize_query(query.location(0));
        
        float expected = slow_fastscan_distance(fsq, query.location(0), data.location(i));
        float actual = qq.distance(encoded[i]);
        
        EXPECT_NEAR(expected, actual, 1e-4);
    }

    // Check distances_all (which uses SIMD chunk kernels)
    MockPointRange query(1, dim);
    for (size_t d = 0; d < dim; ++d) {
        query.data_[d] = dist(rng);
    }
    Quantized_Query<Metric> qq = fsq.quantize_query(query.location(0));
    
    std::vector<float> all_dists(n_points, -1.0f);
    qq.distances_all(encoded, all_dists.data());
    
    for (size_t i = 0; i < n_points; ++i) {
        float expected = slow_fastscan_distance(fsq, query.location(0), data.location(i));
        EXPECT_NEAR(expected, all_dists[i], 1e-4);
    }
}

template<bool Metric>
void RunFastScanBatchQueryTests(size_t n_queries, size_t dim_per_block, size_t num_blocks) {
    size_t dim = num_blocks * dim_per_block;
    MockPointRange queries(n_queries, dim);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (size_t i = 0; i < n_queries * dim; ++i) {
        queries.data_[i] = dist(rng);
    }

    Model<Metric> fsq;
    fsq.num_blocks = num_blocks;
    fsq.dim = dim;
    fsq.dim_per_block = dim_per_block;
    fsq.codebooks.resize(num_blocks);
    fsq.codebook_norms.resize(num_blocks);
    
    for (size_t b = 0; b < num_blocks; ++b) {
        fsq.codebooks[b] = Eigen::MatrixXf(Model<Metric>::K, dim_per_block);
        for (size_t i = 0; i < Model<Metric>::K; ++i) {
            for (size_t j = 0; j < dim_per_block; ++j) {
                fsq.codebooks[b](i, j) = dist(rng);
            }
        }
        fsq.codebook_norms[b] = fsq.codebooks[b].rowwise().squaredNorm();
    }

    parlay::sequence<Quantized_Query<Metric>> luts;
    fsq.quantize_query_batch(queries, luts);

    ASSERT_EQ(luts.size(), n_queries);
    for (size_t i = 0; i < n_queries; ++i) {
        Quantized_Query<Metric> qq_single = fsq.quantize_query(queries.location(i));
        
        EXPECT_NEAR(qq_single.min_dist, luts[i].min_dist, 1e-5);
        EXPECT_NEAR(qq_single.scale, luts[i].scale, 1e-5);
        for (size_t j = 0; j < luts[i].int_lut.size(); ++j) {
            EXPECT_EQ(qq_single.int_lut[j], luts[i].int_lut[j]);
        }
    }
}

TEST(FastScanTest, CorrectnessL2_Small) {
    RunFastScanTests<true>(10, 4, 8);
}

TEST(FastScanTest, CorrectnessIP_Small) {
    RunFastScanTests<false>(10, 4, 8);
}

TEST(FastScanTest, CorrectnessL2_Large_EdgeCase) {
    // 150 points tests exactly 2 full strips (128 elements) and a tail of 22
    RunFastScanTests<true>(150, 2, 4); 
}

TEST(FastScanTest, CorrectnessIP_Large_EdgeCase) {
    RunFastScanTests<false>(131, 8, 16); 
}

TEST(FastScanTest, BatchQueryL2) {
    RunFastScanBatchQueryTests<true>(50, 4, 8);
}

TEST(FastScanTest, BatchQueryIP) {
    RunFastScanBatchQueryTests<false>(50, 4, 8);
}
