// test_vnni_chamfer.cc
//
// Correctness test for the VNNI GEMM Chamfer kernel.
// Compares chamfer_vnni_gemm() against a scalar reference that manually
// extracts per-point bytes from the strip layout and computes dot products.

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

#include "mvsic/core/quantization/turboquant_4bit.h"

namespace mvsic {
namespace turboquant_4bit {
namespace {

struct FlatPointRange {
  std::vector<float> data_;
  size_t n_ = 0;
  size_t dim_ = 0;
  FlatPointRange() = default;
  FlatPointRange(size_t n, size_t d) : data_(n * d, 0.0f), n_(n), dim_(d) {}
  size_t size() const { return n_; }
  uint32_t get_dims() const { return static_cast<uint32_t>(dim_); }
  const float* location(size_t i) const { return data_.data() + i * dim_; }
};

// Scalar reference: compute Chamfer distance by extracting per-point bytes
// from strip layout and computing int8 dot products directly.
template<bool Metric>
float scalar_chamfer_reference(const Quantized_Query<Metric>* const* query_ptrs, size_t num_queries,
                               const Quantized_Point_Range<FlatPointRange, Metric>& encoded,
                               size_t cloud_size) {

  const size_t num_bytes = encoded.num_bytes_per_datapoint;
  const size_t strip_stride = encoded.stride;
  float total_chamfer = 0.0f;

  for (size_t qi = 0; qi < num_queries; ++qi) {
    const auto& qq = *query_ptrs[qi];
    float min_dist = std::numeric_limits<float>::max();

    for (size_t pi = 0; pi < cloud_size; ++pi) {
      // Extract per-point bytes from strip layout.
      const size_t strip = pi / 64;
      const size_t lane = pi % 64;

      int32_t dot = 0;
      for (size_t j = 0; j < num_bytes; ++j) {
        const uint8_t byte = encoded.packed_codes[strip * strip_stride + j * 64 + lane];
        const uint8_t b_even = byte & 0xF;
        const uint8_t b_odd = byte >> 4;
        dot +=
            static_cast<int32_t>(internal::kTurboQuantCentroidsInt8[b_even]) * qq.query_data[2 * j];
        dot += static_cast<int32_t>(internal::kTurboQuantCentroidsInt8[b_odd]) *
               qq.query_data[2 * j + 1];
      }

      float neg_dot =
          -static_cast<float>(dot) * encoded.norm_scaling_factors[pi] * qq.norm_scaling_factor;
      float dist;
      if constexpr (Metric) {
        dist = encoded.unquantized_squared_norms[pi] + 2.0f * neg_dot + qq.unquantized_squared_norm;
      } else {
        dist = neg_dot;
      }

      if (dist < min_dist) min_dist = dist;
    }
    total_chamfer += min_dist;
  }

  return total_chamfer;
}

template<bool Metric>
bool run_test(size_t dim, size_t cloud_size, size_t num_queries, int seed) {
  const char* metric_name = Metric ? "L2" : "IP";

  // Generate random data.
  const float scale = 1.0f / std::sqrt(static_cast<float>(dim));
  std::mt19937_64 gen(seed);
  std::normal_distribution<float> dist(0.0f, scale);

  FlatPointRange data(cloud_size, dim);
  for (float& x : data.data_)
    x = dist(gen);

  // Train model and encode.
  Model<Metric> model;
  model.train(data);
  auto encoded = model.encode(data);

  // Quantize queries.
  std::vector<Quantized_Query<Metric>> queries;
  std::vector<float> qvec(dim);
  for (size_t i = 0; i < num_queries; ++i) {
    for (size_t d = 0; d < dim; ++d)
      qvec[d] = dist(gen);
    queries.push_back(model.quantize_query(qvec.data()));
  }

  // Build query pointer array.
  std::vector<const Quantized_Query<Metric>*> qptrs(num_queries);
  for (size_t i = 0; i < num_queries; ++i)
    qptrs[i] = &queries[i];

  // Compute VNNI GEMM Chamfer.
  const size_t n_strips = (cloud_size + 63) / 64;
  float vnni_result = chamfer_vnni_gemm<Metric>(
      qptrs.data(), num_queries, encoded.packed_codes.data(), encoded.norm_scaling_factors.data(),
      encoded.unquantized_squared_norms.data(), encoded.stride, n_strips,
      encoded.num_bytes_per_datapoint, cloud_size);

  // Compute scalar reference.
  float scalar_result =
      scalar_chamfer_reference<Metric>(qptrs.data(), num_queries, encoded, cloud_size);

  // Compare.
  float diff = std::abs(vnni_result - scalar_result);
  float denom = std::max(std::abs(scalar_result), 1e-6f);
  float rel_err = diff / denom;

  bool pass = rel_err < 1e-3f;

  std::printf("  dim=%-4zu  N=%-4zu  Nq=%-3zu  %-2s  vnni=%.6f  scalar=%.6f  "
              "rel_err=%.2e  %s\n",
              dim, cloud_size, num_queries, metric_name, vnni_result, scalar_result, rel_err,
              pass ? "PASS" : "FAIL");

  return pass;
}

}  // namespace
}  // namespace turboquant_4bit
}  // namespace mvsic

int main() {
  using namespace mvsic::turboquant_4bit;

  std::printf("=== VNNI GEMM Chamfer Correctness Test ===\n\n");

  bool all_pass = true;
  int seed = 42;

  // Test across dimensions, cloud sizes, and metrics.
  for (size_t dim : {64, 128, 256}) {
    for (size_t cloud_size : {16, 64, 100}) {
      for (size_t num_queries : {1, 5, 8, 12}) {
        all_pass &= run_test<false>(dim, cloud_size, num_queries, seed++);
        all_pass &= run_test<true>(dim, cloud_size, num_queries, seed++);
      }
    }
  }

  std::printf("\n%s\n", all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
  return all_pass ? 0 : 1;
}
