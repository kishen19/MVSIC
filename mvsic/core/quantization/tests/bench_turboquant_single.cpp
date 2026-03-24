// bench_turboquant_single.cpp
//
// Microbenchmark for single-vector TurboQuant distances.
// Comparing turboquant.h vs turboquant_4bit.h

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>
#include <iostream>

#include "benchmark/benchmark.h"
#include "mvsic/core/quantization/turboquant.h"
#include "mvsic/core/quantization/other_methods/turboquant_4bit.h"

namespace mvsic {
namespace {

constexpr int kNumDatapoints = 10000;
constexpr int kNumQueries = 100;
constexpr int kDimensionalities[] = {64, 128, 256, 512};

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

template<bool Metric>
struct CachedState {
  size_t dimensionality = 0;
  size_t num_datapoints = 0;

  // New Single-Vector State
  turboquant::Model<Metric> new_model;
  turboquant::Quantized_Point_Range<FlatPointRange, Metric> new_encoded;
  std::vector<turboquant::Quantized_Query<Metric>> new_queries;

  // Old Monolithic State
  turboquant_4bit::Model<Metric> old_model;
  turboquant_4bit::Quantized_Point_Range<FlatPointRange, Metric> old_encoded;
  std::vector<turboquant_4bit::Quantized_Query<Metric>> old_queries;

  std::vector<std::vector<float>> exact_distances;
};

template<bool Metric>
CachedState<Metric>* GetCachedState(size_t dimensionality, size_t num_datapoints) {
  static CachedState<Metric> cache;
  if (cache.dimensionality == dimensionality && cache.num_datapoints == num_datapoints)
    return &cache;

  FlatPointRange data(num_datapoints, dimensionality);
  FlatPointRange query_data(kNumQueries, dimensionality);
  {
    const float scale = 1.0f / std::sqrt(static_cast<float>(dimensionality));
    std::mt19937_64 gen(42);
    std::normal_distribution<float> dist(0.0f, scale);
    for (float& x : data.data_) x = dist(gen);
    for (float& x : query_data.data_) x = dist(gen);
  }

  // Train and encode (New)
  cache.new_model.train(data);
  cache.new_encoded = cache.new_model.encode(data);

  // Train and encode (Old)
  cache.old_model.train(data);
  cache.old_encoded = cache.old_model.encode(data);

  cache.new_queries.clear();
  cache.old_queries.clear();
  for (int i = 0; i < kNumQueries; ++i) {
    cache.new_queries.push_back(cache.new_model.quantize_query(query_data.location(i)));
    cache.old_queries.push_back(cache.old_model.quantize_query(query_data.location(i)));
  }

  // Exact Distances for MAE/MSE (Simplified to IP for checking if metric is false)
  cache.exact_distances.resize(kNumQueries, std::vector<float>(num_datapoints));
  for (int q = 0; q < kNumQueries; ++q) {
    for (size_t c = 0; c < num_datapoints; ++c) {
      float dot = 0.0f;
      for (size_t d = 0; d < dimensionality; ++d) {
        dot += query_data.location(q)[d] * data.location(c)[d];
      }
      cache.exact_distances[q][c] = Metric ? 0.0f /* not computing precise L2 here */ : -dot;
    }
  }

  // --- Print Correctness Explicitly ---
  if (!Metric) {
    double old_mae = 0, old_mse = 0;
    double new_mae = 0, new_mse = 0;
    size_t count = 0;
    for (int q = 0; q < std::min(kNumQueries, 100); ++q) {
      for (size_t c = 0; c < std::min(num_datapoints, (size_t)1000); ++c) {
        float exact = cache.exact_distances[q][c];
        float old_dist = cache.old_encoded[c].distance(cache.old_queries[q]);
        float new_dist = cache.new_encoded[c].distance(cache.new_queries[q]);
        old_mae += std::abs(old_dist - exact);
        old_mse += (old_dist - exact) * (old_dist - exact);
        new_mae += std::abs(new_dist - exact);
        new_mse += (new_dist - exact) * (new_dist - exact);
        count++;
      }
    }
    std::cout << "--- [Correctness Check: Dim " << dimensionality << "] ---\n"
              << "  Old -> MAE: " << old_mae/count << ", MSE: " << old_mse/count << "\n"
              << "  New -> MAE: " << new_mae/count << ", MSE: " << new_mse/count << "\n"
              << "  Correctness Satisfied? " << (new_mae/count < 0.02 && old_mae/count < 0.02 ? "YES! ✔️ (Both MAE < 0.02)" : "NO! ❌") << "\n----------------------------------------\n";
  }

  cache.dimensionality = dimensionality;
  cache.num_datapoints = num_datapoints;
  return &cache;
}

template<bool Metric>
void BM_NewSingle(benchmark::State& state) {
  const size_t dimensionality = state.range(0);
  const size_t num_datapoints = state.range(1);

  auto* cache = GetCachedState<Metric>(dimensionality, num_datapoints);

  for (auto _ : state) {
    for (int q = 0; q < kNumQueries; ++q) {
      const auto& query = cache->new_queries[q];
      for (size_t c = 0; c < num_datapoints; ++c) {
        const auto pt = cache->new_encoded[c];
        float dist = pt.distance(query);
        benchmark::DoNotOptimize(dist);
      }
    }
  }

  const size_t items_per_iter = kNumQueries * num_datapoints;
  state.counters["items_per_second"] = benchmark::Counter(
      state.iterations() * items_per_iter, benchmark::Counter::kIsRate);
}

template<bool Metric>
void BM_OldSingle(benchmark::State& state) {
  const size_t dimensionality = state.range(0);
  const size_t num_datapoints = state.range(1);

  auto* cache = GetCachedState<Metric>(dimensionality, num_datapoints);

  for (auto _ : state) {
    for (int q = 0; q < kNumQueries; ++q) {
      const auto& query = cache->old_queries[q];
      for (size_t c = 0; c < num_datapoints; ++c) {
        const auto pt = cache->old_encoded[c];
        float dist = pt.distance(query);
        benchmark::DoNotOptimize(dist);
      }
    }
  }

  const size_t items_per_iter = kNumQueries * num_datapoints;
  state.counters["items_per_second"] = benchmark::Counter(
      state.iterations() * items_per_iter, benchmark::Counter::kIsRate);
}

void CustomArguments(::benchmark::Benchmark* b) {
  for (int dim : kDimensionalities)
    b->Args({dim, kNumDatapoints});
}

BENCHMARK_TEMPLATE(BM_OldSingle, false)->Apply(CustomArguments)->Name("BM_OldSingle_IP")->Unit(benchmark::kMillisecond)->MinWarmUpTime(0.5)->Repetitions(1);
BENCHMARK_TEMPLATE(BM_NewSingle, false)->Apply(CustomArguments)->Name("BM_NewSingle_IP")->Unit(benchmark::kMillisecond)->MinWarmUpTime(0.5)->Repetitions(1);

}  // namespace
}  // namespace mvsic

BENCHMARK_MAIN();