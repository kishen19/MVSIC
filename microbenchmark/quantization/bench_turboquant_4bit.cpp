// turboquant_4bit_benchmark.cc
//
// Microbenchmark for the strip-based turboquant_4bit kernel.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "benchmark/benchmark.h"
#include "mvsic/core/quantization/turboquant_4bit.h"

namespace mvsic {
namespace turboquant_4bit {
namespace {

constexpr int kNumDatapoints = 1000000;
constexpr int kNumQueries = 1000;
constexpr int kNumDatapointsScored = 1000;
constexpr int kDimensionalities[] = {64, 96, 100, 128, 256, 512, 768, 1536, 3072, 4096, 8192};

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
  Model<Metric> model;
  Quantized_Point_Range<FlatPointRange, Metric> encoded;
  std::vector<Quantized_Query<Metric>> queries;
};

template<bool Metric>
CachedState<Metric>* GetCachedState(size_t dimensionality, size_t num_datapoints) {
  static CachedState<Metric> cache;
  if (cache.dimensionality == dimensionality && cache.num_datapoints == num_datapoints)
    return &cache;

  FlatPointRange data(num_datapoints, dimensionality);
  {
    const float scale = 1.0f / std::sqrt(static_cast<float>(dimensionality));
    std::mt19937_64 gen(42);
    std::normal_distribution<float> dist(0.0f, scale);
    for (float& x : data.data_)
      x = dist(gen);
  }

  cache.model.train(data);
  cache.encoded = cache.model.encode(data);

  cache.queries.clear();
  {
    const float scale = 1.0f / std::sqrt(static_cast<float>(dimensionality));
    std::mt19937_64 gen(123);
    std::normal_distribution<float> dist(0.0f, scale);
    std::vector<float> q(dimensionality);
    for (int i = 0; i < kNumQueries; ++i) {
      for (size_t d = 0; d < dimensionality; ++d)
        q[d] = dist(gen);
      cache.queries.push_back(cache.model.quantize_query(q.data()));
    }
  }

  cache.dimensionality = dimensionality;
  cache.num_datapoints = num_datapoints;
  return &cache;
}

// Benchmark: strip-based batch scoring (distances_contiguous)
template<bool Metric>
void BM_StripBatch(benchmark::State& state) {
  const size_t dimensionality = state.range(0);
  const size_t num_datapoints = state.range(1);
  const int num_queries = state.range(2);
  const int num_datapoints_scored = state.range(3);

  auto* cache = GetCachedState<Metric>(dimensionality, num_datapoints);
  const auto& encoded = cache->encoded;
  const auto& queries = cache->queries;

  state.SetLabel(Metric ? "SquaredL2, strip" : "DotProduct, strip");

  // Use first N points (strip-padded).
  const size_t N_padded = ((num_datapoints_scored + 63) / 64) * 64;
  const size_t n_strips = N_padded / 64;
  const size_t strip_stride = encoded.stride;
  const size_t strip_bytes = n_strips * strip_stride;

  const uint8_t* strip_data = encoded.packed_codes.data();
  const float* norms = encoded.norm_scaling_factors.data();
  const float* sqn = encoded.unquantized_squared_norms.data();

  std::vector<float> scores(N_padded);

  for (auto _ : state) {
    for (int q = 0; q < num_queries; ++q) {
      queries[q].distances_contiguous(strip_data, norms, sqn, strip_stride, num_datapoints_scored,
                                      scores.data());
    }
    benchmark::DoNotOptimize(scores.data());
    benchmark::ClobberMemory();
  }

  const size_t items_per_iter = num_queries * num_datapoints_scored;
  const size_t nb = encoded.num_bytes_per_datapoint;
  const size_t bytes_per_item = nb + sizeof(float) + (Metric ? sizeof(float) : 0);
  state.SetBytesProcessed(state.iterations() * items_per_iter * bytes_per_item);
  state.SetItemsProcessed(state.iterations() * items_per_iter);
  state.counters["ns/dp"] = benchmark::Counter(
      static_cast<double>(items_per_iter),
      benchmark::Counter::kIsIterationInvariantRate | benchmark::Counter::kInvert,
      benchmark::Counter::kIs1000);
}

void CustomArguments(::benchmark::Benchmark* b) {
  for (int dim : kDimensionalities)
    b->Args({dim, kNumDatapoints, kNumQueries, kNumDatapointsScored});
}

BENCHMARK(BM_StripBatch<false>)->Apply(CustomArguments)->Name("BM_Strip_IP");
BENCHMARK(BM_StripBatch<true>)->Apply(CustomArguments)->Name("BM_Strip_L2");

}  // namespace
}  // namespace turboquant_4bit
}  // namespace mvsic

BENCHMARK_MAIN();
