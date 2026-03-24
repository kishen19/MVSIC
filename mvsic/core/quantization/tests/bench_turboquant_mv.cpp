#include <benchmark/benchmark.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include "mvsic/core/quantization/other_methods/turboquant_4bit.h"
#include "mvsic/core/quantization/other_methods/wrapper.h"
#include "mvsic/core/quantization/turboquant_mv.h"
#include "mvsic/core/types/chamfer_ip_point.h"
#include "mvsic/core/types/chamfer_l2_point.h"
#include "mvsic/core/types/point_cloud_set.h"
#include "parlay/parallel.h"
#include "parlay/sequence.h"

namespace mvsic {
namespace {

constexpr int kNumDbClouds = 10000;
constexpr int kNumQueries = 1000;
constexpr int kDbCloudSize = 64;
constexpr int kQueryCloudSize = 32;
constexpr int kDimensionality = 128;

template<bool Metric>
using PointType = std::conditional_t<Metric, ChamferL2_Point, ChamferIP_Point>;

template<bool Metric>
using PointCloudSetType = PointCloudSet<PointType<Metric>>;

template<bool Metric>
PointCloudSetType<Metric> fill_random_point_cloud_set(size_t n_clouds, size_t cloud_size,
                                                      size_t dim) {
  PointCloudSetType<Metric> pcs(n_clouds, cloud_size, dim);
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  float* ptr = pcs.data();
  for (size_t i = 0; i < n_clouds * cloud_size * dim; ++i) {
    ptr[i] = dist(gen);
  }
  return pcs;
}

template<bool Metric>
struct CachedState {
  using PCSet = PointCloudSetType<Metric>;
  using FlatPCSet = FlattenedPCRange<PCSet>;
  using Tq4_Model = turboquant_4bit::Model<Metric>;
  using Tq4_Range = turboquant_4bit::Quantized_Point_Range<FlatPCSet, Metric>;
  using Tq4_Query = turboquant_4bit::Quantized_Query<Metric>;

  PCSet db_pcs;
  PCSet query_pcs;

  // Wrapper-based state
  std::unique_ptr<MultiVecQuantizer<Tq4_Model, Metric>> wrapper_model;
  std::unique_ptr<Quantized_Point_Cloud_Set<Tq4_Range, Metric>> wrapper_set;
  std::vector<Quantized_Query_Point_Cloud<Tq4_Query, Metric>> wrapper_queries;

  // MV-based state
  std::unique_ptr<turboquant_mv::Model<Metric>> mv_model;
  std::unique_ptr<turboquant_mv::Quantized_Point_Cloud_Set<Metric>> mv_set;
  std::vector<turboquant_mv::Quantized_Query_Point_Cloud<Metric>> mv_queries;

  std::vector<std::vector<float>> exact_distances;

  CachedState() {}
};

template<bool Metric>
CachedState<Metric>* GetCachedState() {
  static CachedState<Metric>* state = nullptr;
  if (state != nullptr) return state;

  state = new CachedState<Metric>();
  state->db_pcs = fill_random_point_cloud_set<Metric>(kNumDbClouds, kDbCloudSize, kDimensionality);
  state->query_pcs =
      fill_random_point_cloud_set<Metric>(kNumQueries, kQueryCloudSize, kDimensionality);

  // 1. Train models
  state->wrapper_model =
      std::make_unique<MultiVecQuantizer<typename CachedState<Metric>::Tq4_Model, Metric>>();
  state->wrapper_model->train(state->db_pcs);

  state->mv_model = std::make_unique<turboquant_mv::Model<Metric>>();
  state->mv_model->train(state->db_pcs);

  // 2. Encode DB
  state->wrapper_set =
      std::make_unique<Quantized_Point_Cloud_Set<typename CachedState<Metric>::Tq4_Range, Metric>>(
          state->wrapper_model->encode(state->db_pcs));

  state->mv_set = std::make_unique<turboquant_mv::Quantized_Point_Cloud_Set<Metric>>(
      state->mv_model->encode(state->db_pcs));

  // 3. Encode Queries
  state->wrapper_queries.resize(kNumQueries);
  state->mv_queries.resize(kNumQueries);
  for (size_t i = 0; i < kNumQueries; ++i) {
    auto q_cloud = state->query_pcs[i];
    state->wrapper_queries[i] = state->wrapper_model->quantize_query(q_cloud);
    state->mv_queries[i] = state->mv_model->quantize_query(q_cloud);
  }

  // 4. Compute exact distances
  state->exact_distances.resize(kNumQueries);
  for (size_t q = 0; q < kNumQueries; ++q) {
    state->exact_distances[q].resize(kNumDbClouds);
    auto q_cloud = state->query_pcs[q];
    for (size_t c = 0; c < kNumDbClouds; ++c) {
      auto db_cloud = state->db_pcs[c];
      state->exact_distances[q][c] = q_cloud.distance(db_cloud);
    }
  }

  // Explicit correctness check
  double mv_mae = 0, mv_mse = 0;
  double wrapper_mae = 0, wrapper_mse = 0;
  size_t count = 0;

  std::vector<std::pair<uint32_t, float>> res_mv(kNumDbClouds);
  std::vector<std::pair<uint32_t, float>> res_wrap(kNumDbClouds);

  size_t num_q_check = std::min<size_t>(kNumQueries, 100);
  size_t num_c_check = std::min<size_t>(kNumDbClouds, 1000);
  for (size_t q = 0; q < num_q_check; ++q) {
    state->mv_set->distances_all(state->mv_queries[q], res_mv.data());
    state->wrapper_set->distances_all(state->wrapper_queries[q], res_wrap.data());
    for (size_t c = 0; c < num_c_check; ++c) {
      float exact = state->exact_distances[q][c];
      double d_mv = res_mv[c].second - exact;
      double d_wrap = res_wrap[c].second - exact;
      mv_mae += std::abs(d_mv);
      mv_mse += d_mv * d_mv;
      wrapper_mae += std::abs(d_wrap);
      wrapper_mse += d_wrap * d_wrap;
      count++;
    }
  }

  std::cout << "--- [Correctness Check: Dim " << kDimensionality << "] ---\n"
            << "  Wrapper -> MAE: " << wrapper_mae / count << ", MSE: " << wrapper_mse / count
            << "\n"
            << "  MV      -> MAE: " << mv_mae / count << ", MSE: " << mv_mse / count << "\n"
            << "  Correctness Satisfied? "
            << (mv_mae / count < 0.08 && wrapper_mae / count < 0.08 ? "YES! ✔️" : "NO! ❌")
            << "\n----------------------------------------\n";

  return state;
}

template<bool Metric, typename SetType, typename QueriesType>
void CheckQuality(benchmark::State& state, const CachedState<Metric>* cache,
                  const std::string& prefix, SetType& set, const QueriesType& queries) {
  double sum_ae = 0;
  double sum_se = 0;
  size_t count = 0;

  std::vector<std::pair<uint32_t, float>> results(kNumDbClouds);
  for (size_t q = 0; q < kNumQueries; ++q) {
    set->distances_all(queries[q], results.data());
    for (size_t c = 0; c < kNumDbClouds; ++c) {
      float approx = results[c].second;
      float exact = cache->exact_distances[q][c];
      double diff = approx - exact;
      sum_ae += std::abs(diff);
      sum_se += diff * diff;
      count++;
    }
  }

  state.counters[prefix + "_MAE"] = sum_ae / count;
  state.counters[prefix + "_MSE"] = sum_se / count;
}

template<bool Metric>
void BM_TurboQuantWrapper(benchmark::State& state) {
  auto* cache = GetCachedState<Metric>();
  std::vector<std::pair<uint32_t, float>> results(kNumDbClouds);

  for (auto _ : state) {
    for (size_t q = 0; q < kNumQueries; ++q) {
      cache->wrapper_set->distances_all(cache->wrapper_queries[q], results.data());
      benchmark::DoNotOptimize(results);
    }
  }

  state.counters["items_per_second"] = benchmark::Counter(
      state.iterations() * kNumQueries * kNumDbClouds, benchmark::Counter::kIsRate);
}

template<bool Metric>
void BM_TurboQuantMV(benchmark::State& state) {
  auto* cache = GetCachedState<Metric>();
  std::vector<std::pair<uint32_t, float>> results(kNumDbClouds);

  for (auto _ : state) {
    for (size_t q = 0; q < kNumQueries; ++q) {
      cache->mv_set->distances_all(cache->mv_queries[q], results.data());
      benchmark::DoNotOptimize(results);
    }
  }

  state.counters["items_per_second"] = benchmark::Counter(
      state.iterations() * kNumQueries * kNumDbClouds, benchmark::Counter::kIsRate);
}

BENCHMARK_TEMPLATE(BM_TurboQuantWrapper, false)
    ->Unit(benchmark::kMillisecond)
    ->MinWarmUpTime(1.0)
    ->Repetitions(10)
    ->DisplayAggregatesOnly(true);

BENCHMARK_TEMPLATE(BM_TurboQuantMV, false)
    ->Unit(benchmark::kMillisecond)
    ->MinWarmUpTime(1.0)
    ->Repetitions(10)
    ->DisplayAggregatesOnly(true);

}  // namespace
}  // namespace mvsic

BENCHMARK_MAIN();