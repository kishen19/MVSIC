#include <benchmark/benchmark.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include "mvsic/core/quantization/other_methods/fastscan.h"
#include "mvsic/core/quantization/other_methods/wrapper.h"
#include "mvsic/core/quantization/fastscan_mv.h"
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
  using Fs_Model = fastscan::Model<Metric>;
  using Fs_Range = fastscan::Quantized_Point_Range<FlatPCSet, Metric>;
  using Fs_Query = fastscan::Quantized_Query<Metric>;

  PCSet db_pcs;
  PCSet query_pcs;

  // Wrapper-based state
  std::unique_ptr<MultiVecQuantizer<Fs_Model, Metric>> wrapper_model;
  std::unique_ptr<Quantized_Point_Cloud_Set<Fs_Range, Metric>> wrapper_set;
  std::vector<Quantized_Query_Point_Cloud<Fs_Query, Metric>> wrapper_queries;

  // MV-based state
  std::unique_ptr<fastscan_mv::Model<Metric>> mv_model;
  std::unique_ptr<fastscan_mv::Quantized_Point_Cloud_Set<Metric>> mv_set;
  std::vector<fastscan_mv::Quantized_Query_Point_Cloud<Metric>> mv_queries;

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
      std::make_unique<MultiVecQuantizer<typename CachedState<Metric>::Fs_Model, Metric>>();
  state->wrapper_model->train(state->db_pcs);

  state->mv_model = std::make_unique<fastscan_mv::Model<Metric>>();
  state->mv_model->train(state->db_pcs);

  // 2. Encode DB
  state->wrapper_set =
      std::make_unique<Quantized_Point_Cloud_Set<typename CachedState<Metric>::Fs_Range, Metric>>(
          state->wrapper_model->encode(state->db_pcs));

  state->mv_set = std::make_unique<fastscan_mv::Quantized_Point_Cloud_Set<Metric>>(
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
  double mv_recall10 = 0, wrapper_recall10 = 0;
  size_t count = 0;

  std::vector<std::pair<uint32_t, float>> res_mv(kNumDbClouds);
  std::vector<std::pair<uint32_t, float>> res_wrap(kNumDbClouds);

  size_t num_q_check = std::min<size_t>(kNumQueries, 100);
  size_t num_c_check = kNumDbClouds;
  for (size_t q = 0; q < num_q_check; ++q) {
    state->mv_set->distances_all(state->mv_queries[q], res_mv.data());
    state->wrapper_set->distances_all(state->wrapper_queries[q], res_wrap.data());

    std::vector<std::pair<uint32_t, float>> exact_dists(num_c_check);
    for (size_t c = 0; c < num_c_check; ++c) {
      exact_dists[c] = {c, state->exact_distances[q][c]};
    }
    std::sort(exact_dists.begin(), exact_dists.end(), [](const auto& a, const auto& b) {
      return a.second < b.second;
    });

    std::vector<uint32_t> true_top10;
    for (int i = 0; i < 10 && i < num_c_check; ++i) {
      true_top10.push_back(exact_dists[i].first);
    }

    std::vector<std::pair<uint32_t, float>> mv_sorted = res_mv;
    std::sort(mv_sorted.begin(), mv_sorted.end(), [](const auto& a, const auto& b) {
      return a.second < b.second;
    });

    std::vector<std::pair<uint32_t, float>> wrap_sorted = res_wrap;
    std::sort(wrap_sorted.begin(), wrap_sorted.end(), [](const auto& a, const auto& b) {
      return a.second < b.second;
    });

    int mv_hits = 0, wrap_hits = 0;
    for (int i = 0; i < 10 && i < num_c_check; ++i) {
      if (std::find(true_top10.begin(), true_top10.end(), mv_sorted[i].first) != true_top10.end()) {
        mv_hits++;
      }
      if (std::find(true_top10.begin(), true_top10.end(), wrap_sorted[i].first) != true_top10.end()) {
        wrap_hits++;
      }
    }
    mv_recall10 += static_cast<double>(mv_hits) / 10.0;
    wrapper_recall10 += static_cast<double>(wrap_hits) / 10.0;

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

  mv_recall10 /= num_q_check;
  wrapper_recall10 /= num_q_check;

  std::cout << "--- [Correctness Check: Dim " << kDimensionality << "] ---\n"
            << "  Wrapper -> MAE: " << wrapper_mae / count << ", MSE: " << wrapper_mse / count << ", Recall@10: " << wrapper_recall10 << "\n"
            << "  MV      -> MAE: " << mv_mae / count << ", MSE: " << mv_mse / count << ", Recall@10: " << mv_recall10 << "\n"
            << "  Correctness Satisfied? "
            << (std::abs(mv_mae - wrapper_mae) / count < 1e-4 ? "YES! ✔️" : "NO! ❌")
            << "\n----------------------------------------\n";

  return state;
}

template<bool Metric>
void BM_FastScanWrapper(benchmark::State& state) {
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
void BM_FastScanMV(benchmark::State& state) {
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

BENCHMARK_TEMPLATE(BM_FastScanWrapper, false)
    ->Unit(benchmark::kMillisecond)
    ->MinWarmUpTime(1.0)
    ->Repetitions(10)
    ->DisplayAggregatesOnly(true);

BENCHMARK_TEMPLATE(BM_FastScanMV, false)
    ->Unit(benchmark::kMillisecond)
    ->MinWarmUpTime(1.0)
    ->Repetitions(10)
    ->DisplayAggregatesOnly(true);

}  // namespace
}  // namespace mvsic

BENCHMARK_MAIN();
