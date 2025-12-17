#include <iostream>
#include <vector>
#include <random>
#include <cassert>
#include <string>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <chrono>
#include <type_traits>  // Required for generic checks

#include "parlay/parallel.h"
#include "parlay/primitives.h"

// Project includes
#include "mvsic/core/utils/parse_command_line.h"
#include "algorithms/utils/euclidian_point.h"
#include "algorithms/utils/point_range.h"

// Quantizer implementations
#include "pq.h"
#include "rabitq_wrapper.h"
#include "scann_wrapper.h"

using namespace mvsic;

// --- Helper: Dead Code Elimination (DCE) Prevention ---
// Forces the compiler to believe the object is used, so it doesn't skip the computation.
template<typename T>
inline void dce_hook(const T& obj) {
  // Cast to volatile void* forces the compiler to treat the address as "escaping"
  // to unknown code, preventing it from optimizing away the object construction.
  const volatile void* p = &obj;
  (void)p;
}

// --- Helper: Generate Data in Memory ---
parlay::sequence<parlay::sequence<float>> generate_data_in_memory(size_t n, size_t d) {
  std::cout << "Generating " << n << " vectors of dim " << d << " in memory..." << std::endl;
  parlay::sequence<parlay::sequence<float>> data(n);
  parlay::parallel_for(0, n, [&](size_t i) {
    std::mt19937 rng(i);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    parlay::sequence<float> vec(d);
    for (size_t j = 0; j < d; ++j)
      vec[j] = dist(rng);
    data[i] = std::move(vec);
  });
  return data;
}

// --- Verification (Accuracy) ---
template<typename QuantizedRangeTy, typename ExactRangeTy>
void run_verification(const QuantizedRangeTy& q_data, const ExactRangeTy& exact_data, size_t n,
                      const std::string& method_name) {
  std::cout << "\n--- Accuracy Check (" << method_name << ") ---" << std::endl;

  auto query_point = exact_data[0];
  auto quantized_query = q_data.quantize_query(query_point);

  double total_rel_error = 0.0;
  size_t num_checks = std::min((size_t)10, (size_t)n);

  printf("%-8s %-12s %-15s %-12s\n", "Idx", "Exact", "Approx", "RelErr");

  for (size_t i = 0; i < num_checks; ++i) {
    float d_ex = exact_data[i].distance(query_point);
    float d_approx = quantized_query.distance(q_data[i]);

    float err = std::abs(d_ex - d_approx);
    float rel = (d_ex > 1e-5) ? (err / d_ex) : 0.0f;
    total_rel_error += rel;

    printf("%-8lu %-12.4f %-15.4f %-12.4f\n", i, d_ex, d_approx, rel);
  }
  std::cout << "Avg Rel Error (sample): " << (total_rel_error / num_checks) * 100.0f << "%"
            << std::endl;
}

// --- Benchmark (Performance) ---
template<typename QuantizedRangeTy, typename ExactRangeTy>
void run_benchmark(const QuantizedRangeTy& q_data, const ExactRangeTy& exact_data, size_t n,
                   size_t d) {
  std::cout << "\n--- Performance Benchmark ---" << std::endl;

  // 1. Generate Random Queries
  size_t num_queries = 100;
  std::vector<std::vector<float>> queries(num_queries, std::vector<float>(d));
  std::mt19937 rng(999);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  for (auto& q : queries)
    for (float& x : q)
      x = dist(rng);

  // 2. Measure Query Preparation Latency
  auto start_prep = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < num_queries; ++i) {
    auto qq = q_data.quantize_query(queries[i]);
    // FIX: Use generic hook instead of checking specific members like g_add
    dce_hook(qq);
  }
  auto end_prep = std::chrono::high_resolution_clock::now();
  double prep_latency_us =
      std::chrono::duration_cast<std::chrono::microseconds>(end_prep - start_prep).count() /
      (double)num_queries;

  // 3. Measure Scan Throughput
  auto qq = q_data.quantize_query(queries[0]);

  // Warmup
  float dummy_sum = 0;
  for (size_t i = 0; i < std::min(n, (size_t)1000); ++i)
    dummy_sum += qq.distance(q_data[i]);

  auto start_scan = std::chrono::high_resolution_clock::now();

  size_t total_ops = 0;
  size_t target_ops = 20000000;  // 20M ops for good resolution

  while (total_ops < target_ops) {
    for (size_t i = 0; i < n; ++i) {
      dummy_sum += qq.distance(q_data[i]);
    }
    total_ops += n;
  }

  auto end_scan = std::chrono::high_resolution_clock::now();
  double total_scan_seconds =
      std::chrono::duration_cast<std::chrono::duration<double>>(end_scan - start_scan).count();

  double mops = (total_ops / total_scan_seconds) / 1e6;  // Million Ops Per Second
  double ns_per_dist = (total_scan_seconds * 1e9) / total_ops;

  // Report
  printf("%-25s : %.3f us\n", "Query Prep Latency", prep_latency_us);
  printf("%-25s : %.3f ns\n", "Time per Dist Calc", ns_per_dist);
  printf("%-25s : %.3f M/s\n", "Throughput (Dists)", mops);

  dce_hook(dummy_sum);  // Use result
}

int main(int argc, char** argv) {
  mvsic::commandLine P(argc, argv,
                       "[-m <num_blocks>] [-k <num_clusters>] [-s <subsample_mult>] "
                       "[-n <num_points>] [-d <dim>] "
                       "[-rabitq] [-rbits <rabitq_total_bits>] "
                       "[-scann] [-T <anisotropic_threshold>]");

  // General Params
  uint32_t n = P.getOptionIntValue("-n", 10000);
  uint32_t d = P.getOptionIntValue("-d", 128);

  bool use_rabitq = P.getOption("-rabitq");
  bool use_scann = P.getOption("-scann");

  // PQ / ScaNN Params
  uint32_t m_blocks = P.getOptionIntValue("-m", 8);
  uint32_t k_clusters = P.getOptionIntValue("-k", 256);
  uint32_t sub_mult = P.getOptionIntValue("-s", 20);
  float scann_T = P.getOptionDoubleValue("-T", 0.2);

  // RaBitQ Params
  uint32_t rabitq_bits = P.getOptionIntValue("-rbits", 8);

  // --- 1. Generate Data ---
  parlay::internal::timer t;
  t.start();
  auto raw_data = generate_data_in_memory(n, d);
  std::cout << "Data Gen Time: " << t.next_time() << "s" << std::endl;

  using PointTy = parlayANN::Euclidian_Point<float>;
  using PointRangeTy = parlayANN::PointRange<PointTy>;

  PointRangeTy data_range(raw_data, d);

  // --- 2. Build & Test Quantizer ---
  if (use_rabitq) {
    // --- RaBitQ ---
    std::cout << "\n=== Running RaBitQ ===" << std::endl;
    if (d < 64) std::cout << "Warning: RaBitQ preferred for d >= 64." << std::endl;

    t.start();
    mvsic::RaBitQ_Point_Range<PointRangeTy> rabitq_data(data_range, rabitq_bits);
    double build_time = t.next_time();

    std::cout << "RaBitQ Build Time: " << build_time << "s (" << (n / build_time) << " vec/s)"
              << std::endl;
    std::cout << "Bits per Dim: " << rabitq_bits << std::endl;

    run_verification(rabitq_data, data_range, n, "RaBitQ");
    run_benchmark(rabitq_data, data_range, n, d);

  } else if (use_scann) {
    // --- ScaNN ---
    std::cout << "\n=== Running ScaNN (Anisotropic VQ) ===" << std::endl;
    std::cout << "Config: M=" << m_blocks << ", K=" << k_clusters << ", T=" << scann_T << std::endl;

    t.start();
    mvsic::ScaNN_Point_Range<PointRangeTy> scann_data(data_range, m_blocks, k_clusters, sub_mult,
                                                      scann_T);
    double build_time = t.next_time();

    std::cout << "ScaNN Build Time: " << build_time << "s (" << (n / build_time) << " vec/s)"
              << std::endl;
    run_verification(scann_data, data_range, n, "ScaNN");
    run_benchmark(scann_data, data_range, n, d);

  } else {
    // --- PQ ---
    std::cout << "\n=== Running Product Quantization (PQ) ===" << std::endl;
    if (d % m_blocks != 0) {
      std::cerr << "Error: Dimension " << d << " not divisible by M=" << m_blocks << std::endl;
      return 1;
    }
    std::cout << "Config: M=" << m_blocks << ", K=" << k_clusters << std::endl;

    t.start();
    mvsic::Quantized_Point_Range<PointRangeTy> pq_data(data_range, m_blocks, k_clusters, sub_mult);
    double build_time = t.next_time();

    std::cout << "PQ Build Time: " << build_time << "s (" << (n / build_time) << " vec/s)"
              << std::endl;
    run_verification(pq_data, data_range, n, "PQ");
    run_benchmark(pq_data, data_range, n, d);
  }

  return 0;
}