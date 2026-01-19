#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include <cmath>

#include "parlay/parallel.h"
#include "parlay/primitives.h"

#include "mvsic/core/utils/parse_command_line.h"
#include "algorithms/utils/euclidian_point.h"

// --- Your Data Types ---
#include "mvsic/core/types/point_cloud_set.h"
#include "mvsic/core/types/chamfer_l2_point.h"
#include "mvsic/core/types/chamfer_ip_point.h"

// --- Quantization Headers ---
#include "wrapper.h"
#include "pq.h"
#include "rabitq.h"
#include "scann.h"

using namespace mvsic;

// Configuration
using PointTy = ChamferL2_Point;
using PCSetTy = PointCloudSet<PointTy>;

// ---------------------------------------------------------
// Data Generation
// ---------------------------------------------------------
struct SyntheticData {
  std::vector<float> values;
  std::vector<size_t> offsets;
  std::vector<uint32_t> ids;
  uint32_t n_clouds;
  uint32_t dim;
};

SyntheticData generate_point_clouds(uint32_t n_clouds, uint32_t dim, uint32_t min_size,
                                    uint32_t max_size,
                                    int seed) {  // <--- Added Seed
  std::cout << "Generating " << n_clouds << " point clouds (Dim=" << dim << ", Seed=" << seed
            << ")..." << std::endl;

  SyntheticData data;
  data.n_clouds = n_clouds;
  data.dim = dim;
  data.offsets.push_back(0);

  std::mt19937 rng(seed);
  std::uniform_int_distribution<uint32_t> size_dist(min_size, max_size);

  size_t total_vectors = 0;

  for (uint32_t i = 0; i < n_clouds; ++i) {
    uint32_t sz = size_dist(rng);
    total_vectors += sz;
    data.offsets.push_back(total_vectors * dim);
    data.ids.push_back(i);
  }

  data.values.resize(total_vectors * dim);

  // Fill with random data [-1, 1]
  parlay::parallel_for(0, total_vectors * dim, [&](size_t i) {
    uint64_t r = parlay::hash64(i + (uint64_t)seed * 999999);
    float v = (float)((r % 20000) / 10000.0) - 1.0f;
    data.values[i] = v;
  });

  std::cout << "Total Vectors: " << total_vectors << std::endl;
  return data;
}

// ---------------------------------------------------------
// Test Runner
// ---------------------------------------------------------
template<typename QPCSet>
void run_tests(const QPCSet& q_pc_set, const PCSetTy& exact_set, const PCSetTy& query_set,
               uint32_t n_queries) {

  n_queries = std::min(n_queries, query_set.size());
  uint32_t target_idx = 0;

  // --- 1. Accuracy Verification ---
  std::cout << "\n--- Accuracy Verification (First 5 Queries) ---" << std::endl;
  printf("%-5s %-12s %-12s %-12s\n", "QIdx", "Exact", "Approx", "RelErr");

  double total_err_accum = 0;
  for (uint32_t i = 0; i < std::min(n_queries, (uint32_t)5); ++i) {
    float d_exact = query_set[i].distance(exact_set[target_idx]);

    // Note: PQ Returns Squared L2. Standard Chamfer uses L2 (Sqrt).
    // If your ChamferL2_Point returns regular L2, we must Square it to match PQ.
    // Or Sqrt the PQ result. Let's compare raw values first.

    auto q_query = q_pc_set.quantize_query(query_set[i]);
    float d_approx = q_query.distance(q_pc_set[target_idx]);

    // Heuristic: If approx is vastly larger than exact, maybe exact is Sqrt'd?
    // Let's assume standard metric behavior (comparing distances directly)

    float err = std::abs(d_exact - d_approx);
    float rel = (std::abs(d_exact) > 1e-6) ? (err / std::abs(d_exact)) : 0.0f;
    total_err_accum += rel;

    printf("%-5u %-12.4f %-12.4f %-12.4f\n", i, d_exact, d_approx, rel);
  }
  (void)total_err_accum;

  // --- 2. Performance Benchmark ---
  std::cout << "\n--- Performance Benchmark (" << n_queries << " queries) ---" << std::endl;

  // Pre-quantize queries to isolate scan time
  std::vector<decltype(q_pc_set.quantize_query(query_set[0]))> prepped_queries;
  prepped_queries.reserve(n_queries);
  for (uint32_t i = 0; i < n_queries; ++i) {
    prepped_queries.push_back(q_pc_set.quantize_query(query_set[i]));
  }

  auto start = std::chrono::high_resolution_clock::now();

  double total_checksum = 0;
  uint32_t scan_size = std::min(exact_set.size(), (uint32_t)100);

  // Verify bounds to prevent Segfault
  if (q_pc_set.n_clouds < scan_size) {
    std::cerr << "Error: DB Size smaller than scan size" << std::endl;
    return;
  }

  for (uint32_t i = 0; i < n_queries; ++i) {
    const auto& q_query = prepped_queries[i];
    for (uint32_t j = 0; j < scan_size; ++j) {
      // Accessing q_pc_set[j] creates a temp handle.
      // q_query.distance iterates this handle.
      total_checksum += q_query.distance(q_pc_set[j]);
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  double elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

  double total_pair_ops = n_queries * scan_size;
  double ms_per_pair = elapsed_ms / total_pair_ops;
  double qps = (total_pair_ops * 1000.0) / (elapsed_ms + 1e-9);

  std::cout << "Total Pairs Checked: " << total_pair_ops << std::endl;
  std::cout << "Total Time: " << elapsed_ms << " ms" << std::endl;
  std::cout << "Latency per Cloud-Pair: " << ms_per_pair << " ms" << std::endl;
  std::cout << "Throughput: " << qps << " pairs/sec" << std::endl;

  if (total_checksum == -1.0) std::cout << "";
}

int main(int argc, char** argv) {
  commandLine P(argc, argv,
                "[-n <n_clouds>] [-d <dim>] [-min <min_pts>] [-max <max_pts>] "
                "[-rabitq] [-rbits <bits>] [-scann] [-T <thresh>] "
                "[-m <blocks>] [-k <clusters>]");

  uint32_t n_clouds = P.getOptionIntValue("-n", 1000);
  uint32_t dim = P.getOptionIntValue("-d", 128);
  uint32_t min_sz = P.getOptionIntValue("-min", 10);
  uint32_t max_sz = P.getOptionIntValue("-max", 50);

  bool use_rabitq = P.getOption("-rabitq");
  bool use_scann = P.getOption("-scann");

  uint32_t rbits = P.getOptionIntValue("-rbits", 8);
  uint32_t m = P.getOptionIntValue("-m", 8);
  uint32_t k = P.getOptionIntValue("-k", 256);
  float T = P.getOptionDoubleValue("-T", 0.2);

  // 1. Generate Data (Using different seeds for DB and Query)
  auto gen_data = generate_point_clouds(n_clouds, dim, min_sz, max_sz, 123);
  auto q_gen_data = generate_point_clouds(50, dim, min_sz, max_sz, 999);

  // 2. Create PointCloudSets using the specific PointTy
  PCSetTy db_set(gen_data.n_clouds, gen_data.dim, gen_data.values.data(), gen_data.offsets.data(),
                 gen_data.ids.data());

  PCSetTy query_set(q_gen_data.n_clouds, q_gen_data.dim, q_gen_data.values.data(),
                    q_gen_data.offsets.data(), q_gen_data.ids.data());

  // 3. Build & Run
  if (use_rabitq) {
    std::cout << "\n=== Testing RaBitQ Point Clouds ===" << std::endl;
    using RaBitQ_Range = RaBitQ_Point_Range<FlattenedPCRange<PCSetTy>>;
    using RaBitQ_PCSet = Quantized_Point_Cloud_Set<RaBitQ_Range>;

    parlay::internal::timer t;
    t.start();
    RaBitQ_PCSet q_db(db_set, rbits);
    std::cout << "Build Time: " << t.next_time() << "s" << std::endl;
    run_tests(q_db, db_set, query_set, 50);

  } else if (use_scann) {
    std::cout << "\n=== Testing ScaNN Point Clouds ===" << std::endl;
    using ScaNN_Range = ScaNN_Point_Range<FlattenedPCRange<PCSetTy>>;
    using ScaNN_PCSet = Quantized_Point_Cloud_Set<ScaNN_Range>;

    parlay::internal::timer t;
    t.start();
    ScaNN_PCSet q_db(db_set, m, k, 20, T);
    std::cout << "Build Time: " << t.next_time() << "s" << std::endl;
    run_tests(q_db, db_set, query_set, 50);

  } else {
    std::cout << "\n=== Testing PQ Point Clouds ===" << std::endl;
    using PQ_Range = Quantized_Point_Range<FlattenedPCRange<PCSetTy>>;
    using PQ_PCSet = Quantized_Point_Cloud_Set<PQ_Range>;

    parlay::internal::timer t;
    t.start();
    PQ_PCSet q_db(db_set, m, k, 20);
    std::cout << "Build Time: " << t.next_time() << "s" << std::endl;
    run_tests(q_db, db_set, query_set, 50);
  }

  return 0;
}