// Microbenchmark: OneToMany Chamfer distance — naive vs batched
//
// Runs num_tasks one-to-many computations in parallel; each is one query vs one
// PointCloudSet of m docs. Compares PCS::distances_naive vs PCS::distances.
//
// Usage (Bazel):
//   Fixed doc size:
//     bazel run -c opt //microbenchmark:one_to_many_bench -- [num_tasks] [m] [d] [q_size] [doc_size] [reps]
//   Variable doc size:
//     bazel run -c opt //microbenchmark:one_to_many_bench -- [num_tasks] [m] [d] [q_size] [doc_lo] [doc_hi] [reps]
//
// Defaults: num_tasks=16  m=500  d=128  q_size=32  doc_lo=16  doc_hi=128  reps=5
//
// Examples:
//   # 16 parallel leaves, each 500 docs, variable doc sizes
//   bazel run -c opt //microbenchmark:one_to_many_bench -- 16 500 128 32 16 128 5
//   # 1 task (no outer parallelism), 500 docs, fixed doc size
//   bazel run -c opt //microbenchmark:one_to_many_bench -- 1 500 128 32 128 5

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"

#include "mvsic/core/types/chamfer_ip_point.h"
#include "mvsic/core/types/point_cloud_set.h"

using ChPoint = mvsic::ChamferIP_Point;
using PCS = mvsic::PointCloudSet<ChPoint>;

// -----------------------------------------------------------------------------
// Timer
// -----------------------------------------------------------------------------
struct Timer {
  using clock = std::chrono::steady_clock;
  clock::time_point t0;
  void start() { t0 = clock::now(); }
  double sec() const { return std::chrono::duration<double>(clock::now() - t0).count(); }
};

// -----------------------------------------------------------------------------
// Synthetic data
// -----------------------------------------------------------------------------
static void fill_random(float* p, size_t n, uint32_t seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> u(-1.f, 1.f);
  for (size_t i = 0; i < n; ++i)
    p[i] = u(gen);
}

// Build PointCloudSet with n point clouds, dim d, each doc has doc_size vectors (fixed).
static PCS make_random_pcs_fixed(size_t n, uint32_t d, size_t doc_size, uint32_t seed) {
  size_t total = n * doc_size * d;
  std::vector<float> values(total);
  fill_random(values.data(), total, seed);

  std::vector<size_t> offsets(n + 1);
  for (size_t i = 0; i <= n; ++i)
    offsets[i] = i * doc_size * d;

  std::vector<uint32_t> ids(n);
  for (size_t i = 0; i < n; ++i)
    ids[i] = static_cast<uint32_t>(i);

  return PCS(static_cast<uint32_t>(n), d, values.data(), offsets.data(), ids.data());
}

// Build PointCloudSet with n point clouds, dim d; doc i has size in [doc_lo, doc_hi] (uniform).
static PCS make_random_pcs_variable(size_t n, uint32_t d, size_t doc_lo, size_t doc_hi,
                                    uint32_t seed) {
  std::mt19937 gen(seed);
  std::uniform_int_distribution<size_t> u(doc_lo, doc_hi > doc_lo ? doc_hi : doc_lo);
  std::vector<size_t> sizes(n);
  for (size_t i = 0; i < n; ++i)
    sizes[i] = u(gen);

  std::vector<size_t> offsets(n + 1);
  offsets[0] = 0;
  for (size_t i = 0; i < n; ++i)
    offsets[i + 1] = offsets[i] + sizes[i] * d;

  std::vector<float> values(offsets[n]);
  fill_random(values.data(), offsets[n], seed + 1);

  std::vector<uint32_t> ids(n);
  for (size_t i = 0; i < n; ++i)
    ids[i] = static_cast<uint32_t>(i);

  return PCS(static_cast<uint32_t>(n), d, values.data(), offsets.data(), ids.data());
}

// One query point cloud with q_size vectors.
static ChPoint make_query(uint32_t d, size_t q_size, uint32_t seed) {
  std::vector<float> buf(q_size * d);
  fill_random(buf.data(), buf.size(), seed);
  return ChPoint(static_cast<uint32_t>(q_size), d, buf.data(), 0);
}

// -----------------------------------------------------------------------------
// Run one variant, return median time in seconds over reps runs.
// -----------------------------------------------------------------------------
static double median_time(std::vector<double>& times) {
  if (times.empty()) return 0;
  size_t mid = times.size() / 2;
  std::nth_element(times.begin(), times.begin() + mid, times.end());
  return times[mid];
}

template<typename Func>
static double bench(Func&& f, int reps) {
  std::vector<double> times;
  times.reserve(reps);
  for (int r = 0; r < reps; ++r) {
    Timer t;
    t.start();
    f();
    times.push_back(t.sec());
  }
  return median_time(times);
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------
int main(int argc, char** argv) {
  size_t num_tasks = 16;
  size_t m = 500;
  uint32_t d = 128;
  size_t q_size = 32;
  size_t doc_lo = 16;
  size_t doc_hi = 128;
  int reps = 5;
  bool variable_sizes = true;

  if (argc >= 2) num_tasks = static_cast<size_t>(std::atoi(argv[1]));
  if (argc >= 3) m = static_cast<size_t>(std::atoi(argv[2]));
  if (argc >= 4) d = static_cast<uint32_t>(std::atoi(argv[3]));
  if (argc >= 5) q_size = static_cast<size_t>(std::atoi(argv[4]));
  if (argc >= 6) doc_lo = static_cast<size_t>(std::atoi(argv[5]));
  // 6 args: num_tasks m d q_size doc_size reps -> fixed
  // 7 args: num_tasks m d q_size doc_lo doc_hi reps -> variable
  if (argc >= 7) reps = std::atoi(argv[6]);
  if (argc >= 8) {
    doc_hi = static_cast<size_t>(std::atoi(argv[6]));
    reps = std::atoi(argv[7]);
    variable_sizes = (doc_hi != doc_lo);
  } else {
    doc_hi = doc_lo;
    variable_sizes = false;
  }

  std::cout << "OneToMany Chamfer (IP) microbenchmark\n"
            << "  num_tasks = " << num_tasks << ", m (docs per task) = " << m << ", d = " << d
            << ", q_size = " << q_size;
  if (variable_sizes) {
    std::cout << ", doc_size in [" << doc_lo << ", " << doc_hi << "] (variable)";
  } else {
    std::cout << ", doc_size = " << doc_lo << " (fixed)";
  }
  std::cout << ", reps = " << reps << "\n\n";

  // One block of query data for all tasks (num_tasks * q_size * d)
  std::vector<float> query_data(num_tasks * q_size * d);
  fill_random(query_data.data(), query_data.size(), 42);

  // One PointCloudSet per task
  std::vector<PCS> B_list(num_tasks);
  parlay::parallel_for(0, num_tasks, [&](size_t t) {
    B_list[t] = variable_sizes ? make_random_pcs_variable(m, d, doc_lo, doc_hi, 43 + t)
                               : make_random_pcs_fixed(m, d, doc_lo, 43 + t);
  });

  // One result buffer per task (flat: num_tasks * m)
  parlay::sequence<std::pair<uint32_t, float>> results(num_tasks * m);

  auto run_naive = [&]() {
    parlay::parallel_for(0, num_tasks, [&](size_t t) {
      ChPoint q_t(static_cast<uint32_t>(q_size), d,
                  query_data.data() + t * q_size * d, 0);
      B_list[t].distances_naive(q_t, results.data() + t * m);
    });
  };

  auto run_batched = [&]() {
    parlay::parallel_for(0, num_tasks, [&](size_t t) {
      ChPoint q_t(static_cast<uint32_t>(q_size), d,
                  query_data.data() + t * q_size * d, 0);
      B_list[t].distances(q_t, results.data() + t * m);
    });
  };

  // Sanity: compare naive vs batched on task 0
  run_naive();
  std::vector<float> ref(m);
  for (size_t i = 0; i < m; ++i)
    ref[i] = results[i].second;
  run_batched();
  for (size_t i = 0; i < m; ++i) {
    if (std::fabs(results[i].second - ref[i]) > 1e-4f) {
      std::cerr << "Mismatch task 0 at " << i << " naive=" << ref[i]
                << " batched=" << results[i].second << "\n";
    }
  }

  double t_naive = bench(run_naive, reps);
  double t_batched = bench(run_batched, reps);

  std::cout << std::fixed << std::setprecision(4);
  std::cout << "variant                median_s    vs_best\n";
  std::cout << "--------------------------------------------\n";
  double best = std::min(t_naive, t_batched);
  auto row = [&](const char* name, double t) {
    std::cout << std::setw(22) << name << " " << std::setw(10) << t << " " << std::setw(8)
              << (t / best) << "x\n";
  };
  row("PCS::distances_naive", t_naive);
  row("PCS::distances", t_batched);
  std::cout << "\nBest: " << (t_naive <= best ? "PCS::distances_naive" : "PCS::distances") << "\n";

  return 0;
}
