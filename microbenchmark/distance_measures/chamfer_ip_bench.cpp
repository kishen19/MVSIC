// Microbenchmark: chamfer_ip_distance vs chamfer_ip_distance_opt
//
// Nested parallelism: num_tasks many query point clouds (each n_a x dim). The
// same num_docs document point clouds are used for every task. Outer parallel_for
// over tasks, inner parallel_for over documents. No granularity set; Parlay decides.
//
// Usage (Bazel):
//   Fixed n_b:  [num_tasks] [num_docs] [dim] [n_a] [n_b] [reps]
//   Variable:  [num_tasks] [num_docs] [dim] [n_a] [n_b_lo] [n_b_hi] [reps]
//
// Defaults: num_tasks=16  num_docs=10000  dim=128  n_a=32  n_b_lo=16  n_b_hi=128  reps=5
//
// Example: 16 10000 128 32 16 128 5  -> 16 tasks, same 10k docs per task, nested parallel_for

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include "mvsic/core/distance_measures/one_to_one.h"
#include "parlay/primitives.h"

static constexpr uint32_t DIM = 128;
static constexpr uint32_t N_A = 32;  // query size

// -----------------------------------------------------------------------------
// Random data
// -----------------------------------------------------------------------------
static void fill_random(float* p, size_t n, uint32_t seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> u(-1.f, 1.f);
  for (size_t i = 0; i < n; ++i)
    p[i] = u(gen);
}

// -----------------------------------------------------------------------------
// Timer
// -----------------------------------------------------------------------------
struct Timer {
  using clock = std::chrono::steady_clock;
  clock::time_point t0;
  void start() { t0 = clock::now(); }
  double sec() const { return std::chrono::duration<double>(clock::now() - t0).count(); }
};

static double median_time(std::vector<double>& times) {
  if (times.empty()) return 0;
  size_t mid = times.size() / 2;
  std::nth_element(times.begin(), times.begin() + mid, times.end());
  return times[mid];
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------
int main(int argc, char** argv) {
  size_t num_tasks = 16;
  size_t num_docs = 10000;
  uint32_t dim = DIM;
  uint32_t n_a = N_A;
  uint32_t n_b_lo = 16;
  uint32_t n_b_hi = 128;
  int reps = 5;
  bool variable_n_b = true;

  if (argc >= 2) num_tasks = static_cast<size_t>(std::atoi(argv[1]));
  if (argc >= 3) num_docs = static_cast<size_t>(std::atoi(argv[2]));
  if (argc >= 4) dim = static_cast<uint32_t>(std::atoi(argv[3]));
  if (argc >= 5) n_a = static_cast<uint32_t>(std::atoi(argv[4]));
  if (argc >= 6) n_b_lo = static_cast<uint32_t>(std::atoi(argv[5]));
  if (argc >= 7) reps = std::atoi(argv[6]);
  if (argc >= 8) {
    n_b_hi = static_cast<uint32_t>(std::atoi(argv[6]));
    reps = std::atoi(argv[7]);
    variable_n_b = (n_b_hi != n_b_lo);
  } else {
    n_b_hi = n_b_lo;
    variable_n_b = false;
  }

  std::cout << "Chamfer IP microbenchmark (nested parallel_for, same docs per task)\n"
            << "  num_tasks = " << num_tasks << " (query clouds " << n_a << " x " << dim << "), "
            << "num_docs = " << num_docs << " (shared across tasks)";
  if (variable_n_b) {
    std::cout << ", n_b in [" << n_b_lo << ", " << n_b_hi << "]";
  } else {
    std::cout << ", n_b = " << n_b_lo;
  }
  std::cout << ", reps = " << reps << "\n\n";

  // One query point cloud per task (num_tasks many n_a x dim)
  std::vector<float> a_data(num_tasks * n_a * dim);
  fill_random(a_data.data(), a_data.size(), 42);

  // One set of document point clouds (same for all tasks). Variable n_b per doc.
  std::vector<uint32_t> n_b_list(num_docs);
  std::mt19937 gen(43);
  std::uniform_int_distribution<uint32_t> u_nb(n_b_lo, n_b_hi > n_b_lo ? n_b_hi : n_b_lo);
  size_t total_b_floats = 0;
  for (size_t d = 0; d < num_docs; ++d) {
    n_b_list[d] = u_nb(gen);
    total_b_floats += n_b_list[d] * dim;
  }
  std::vector<float> b_data(total_b_floats);
  fill_random(b_data.data(), total_b_floats, 44);

  std::vector<size_t> b_offsets(num_docs + 1);
  b_offsets[0] = 0;
  for (size_t d = 0; d < num_docs; ++d) {
    b_offsets[d + 1] = b_offsets[d] + n_b_list[d] * dim;
  }

  // Sanity: compare one pair (task 0, doc 0)
  float ref = mvsic::chamfer_ip_distance(a_data.data(), n_a, b_data.data(), n_b_list[0], dim);
  float opt = mvsic::chamfer_ip_distance_opt(a_data.data(), n_a, b_data.data(), n_b_list[0], dim);
  float eigen_opt =
      mvsic::chamfer_ip_distance_eigen_opt(a_data.data(), n_a, b_data.data(), n_b_list[0], dim);
  if (std::fabs(ref - opt) > 1e-4f) {
    std::cerr << "Mismatch pair 0: ref=" << ref << " opt=" << opt << "\n";
  }
  if (std::fabs(ref - eigen_opt) > 1e-4f) {
    std::cerr << "Mismatch pair 0: ref=" << ref << " eigen_opt=" << eigen_opt << "\n";
  }

  std::cout << "Num Workers: " << parlay::num_workers() << std::endl;

  const size_t total_pairs = num_tasks * num_docs;

  auto run_original = [&]() {
    parlay::parallel_for(0, num_tasks, [&](size_t t) {
      float* a = a_data.data() + t * n_a * dim;
      parlay::parallel_for(0, num_docs, [&](size_t d) {
        float* b = b_data.data() + b_offsets[d];
        (void)mvsic::chamfer_ip_distance(a, n_a, b, n_b_list[d], dim);
      });
    });
  };

  auto run_opt = [&]() {
    parlay::parallel_for(0, num_tasks, [&](size_t t) {
      float* a = a_data.data() + t * n_a * dim;
      parlay::parallel_for(0, num_docs, [&](size_t d) {
        float* b = b_data.data() + b_offsets[d];
        (void)mvsic::chamfer_ip_distance_opt(a, n_a, b, n_b_list[d], dim);
      });
    });
  };

  auto run_eigen_opt = [&]() {
    parlay::parallel_for(0, num_tasks, [&](size_t t) {
      float* a = a_data.data() + t * n_a * dim;
      parlay::parallel_for(0, num_docs, [&](size_t d) {
        float* b = b_data.data() + b_offsets[d];
        (void)mvsic::chamfer_ip_distance_eigen_opt(a, n_a, b, n_b_list[d], dim);
      });
    });
  };

  std::vector<double> times_orig(reps), times_opt(reps), times_eigen_opt(reps);
  for (int r = 0; r < reps; ++r) {
    Timer t;
    t.start();
    run_original();
    times_orig[r] = t.sec();
  }
  for (int r = 0; r < reps; ++r) {
    Timer t;
    t.start();
    run_opt();
    times_opt[r] = t.sec();
  }
  for (int r = 0; r < reps; ++r) {
    Timer t;
    t.start();
    run_eigen_opt();
    times_eigen_opt[r] = t.sec();
  }

  double t_orig = median_time(times_orig);
  double t_opt = median_time(times_opt);
  double t_eigen_opt = median_time(times_eigen_opt);
  double best = std::min({t_orig, t_opt, t_eigen_opt});

  std::cout << std::fixed << std::setprecision(4);
  std::cout << "variant                    median_s    vs_best   pairs/s\n";
  std::cout << "------------------------------------------------------------\n";
  auto row = [&](const char* name, double t) {
    std::cout << std::setw(26) << name << " " << std::setw(10) << t << " " << std::setw(8)
              << (t / best) << "x  " << std::setw(12) << (total_pairs / t) << "\n";
  };
  row("chamfer_ip_distance", t_orig);
  row("chamfer_ip_distance_opt", t_opt);
  row("chamfer_ip_distance_eigen_opt", t_eigen_opt);
  std::cout << "\nBest: ";
  if (t_orig <= best) std::cout << "chamfer_ip_distance ";
  if (t_opt <= best) std::cout << "chamfer_ip_distance_opt ";
  if (t_eigen_opt <= best) std::cout << "chamfer_ip_distance_eigen_opt";
  std::cout << "\n";

  return 0;
}
