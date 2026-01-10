// bench_pq_fastscan.cpp
//
// Benchmarks (full scan distance computation):
//   1) Exact (unquantized) float L2 distance via efanna2e::DistanceL2 (NSGDist)
//   2) Naive PQ with K=16 (scalar LUT lookup)
//   3) FastScan (K=16) AVX-512 kernel
//
// Usage (Bazel):
//   PARLAY_NUM_THREADS=16 bazel run -c opt //:bench_pq_fastscan -- [N] [Q] [D] [pq_block]
//   [fs_block] [REPS]
//
// Defaults:
//   N=1,000,000  Q=1,000  D=128  pq_block=64  fs_block=64  REPS=2
//
// Notes:
// - FastScan requires AVX-512 (AVX512F + AVX512BW recommended).
// - PQ uses K=16 to match FastScan's K=16 exactly.
// - Metric is fixed to L2 (Metric=true). If you want IP, change constexpr Metric=false.

#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>
#include <random>
#include <algorithm>
#include <iostream>
#include <limits>

#include "parlay/primitives.h"
#include "parlay/parallel.h"
#include "parlay/sequence.h"

#include "mvsic/core/quantization/pq.h"
#include "mvsic/core/quantization/fastscan.h"
#include "mvsic/core/distance_measures/one_to_one.h"  // brings in NSGDist.h

#if !defined(__AVX512F__)
#warning "AVX-512 not enabled by compiler flags; fastscan benchmarks will be skipped."
#endif

// ---------------------------
// Timing helpers
// ---------------------------
struct Timer {
  using clock = std::chrono::steady_clock;
  clock::time_point t0;
  void start() { t0 = clock::now(); }
  double sec() const {
    auto t1 = clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
  }
};

static inline double ns_per_dist(double seconds, uint64_t num_dists) {
  return (seconds * 1e9) / double(num_dists);
}

static inline void print_scan_stats(const char* name, double best_s, uint64_t num_dists, size_t Q) {
  const double gdist_s = (double(num_dists) / best_s) / 1e9;
  const double nsdist = ns_per_dist(best_s, num_dists);
  const double ms_per_query = (best_s * 1e3) / double(Q);

  std::cout << name << ": total " << best_s << " s"
            << " | latency " << ms_per_query << " ms/query"
            << " | " << gdist_s << " Gdist/s"
            << " | " << nsdist << " ns/dist\n";
}

// ---------------------------
// Synthetic PointRange adapter
// ---------------------------
struct DensePointRange {
  std::vector<float> buf;  // contiguous [n * dim]
  size_t n = 0;
  uint32_t dim = 0;

  DensePointRange() = default;
  DensePointRange(size_t n_, uint32_t d_) : buf(n_ * size_t(d_)), n(n_), dim(d_) {}

  size_t size() const { return n; }
  uint32_t get_dims() const { return dim; }

  // pq/fastscan expect location(i) as uint8_t* pointing to float data.
  const uint8_t* location(size_t i) const {
    return reinterpret_cast<const uint8_t*>(buf.data() + i * size_t(dim));
  }

  const float* data() const { return buf.data(); }
  float* data() { return buf.data(); }
};

// Fill with N(0,1) and optionally L2-normalize.
static void fill_random(DensePointRange& r, uint64_t seed, bool l2_normalize) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> nd(0.0f, 1.0f);

  for (size_t i = 0; i < r.n; ++i) {
    float* v = r.data() + i * size_t(r.dim);
    float ss = 0.0f;
    for (uint32_t j = 0; j < r.dim; ++j) {
      float x = nd(rng);
      v[j] = x;
      ss += x * x;
    }
    if (l2_normalize) {
      float inv = 1.0f / std::sqrt(std::max(ss, 1e-12f));
      for (uint32_t j = 0; j < r.dim; ++j)
        v[j] *= inv;
    }
  }
}

// ---------------------------
// Full scan kernels (one query)
// ---------------------------

// Exact float L2 distance using efanna2e::DistanceL2 (from NSGDist.h).
static inline float full_scan_sum_exact_l2(const DensePointRange& db, const float* q, uint32_t D) {
  // Make sure we don't construct the distance functor per comparison.
  static thread_local efanna2e::DistanceL2 distfunc;

  float sum = 0.0f;
  const float* base = db.data();
  const size_t N = db.size();
  for (size_t i = 0; i < N; ++i) {
    const float* p = base + i * size_t(D);
    sum += distfunc.compare(q, p, D);
  }
  return sum;
}

// Full scan over PQ (scalar LUT + scalar code loop)
template<typename PQRange, typename PQQuery>
static inline float full_scan_sum_pq(const PQRange& db, const PQQuery& q) {
  float sum = 0.0f;
  const size_t N = db.size();
  for (size_t i = 0; i < N; ++i)
    sum += q.distance(db[i]);
  return sum;
}

#if defined(__AVX512F__)
// Full scan over FastScan: use scan_64_chunk for full strips, scalar tail.
template<typename FSRange, typename FSQuery>
static inline float full_scan_sum_fastscan(const FSRange& db, const FSQuery& q) {
  float sum = 0.0f;
  const size_t N = db.size();

  const size_t strip_stride = size_t(db.num_blocks) * 32;  // bytes per strip
  const size_t full = N & ~size_t(63);

  alignas(64) float out[64];

  // full strips
  for (size_t i = 0; i < full; i += 64) {
    const size_t strip = i / 64;
    const uint8_t* codes_ptr = db.packed_codes.data() + strip * strip_stride;
    db.scan_64_chunk(q, codes_ptr, out);
    for (int t = 0; t < 64; ++t)
      sum += out[t];
  }

  // scalar tail
  for (size_t i = full; i < N; ++i)
    sum += q.distance(db[i]);
  return sum;
}
#endif

// ---------------------------
// Query-parallel runners
// ---------------------------
static inline float scan_all_queries_exact_qpar(const DensePointRange& db,
                                                const DensePointRange& queries, uint32_t D,
                                                parlay::sequence<double>& per_q_sum) {
  const size_t Q = queries.size();
  const float* qbase = queries.data();

  parlay::parallel_for(0, Q, [&](size_t qi) {
    const float* q = qbase + qi * size_t(D);
    per_q_sum[qi] = double(full_scan_sum_exact_l2(db, q, D));
  });

  double total = 0.0;
  for (size_t i = 0; i < Q; ++i)
    total += per_q_sum[i];
  return float(total);
}

template<typename PQRange, typename PQQueryVec>
static inline float scan_all_queries_pq_qpar(const PQRange& db, const PQQueryVec& qvec, size_t Q,
                                             parlay::sequence<double>& per_q_sum) {
  parlay::parallel_for(0, Q,
                       [&](size_t qi) { per_q_sum[qi] = double(full_scan_sum_pq(db, qvec[qi])); });

  double total = 0.0;
  for (size_t i = 0; i < Q; ++i)
    total += per_q_sum[i];
  return float(total);
}

#if defined(__AVX512F__)
template<typename FSRange, typename FSQueryVec>
static inline float scan_all_queries_fastscan_qpar(const FSRange& db, const FSQueryVec& qvec,
                                                   size_t Q, parlay::sequence<double>& per_q_sum) {
  parlay::parallel_for(
      0, Q, [&](size_t qi) { per_q_sum[qi] = double(full_scan_sum_fastscan(db, qvec[qi])); });

  double total = 0.0;
  for (size_t i = 0; i < Q; ++i)
    total += per_q_sum[i];
  return float(total);
}
#endif

// ---------------------------
// Benchmark runner
// ---------------------------
int main(int argc, char** argv) {
  size_t N = (argc > 1 ? std::stoull(argv[1]) : 1'000'000);
  size_t Q = (argc > 2 ? std::stoull(argv[2]) : 1000);
  uint32_t D = (argc > 3 ? uint32_t(std::stoul(argv[3])) : 128);

  uint32_t pq_block = (argc > 4 ? uint32_t(std::stoul(argv[4])) : 64);
  uint32_t fs_block = (argc > 5 ? uint32_t(std::stoul(argv[5])) : 64);

  int REPS = (argc > 6 ? int(std::stoi(argv[6])) : 2);
  if (REPS < 1) REPS = 1;

  // L2 mode: matches your typical PQ usage for L2 LUTs.
  constexpr bool Metric = true;
  constexpr uint32_t K16 = 16;

  std::cout << "N=" << N << " Q=" << Q << " D=" << D << " pq_block=" << pq_block
            << " fs_block=" << fs_block << " K=16"
            << " (Metric=" << (Metric ? "L2" : "IP") << ")\n";

  if (D % pq_block != 0) {
    std::cerr << "ERROR: D not divisible by pq_block.\n";
    return 1;
  }
  if (D % fs_block != 0) {
    std::cerr << "ERROR: D not divisible by fs_block.\n";
    return 1;
  }

  DensePointRange db(N, D);
  DensePointRange queries(Q, D);

  fill_random(db, 12345, /*l2_normalize=*/false);
  fill_random(queries, 999, /*l2_normalize=*/false);

  Timer t;

  // ---------------------------
  // Build PQ (K=16)
  // ---------------------------
  t.start();
  mvsic::pq::Quantized_Point_Range<DensePointRange, Metric> pq16(db, pq_block, K16,
                                                                 /*subsample_mult=*/20);
  double pq_build_s = t.sec();

  // ---------------------------
  // Build FastScan
  // ---------------------------
  double fs_build_s = 0.0;
#if defined(__AVX512F__)
  std::vector<size_t> cloud_offsets_float = {0, size_t(N) * size_t(D)};
  t.start();
  mvsic::fastscan::Quantized_Point_Range<DensePointRange, Metric> fs(db, cloud_offsets_float,
                                                                     fs_block);
  fs_build_s = t.sec();
#endif

  std::cout << "\n=== Build / Encode time ===\n";
  std::cout << "PQ(K=16):        " << pq_build_s << " s\n";
#if defined(__AVX512F__)
  std::cout << "FastScan(K=16):  " << fs_build_s << " s\n";
#else
  std::cout << "FastScan(K=16):  (skipped, no AVX512)\n";
#endif
  std::cout << "Exact (float):   (no build)\n";

  // ---------------------------
  // LUT construction time
  // ---------------------------
  std::vector<mvsic::pq::Quantized_Query<Metric>> pq_q;
  pq_q.reserve(Q);

#if defined(__AVX512F__)
  std::vector<mvsic::fastscan::Quantized_Query<Metric>> fs_q;
  fs_q.reserve(Q);
#endif

  t.start();
  for (size_t i = 0; i < Q; ++i) {
    pq_q.push_back(pq16.quantize_query(queries.data() + i * size_t(D)));
  }
  double pq_lut_s = t.sec();

#if defined(__AVX512F__)
  t.start();
  for (size_t i = 0; i < Q; ++i) {
    fs_q.push_back(fs.quantize_query(queries.data() + i * size_t(D)));
  }
  double fs_lut_s = t.sec();
#endif

  std::cout << "\n=== Query LUT time (Q=" << Q << ") ===\n";
  std::cout << "PQ(K=16):        " << pq_lut_s << " s  (" << (pq_lut_s * 1e6 / Q) << " us/query)\n";
#if defined(__AVX512F__)
  std::cout << "FastScan(K=16):  " << fs_lut_s << " s  (" << (fs_lut_s * 1e6 / Q) << " us/query)\n";
#else
  std::cout << "FastScan(K=16):  (skipped, no AVX512)\n";
#endif
  std::cout << "Exact (float):   (no LUT)\n";

  // ---------------------------
  // Full scan kernel time (Q * N distances)
  // ---------------------------
  volatile float sink = 0.0f;

  // Warmup
  for (size_t i = 0; i < std::min<size_t>(Q, 2); ++i) {
    sink += full_scan_sum_exact_l2(db, queries.data() + i * size_t(D), D);
    sink += full_scan_sum_pq(pq16, pq_q[i]);
  }
#if defined(__AVX512F__)
  for (size_t i = 0; i < std::min<size_t>(Q, 2); ++i) {
    sink += full_scan_sum_fastscan(fs, fs_q[i]);
  }
#endif

  parlay::sequence<double> per_q_sum(Q, 0.0);
  const uint64_t num_dists = uint64_t(N) * uint64_t(Q);

  std::cout << "\n=== Full scan kernel (Q * N distances) ===\n";

  // Exact
  {
    double best = 1e100;
    float acc = 0.0f;
    for (int r = 0; r < REPS; ++r) {
      t.start();
      float local = scan_all_queries_exact_qpar(db, queries, D, per_q_sum);
      double s = t.sec();
      best = std::min(best, s);
      acc = local;
    }
    sink += acc;
    print_scan_stats("Exact float L2 (NSGDist)", best, num_dists, Q);
  }

  // PQ
  {
    double best = 1e100;
    float acc = 0.0f;
    for (int r = 0; r < REPS; ++r) {
      t.start();
      float local = scan_all_queries_pq_qpar(pq16, pq_q, Q, per_q_sum);
      double s = t.sec();
      best = std::min(best, s);
      acc = local;
    }
    sink += acc;
    print_scan_stats("PQ scalar K=16", best, num_dists, Q);
  }

  // FastScan
#if defined(__AVX512F__)
  {
    double best = 1e100;
    float acc = 0.0f;
    for (int r = 0; r < REPS; ++r) {
      t.start();
      float local = scan_all_queries_fastscan_qpar(fs, fs_q, Q, per_q_sum);
      double s = t.sec();
      best = std::min(best, s);
      acc = local;
    }
    sink += acc;
    print_scan_stats("FastScan K=16 (AVX-512)", best, num_dists, Q);
  }
#else
  std::cout << "FastScan K=16 (AVX-512): (skipped, no AVX512)\n";
#endif

  std::cout << "\n(sink=" << sink << ")\n";
  return 0;
}
