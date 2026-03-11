// bench_pq_fastscan.cpp
//
// Benchmarks (full scan distance computation):
//   1) Exact (unquantized) float IP distance via efanna2e::DistanceInnerProduct (NSGDist)
//   2) FastScan (K=16) using Quantized_Query::distances_all
//   3) RaBitQ using Quantized_Query::distances_all
//   4) TQ-4bit: Quantized_Query::distances_contiguous (batch strip scan)
//   5) Byte TQ (int8): per-point distance only (no batch API)
//   6) TQ-PQ-4bit: Quantized_Query::distances_all (FastScan-style shuffle scan)
//
// Usage (Bazel):
//   ... -- [-i <db_file>] [-q <query_file>] [-N <n>] [-Q <q>] [-D <d>]
//          [-fs_block <b>] [-rbits <b>] [-reps <r>] [-seed_db <s>] [-seed_q <s>]
//
// With -i and -q: load real data (PointRange binary = uint32 n, uint32 d, n*d floats).
// Without: synthetic data. Defaults: N=1000000, Q=1000, D=128, fs_block=64, rbits=8, reps=2.
//
// Notes:
// - FastScan requires AVX-512 (AVX512F + AVX512BW recommended).
// - Metric is fixed to IP (Metric=false). Distance = -inner_product (smaller = more similar).
// - Since distances_all() is parlay-parallel internally, we run queries serially to avoid nested
//   parallelism.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"

#include "mvsic/core/distance_measures/one_to_one.h"  // brings in NSGDist.h
#include "mvsic/core/quantization/fastscan.h"
#include "mvsic/core/quantization/rabitq.h"
#include "mvsic/core/quantization/turboquant_4bit.h"
#include "mvsic/core/quantization/turboquant_byte.h"
#include "mvsic/core/quantization/turboquant_pq_4bit.h"
#include "mvsic/core/types/ip_point.h"
#include "mvsic/core/types/point_range.h"
#include "mvsic/core/utils/parse_command_line.h"

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

struct MethodStats {
  std::string name;
  double best_s = 0.0;
};

static inline void print_scan_stats_table(const std::vector<MethodStats>& rows, uint64_t num_dists,
                                          size_t Q) {
  if (rows.empty()) return;
  const double exact_s = rows.front().best_s;

  std::cout << "\n=== Scan performance summary ===\n";
  std::cout << "Ndist = " << num_dists << ", Q = " << Q << "\n\n";

  std::cout << std::left << std::setw(30) << "Method" << std::right << std::setw(12) << "time [s]"
            << std::setw(14) << "ms / query" << std::setw(14) << "Gdist/s" << std::setw(14)
            << "ns / dist" << std::setw(14) << "speedup\n";
  std::cout << std::string(30 + 12 + 14 * 4, '-') << "\n";

  for (const auto& r : rows) {
    const double t = r.best_s;
    const double gdist_s = (double(num_dists) / t) / 1e9;
    const double nsdist = ns_per_dist(t, num_dists);
    const double ms_per_query = (t * 1e3) / double(Q);
    const double speedup = exact_s / t;

    std::cout << std::left << std::setw(30) << r.name << std::right << std::setw(12) << std::fixed
              << std::setprecision(4) << t << std::setw(14) << std::setprecision(3) << ms_per_query
              << std::setw(14) << std::setprecision(3) << gdist_s << std::setw(14)
              << std::setprecision(3) << nsdist << std::setw(14) << std::setprecision(3) << speedup
              << "\n";
  }
  std::cout << "\n";
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

  // pq/fastscan/rabitq expect location(i) as uint8_t* pointing to float data.
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
// Exact: DB-parallel sum for one query (IP: distance = -inner_product)
// Works with any range that has size(), get_dims(), location(i).
// ---------------------------
template<typename DBRange>
static inline float full_scan_sum_exact_ip_dbpar(const DBRange& db, const float* q, uint32_t D,
                                                 size_t grain = 4096) {
  const size_t N = db.size();
  const size_t num_blocks = (N + grain - 1) / grain;
  parlay::sequence<double> block_sums(num_blocks, 0.0);

  parlay::parallel_for(0, num_blocks, [&](size_t bi) {
    static thread_local efanna2e::DistanceInnerProduct distfunc;
    const size_t s = bi * grain;
    const size_t e = std::min(N, s + grain);
    double local = 0.0;
    for (size_t i = s; i < e; ++i) {
      const float* p = reinterpret_cast<const float*>(db.location(i));
      local += double(-distfunc.compare(q, p, D));  // distance = -IP
    }
    block_sums[bi] = local;
  });

  double total = 0.0;
  for (size_t bi = 0; bi < num_blocks; ++bi)
    total += block_sums[bi];
  return float(total);
}

template<typename DBRange, typename QRange>
static inline float scan_all_queries_exact_serial(const DBRange& db, const QRange& queries,
                                                  uint32_t D, size_t Q,
                                                  parlay::sequence<double>& per_q_sum) {
  for (size_t qi = 0; qi < Q; ++qi) {
    const float* q = reinterpret_cast<const float*>(queries.location(qi));
    per_q_sum[qi] = double(full_scan_sum_exact_ip_dbpar(db, q, D));
  }
  double total = 0.0;
  for (size_t i = 0; i < Q; ++i)
    total += per_q_sum[i];
  return float(total);
}

template<typename EncRange, typename QueryVec>
static inline float scan_all_queries_distances_all_serial(const EncRange& db, const QueryVec& qvec,
                                                          size_t Q, float* scratch,
                                                          size_t sum_N_unpadded,
                                                          parlay::sequence<double>& per_q_sum) {
  for (size_t qi = 0; qi < Q; ++qi) {
    qvec[qi].distances_all(db, scratch);

    double s = 0.0;
    for (size_t i = 0; i < sum_N_unpadded; ++i)
      s += double(scratch[i]);

    per_q_sum[qi] = s;
  }

  double total = 0.0;
  for (size_t i = 0; i < Q; ++i)
    total += per_q_sum[i];
  return float(total);
}

// TQ-PQ-4bit: use distances_all (parallel over strips, like FastScan).
template<typename TQEnc, typename TQQueryVec>
static inline float scan_all_queries_tqpq_serial(const TQEnc& enc, const TQQueryVec& qvec, size_t Q,
                                                 float* scratch, size_t N,
                                                 parlay::sequence<double>& per_q_sum) {
  for (size_t qi = 0; qi < Q; ++qi) {
    qvec[qi].distances_all(enc, scratch);
    double s = 0.0;
    for (size_t i = 0; i < N; ++i)
      s += double(scratch[i]);
    per_q_sum[qi] = s;
  }
  double total = 0.0;
  for (size_t i = 0; i < Q; ++i)
    total += per_q_sum[i];
  return float(total);
}

// TQ-4bit: use distances_contiguous_parallel when available (parallel over strips; no generic
// distances_all(enc) API).
template<typename TQEnc, typename TQQueryVec>
static inline float scan_all_queries_tq4bit_serial(const TQEnc& enc, const TQQueryVec& qvec,
                                                   size_t Q, float* scratch, size_t N,
                                                   parlay::sequence<double>& per_q_sum) {
  for (size_t qi = 0; qi < Q; ++qi) {
    qvec[qi].distances_contiguous_parallel(enc.packed_codes.data(), enc.norm_scaling_factors.data(),
                                           enc.unquantized_squared_norms.data(), enc.stride, N,
                                           scratch);
    double s = 0.0;
    for (size_t i = 0; i < N; ++i)
      s += double(scratch[i]);
    per_q_sum[qi] = s;
  }
  double total = 0.0;
  for (size_t i = 0; i < Q; ++i)
    total += per_q_sum[i];
  return float(total);
}

// Byte TQ: per-point distance only (no batch distances_all API).
template<typename BTQEnc, typename BTQQueryVec>
static inline float scan_all_queries_byte_tq_serial(const BTQEnc& enc, const BTQQueryVec& qvec,
                                                    size_t Q, float* scratch, size_t N,
                                                    parlay::sequence<double>& per_q_sum) {
  for (size_t qi = 0; qi < Q; ++qi) {
    parlay::parallel_for(0, N, [&](size_t i) { scratch[i] = qvec[qi].distance(enc[i]); });
    double s = 0.0;
    for (size_t i = 0; i < N; ++i)
      s += double(scratch[i]);
    per_q_sum[qi] = s;
  }
  double total = 0.0;
  for (size_t i = 0; i < Q; ++i)
    total += per_q_sum[i];
  return float(total);
}

// ---------------------------
// Benchmark runner (template: works with DensePointRange or PointRange<float, IPPoint>)
// ---------------------------
template<typename DBRange, typename QRange>
static int run_bench(const DBRange& db, const QRange& queries, uint32_t fs_block, size_t rbits,
                     int REPS, bool run_rabitq) {
  const size_t N = db.size();
  const size_t Q_total = queries.size();
  const size_t Q = std::min<size_t>(Q_total, 100);
  const uint32_t D = static_cast<uint32_t>(db.get_dims());
  constexpr bool Metric = false;

  std::cout << "N=" << N << " Q=" << Q << " D=" << D << " fs_block=" << fs_block
            << " rbits=" << rbits << " (Metric=IP)\n";
  if (Q_total > Q) {
    std::cout << "  (subsampling " << Q << " / " << Q_total << " queries)\n";
  }

  if (D % fs_block != 0) {
    std::cerr << "ERROR: D not divisible by fs_block.\n";
    return 1;
  }

  Timer t;

  // ---------------------------
  // Build FastScan model + encode (AVX-512 / AVX2 / VNNI)
  // ---------------------------
  double fs_build_s = 0.0;
#if defined(__AVX512F__) || defined(__AVX2__)
  t.start();
  mvsic::fastscan::Model<Metric> fs_model;
  fs_model.train(db, fs_block);
  auto fs = fs_model.encode(db);
  fs_build_s = t.sec();
#endif

  // ---------------------------
  // Build RaBitQ model + encode (optional, controlled by -rabitq flag)
  // ---------------------------
  mvsic::rabitq::Model<Metric> rq_model;
  using RQ_DB = decltype(rq_model.encode(db));
  RQ_DB rq;
  double rq_build_s = 0.0;
  if (run_rabitq) {
    t.start();
    rq_model.train(db, rbits);
    rq = rq_model.encode(db);
    rq_build_s = t.sec();
  }

  // ---------------------------
  // Build turboquant_4bit (4-bit TQ) model + encode
  // ---------------------------
  t.start();
  mvsic::turboquant_4bit::Model<Metric> tq_model;
  tq_model.train(db);
  auto tq = tq_model.encode(db);
  double tq_build_s = t.sec();

  // ---------------------------
  // Build turboquant_byte (int8) model + encode
  // ---------------------------
  t.start();
  mvsic::turboquant_byte::Model<Metric> btq_model;
  btq_model.train(db);
  auto btq = btq_model.encode(db);
  double btq_build_s = t.sec();

  // TurboQuant PQ 4-bit: build + LUT + scan for multiple block sizes. Requires AVX-512.
  static constexpr uint32_t tqpq_blocks[] = {1, 2, 4, 8};
  static constexpr size_t num_tqpq = sizeof(tqpq_blocks) / sizeof(tqpq_blocks[0]);
  double tqpq_build_s[num_tqpq] = {0.0, 0.0, 0.0, 0.0};
  double tqpq_lut_s[num_tqpq] = {0.0, 0.0, 0.0, 0.0};
  double tqpq_scan_best_s[num_tqpq] = {0.0, 0.0, 0.0, 0.0};
  parlay::sequence<double> tqpq_per_q_sum(Q, 0.0);
  volatile float tqpq_sink = 0.0f;
#if defined(__AVX512F__)
  for (size_t bi = 0; bi < num_tqpq; ++bi) {
    const uint32_t bsz = tqpq_blocks[bi];
    t.start();
    if (bsz == 1) {
      mvsic::turboquant_pq_4bit::Model<Metric, 1> tqpq_m;
      tqpq_m.train(db);
      auto tqpq_enc = tqpq_m.encode(db);
      tqpq_build_s[bi] = t.sec();
      std::vector<mvsic::turboquant_pq_4bit::Quantized_Query<Metric, 1>> tqpq_q;
      tqpq_q.reserve(Q);
      t.start();
      for (size_t i = 0; i < Q; ++i)
        tqpq_q.push_back(
            tqpq_m.quantize_query(reinterpret_cast<const float*>(queries.location(i))));
      tqpq_lut_s[bi] = t.sec();
      std::vector<float> tqpq_scratch(N);
      double best = 1e100;
      for (int r = 0; r < REPS; ++r) {
        t.start();
        float acc = scan_all_queries_tqpq_serial(tqpq_enc, tqpq_q, Q, tqpq_scratch.data(), N,
                                                 tqpq_per_q_sum);
        best = std::min(best, t.sec());
        tqpq_sink += acc;
      }
      tqpq_scan_best_s[bi] = best;
    } else if (bsz == 2) {
      mvsic::turboquant_pq_4bit::Model<Metric, 2> tqpq_m;
      tqpq_m.train(db);
      auto tqpq_enc = tqpq_m.encode(db);
      tqpq_build_s[bi] = t.sec();
      std::vector<mvsic::turboquant_pq_4bit::Quantized_Query<Metric, 2>> tqpq_q;
      tqpq_q.reserve(Q);
      t.start();
      for (size_t i = 0; i < Q; ++i)
        tqpq_q.push_back(
            tqpq_m.quantize_query(reinterpret_cast<const float*>(queries.location(i))));
      tqpq_lut_s[bi] = t.sec();
      std::vector<float> tqpq_scratch(N);
      double best = 1e100;
      for (int r = 0; r < REPS; ++r) {
        t.start();
        float acc = scan_all_queries_tqpq_serial(tqpq_enc, tqpq_q, Q, tqpq_scratch.data(), N,
                                                 tqpq_per_q_sum);
        best = std::min(best, t.sec());
        tqpq_sink += acc;
      }
      tqpq_scan_best_s[bi] = best;
    } else if (bsz == 4) {
      mvsic::turboquant_pq_4bit::Model<Metric, 4> tqpq_m;
      tqpq_m.train(db);
      auto tqpq_enc = tqpq_m.encode(db);
      tqpq_build_s[bi] = t.sec();
      std::vector<mvsic::turboquant_pq_4bit::Quantized_Query<Metric, 4>> tqpq_q;
      tqpq_q.reserve(Q);
      t.start();
      for (size_t i = 0; i < Q; ++i)
        tqpq_q.push_back(
            tqpq_m.quantize_query(reinterpret_cast<const float*>(queries.location(i))));
      tqpq_lut_s[bi] = t.sec();
      std::vector<float> tqpq_scratch(N);
      double best = 1e100;
      for (int r = 0; r < REPS; ++r) {
        t.start();
        float acc = scan_all_queries_tqpq_serial(tqpq_enc, tqpq_q, Q, tqpq_scratch.data(), N,
                                                 tqpq_per_q_sum);
        best = std::min(best, t.sec());
        tqpq_sink += acc;
      }
      tqpq_scan_best_s[bi] = best;
    } else if (bsz == 8) {
      mvsic::turboquant_pq_4bit::Model<Metric, 8> tqpq_m;
      tqpq_m.train(db);
      auto tqpq_enc = tqpq_m.encode(db);
      tqpq_build_s[bi] = t.sec();
      std::vector<mvsic::turboquant_pq_4bit::Quantized_Query<Metric, 8>> tqpq_q;
      tqpq_q.reserve(Q);
      t.start();
      for (size_t i = 0; i < Q; ++i)
        tqpq_q.push_back(
            tqpq_m.quantize_query(reinterpret_cast<const float*>(queries.location(i))));
      tqpq_lut_s[bi] = t.sec();
      std::vector<float> tqpq_scratch(N);
      double best = 1e100;
      for (int r = 0; r < REPS; ++r) {
        t.start();
        float acc = scan_all_queries_tqpq_serial(tqpq_enc, tqpq_q, Q, tqpq_scratch.data(), N,
                                                 tqpq_per_q_sum);
        best = std::min(best, t.sec());
        tqpq_sink += acc;
      }
      tqpq_scan_best_s[bi] = best;
    }
  }
#endif  // __AVX512F__

  std::cout << "\n=== Build / Encode time ===\n";
#if defined(__AVX512F__) || defined(__AVX2__)
  std::cout << "FastScan(K=16):   " << fs_build_s << " s\n";
#else
  std::cout << "FastScan(K=16):  (skipped, no AVX512/AVX2)\n";
#endif
  if (run_rabitq) {
    std::cout << "RaBitQ(bits=" << rbits << "): " << rq_build_s << " s\n";
  } else {
    std::cout << "RaBitQ:          (skipped, pass -rabitq to enable)\n";
  }
  std::cout << "TQ-4bit(K=16):    " << tq_build_s << " s\n";
  std::cout << "Byte TQ:          " << btq_build_s << " s\n";
#if defined(__AVX512F__)
  for (size_t bi = 0; bi < num_tqpq; ++bi) {
    std::cout << "TQ-PQ(K=16,B=" << tqpq_blocks[bi] << "): " << tqpq_build_s[bi] << " s\n";
  }
#endif
  std::cout << "Exact (NSGDist):  (no build)\n";

  // ---------------------------
  // LUT construction time
  // ---------------------------
#if defined(__AVX512F__) || defined(__AVX2__)
  std::vector<mvsic::fastscan::Quantized_Query<Metric>> fs_q;
  fs_q.reserve(Q);
#endif

  std::vector<mvsic::rabitq::Quantized_Query<Metric>> rq_q;
  if (run_rabitq) rq_q.reserve(Q);

  std::vector<mvsic::turboquant_4bit::Quantized_Query<Metric>> tq_q;
  tq_q.reserve(Q);

  std::vector<mvsic::turboquant_byte::Quantized_Query<Metric>> btq_q;
  btq_q.reserve(Q);

#if defined(__AVX512F__) || defined(__AVX2__)
  t.start();
  for (size_t i = 0; i < Q; ++i) {
    fs_q.push_back(fs_model.quantize_query(reinterpret_cast<const float*>(queries.location(i))));
  }
  double fs_lut_s = t.sec();
#endif

  double rq_lut_s = 0.0;
  if (run_rabitq) {
    t.start();
    for (size_t i = 0; i < Q; ++i) {
      rq_q.push_back(rq_model.quantize_query(reinterpret_cast<const float*>(queries.location(i))));
    }
    rq_lut_s = t.sec();
  }

  t.start();
  for (size_t i = 0; i < Q; ++i) {
    tq_q.push_back(tq_model.quantize_query(reinterpret_cast<const float*>(queries.location(i))));
  }
  double tq_lut_s = t.sec();

  t.start();
  for (size_t i = 0; i < Q; ++i) {
    btq_q.push_back(btq_model.quantize_query(reinterpret_cast<const float*>(queries.location(i))));
  }
  double btq_lut_s = t.sec();

  const uint64_t num_dists = uint64_t(N) * uint64_t(Q);

  std::cout << "\n=== Query LUT time (Q=" << Q << ") ===\n";
#if defined(__AVX512F__) || defined(__AVX2__)
  std::cout << "FastScan(K=16):  " << fs_lut_s << " s  (" << (fs_lut_s * 1e6 / Q) << " us/query)\n";
#else
  std::cout << "FastScan(K=16):  (skipped, no AVX512/AVX2)\n";
#endif
  if (run_rabitq) {
    std::cout << "RaBitQ:          " << rq_lut_s << " s  (" << (rq_lut_s * 1e6 / Q)
              << " us/query)\n";
  } else {
    std::cout << "RaBitQ:          (skipped, pass -rabitq to enable)\n";
  }
  std::cout << "TQ-4bit: " << tq_lut_s << " s  (" << (tq_lut_s * 1e6 / Q) << " us/query)\n";
  std::cout << "Byte TQ:  " << btq_lut_s << " s  (" << (btq_lut_s * 1e6 / Q) << " us/query)\n";
#if defined(__AVX512F__)
  for (size_t bi = 0; bi < num_tqpq; ++bi) {
    std::cout << "TQ-PQ-16-" << tqpq_blocks[bi] << ": " << tqpq_lut_s[bi] << " s  ("
              << (tqpq_lut_s[bi] * 1e6 / Q) << " us/query)\n";
  }
#endif
  std::cout << "Exact (float IP): (no LUT)\n";

  // ---------------------------
  // Full scan kernel time (Q * N distances)
  // ---------------------------
  volatile float sink = 0.0f;

  // scratch buffers for distances_all() / distances_contiguous()
  std::vector<float> rq_out(N);
  std::vector<float> tq_out(N);
  std::vector<float> btq_out(N);

#if defined(__AVX512F__) || defined(__AVX2__)
  const size_t fs_N_total = static_cast<size_t>(fs.size());
  std::vector<float> fs_out(std::max(fs_N_total, N));
#endif

  // Warmup (avoid cold-start effects)
  for (size_t i = 0; i < std::min<size_t>(Q, 2); ++i) {
    sink +=
        full_scan_sum_exact_ip_dbpar(db, reinterpret_cast<const float*>(queries.location(i)), D);

#if defined(__AVX512F__) || defined(__AVX2__)
    fs_q[i].distances_all(fs, fs_out.data());
    for (size_t j = 0; j < N; ++j)
      sink += fs_out[j];
#endif

    rq_q[i].distances_all(rq, rq_out.data());
    for (size_t j = 0; j < N; ++j)
      sink += rq_out[j];

#if defined(__AVX512F__)
    tq_q[i].distances_contiguous(tq.packed_codes.data(), tq.norm_scaling_factors.data(),
                                 tq.unquantized_squared_norms.data(), tq.stride, N, tq_out.data());
#else
    parlay::parallel_for(0, N, [&](size_t j) { tq_out[j] = tq_q[i].distance(tq[j]); });
#endif
    for (size_t j = 0; j < N; ++j)
      sink += tq_out[j];

    parlay::parallel_for(0, N, [&](size_t j) { btq_out[j] = btq_q[i].distance(btq[j]); });
    for (size_t j = 0; j < N; ++j)
      sink += btq_out[j];
  }
#if defined(__AVX512F__)
  sink += tqpq_sink;
#endif

  parlay::sequence<double> per_q_sum(Q, 0.0);

  std::cout << "\n=== Full scan kernel (Q * N distances) ===\n";
  std::cout << "(queries run serially; distances_all() is parlay-parallel inside)\n";

  std::vector<MethodStats> stats;

  // Exact
  {
    double best = 1e100;
    float acc = 0.0f;
    for (int r = 0; r < REPS; ++r) {
      t.start();
      float local = scan_all_queries_exact_serial(db, queries, D, Q, per_q_sum);
      double s = t.sec();
      best = std::min(best, s);
      acc = local;
    }
    sink += acc;
    stats.push_back({"Exact (NSGDist)", best});
  }

  // FastScan (AVX-512 / AVX2 / VNNI)
#if defined(__AVX512F__) || defined(__AVX2__)
  {
    double best = 1e100;
    float acc = 0.0f;
    for (int r = 0; r < REPS; ++r) {
      t.start();
      float local = scan_all_queries_distances_all_serial(fs, fs_q, Q, fs_out.data(), N, per_q_sum);
      double s = t.sec();
      best = std::min(best, s);
      acc = local;
    }
    sink += acc;
    stats.push_back({"FastScan (K=16)", best});
  }
#else
  std::cout << "FastScan K=16: (skipped, no AVX512/AVX2)\n";
#endif

  // RaBitQ (optional)
  if (run_rabitq) {
    double best = 1e100;
    float acc = 0.0f;
    for (int r = 0; r < REPS; ++r) {
      t.start();
      float local = scan_all_queries_distances_all_serial(rq, rq_q, Q, rq_out.data(), N, per_q_sum);
      double s = t.sec();
      best = std::min(best, s);
      acc = local;
    }
    sink += acc;
    stats.push_back({"RaBitQ", best});
  }

  // TQ-4bit: distances_contiguous when AVX512, else per-point (no batch API without AVX512)
  {
    double best = 1e100;
    float acc = 0.0f;
    for (int r = 0; r < REPS; ++r) {
      t.start();
#if defined(__AVX512F__)
      float local = scan_all_queries_tq4bit_serial(tq, tq_q, Q, tq_out.data(), N, per_q_sum);
#else
      float local = scan_all_queries_byte_tq_serial(tq, tq_q, Q, tq_out.data(), N, per_q_sum);
#endif
      double s = t.sec();
      best = std::min(best, s);
      acc = local;
    }
    sink += acc;
    stats.push_back({"TQ-4bit (K=16)", best});
  }

  // Byte TQ: per-point distance
  {
    double best = 1e100;
    float acc = 0.0f;
    for (int r = 0; r < REPS; ++r) {
      t.start();
      float local = scan_all_queries_byte_tq_serial(btq, btq_q, Q, btq_out.data(), N, per_q_sum);
      double s = t.sec();
      best = std::min(best, s);
      acc = local;
    }
    sink += acc;
    stats.push_back({"Byte TQ", best});
  }

  // TQ-PQ-4bit (precomputed in build section)
#if defined(__AVX512F__)
  for (size_t bi = 0; bi < num_tqpq; ++bi) {
    stats.push_back(
        {"TQ-PQ (K=16, B=" + std::to_string(tqpq_blocks[bi]) + ")", tqpq_scan_best_s[bi]});
  }
#endif

  if (!stats.empty()) print_scan_stats_table(stats, num_dists, Q);

  // ---------------------------
  // Optional: correctness check (TQ 4-bit vs exact for a small subset)
  // ---------------------------
  const size_t check_queries = std::min<size_t>(10, Q);
  const size_t check_db = std::min<size_t>(5000, N);
  double tq_max_ae = 0.0;
  double tq_mean_ae = 0.0;
  uint64_t check_count = 0;
  for (size_t qi = 0; qi < check_queries; ++qi) {
    tq_q[qi].distances_contiguous(tq.packed_codes.data(), tq.norm_scaling_factors.data(),
                                  tq.unquantized_squared_norms.data(), tq.stride, N, tq_out.data());
    const float* q = reinterpret_cast<const float*>(queries.location(qi));
    for (size_t i = 0; i < check_db; ++i) {
      static thread_local efanna2e::DistanceInnerProduct distfunc;
      float exact = -distfunc.compare(q, reinterpret_cast<const float*>(db.location(i)),
                                      D);  // distance = -IP
      float approx = tq_out[i];
      double ae = std::fabs(static_cast<double>(exact) - static_cast<double>(approx));
      tq_max_ae = std::max(tq_max_ae, ae);
      tq_mean_ae += ae;
      check_count++;
    }
  }
  if (check_count > 0) {
    tq_mean_ae /= static_cast<double>(check_count);
    std::cout << "\n=== Correctness (TurboQuant 4b vs exact IP, sample " << check_queries << " q x "
              << check_db << " db) ===\n";
    std::cout << "TQ 4b max |approx - exact|: " << tq_max_ae
              << "  mean |approx - exact|: " << tq_mean_ae << "\n";
  }

  std::cout << "\n(sink=" << sink << ")\n";
  return 0;
}

int main(int argc, char** argv) {
  using namespace mvsic;
  commandLine P(argc, argv,
                "[-i <db_file>] [-q <query_file>] "
                "[-N <n>] [-Q <q>] [-D <d>] [-seed_db <s>] [-seed_q <s>] "
                "[-fs_block <b>] [-rbits <b>] [-reps <r>] "
                "[-rabitq]");

  // Common options
  uint32_t fs_block = static_cast<uint32_t>(P.getOptionIntValue("-fs_block", 64));
  size_t rbits = static_cast<size_t>(P.getOptionLongValue("-rbits", 8));
  int reps = std::max(1, P.getOptionIntValue("-reps", 2));
  bool run_rabitq = P.getOption("-rabitq");

  char* db_file = P.getOptionValue("-i");
  char* q_file = P.getOptionValue("-q");

  if (db_file && q_file) {
    // Real-world input (PointRange format: binary = uint32 n, uint32 d, n*d floats)
    mvsic::PointRange<float, mvsic::IP_Point<float>> db(db_file);
    mvsic::PointRange<float, mvsic::IP_Point<float>> queries(q_file);
    if (queries.get_dims() != db.get_dims()) {
      std::cerr << "ERROR: DB and query dimensions differ.\n";
      return 1;
    }
    return run_bench(db, queries, fs_block, rbits, reps, run_rabitq);
  }

  if (db_file || q_file) {
    std::cerr << "ERROR: Provide both -i <db_file> and -q <query_file> for file mode.\n";
    return 1;
  }

  // Synthetic mode: all -flag
  size_t N = static_cast<size_t>(P.getOptionLongValue("-N", 1000000));
  size_t Q = static_cast<size_t>(P.getOptionLongValue("-Q", 1000));
  uint32_t D = static_cast<uint32_t>(P.getOptionIntValue("-D", 128));
  uint64_t seed_db = 12345ULL;
  uint64_t seed_q = 999ULL;
  if (char* s = P.getOptionValue("-seed_db")) seed_db = static_cast<uint64_t>(std::stoull(s));
  if (char* s = P.getOptionValue("-seed_q")) seed_q = static_cast<uint64_t>(std::stoull(s));

  DensePointRange db(N, D);
  DensePointRange queries(Q, D);
  fill_random(db, seed_db, /*l2_normalize=*/false);
  fill_random(queries, seed_q, /*l2_normalize=*/false);

  return run_bench(db, queries, fs_block, rbits, reps, run_rabitq);
}
