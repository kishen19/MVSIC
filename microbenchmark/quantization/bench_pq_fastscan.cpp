// bench_pq_fastscan.cpp
//
// Benchmarks (full scan distance computation):
//   1) Exact (unquantized) float IP distance via efanna2e::DistanceInnerProduct (NSGDist)
//   2) FastScan (K=16) using Quantized_Query::distances_all
//   3) RaBitQ using Quantized_Query::distances_all
//   4) TQ-4bit: per-point distance (same path as Byte TQ)
//   5) Byte TQ (int8) per-point distance
//   6) TQ-PQ-4bit: block-wise k-means, uint8 LUT, FastScan-style shuffle scan
//
// Usage (Bazel):
//   PARLAY_NUM_THREADS=16 bazel run -c opt //microbenchmark/quantization:bench_pq_fastscan -- [N] [Q] [D]
//   [fs_block] [rbits] [tqpq_block] [REPS]
//
// Defaults:
//   N=1,000,000  Q=1,000  D=128  fs_block=64  rbits=8  tqpq_block=4  REPS=2
//
// Notes:
// - FastScan requires AVX-512 (AVX512F + AVX512BW recommended).
// - Metric is fixed to IP (Metric=false). Distance = -inner_product (smaller = more similar).
// - tqpq_block: block size for TurboQuant PQ 4-bit (1, 2, 4, 8, or 16).
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
// ---------------------------
static inline float full_scan_sum_exact_ip_dbpar(const DensePointRange& db, const float* q,
                                                 uint32_t D, size_t grain = 4096) {
  const size_t N = db.size();
  const float* base = db.data();

  const size_t num_blocks = (N + grain - 1) / grain;
  parlay::sequence<double> block_sums(num_blocks, 0.0);

  parlay::parallel_for(0, num_blocks, [&](size_t bi) {
    static thread_local efanna2e::DistanceInnerProduct distfunc;

    const size_t s = bi * grain;
    const size_t e = std::min(N, s + grain);

    double local = 0.0;
    for (size_t i = s; i < e; ++i) {
      const float* p = base + i * size_t(D);
      local += double(-distfunc.compare(q, p, D));  // distance = -IP
    }
    block_sums[bi] = local;
  });

  double total = 0.0;
  for (size_t bi = 0; bi < num_blocks; ++bi)
    total += block_sums[bi];
  return float(total);
}

// ---------------------------
// Query-serial runners (avoid nested parallelism)
// ---------------------------
static inline float scan_all_queries_exact_serial(const DensePointRange& db,
                                                  const DensePointRange& queries, uint32_t D,
                                                  parlay::sequence<double>& per_q_sum) {
  const size_t Q = queries.size();
  const float* qbase = queries.data();

  for (size_t qi = 0; qi < Q; ++qi) {
    const float* q = qbase + qi * size_t(D);
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
static inline float scan_all_queries_tqpq_serial(const TQEnc& enc, const TQQueryVec& qvec,
                                                 size_t Q, float* scratch, size_t N,
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

// TQ-4bit / Byte TQ: per-point distance (same pattern for both).
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
// Benchmark runner
// ---------------------------
int main(int argc, char** argv) {
  size_t N = (argc > 1 ? std::stoull(argv[1]) : 1'000'000);
  size_t Q = (argc > 2 ? std::stoull(argv[2]) : 1000);
  uint32_t D = (argc > 3 ? uint32_t(std::stoul(argv[3])) : 128);

  uint32_t fs_block = (argc > 4 ? uint32_t(std::stoul(argv[4])) : 64);
  size_t rbits = (argc > 5 ? size_t(std::stoull(argv[5])) : size_t(8));
  uint32_t tqpq_block = (argc > 6 ? uint32_t(std::stoul(argv[6])) : 4);

  int REPS = (argc > 7 ? int(std::stoi(argv[7])) : 2);
  if (REPS < 1) REPS = 1;

  // IP mode (distance = -inner_product; smaller = more similar)
  constexpr bool Metric = false;

  std::cout << "N=" << N << " Q=" << Q << " D=" << D << " fs_block=" << fs_block
            << " rbits=" << rbits << " tqpq_block=" << tqpq_block << " (Metric=IP)\n";

  if (D % fs_block != 0) {
    std::cerr << "ERROR: D not divisible by fs_block.\n";
    return 1;
  }
  if (tqpq_block != 1 && tqpq_block != 2 && tqpq_block != 4 && tqpq_block != 8 &&
      tqpq_block != 16) {
    std::cerr << "ERROR: tqpq_block must be 1, 2, 4, 8, or 16.\n";
    return 1;
  }

  DensePointRange db(N, D);
  DensePointRange queries(Q, D);

  fill_random(db, 12345, /*l2_normalize=*/false);
  fill_random(queries, 999, /*l2_normalize=*/false);

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
  // Build RaBitQ model + encode
  // ---------------------------
  t.start();
  mvsic::rabitq::Model<Metric> rq_model;
  rq_model.train(db, rbits);
  auto rq = rq_model.encode(db);
  double rq_build_s = t.sec();

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

  // TurboQuant PQ 4-bit: build + LUT + scan (run once for selected block size). Requires AVX-512.
  double tqpq_build_s = 0.0;
  double tqpq_lut_s = 0.0;
  double tqpq_scan_best_s = 0.0;
  parlay::sequence<double> tqpq_per_q_sum(Q, 0.0);
  volatile float tqpq_sink = 0.0f;
#if defined(__AVX512F__)
  switch (tqpq_block) {
    case 1: {
      t.start();
      mvsic::turboquant_pq_4bit::Model<Metric, 1> tqpq_m;
      tqpq_m.train(db);
      auto tqpq_enc = tqpq_m.encode(db);
      tqpq_build_s = t.sec();
      std::vector<mvsic::turboquant_pq_4bit::Quantized_Query<Metric, 1>> tqpq_q;
      tqpq_q.reserve(Q);
      t.start();
      for (size_t i = 0; i < Q; ++i)
        tqpq_q.push_back(tqpq_m.quantize_query(queries.data() + i * size_t(D)));
      tqpq_lut_s = t.sec();
      std::vector<float> tqpq_scratch(N);
      double best = 1e100;
      for (int r = 0; r < REPS; ++r) {
        t.start();
        float acc =
            scan_all_queries_tqpq_serial(tqpq_enc, tqpq_q, Q, tqpq_scratch.data(), N, tqpq_per_q_sum);
        best = std::min(best, t.sec());
        tqpq_sink += acc;
      }
      tqpq_scan_best_s = best;
      break;
    }
    case 2: {
      t.start();
      mvsic::turboquant_pq_4bit::Model<Metric, 2> tqpq_m;
      tqpq_m.train(db);
      auto tqpq_enc = tqpq_m.encode(db);
      tqpq_build_s = t.sec();
      std::vector<mvsic::turboquant_pq_4bit::Quantized_Query<Metric, 2>> tqpq_q;
      tqpq_q.reserve(Q);
      t.start();
      for (size_t i = 0; i < Q; ++i)
        tqpq_q.push_back(tqpq_m.quantize_query(queries.data() + i * size_t(D)));
      tqpq_lut_s = t.sec();
      std::vector<float> tqpq_scratch(N);
      double best = 1e100;
      for (int r = 0; r < REPS; ++r) {
        t.start();
        float acc = scan_all_queries_tqpq_serial(tqpq_enc, tqpq_q, Q, tqpq_scratch.data(), N, tqpq_per_q_sum);
        best = std::min(best, t.sec());
        tqpq_sink += acc;
      }
      tqpq_scan_best_s = best;
      break;
    }
    case 4: {
      t.start();
      mvsic::turboquant_pq_4bit::Model<Metric, 4> tqpq_m;
      tqpq_m.train(db);
      auto tqpq_enc = tqpq_m.encode(db);
      tqpq_build_s = t.sec();
      std::vector<mvsic::turboquant_pq_4bit::Quantized_Query<Metric, 4>> tqpq_q;
      tqpq_q.reserve(Q);
      t.start();
      for (size_t i = 0; i < Q; ++i)
        tqpq_q.push_back(tqpq_m.quantize_query(queries.data() + i * size_t(D)));
      tqpq_lut_s = t.sec();
      std::vector<float> tqpq_scratch(N);
      double best = 1e100;
      for (int r = 0; r < REPS; ++r) {
        t.start();
        float acc = scan_all_queries_tqpq_serial(tqpq_enc, tqpq_q, Q, tqpq_scratch.data(), N, tqpq_per_q_sum);
        best = std::min(best, t.sec());
        tqpq_sink += acc;
      }
      tqpq_scan_best_s = best;
      break;
    }
    case 8: {
      t.start();
      mvsic::turboquant_pq_4bit::Model<Metric, 8> tqpq_m;
      tqpq_m.train(db);
      auto tqpq_enc = tqpq_m.encode(db);
      tqpq_build_s = t.sec();
      std::vector<mvsic::turboquant_pq_4bit::Quantized_Query<Metric, 8>> tqpq_q;
      tqpq_q.reserve(Q);
      t.start();
      for (size_t i = 0; i < Q; ++i)
        tqpq_q.push_back(tqpq_m.quantize_query(queries.data() + i * size_t(D)));
      tqpq_lut_s = t.sec();
      std::vector<float> tqpq_scratch(N);
      double best = 1e100;
      for (int r = 0; r < REPS; ++r) {
        t.start();
        float acc = scan_all_queries_tqpq_serial(tqpq_enc, tqpq_q, Q, tqpq_scratch.data(), N, tqpq_per_q_sum);
        best = std::min(best, t.sec());
        tqpq_sink += acc;
      }
      tqpq_scan_best_s = best;
      break;
    }
    case 16: {
      t.start();
      mvsic::turboquant_pq_4bit::Model<Metric, 16> tqpq_m;
      tqpq_m.train(db);
      auto tqpq_enc = tqpq_m.encode(db);
      tqpq_build_s = t.sec();
      std::vector<mvsic::turboquant_pq_4bit::Quantized_Query<Metric, 16>> tqpq_q;
      tqpq_q.reserve(Q);
      t.start();
      for (size_t i = 0; i < Q; ++i)
        tqpq_q.push_back(tqpq_m.quantize_query(queries.data() + i * size_t(D)));
      tqpq_lut_s = t.sec();
      std::vector<float> tqpq_scratch(N);
      double best = 1e100;
      for (int r = 0; r < REPS; ++r) {
        t.start();
        float acc = scan_all_queries_tqpq_serial(tqpq_enc, tqpq_q, Q, tqpq_scratch.data(), N, tqpq_per_q_sum);
        best = std::min(best, t.sec());
        tqpq_sink += acc;
      }
      tqpq_scan_best_s = best;
      break;
    }
    default:
      break;
  }
#endif  // __AVX512F__

  std::cout << "\n=== Build / Encode time ===\n";
#if defined(__AVX512F__) || defined(__AVX2__)
  std::cout << "FastScan(K=16):  " << fs_build_s << " s\n";
#else
  std::cout << "FastScan(K=16):  (skipped, no AVX512/AVX2)\n";
#endif
  std::cout << "RaBitQ(bits=" << rbits << "): " << rq_build_s << " s\n";
  std::cout << "TQ-4bit: " << tq_build_s << " s\n";
  std::cout << "Byte TQ: " << btq_build_s << " s\n";
#if defined(__AVX512F__)
  std::cout << "TQ-PQ-4bit(block=" << tqpq_block << "): " << tqpq_build_s << " s\n";
#endif
  std::cout << "Exact (float IP): (no build)\n";

  // ---------------------------
  // LUT construction time
  // ---------------------------
#if defined(__AVX512F__) || defined(__AVX2__)
  std::vector<mvsic::fastscan::Quantized_Query<Metric>> fs_q;
  fs_q.reserve(Q);
#endif

  std::vector<mvsic::rabitq::Quantized_Query<Metric>> rq_q;
  rq_q.reserve(Q);

  std::vector<mvsic::turboquant_4bit::Quantized_Query<Metric>> tq_q;
  tq_q.reserve(Q);

  std::vector<mvsic::turboquant_byte::Quantized_Query<Metric>> btq_q;
  btq_q.reserve(Q);

#if defined(__AVX512F__) || defined(__AVX2__)
  t.start();
  for (size_t i = 0; i < Q; ++i) {
    fs_q.push_back(fs_model.quantize_query(queries.data() + i * size_t(D)));
  }
  double fs_lut_s = t.sec();
#endif

  t.start();
  for (size_t i = 0; i < Q; ++i) {
    rq_q.push_back(rq_model.quantize_query(queries.data() + i * size_t(D)));
  }
  double rq_lut_s = t.sec();

  t.start();
  for (size_t i = 0; i < Q; ++i) {
    tq_q.push_back(tq_model.quantize_query(queries.data() + i * size_t(D)));
  }
  double tq_lut_s = t.sec();

  t.start();
  for (size_t i = 0; i < Q; ++i) {
    btq_q.push_back(btq_model.quantize_query(queries.data() + i * size_t(D)));
  }
  double btq_lut_s = t.sec();

  const uint64_t num_dists = uint64_t(N) * uint64_t(Q);

  std::cout << "\n=== Query LUT time (Q=" << Q << ") ===\n";
#if defined(__AVX512F__) || defined(__AVX2__)
  std::cout << "FastScan(K=16):  " << fs_lut_s << " s  (" << (fs_lut_s * 1e6 / Q) << " us/query)\n";
#else
  std::cout << "FastScan(K=16):  (skipped, no AVX512/AVX2)\n";
#endif
  std::cout << "RaBitQ:          " << rq_lut_s << " s  (" << (rq_lut_s * 1e6 / Q) << " us/query)\n";
  std::cout << "TQ-4bit: " << tq_lut_s << " s  (" << (tq_lut_s * 1e6 / Q)
            << " us/query)\n";
  std::cout << "Byte TQ:  " << btq_lut_s << " s  (" << (btq_lut_s * 1e6 / Q)
            << " us/query)\n";
#if defined(__AVX512F__)
  std::cout << "TQ-PQ-4bit: " << tqpq_lut_s << " s  (" << (tqpq_lut_s * 1e6 / Q)
            << " us/query)\n";
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
    sink += full_scan_sum_exact_ip_dbpar(db, queries.data() + i * size_t(D), D);

#if defined(__AVX512F__) || defined(__AVX2__)
    fs_q[i].distances_all(fs, fs_out.data());
    for (size_t j = 0; j < N; ++j)
      sink += fs_out[j];
#endif

    rq_q[i].distances_all(rq, rq_out.data());
    for (size_t j = 0; j < N; ++j)
      sink += rq_out[j];

    tq_q[i].distances_contiguous(tq.packed_codes.data(), tq.norm_scaling_factors.data(),
                                 tq.unquantized_squared_norms.data(), tq.stride, N, tq_out.data());
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

  // Exact
  {
    double best = 1e100;
    float acc = 0.0f;
    for (int r = 0; r < REPS; ++r) {
      t.start();
      float local = scan_all_queries_exact_serial(db, queries, D, per_q_sum);
      double s = t.sec();
      best = std::min(best, s);
      acc = local;
    }
    sink += acc;
    print_scan_stats("Exact float IP (NSGDist)", best, num_dists, Q);
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
    print_scan_stats("FastScan distances_all K=16 (AVX2/VNNI)", best, num_dists, Q);
  }
#else
  std::cout << "FastScan K=16: (skipped, no AVX512/AVX2)\n";
#endif

  // RaBitQ
  {
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
    print_scan_stats("RaBitQ distances_all", best, num_dists, Q);
  }

  // TQ-4bit: per-point distance (same as Byte TQ path)
  {
    double best = 1e100;
    float acc = 0.0f;
    for (int r = 0; r < REPS; ++r) {
      t.start();
      float local = scan_all_queries_byte_tq_serial(tq, tq_q, Q, tq_out.data(), N, per_q_sum);
      double s = t.sec();
      best = std::min(best, s);
      acc = local;
    }
    sink += acc;
    print_scan_stats("TQ-4bit per-point (4b)", best, num_dists, Q);
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
    print_scan_stats("Byte TQ per-point (int8)", best, num_dists, Q);
  }

  // TQ-PQ-4bit (precomputed in build section)
#if defined(__AVX512F__)
  print_scan_stats("TQ-PQ-4bit distances_all", tqpq_scan_best_s, num_dists, Q);
#endif

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
    const float* q = queries.data() + qi * size_t(D);
    for (size_t i = 0; i < check_db; ++i) {
      static thread_local efanna2e::DistanceInnerProduct distfunc;
      float exact = -distfunc.compare(q, db.data() + i * size_t(D), D);  // distance = -IP
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
