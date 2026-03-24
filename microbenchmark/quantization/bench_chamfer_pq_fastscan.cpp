// bench_chamfer_pq_fastscan.cpp
//
// Point-cloud benchmark: Exact vs PQ_mv(K=16) vs FastScan_mv vs RaBitQ_mv vs TurboQuant_mv
// vs TurboQuant PQ (K=16, various block sizes) via wrapper.
// Measures time for "query cloud -> ALL db clouds" using distances_all() for quantized,
// and PointCloudSet::distances() for exact.
//
// FastScan runs on AVX-512 (VNNI) or AVX2. RaBitQ requires AVX-512.
// On non-AVX512 machines, Exact, PQ, FastScan (AVX2), and TurboQuant are benchmarked.
//
// Threads:
//   PARLAY_NUM_THREADS=16 bazel run //path/to:bench -- [args]
//
// Args (parse_command_line.h style):
//   File mode:
//     -i <dbFile>      (PointCloudSet binary)
//     -q <qFile>       (PointCloudSet binary)
//     -mm              (mmap-load DB; queries loaded normally)
//
//   Synthetic mode (default if -i/-q not provided):
//     -N_db <u32>      (default 20000)
//     -N_q  <u32>      (default 200)
//     -K_db <u32>      (default 64)
//     -D    <u32>      (default 128)
//     -seed_db <u64>   (default 12345)
//     -seed_q  <u64>   (default 999)
//     NOTE: query vectors-per-cloud K_q is FIXED to 32 (standard).
//
//   Common:
//     -dist_func <L2|IP> (default IP)
//     -pq_block <u32>    (default 8)
//     -fs_block <u32>    (default 8)
//     -rbits <u32>       (default 4)
//     -reps <u32>        (default 3)

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "parlay/parallel.h"
#include "parlay/primitives.h"

#ifdef __AVX512F__
#include "mvsic/core/quantization/rabitq_mv.h"
#endif
#include "mvsic/core/quantization/fastscan_mv.h"
#include "mvsic/core/quantization/pq_mv.h"
#include "mvsic/core/quantization/turboquant_mv.h"
#include "mvsic/core/quantization/other_methods/turboquant_pq_4bit.h"
#include "mvsic/core/quantization/other_methods/wrapper.h"

#include "mvsic/core/types/chamfer_ip_point.h"
#include "mvsic/core/types/chamfer_l2_point.h"
#include "mvsic/core/types/point_cloud_set.h"

#include "mvsic/core/utils/parse_command_line.h"

using namespace mvsic;

// ---------------------------
// Timing helper
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

static inline double ns_per_op(double seconds, uint64_t ops) {
  return (seconds * 1e9) / double(ops);
}

// ---------------------------
// Synthetic data gen for PointCloudSet
// ---------------------------
template<typename PC>
static void fill_random_point_cloud_set(PC& pcs, uint64_t seed, bool l2_normalize_vectors) {
  std::normal_distribution<float> nd(0.0f, 1.0f);

  const uint32_t D = pcs.get_dims();
  const size_t total_vecs = pcs.total_size();
  float* base = pcs.data();

  // Deterministic per-(vi,j) RNG so parallel_for is stable.
  parlay::parallel_for(0, total_vecs, [&](size_t vi) {
    float* v = base + vi * size_t(D);
    float ss = 0.0f;
    for (uint32_t j = 0; j < D; ++j) {
      uint64_t s = seed ^ (vi * 0x9e3779b97f4a7c15ULL) ^ (uint64_t(j) << 32);
      std::mt19937_64 rlocal(s);
      float x = nd(rlocal);
      v[j] = x;
      ss += x * x;
    }
    if (l2_normalize_vectors) {
      float inv = 1.0f / std::sqrt(std::max(ss, 1e-12f));
      for (uint32_t j = 0; j < D; ++j)
        v[j] *= inv;
    }
  });
}

// ---------------------------
// Benchmark: run "all distances" repeatedly, keep best
// ---------------------------
template<typename PCSet>
static double bench_exact_all(const PCSet& db, const PCSet& queries,
                              std::vector<std::pair<uint32_t, float>>& out, int reps,
                              volatile double& sink) {
  Timer t;
  double best = 1e100;

  // warmup
  for (size_t i = 0; i < std::min<size_t>(queries.size(), 2); ++i) {
    db.distances(queries[i], out.data());
    sink += out[0].second;
  }

  for (int r = 0; r < reps; ++r) {
    t.start();
    for (size_t qi = 0; qi < queries.size(); ++qi) {
      db.distances(queries[qi], out.data());
      sink += out[qi % out.size()].second;
    }
    best = std::min(best, t.sec());
  }
  return best;
}

// Quantized benchmark: model builds per-query LUT cloud, encoded set computes distances_all.
template<typename QModel, typename EncSet, typename PCSet>
static double bench_quant_all(const QModel& model, const EncSet& qdb, const PCSet& queries,
                              std::vector<std::pair<uint32_t, float>>& out, int reps,
                              volatile double& sink) {
  Timer t;
  double best = 1e100;

  // warmup
  for (size_t i = 0; i < std::min<size_t>(queries.size(), 2); ++i) {
    auto qq = model.quantize_query(queries[i]);
    qdb.distances_all(qq, out.data());  // IMPORTANT: all db clouds
    sink += out[0].second;
  }

  for (int r = 0; r < reps; ++r) {
    t.start();
    for (size_t qi = 0; qi < queries.size(); ++qi) {
      auto qq = model.quantize_query(queries[qi]);
      qdb.distances_all(qq, out.data());  // IMPORTANT: all db clouds
      sink += out[qi % out.size()].second;
    }
    best = std::min(best, t.sec());
  }
  return best;
}

// Helper to print a benchmark result.
static void print_result(const char* label, double best, uint64_t ops, double exact_best) {
  double dps = double(ops) / best;
  std::cout << label << ":\n";
  std::cout << "  total_time : " << best << " s\n";
  std::cout << "  throughput : " << std::fixed << std::setprecision(3) << (dps / 1e6)
            << " M cloud-dists/s\n";
  std::cout << "  latency    : " << std::fixed << std::setprecision(3) << ns_per_op(best, ops)
            << " ns / cloud-dist\n";
  if (exact_best > 0) {
    std::cout << "  speedup: " << std::fixed << std::setprecision(2) << (exact_best / best)
              << "x\n";
  }
}

struct BenchRow {
  std::string name;
  double best_s = 0.0;
};

static void print_bench_table(const std::vector<BenchRow>& rows, uint64_t ops, double exact_best) {
  if (rows.empty()) return;
  std::cout << "\n=== Summary (All-cloud distances) ===\n";
  std::cout << std::left << std::setw(28) << "Method" << std::right << std::setw(12) << "time [s]"
            << std::setw(14) << "M dists/s" << std::setw(14) << "ns / dist" << std::setw(12)
            << "speedup\n";
  std::cout << std::string(28 + 12 + 14 * 3 + 12, '-') << "\n";
  for (const auto& r : rows) {
    const double t = r.best_s;
    const double dps = double(ops) / t;
    const double ns = ns_per_op(t, ops);
    const double speedup = (exact_best > 0.0) ? (exact_best / t) : 0.0;
    std::cout << std::left << std::setw(28) << r.name << std::right << std::setw(12) << std::fixed
              << std::setprecision(4) << t << std::setw(14) << std::setprecision(3) << (dps / 1e6)
              << std::setw(14) << std::setprecision(1) << ns << std::setw(12)
              << std::setprecision(2) << speedup << "\n";
  }
  std::cout << "\n";
}

// ---------------------------
// Core runner given concrete DB/Q sets
// ---------------------------
template<typename ChPoint>
static int run_from_sets(const PointCloudSet<ChPoint>& db, const PointCloudSet<ChPoint>& queries,
                         uint32_t pq_block, uint32_t fs_block, uint32_t rbits, int reps,
                         bool run_pq, bool run_rabitq) {
  using PC = PointCloudSet<ChPoint>;
  constexpr bool Metric = ChPoint::is_metric();

  const uint32_t D = db.get_dims();
  if (queries.get_dims() != D) {
    std::cerr << "ERROR: DB dims (" << D << ") != Query dims (" << queries.get_dims() << ")\n";
    return 1;
  }
  if (D % pq_block != 0) {
    std::cerr << "ERROR: D must be divisible by pq_block.\n";
    return 1;
  }
  if (D % fs_block != 0) {
    std::cerr << "ERROR: D must be divisible by fs_block.\n";
    return 1;
  }

  std::cout << "DB: clouds=" << db.size() << "  dims=" << D << "  total_vecs=" << db.total_size()
            << "  avg_k=" << std::fixed << std::setprecision(2) << db.average_size() << "\n";
  std::cout << "Q : clouds=" << queries.size() << "  dims=" << D
            << "  total_vecs=" << queries.total_size() << "  avg_k=" << std::fixed
            << std::setprecision(2) << queries.average_size() << "\n";
  std::cout << "pq_block=" << pq_block << "  fs_block=" << fs_block << "  rbits=" << rbits
            << "  dist=" << (Metric ? "L2" : "IP") << "  reps=" << reps << "\n";
#ifdef __AVX512F__
  std::cout << "AVX-512: enabled (VNNI GEMM + FastScan + RaBitQ)\n";
#elif defined(__AVX2__)
  std::cout << "AVX-512: NOT available; AVX2: enabled (FastScan)\n";
#else
  std::cout << "AVX-512/AVX2: NOT available (only Exact + PQ + TurboQuant)\n";
#endif

  // ---------------------------
  // Train + Encode quantized DBs
  // ---------------------------
  Timer t;

  const uint32_t PQ_K = 16;
  const uint32_t PQ_S = 20;

  // PQ (optional, controlled by -pq flag)
  pq_mv::Model<Metric> pq_model;
  double pq_train_s = 0.0, pq_encode_s = 0.0;
  pq_mv::Quantized_Point_Cloud_Set<Metric> pq_db;
  if (run_pq) {
    t.start();
    pq_model.train(db, pq_block, PQ_K, PQ_S);
    pq_train_s = t.sec();

    t.start();
    pq_db = pq_model.encode(db);
    pq_encode_s = t.sec();
  }

#ifdef __AVX512F__
  rabitq_mv::Model<Metric> rq_model;
  double rq_train_s = 0.0, rq_encode_s = 0.0;
  rabitq_mv::Quantized_Point_Cloud_Set<Metric> rq_db;
  if (run_rabitq) {
    t.start();
    rq_model.train(db, rbits);
    rq_train_s = t.sec();

    t.start();
    rq_db = rq_model.encode(db);
    rq_encode_s = t.sec();
  }
#endif

#if defined(__AVX512F__) || defined(__AVX2__)
  fastscan_mv::Model<Metric> fs_model;
  t.start();
  fs_model.train(db, fs_block);
  double fs_train_s = t.sec();

  t.start();
  auto fs_db = fs_model.encode(db);
  double fs_encode_s = t.sec();
#endif

  turboquant_mv::Model<Metric> tq_model;
  t.start();
  tq_model.train(db);
  double tq_train_s = t.sec();

  t.start();
  auto tq_db = tq_model.encode(db);
  double tq_encode_s = t.sec();

  MultiVecQuantizer<turboquant_pq_4bit::Model<Metric, 1>, Metric> tqpq1_model;
  MultiVecQuantizer<turboquant_pq_4bit::Model<Metric, 2>, Metric> tqpq2_model;
  MultiVecQuantizer<turboquant_pq_4bit::Model<Metric, 4>, Metric> tqpq4_model;
  MultiVecQuantizer<turboquant_pq_4bit::Model<Metric, 8>, Metric> tqpq8_model;

  t.start();
  tqpq1_model.train(db);
  double tqpq1_train_s = t.sec();
  t.start();
  auto tqpq1_db = tqpq1_model.encode(db);
  double tqpq1_encode_s = t.sec();

  t.start();
  tqpq2_model.train(db);
  double tqpq2_train_s = t.sec();
  t.start();
  auto tqpq2_db = tqpq2_model.encode(db);
  double tqpq2_encode_s = t.sec();

  t.start();
  tqpq4_model.train(db);
  double tqpq4_train_s = t.sec();
  t.start();
  auto tqpq4_db = tqpq4_model.encode(db);
  double tqpq4_encode_s = t.sec();

  t.start();
  tqpq8_model.train(db);
  double tqpq8_train_s = t.sec();
  t.start();
  auto tqpq8_db = tqpq8_model.encode(db);
  double tqpq8_encode_s = t.sec();

  std::cout << "\n=== Train / Encode ===\n";
  if (run_pq) {
    std::cout << "PQ(K=16) train  : " << pq_train_s << " s\n";
    std::cout << "PQ(K=16) encode  : " << pq_encode_s << " s\n";
    std::cout << "PQ(K=16) total   : " << (pq_train_s + pq_encode_s) << " s\n";
  } else {
    std::cout << "PQ(K=16)        : (skipped, pass -pq to enable)\n";
  }
#if defined(__AVX512F__) || defined(__AVX2__)
  std::cout << "FastScan train   : " << fs_train_s << " s\n";
  std::cout << "FastScan encode  : " << fs_encode_s << " s\n";
  std::cout << "FastScan total   : " << (fs_train_s + fs_encode_s) << " s\n";
#endif
#ifdef __AVX512F__
  if (run_rabitq) {
    std::cout << "RaBitQ train   : " << rq_train_s << " s\n";
    std::cout << "RaBitQ encode  : " << rq_encode_s << " s\n";
    std::cout << "RaBitQ total   : " << (rq_train_s + rq_encode_s) << " s\n";
  } else {
    std::cout << "RaBitQ          : (skipped, pass -rabitq to enable)\n";
  }
#endif
  std::cout << "TurboQuant_mv train : " << tq_train_s << " s\n";
  std::cout << "TurboQuant_mv encode: " << tq_encode_s << " s\n";
  std::cout << "TurboQuant_mv total : " << (tq_train_s + tq_encode_s) << " s\n";
  std::cout << "TQ-PQ(K=16,B=1)   : " << (tqpq1_train_s + tqpq1_encode_s) << " s\n";
  std::cout << "TQ-PQ(K=16,B=2)   : " << (tqpq2_train_s + tqpq2_encode_s) << " s\n";
  std::cout << "TQ-PQ(K=16,B=4)   : " << (tqpq4_train_s + tqpq4_encode_s) << " s\n";
  std::cout << "TQ-PQ(K=16,B=8)   : " << (tqpq8_train_s + tqpq8_encode_s) << " s\n";

  // ---------------------------
  // Benchmark: distances to ALL clouds
  // ---------------------------
  const uint64_t ops = uint64_t(db.size()) * uint64_t(queries.size());
  std::vector<std::pair<uint32_t, float>> results(db.size());
  volatile double sink = 0.0;

  std::cout << "\n=== All-cloud distance time (Qclouds * Nclouds) ===\n";
  std::vector<BenchRow> summary;

  double exact_best = bench_exact_all(db, queries, results, reps, sink);
  print_result("Exact (PointCloudSet::distances)", exact_best, ops, 0);
  summary.push_back({"Exact", exact_best});

  if (run_pq) {
    double best = bench_quant_all(pq_model, pq_db, queries, results, reps, sink);
    print_result("PQ_mv(K=16) (distances_all)", best, ops, exact_best);
    summary.push_back({"PQ_mv (K=16)", best});
  }

#if defined(__AVX512F__) || defined(__AVX2__)
  {
    double best = bench_quant_all(fs_model, fs_db, queries, results, reps, sink);
    print_result("FastScan_mv (distances_all)", best, ops, exact_best);
    summary.push_back({"FastScan_mv (K=16)", best});
  }
#endif

#ifdef __AVX512F__
  if (run_rabitq) {
    double best = bench_quant_all(rq_model, rq_db, queries, results, reps, sink);
    print_result("RaBitQ_mv (distances_all)", best, ops, exact_best);
    summary.push_back({"RaBitQ_mv", best});
  }
#endif

  {
    double best = bench_quant_all(tq_model, tq_db, queries, results, reps, sink);
    print_result("TurboQuant_mv (distances_all)", best, ops, exact_best);
    summary.push_back({"TurboQuant_mv", best});
  }

  {
    double best = bench_quant_all(tqpq1_model, tqpq1_db, queries, results, reps, sink);
    print_result("TQ-PQ(K=16,B=1) (wrapper::distances_all)", best, ops, exact_best);
    summary.push_back({"TQ-PQ (B=1)", best});
  }
  {
    double best = bench_quant_all(tqpq2_model, tqpq2_db, queries, results, reps, sink);
    print_result("TQ-PQ(K=16,B=2) (wrapper::distances_all)", best, ops, exact_best);
    summary.push_back({"TQ-PQ (B=2)", best});
  }
  {
    double best = bench_quant_all(tqpq4_model, tqpq4_db, queries, results, reps, sink);
    print_result("TQ-PQ(K=16,B=4) (wrapper::distances_all)", best, ops, exact_best);
    summary.push_back({"TQ-PQ (B=4)", best});
  }
  {
    double best = bench_quant_all(tqpq8_model, tqpq8_db, queries, results, reps, sink);
    print_result("TQ-PQ(K=16,B=8) (wrapper::distances_all)", best, ops, exact_best);
    summary.push_back({"TQ-PQ (B=8)", best});
  }

  print_bench_table(summary, ops, exact_best);

  std::cout << "\n(sink=" << sink << ")\n";
  return 0;
}

// ---------------------------
// Synthetic mode
// ---------------------------
template<typename ChPoint>
static int run_synth(uint32_t N_db, uint32_t N_q, uint32_t K_db, uint32_t D, uint64_t seed_db,
                     uint64_t seed_q, uint32_t pq_block, uint32_t fs_block, uint32_t rbits,
                     int reps, bool run_pq, bool run_rabitq) {
  constexpr bool Metric = ChPoint::is_metric();
  using PC = PointCloudSet<ChPoint>;

  const uint32_t K_q = 32;  // fixed per request

  PC db(N_db, K_db, D);
  PC queries(N_q, K_q, D);

  const bool l2_normalize_vectors = !Metric;  // IP case
  fill_random_point_cloud_set(db, seed_db, l2_normalize_vectors);
  fill_random_point_cloud_set(queries, seed_q, l2_normalize_vectors);

  std::cout << "Mode: synthetic (K_q fixed to 32)\n";
  return run_from_sets<ChPoint>(db, queries, pq_block, fs_block, rbits, reps, run_pq, run_rabitq);
}

// ---------------------------
// File mode
// ---------------------------
template<typename ChPoint>
static int run_files(commandLine& P, uint32_t pq_block, uint32_t fs_block, uint32_t rbits, int reps,
                     bool run_pq, bool run_rabitq) {
  using PC = PointCloudSet<ChPoint>;

  char* dbFile = P.getOptionValue("-i");
  if (dbFile == nullptr) {
    std::cerr << "ERROR: file mode requires -i <dbFile>\n";
    return 1;
  }
  char* qFile = P.getOptionValue("-q");
  if (qFile == nullptr) {
    std::cerr << "ERROR: file mode requires -q <qFile>\n";
    return 1;
  }

  bool mm = P.getOption("-mm");
  auto db = PC(dbFile, mm);
  auto queries = PC(qFile, /*is_mmap=*/false);

  std::cout << "Mode: file\n";
  std::cout << "  db=" << dbFile << (mm ? " (mmap)\n" : "\n");
  std::cout << "  q =" << qFile << "\n";
  return run_from_sets<ChPoint>(db, queries, pq_block, fs_block, rbits, reps, run_pq, run_rabitq);
}

int main(int argc, char** argv) {
  commandLine P(argc, argv,
                "[-i <dbFile>] [-q <qFile>] [-mm] "
                "[-N_db <n>] [-N_q <n>] [-K_db <k>] [-D <d>] [-seed_db <s>] [-seed_q <s>] "
                "[-dist_func <L2|IP>] [-pq_block <b>] [-fs_block <b>] [-rbits <b>] [-reps <r>] "
                "[-pq] [-rabitq]");

  // Common
  std::string df = P.getOptionValue("-dist_func", "IP");
  uint32_t pq_block = static_cast<uint32_t>(P.getOptionIntValue("-pq_block", 8));
  uint32_t fs_block = static_cast<uint32_t>(P.getOptionIntValue("-fs_block", 8));
  uint32_t rbits = static_cast<uint32_t>(P.getOptionIntValue("-rbits", 4));
  int reps = std::max(1, P.getOptionIntValue("-reps", 3));
  bool run_pq = P.getOption("-pq");
  bool run_rabitq = P.getOption("-rabitq");

  // Decide mode: if both -i and -q are present => file mode, else synthetic
  const bool file_mode = (P.getOptionValue("-i") != nullptr) || (P.getOptionValue("-q") != nullptr);
  if (file_mode) {
    // Require both.
    if (P.getOptionValue("-i") == nullptr || P.getOptionValue("-q") == nullptr) {
      std::cerr << "ERROR: Provide both -i <dbFile> and -q <qFile> for file mode.\n";
      return 1;
    }

    if (df == "IP" || df == "ip")
      return run_files<ChamferIP_Point>(P, pq_block, fs_block, rbits, reps, run_pq, run_rabitq);
    return run_files<ChamferL2_Point>(P, pq_block, fs_block, rbits, reps, run_pq, run_rabitq);
  }

  // Synthetic mode args
  uint32_t N_db = static_cast<uint32_t>(P.getOptionIntValue("-N_db", 20000));
  uint32_t N_q = static_cast<uint32_t>(P.getOptionIntValue("-N_q", 200));
  uint32_t K_db = static_cast<uint32_t>(P.getOptionIntValue("-K_db", 64));
  uint32_t D = static_cast<uint32_t>(P.getOptionIntValue("-D", 128));

  // parse_command_line doesn't have u64 helpers in the snippet; take as string if provided.
  uint64_t seed_db = 12345ULL;
  uint64_t seed_q = 999ULL;
  if (char* s = P.getOptionValue("-seed_db")) seed_db = static_cast<uint64_t>(std::stoull(s));
  if (char* s = P.getOptionValue("-seed_q")) seed_q = static_cast<uint64_t>(std::stoull(s));

  if (df == "IP" || df == "ip") {
    return run_synth<ChamferIP_Point>(N_db, N_q, K_db, D, seed_db, seed_q, pq_block, fs_block,
                                      rbits, reps, run_pq, run_rabitq);
  }
  return run_synth<ChamferL2_Point>(N_db, N_q, K_db, D, seed_db, seed_q, pq_block, fs_block, rbits,
                                    reps, run_pq, run_rabitq);
}
