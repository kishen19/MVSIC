// bench_chamfer_pq_fastscan.cpp
//
// Point-cloud benchmark: Exact vs PQ(K=16) vs FastScan(K=16) vs RaBitQ vs TurboQuant
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
//     -dist_func <L2|IP> (default L2)
//     -pq_block <u32>    (default 8)
//     -fs_block <u32>    (default 8)
//     -rbits <u32>       (default 2)
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
#include "mvsic/core/quantization/rabitq.h"
#endif  // __AVX512F__
#include "mvsic/core/quantization/fastscan.h"
#include "mvsic/core/quantization/pq.h"
#include "mvsic/core/quantization/turboquant.h"
#include "mvsic/core/quantization/one_to_many_turboquant.h"
#include "mvsic/core/quantization/byte_turboquant.h"
#include "mvsic/core/quantization/wrapper.h"

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

// Verification: compare GEMM (batch) distances against scalar per-point reference.
// For each query cloud × each DB cloud, computes distance both ways and checks.
template<typename QModel, typename EncSet, typename PCSet>
static void verify_quant(const char* label, const QModel& model, const EncSet& qdb,
                         const PCSet& queries, size_t max_q_clouds = 0, size_t max_db_clouds = 0) {
  const size_t nq = queries.size();
  const size_t ndb = qdb.n_clouds;
  if (max_q_clouds == 0 || max_q_clouds > nq) max_q_clouds = nq;
  if (max_db_clouds == 0 || max_db_clouds > ndb) max_db_clouds = ndb;

  std::cout << "\n=== Verifying " << label << " ==="
            << " (" << max_q_clouds << " query clouds × " << max_db_clouds << " DB clouds)\n";

  double max_rel_err = 0.0;
  double max_abs_err = 0.0;
  size_t n_mismatches = 0;
  size_t n_checked = 0;

  for (size_t qi = 0; qi < max_q_clouds; ++qi) {
    auto qq = model.quantize_query(queries[qi]);

    for (size_t di = 0; di < max_db_clouds; ++di) {
      auto cloud = qdb[di];

      // GEMM path (batch).
      float d_gemm = qq.distance(cloud);

      // Scalar per-point path.
      float d_scalar = qq.distance_perpoint(cloud);

      float abs_err = std::fabs(d_gemm - d_scalar);
      float denom = std::max(std::fabs(d_scalar), 1e-8f);
      float rel_err = abs_err / denom;

      if (abs_err > max_abs_err) max_abs_err = abs_err;
      if (rel_err > max_rel_err) max_rel_err = rel_err;

      if (rel_err > 1e-4f && abs_err > 1e-3f) {
        if (n_mismatches < 10) {
          std::cout << "  MISMATCH q=" << qi << " db=" << di << " gemm=" << d_gemm
                    << " scalar=" << d_scalar << " rel_err=" << rel_err << " abs_err=" << abs_err
                    << "\n";
        }
        ++n_mismatches;
      }
      ++n_checked;
    }
  }

  std::cout << "  checked: " << n_checked << " pairs\n";
  std::cout << "  max_rel_err: " << max_rel_err << "\n";
  std::cout << "  max_abs_err: " << max_abs_err << "\n";
  if (n_mismatches > 0) {
    std::cout << "  *** " << n_mismatches << " MISMATCHES ***\n";
  } else {
    std::cout << "  PASS\n";
  }
}

// -------------------------------------------------------------------
// Brute-force quality: use model's encode_single + quantize_query
// directly, then compare per-vector dot products against exact.
// -------------------------------------------------------------------
template<bool Metric>
static void brute_force_quality_check(const char* label,
                                      const one_to_many_turboquant::Model<Metric>& tq,
                                      const float* db_vecs, const float* q_vecs, size_t n_db,
                                      size_t n_q, size_t D) {
  using namespace mvsic::one_to_many_turboquant::internal;

  const size_t padded_dim = tq.padded_dim;
  const size_t num_bytes = tq.num_bytes_per_datapoint;

  std::cout << "\n=== Brute-force TQ quality: " << label << " ===\n";
  std::cout << "  D=" << D << " padded_dim=" << padded_dim << "\n";

  size_t n_check_db = std::min(n_db, size_t(20));
  size_t n_check_q = std::min(n_q, size_t(10));

  // Encode DB vectors.
  std::vector<std::vector<uint8_t>> db_codes(n_check_db);
  std::vector<float> db_nsf(n_check_db), db_sqn(n_check_db);
  std::vector<float> ws(padded_dim);
  for (size_t i = 0; i < n_check_db; ++i) {
    db_codes[i].resize(num_bytes, 0);
    std::vector<float> padded(padded_dim, 0.0f);
    for (size_t d = 0; d < D; ++d)
      padded[d] = db_vecs[i * D + d];
    auto [sqn, nsf] = tq.encode_single(padded.data(), db_codes[i].data(), ws);
    db_nsf[i] = nsf;
    db_sqn[i] = sqn;
  }

  // Quantize query vectors.
  struct QInfo {
    std::vector<int8_t> data;
    float nsf, sqn;
  };
  std::vector<QInfo> q_info(n_check_q);
  for (size_t i = 0; i < n_check_q; ++i) {
    std::vector<float> padded(padded_dim, 0.0f);
    for (size_t d = 0; d < D; ++d)
      padded[d] = q_vecs[i * D + d];
    auto qq = tq.quantize_query(padded.data());
    q_info[i].data.resize(padded_dim);
    for (size_t d = 0; d < padded_dim; ++d)
      q_info[i].data[d] = qq.query_data[d];
    q_info[i].nsf = qq.norm_scaling_factor;
    q_info[i].sqn = qq.unquantized_squared_norm;
  }

  // Compare.
  double total_rel = 0, max_rel = 0;
  size_t n_pairs = 0, n_bad = 0;

  for (size_t qi = 0; qi < n_check_q; ++qi) {
    for (size_t di = 0; di < n_check_db; ++di) {
      float exact = 0;
      for (size_t d = 0; d < D; ++d)
        exact += q_vecs[qi * D + d] * db_vecs[di * D + d];

      int32_t tq_int = 0;
      for (size_t j = 0; j < num_bytes; ++j) {
        uint8_t b = db_codes[di][j];
        tq_int += int32_t(kTurboQuantCentroidsInt8[b & 0xF]) * int32_t(q_info[qi].data[2 * j]);
        if (2 * j + 1 < padded_dim)
          tq_int += int32_t(kTurboQuantCentroidsInt8[b >> 4]) * int32_t(q_info[qi].data[2 * j + 1]);
      }
      float tq_approx = float(tq_int) * db_nsf[di] * q_info[qi].nsf;

      float denom = std::max(std::fabs(exact), 1e-8f);
      float rel = std::fabs(tq_approx - exact) / denom;
      total_rel += rel;
      if (rel > max_rel) max_rel = rel;
      if (rel > 0.5) ++n_bad;

      if (n_pairs < 10) {
        std::cout << "  q=" << qi << " db=" << di << "  exact=" << std::setw(10)
                  << std::setprecision(5) << exact << "  tq=" << std::setw(10) << tq_approx
                  << "  ratio=" << std::setw(8) << std::setprecision(4)
                  << (std::fabs(exact) > 1e-8f ? tq_approx / exact : 0.f)
                  << "  nsf_db=" << std::setprecision(6) << db_nsf[di]
                  << "  nsf_q=" << q_info[qi].nsf << "\n";
      }
      ++n_pairs;
    }
  }

  std::cout << "  mean_rel=" << std::setprecision(4) << (total_rel / n_pairs)
            << "  max_rel=" << max_rel << "  bad(>50%)=" << n_bad << "/" << n_pairs << "\n";
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

// ---------------------------
// Core runner given concrete DB/Q sets
// ---------------------------
template<typename ChPoint>
static int run_from_sets(const PointCloudSet<ChPoint>& db, const PointCloudSet<ChPoint>& queries,
                         uint32_t pq_block, uint32_t fs_block, uint32_t rbits, int reps,
                         bool verify) {
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

  // PQ
  MultiVecQuantizer<pq::Model<Metric>, Metric> pq_model;
  t.start();
  pq_model.train(db, pq_block, PQ_K, PQ_S);
  double pq_train_s = t.sec();

  t.start();
  auto pq_db = pq_model.encode(db);
  double pq_encode_s = t.sec();

#ifdef __AVX512F__
  // RaBitQ (requires AVX-512)
  MultiVecQuantizer<rabitq::Model<Metric>, Metric> rq_model;
  t.start();
  rq_model.train(db, rbits);
  double rq_train_s = t.sec();

  t.start();
  auto rq_db = rq_model.encode(db);
  double rq_encode_s = t.sec();
#endif  // __AVX512F__

#if defined(__AVX512F__) || defined(__AVX2__)
  // FastScan (VNNI when AVX-512, AVX2 fallback)
  MultiVecQuantizer<fastscan::Model<Metric>, Metric> fs_model;
  t.start();
  fs_model.train(db, fs_block);
  double fs_train_s = t.sec();

  t.start();
  auto fs_db = fs_model.encode(db);
  double fs_encode_s = t.sec();
#endif

  // TurboQuant (4-bit)
  MultiVecQuantizer<one_to_many_turboquant::Model<Metric>, Metric> tq_model;
  t.start();
  tq_model.train(db);
  double tq_train_s = t.sec();

  t.start();
  auto tq_db = tq_model.encode(db);
  double tq_encode_s = t.sec();

  // Byte TurboQuant (int8)
  MultiVecQuantizer<byte_turboquant::Model<Metric>, Metric> btq_model;
  t.start();
  btq_model.train(db);
  double btq_train_s = t.sec();

  t.start();
  auto btq_db = btq_model.encode(db);
  double btq_encode_s = t.sec();

  std::cout << "\n=== Train / Encode ===\n";
  std::cout << "PQ(K=16) train  : " << pq_train_s << " s\n";
  std::cout << "PQ(K=16) encode  : " << pq_encode_s << " s\n";
  std::cout << "PQ(K=16) total   : " << (pq_train_s + pq_encode_s) << " s\n";
#ifdef __AVX512F__
  std::cout << "FastScan train   : " << fs_train_s << " s\n";
  std::cout << "FastScan encode  : " << fs_encode_s << " s\n";
  std::cout << "FastScan total   : " << (fs_train_s + fs_encode_s) << " s\n";
  std::cout << "RaBitQ train   : " << rq_train_s << " s\n";
  std::cout << "RaBitQ encode  : " << rq_encode_s << " s\n";
  std::cout << "RaBitQ total   : " << (rq_train_s + rq_encode_s) << " s\n";
#endif
#if defined(__AVX512F__) || defined(__AVX2__)
  std::cout << "FastScan train   : " << fs_train_s << " s\n";
  std::cout << "FastScan encode  : " << fs_encode_s << " s\n";
  std::cout << "FastScan total   : " << (fs_train_s + fs_encode_s) << " s\n";
#endif
  std::cout << "TurboQuant train : " << tq_train_s << " s\n";
  std::cout << "TurboQuant encode: " << tq_encode_s << " s\n";
  std::cout << "TurboQuant total : " << (tq_train_s + tq_encode_s) << " s\n";
  std::cout << "ByteTQ train     : " << btq_train_s << " s\n";
  std::cout << "ByteTQ encode    : " << btq_encode_s << " s\n";
  std::cout << "ByteTQ total     : " << (btq_train_s + btq_encode_s) << " s\n";

  // ---------------------------
  // Benchmark: distances to ALL clouds
  // ---------------------------
  const uint64_t ops = uint64_t(db.size()) * uint64_t(queries.size());
  std::vector<std::pair<uint32_t, float>> results(db.size());
  volatile double sink = 0.0;

  std::cout << "\n=== All-cloud distance time (Qclouds * Nclouds) ===\n";

  double exact_best = bench_exact_all(db, queries, results, reps, sink);
  print_result("Exact (PointCloudSet::distances)", exact_best, ops, 0);

  {
    double best = bench_quant_all(pq_model, pq_db, queries, results, reps, sink);
    print_result("PQ(K=16) (wrapper::distances_all)", best, ops, exact_best);
  }

#if defined(__AVX512F__) || defined(__AVX2__)
  {
    double best = bench_quant_all(fs_model, fs_db, queries, results, reps, sink);
    print_result("FastScan(K=16) (wrapper::distances_all)", best, ops, exact_best);
  }
#endif

#ifdef __AVX512F__
  {
    double best = bench_quant_all(rq_model, rq_db, queries, results, reps, sink);
    print_result("RaBitQ (wrapper::distances_all)", best, ops, exact_best);
  }
#endif  // __AVX512F__

  {
    double best = bench_quant_all(tq_model, tq_db, queries, results, reps, sink);
    print_result("TurboQuant-4bit (wrapper::distances_all)", best, ops, exact_best);
  }

  {
    double best = bench_quant_all(btq_model, btq_db, queries, results, reps, sink);
    print_result("ByteTQ-int8 (wrapper::distances_all)", best, ops, exact_best);
  }

  // ---------------------------
  // Verification (if requested)
  // ---------------------------
  if (verify) {
    // Limit to first 20 query clouds × first 100 DB clouds for speed.
    verify_quant("TurboQuant-4bit", tq_model, tq_db, queries, 20, 100);
    verify_quant("ByteTQ-int8", btq_model, btq_db, queries, 20, 100);

    // Brute-force per-vector quality check.
    // Uses the first N raw float vectors from DB and queries.
    const float* db_data = reinterpret_cast<const float*>(db.data());
    const float* q_data = reinterpret_cast<const float*>(queries.data());
    brute_force_quality_check<Metric>("TurboQuant-4bit", tq_model.vec_model, db_data, q_data,
                                      std::min(db.total_size(), size_t(50)),
                                      std::min(queries.total_size(), size_t(20)), D);
  }

  std::cout << "\n(sink=" << sink << ")\n";
  return 0;
}

// ---------------------------
// Synthetic mode
// ---------------------------
template<typename ChPoint>
static int run_synth(uint32_t N_db, uint32_t N_q, uint32_t K_db, uint32_t D, uint64_t seed_db,
                     uint64_t seed_q, uint32_t pq_block, uint32_t fs_block, uint32_t rbits,
                     int reps, bool verify) {
  constexpr bool Metric = ChPoint::is_metric();
  using PC = PointCloudSet<ChPoint>;

  const uint32_t K_q = 32;  // fixed per request

  PC db(N_db, K_db, D);
  PC queries(N_q, K_q, D);

  const bool l2_normalize_vectors = !Metric;  // IP case
  fill_random_point_cloud_set(db, seed_db, l2_normalize_vectors);
  fill_random_point_cloud_set(queries, seed_q, l2_normalize_vectors);

  std::cout << "Mode: synthetic (K_q fixed to 32)\n";
  return run_from_sets<ChPoint>(db, queries, pq_block, fs_block, rbits, reps, verify);
}

// ---------------------------
// File mode
// ---------------------------
template<typename ChPoint>
static int run_files(commandLine& P, uint32_t pq_block, uint32_t fs_block, uint32_t rbits, int reps,
                     bool verify) {
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
  return run_from_sets<ChPoint>(db, queries, pq_block, fs_block, rbits, reps, verify);
}

int main(int argc, char** argv) {
  commandLine P(argc, argv,
                "[-i <dbFile>] [-q <qFile>] [-mm] "
                "[-N_db <n>] [-N_q <n>] [-K_db <k>] [-D <d>] [-seed_db <s>] [-seed_q <s>] "
                "[-dist_func <L2|IP>] [-pq_block <b>] [-fs_block <b>] [-rbits <b>] [-reps <r>] "
                "[-verify]");

  // Common
  std::string df = P.getOptionValue("-dist_func", "L2");
  uint32_t pq_block = static_cast<uint32_t>(P.getOptionIntValue("-pq_block", 8));
  uint32_t fs_block = static_cast<uint32_t>(P.getOptionIntValue("-fs_block", 8));
  uint32_t rbits = static_cast<uint32_t>(P.getOptionIntValue("-rbits", 2));
  int reps = std::max(1, P.getOptionIntValue("-reps", 3));
  bool verify = P.getOption("-verify");

  // Decide mode: if both -i and -q are present => file mode, else synthetic
  const bool file_mode = (P.getOptionValue("-i") != nullptr) || (P.getOptionValue("-q") != nullptr);
  if (file_mode) {
    // Require both.
    if (P.getOptionValue("-i") == nullptr || P.getOptionValue("-q") == nullptr) {
      std::cerr << "ERROR: Provide both -i <dbFile> and -q <qFile> for file mode.\n";
      return 1;
    }

    if (df == "IP" || df == "ip")
      return run_files<ChamferIP_Point>(P, pq_block, fs_block, rbits, reps, verify);
    return run_files<ChamferL2_Point>(P, pq_block, fs_block, rbits, reps, verify);
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
                                      rbits, reps, verify);
  }
  return run_synth<ChamferL2_Point>(N_db, N_q, K_db, D, seed_db, seed_q, pq_block, fs_block, rbits,
                                    reps, verify);
}
