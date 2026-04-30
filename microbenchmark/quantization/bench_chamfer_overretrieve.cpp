// bench_chamfer_overretrieve.cpp
//
// Quality microbenchmark: Exact vs (optional) PQ(K=16) vs FastScan(K=16) vs (optional) RaBitQ vs
// TurboQuant-4bit vs TurboQuant-8bit vs TurboQuantPQ-4bit (various block sizes).
// Reports recall@k as a function of the candidate budget k' (multiples of k).
//
// For each query cloud:
//  1) Brute-force exact distances to ALL db clouds; obtain exact top-K ids.
//  2) For each quantized method: compute approximate distances to ALL db clouds; sort by score.
//  3) For each candidate budget k' in {k,2k,4k,...}, compute recall@k:
//       | approx_top_k' ∩ exact_top_k | / k
//
// Args (parse_command_line.h style):
//   File mode:
//     -i <dbFile>      (PointCloudSet binary)
//     -q <qFile>       (PointCloudSet binary)
//     -mm              (mmap-load DB; queries loaded normally)
//
//   Synthetic mode (default if -i/-q not provided):
//     -N_db <u32>      (default 20000)   // number of db clouds
//     -N_q  <u32>      (default 200)     // number of query clouds
//     -K_db <u32>      (default 64)      // DB vectors per cloud
//     -D    <u32>      (default 128)
//     -seed_db <u64>   (default 12345)
//     -seed_q  <u64>   (default 999)
//     NOTE: query vectors-per-cloud K_q is fixed to 32.
//
//   Common:
//     -dist_func <L2|IP> (default L2)
//     -pq_block <u32>    (default 8)
//     -fs_block <u32>    (default 8)
//     -rbits <u32>       (default 2)
//     -k <u32>           (default 10)    // report recall@k
//     -pq                (enable PQ; default off)
//     -rabitq            (enable RaBitQ; default off)
//
// Notes:
// - k' grid: {k, 2k, 4k, 8k, ...} capped at min(Nclouds, 64k).
// - This benchmark is meant for *quality* not speed; it sorts full Nclouds lists.

#include <algorithm>
#include <cmath>
#include <fstream>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "parlay/parallel.h"

#include "mvsic/core/quantization/fastscan_mv.h"
#include "mvsic/core/quantization/pq_mv.h"
#include "mvsic/core/quantization/rabitq_mv.h"
#include "mvsic/core/quantization/turboquant_mv.h"
#include "mvsic/core/quantization/turboquant_8bit_mv.h"
#include "mvsic/core/quantization/other_methods/turboquant_pq_4bit.h"
#include "mvsic/core/quantization/other_methods/wrapper.h"

#include "mvsic/core/types/chamfer_ip_point.h"
#include "mvsic/core/types/chamfer_l2_point.h"
#include "mvsic/core/types/point_cloud_set.h"

#include "mvsic/core/utils/parse_command_line.h"

using namespace mvsic;

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

static std::vector<uint32_t> default_kprime_grid(uint32_t k, uint32_t Nclouds_cap) {
  // Report recall@k while varying the candidate budget k' in reasonable multiples of k.
  // Default: {k, 2k, 5k, 10k, 25k, 50k} (capped by Nclouds).
  static constexpr std::array<uint32_t, 6> mult = {1, 2, 5, 10, 25, 50};
  std::vector<uint32_t> out;
  if (k == 0) return out;
  for (uint32_t m : mult) {
    uint64_t kp64 = uint64_t(k) * uint64_t(m);
    if (kp64 == 0) continue;
    uint32_t kp = static_cast<uint32_t>(std::min<uint64_t>(kp64, uint64_t(Nclouds_cap)));
    if (kp == 0) continue;
    if (out.empty() || kp > out.back()) out.push_back(kp);
  }
  return out;
}

// Load pre-computed ground truth from binary file produced by compute_ground_truth.
// File format: int k_gt (header), then for each query k_gt pairs of (float dist, uint32_t id)
// sorted by ascending distance.
// Returns per-query sorted (id, dist) pairs. Neighbor `id` must match PointCloudSet logical ids
// (same as db.get_id(i) for cloud index i), unless -gt_neighbor_indices is set (then id is i).
static std::vector<std::vector<std::pair<uint32_t, float>>> load_ground_truth(const char* path,
                                                                              size_t num_queries,
                                                                              uint32_t k) {
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    std::cerr << "ERROR: cannot open ground truth file: " << path << "\n";
    std::exit(1);
  }

  int k_gt = 0;
  in.read(reinterpret_cast<char*>(&k_gt), sizeof(int));
  if (k_gt < static_cast<int>(k)) {
    std::cerr << "ERROR: ground truth k (" << k_gt << ") < k (" << k << "); not enough neighbors.\n";
    std::exit(1);
  }

  std::vector<std::vector<std::pair<uint32_t, float>>> gt(num_queries);
  for (size_t qi = 0; qi < num_queries; ++qi) {
    gt[qi].resize(k);
    for (uint32_t j = 0; j < k; ++j) {
      float dist;
      uint32_t id;
      in.read(reinterpret_cast<char*>(&dist), sizeof(float));
      in.read(reinterpret_cast<char*>(&id), sizeof(uint32_t));
      gt[qi][j] = {id, dist};
    }
    // Skip remaining neighbors, if any.
    for (int j = static_cast<int>(k); j < k_gt; ++j) {
      float dist;
      uint32_t id;
      in.read(reinterpret_cast<char*>(&dist), sizeof(float));
      in.read(reinterpret_cast<char*>(&id), sizeof(uint32_t));
    }
  }

  if (!in) {
    std::cerr << "ERROR: ground truth file too short for " << num_queries << " queries.\n";
    std::exit(1);
  }

  std::cout << "Loaded ground truth from " << path << " (k_gt=" << k_gt
            << ", queries=" << num_queries << ")\n";
  return gt;
}

template<typename ChPoint>
static int run_from_sets(const PointCloudSet<ChPoint>& db, const PointCloudSet<ChPoint>& queries,
                         uint32_t pq_block, uint32_t pq_k, uint32_t fs_block, uint32_t rbits,
                         uint32_t k, const char* gt_file = nullptr, bool gt_neighbor_indices = false,
                         bool run_pq = true, bool run_rabitq = true) {
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

  const size_t Nclouds = db.size();
  const size_t Qclouds = queries.size();
  if (k == 0 || k > Nclouds) {
    std::cerr << "ERROR: invalid -k (k=" << k << ", Nclouds=" << Nclouds << ")\n";
    return 1;
  }
  const auto kprime_grid = default_kprime_grid(k, static_cast<uint32_t>(Nclouds));

  std::cout << "DB: clouds=" << Nclouds << " dims=" << D << " total_vecs=" << db.total_size()
            << " avg_k=" << std::fixed << std::setprecision(2) << db.average_size() << "\n";
  std::cout << "Q : clouds=" << Qclouds << " dims=" << D << " total_vecs=" << queries.total_size()
            << " avg_k=" << std::fixed << std::setprecision(2) << queries.average_size() << "\n";
  std::cout << "pq_block=" << pq_block << " fs_block=" << fs_block << " rbits=" << rbits
            << " dist=" << (Metric ? "L2" : "IP") << "\n";
  std::cout << "k=" << k << "  k' grid: ";
  for (auto kp : kprime_grid) std::cout << kp << " ";
  std::cout << "\n";
  if (gt_file) {
    std::cout << "Ground truth: " << gt_file;
    if (gt_neighbor_indices) std::cout << " (neighbor ids = DB row indices, mapped with get_id)";
    std::cout << "\n";
  }

  // Train + Encode quantized DBs (same as bench_chamfer_pq_fastscan).
  const uint32_t PQ_S = 20;

  pq_mv::Model<Metric> pq_model;
  pq_mv::Quantized_Point_Cloud_Set<Metric> pq_db;
  if (run_pq) {
    pq_model.train(db, pq_block, pq_k, PQ_S);
    pq_db = pq_model.encode(db);
  }

  fastscan_mv::Model<Metric> fs_model;
  fs_model.train(db, fs_block);
  auto fs_db = fs_model.encode(db);

  rabitq_mv::Model<Metric> rq_model;
  rabitq_mv::Quantized_Point_Cloud_Set<Metric> rq_db;
  if (run_rabitq) {
    rq_model.train(db, rbits);
    rq_db = rq_model.encode(db);
  }

  turboquant_mv::Model<Metric> tq_model;
  tq_model.train(db);
  auto tq_db = tq_model.encode(db);

  turboquant_8bit_mv::Model<Metric> tq8_model;
  tq8_model.train(db);
  auto tq8_db = tq8_model.encode(db);

  // TurboQuant PQ 4-bit (B=1/2/4/8)
  MultiVecQuantizer<turboquant_pq_4bit::Model<Metric, 1>, Metric> tqpq1_model;
  MultiVecQuantizer<turboquant_pq_4bit::Model<Metric, 2>, Metric> tqpq2_model;
  MultiVecQuantizer<turboquant_pq_4bit::Model<Metric, 4>, Metric> tqpq4_model;
  MultiVecQuantizer<turboquant_pq_4bit::Model<Metric, 8>, Metric> tqpq8_model;
  tqpq1_model.train(db);
  tqpq2_model.train(db);
  tqpq4_model.train(db);
  tqpq8_model.train(db);
  auto tqpq1_db = tqpq1_model.encode(db);
  auto tqpq2_db = tqpq2_model.encode(db);
  auto tqpq4_db = tqpq4_model.encode(db);
  auto tqpq8_db = tqpq8_model.encode(db);

  std::vector<std::pair<uint32_t, float>> exact_scores(Nclouds);
  std::vector<std::pair<uint32_t, float>> approx_scores(Nclouds);

  // Optionally load pre-computed ground truth.
  std::vector<std::vector<std::pair<uint32_t, float>>> gt_data;
  if (gt_file) {
    gt_data = load_ground_truth(gt_file, Qclouds, k);
  }

  enum Method {
    PQ = 0,
    FASTSCAN = 1,
    RABITQ = 2,
    TURBOQUANT_MV = 3,
    TURBOQUANT_8BIT_MV = 4,
    TQPQ_B1 = 5,
    TQPQ_B2 = 6,
    TQPQ_B4 = 7,
    TQPQ_B8 = 8,
    NUM_METHODS = 9
  };
  const char* method_names[NUM_METHODS] = {
      "PQ (K=16)",
      "FastScan (K=16)",
      "RaBitQ",
      "TurboQuant_mv (4bit)",
      "TurboQuant_mv (8bit)",
      "TQ-PQ (K=16,B=1)",
      "TQ-PQ (K=16,B=2)",
      "TQ-PQ (K=16,B=4)",
      "TQ-PQ (K=16,B=8)",
  };

  std::vector<double> sum_recall(NUM_METHODS * kprime_grid.size(), 0.0);
  auto idx2 = [&](Method m, size_t i) { return static_cast<size_t>(m) * kprime_grid.size() + i; };

  std::cout << "Outer iterations: " << Qclouds << std::endl;
  for (size_t qi = 0; qi < Qclouds; ++qi) {
    if (gt_file) {
      // Use pre-computed ground truth (already sorted by ascending distance).
    } else {
      db.distances(queries[qi], exact_scores.data());
      std::sort(exact_scores.begin(), exact_scores.end(),
                [](const auto& a, const auto& b) { return a.second < b.second; });
    }

    const auto& sorted_exact = gt_file ? gt_data[qi] : exact_scores;

    std::unordered_set<uint32_t> exact_set;
    exact_set.reserve(static_cast<size_t>(k) * 2);
    for (uint32_t j = 0; j < k; ++j) {
      uint32_t nid = sorted_exact[j].first;
      if (gt_file && gt_neighbor_indices) {
        if (nid >= Nclouds) {
          std::cerr << "ERROR: ground-truth neighbor index " << nid << " >= Nclouds " << Nclouds
                    << " (query " << qi << ")\n";
          return 1;
        }
        nid = db.get_id(static_cast<size_t>(nid));
      }
      exact_set.insert(nid);
    }

    auto eval_method = [&](Method meth) {
      std::sort(approx_scores.begin(), approx_scores.end(),
                [](const auto& a, const auto& b) { return a.second < b.second; });
      // distances_all() already stores logical cloud ids (same as db.get_id(cid)) in .first.
      std::vector<uint32_t> approx_ids;
      approx_ids.reserve(Nclouds);
      for (const auto& p : approx_scores) approx_ids.push_back(p.first);

      for (size_t i = 0; i < kprime_grid.size(); ++i) {
        const size_t kp = static_cast<size_t>(kprime_grid[i]);
        size_t hits = 0;
        const size_t limit = std::min(kp, approx_ids.size());
        for (size_t j = 0; j < limit; ++j) {
          if (exact_set.find(approx_ids[j]) != exact_set.end()) ++hits;
        }
        sum_recall[idx2(meth, i)] += static_cast<double>(hits) / static_cast<double>(k);
      }
    };

    if (run_pq) {
      auto qq = pq_model.quantize_query(queries[qi]);
      pq_db.distances_all(qq, approx_scores.data());
      eval_method(PQ);
    }
    {
      auto qq = fs_model.quantize_query(queries[qi]);
      fs_db.distances_all(qq, approx_scores.data());
      eval_method(FASTSCAN);
    }
    if (run_rabitq) {
      auto qq = rq_model.quantize_query(queries[qi]);
      rq_db.distances_all(qq, approx_scores.data());
      eval_method(RABITQ);
    }

    {
      auto qq = tq_model.quantize_query(queries[qi]);
      tq_db.distances_all(qq, approx_scores.data());
      eval_method(TURBOQUANT_MV);
    }

    {
      auto qq = tq8_model.quantize_query(queries[qi]);
      tq8_db.distances_all(qq, approx_scores.data());
      eval_method(TURBOQUANT_8BIT_MV);
    }

    // TQ-Scalar: same encoding as TQ4 but using per-point scalar distance
    // instead of VNNI GEMM batch distance. Isolates GEMM kernel issues.
    //    {
    //      auto qq = tq_model.quantize_query(queries[qi]);
    //      parlay::parallel_for(0, Nclouds, [&](size_t cid) {
    //        float d = qq.distance_perpoint(tq_db[cid]);
    //        approx_scores[cid] = {static_cast<uint32_t>(cid), d};
    //      });
    //      eval_method(TQ_SCALAR);
    //    }

    // TQ-PQ (wrapper / multi-vector)
    {
      auto qq = tqpq1_model.quantize_query(queries[qi]);
      tqpq1_db.distances_all(qq, approx_scores.data());
      eval_method(TQPQ_B1);
    }
    {
      auto qq = tqpq2_model.quantize_query(queries[qi]);
      tqpq2_db.distances_all(qq, approx_scores.data());
      eval_method(TQPQ_B2);
    }
    {
      auto qq = tqpq4_model.quantize_query(queries[qi]);
      tqpq4_db.distances_all(qq, approx_scores.data());
      eval_method(TQPQ_B4);
    }
    {
      auto qq = tqpq8_model.quantize_query(queries[qi]);
      tqpq8_db.distances_all(qq, approx_scores.data());
      eval_method(TQPQ_B8);
    }
  }

  std::cout << "\n=== Recall@" << k << " vs candidate budget k' ===\n";
  std::cout << "Averages over Q=" << Qclouds << " query clouds.\n\n";

  // Pretty table: rows = methods, cols = k' grid, entries = avg recall@k.
  std::cout << std::left << std::setw(24) << "Method";
  for (uint32_t kp : kprime_grid)
    std::cout << std::right << std::setw(10) << ("k'=" + std::to_string(kp));
  std::cout << "\n";
  std::cout << std::string(24 + 10 * kprime_grid.size(), '-') << "\n";

  for (int meth = 0; meth < NUM_METHODS; ++meth) {
    if (meth == PQ && !run_pq) continue;
    if (meth == RABITQ && !run_rabitq) continue;
    std::cout << std::left << std::setw(24) << method_names[meth];
    for (size_t i = 0; i < kprime_grid.size(); ++i) {
      const double avg_r =
          sum_recall[idx2(static_cast<Method>(meth), i)] / std::max<size_t>(1, Qclouds);
      std::cout << std::right << std::setw(10) << std::fixed << std::setprecision(2) << avg_r;
    }
    std::cout << "\n";
  }

  return 0;
}

template<typename ChPoint>
static int run_synth(uint32_t N_db, uint32_t N_q, uint32_t K_db, uint32_t D, uint64_t seed_db,
                     uint64_t seed_q, uint32_t pq_block, uint32_t pq_k, uint32_t fs_block,
                     uint32_t rbits, uint32_t k, bool run_pq, bool run_rabitq) {
  constexpr bool Metric = ChPoint::is_metric();
  using PC = PointCloudSet<ChPoint>;

  const uint32_t K_q = 32;
  PC db(N_db, K_db, D);
  PC queries(N_q, K_q, D);

  const bool l2_normalize_vectors = !Metric;
  fill_random_point_cloud_set(db, seed_db, l2_normalize_vectors);
  fill_random_point_cloud_set(queries, seed_q, l2_normalize_vectors);

  std::cout << "Mode: synthetic (K_q fixed to 32)\n";
  return run_from_sets<ChPoint>(db, queries, pq_block, pq_k, fs_block, rbits, k,
                                /*gt_file=*/nullptr, /*gt_neighbor_indices=*/false, run_pq, run_rabitq);
}

template<typename ChPoint>
static int run_files(commandLine& P, uint32_t pq_block, uint32_t pq_k, uint32_t fs_block,
                     uint32_t rbits, uint32_t k, const char* gt_file, bool run_pq,
                     bool run_rabitq) {
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
  // auto queries = PC(qFile, /*is_mmap=*/false);
  auto queries_full = PC(qFile, /*is_mmap=*/false);
  auto sample = parlay::delayed_tabulate(100, [&](size_t i) { return queries_full[i]; });
  auto queries = PC(sample, queries_full.get_dims());

  std::cout << "Mode: file\n";
  std::cout << "  db=" << dbFile << (mm ? " (mmap)\n" : "\n");
  std::cout << "  q =" << qFile << "\n";
  const bool gt_neighbor_indices = P.getOption("-gt_neighbor_indices");
  return run_from_sets<ChPoint>(db, queries, pq_block, pq_k, fs_block, rbits, k, gt_file,
                                gt_neighbor_indices, run_pq, run_rabitq);
}

int main(int argc, char** argv) {
  commandLine P(argc, argv,
                "[-i <dbFile>] [-q <qFile>] [-mm] [-gt <gtFile>] [-gt_neighbor_indices] "
                "[-N_db <n>] [-N_q <n>] [-K_db <k>] [-D <d>] [-seed_db <s>] [-seed_q <s>] "
                "[-dist_func <L2|IP>] [-pq_block <b>] [-pq_k <k>] [-fs_block <b>] [-rbits <b>] "
                "[-k <k>] [-pq] [-rabitq]");

  std::string df = P.getOptionValue("-dist_func", "L2");
  uint32_t pq_block = static_cast<uint32_t>(P.getOptionIntValue("-pq_block", 8));
  uint32_t pq_k = static_cast<uint32_t>(P.getOptionIntValue("-pq_k", 16));
  uint32_t fs_block = static_cast<uint32_t>(P.getOptionIntValue("-fs_block", 8));
  uint32_t rbits = static_cast<uint32_t>(P.getOptionIntValue("-rbits", 2));
  uint32_t k = static_cast<uint32_t>(P.getOptionIntValue("-k", 10));
  const char* gt_file = P.getOptionValue("-gt");
  bool run_pq = P.getOption("-pq");
  bool run_rabitq = P.getOption("-rabitq");

  const bool file_mode = (P.getOptionValue("-i") != nullptr) || (P.getOptionValue("-q") != nullptr);
  if (file_mode) {
    if (P.getOptionValue("-i") == nullptr || P.getOptionValue("-q") == nullptr) {
      std::cerr << "ERROR: Provide both -i <dbFile> and -q <qFile> for file mode.\n";
      return 1;
    }
    if (df == "IP" || df == "ip")
      return run_files<ChamferIP_Point>(P, pq_block, pq_k, fs_block, rbits, k, gt_file, run_pq,
                                        run_rabitq);
    return run_files<ChamferL2_Point>(P, pq_block, pq_k, fs_block, rbits, k, gt_file, run_pq,
                                      run_rabitq);
  }

  uint32_t N_db = static_cast<uint32_t>(P.getOptionIntValue("-N_db", 20000));
  uint32_t N_q = static_cast<uint32_t>(P.getOptionIntValue("-N_q", 200));
  uint32_t K_db = static_cast<uint32_t>(P.getOptionIntValue("-K_db", 64));
  uint32_t D = static_cast<uint32_t>(P.getOptionIntValue("-D", 128));

  uint64_t seed_db = 12345ULL;
  uint64_t seed_q = 999ULL;
  if (char* s = P.getOptionValue("-seed_db")) seed_db = static_cast<uint64_t>(std::stoull(s));
  if (char* s = P.getOptionValue("-seed_q")) seed_q = static_cast<uint64_t>(std::stoull(s));

  if (df == "IP" || df == "ip") {
    return run_synth<ChamferIP_Point>(N_db, N_q, K_db, D, seed_db, seed_q, pq_block, pq_k, fs_block,
                                      rbits, k, run_pq, run_rabitq);
  }
  return run_synth<ChamferL2_Point>(N_db, N_q, K_db, D, seed_db, seed_q, pq_block, pq_k, fs_block,
                                    rbits, k, run_pq, run_rabitq);
}
