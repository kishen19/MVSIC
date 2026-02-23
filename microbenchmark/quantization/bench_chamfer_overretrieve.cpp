// bench_chamfer_overretrieve.cpp
//
// Quality microbenchmark: Exact vs PQ(K=16) vs FastScan(K=16) vs RaBitQ vs TurboQuant-4bit vs
// ByteTQ. Reports average overretrieval (%) needed to achieve target recall@K (e.g. 90%, 95%).
//
// For each query cloud:
//  1) Brute-force exact distances to ALL db clouds; obtain exact top-K ids.
//  2) For each quantized method: compute approximate distances to ALL db clouds; sort by score.
//  3) For each K and recall target r: find minimal M such that top-M approx contains
//     at least ceil(r*K) of the exact top-K ids. Overretrieval% = (M-K)/K * 100.
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
//     -Kmax <u32>        (default 100)   // largest K evaluated; K grid is derived from this
//     -rec99 <f>         (default 0.99)
//
// Notes:
// - K grid: {1, 5, 10, 20, 50, 100} intersected with [1..Kmax].
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

#include "mvsic/core/quantization/fastscan.h"
#include "mvsic/core/quantization/pq.h"
#include "mvsic/core/quantization/rabitq.h"
#include "mvsic/core/quantization/one_to_many_turboquant.h"
#include "mvsic/core/quantization/byte_turboquant.h"
#include "mvsic/core/quantization/wrapper.h"

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

static std::vector<uint32_t> default_K_grid(uint32_t Kmax) {
  std::vector<uint32_t> ks = {1, 5, 10, 20, 50, 100};
  std::vector<uint32_t> out;
  out.reserve(ks.size());
  for (uint32_t k : ks) {
    if (k <= Kmax) out.push_back(k);
  }
  if (out.empty()) out.push_back(std::max<uint32_t>(1, Kmax));
  return out;
}

// Returns minimal M (1..N) such that top-M approx contains >= need hits from exact_topK_set.
static size_t min_M_for_recall(const std::vector<uint32_t>& approx_ranked_ids,
                               const std::unordered_set<uint32_t>& exact_topK_set, size_t need) {
  if (need == 0) return 0;
  size_t hits = 0;
  for (size_t m = 0; m < approx_ranked_ids.size(); ++m) {
    if (exact_topK_set.find(approx_ranked_ids[m]) != exact_topK_set.end()) {
      ++hits;
      if (hits >= need) return m + 1;  // M is 1-based count
    }
  }
  return approx_ranked_ids.size();
}

// Load pre-computed ground truth from binary file produced by compute_ground_truth.
// File format: int k_gt (header), then for each query k_gt pairs of (float dist, uint32_t id)
// sorted by ascending distance.
// Returns per-query sorted (id, dist) pairs.
static std::vector<std::vector<std::pair<uint32_t, float>>> load_ground_truth(const char* path,
                                                                              size_t num_queries,
                                                                              uint32_t Kmax) {
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    std::cerr << "ERROR: cannot open ground truth file: " << path << "\n";
    std::exit(1);
  }

  int k_gt = 0;
  in.read(reinterpret_cast<char*>(&k_gt), sizeof(int));
  if (k_gt < static_cast<int>(Kmax)) {
    std::cerr << "ERROR: ground truth k (" << k_gt << ") < Kmax (" << Kmax
              << "); not enough neighbors.\n";
    std::exit(1);
  }

  std::vector<std::vector<std::pair<uint32_t, float>>> gt(num_queries);
  for (size_t qi = 0; qi < num_queries; ++qi) {
    gt[qi].resize(k_gt);
    for (int j = 0; j < k_gt; ++j) {
      float dist;
      uint32_t id;
      in.read(reinterpret_cast<char*>(&dist), sizeof(float));
      in.read(reinterpret_cast<char*>(&id), sizeof(uint32_t));
      gt[qi][j] = {id, dist};
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
                         uint32_t Kmax, float rec99, const char* gt_file = nullptr) {
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

  const auto Kgrid = default_K_grid(Kmax);
  const size_t Nclouds = db.size();
  const size_t Qclouds = queries.size();

  std::cout << "DB: clouds=" << Nclouds << " dims=" << D << " total_vecs=" << db.total_size()
            << " avg_k=" << std::fixed << std::setprecision(2) << db.average_size() << "\n";
  std::cout << "Q : clouds=" << Qclouds << " dims=" << D << " total_vecs=" << queries.total_size()
            << " avg_k=" << std::fixed << std::setprecision(2) << queries.average_size() << "\n";
  std::cout << "pq_block=" << pq_block << " fs_block=" << fs_block << " rbits=" << rbits
            << " dist=" << (Metric ? "L2" : "IP") << "\n";
  std::cout << "Kgrid: ";
  for (auto k : Kgrid)
    std::cout << k << " ";
  std::cout << "\n";
  if (gt_file) std::cout << "Ground truth: " << gt_file << "\n";
  std::cout << "Recall target: " << rec99 << "\n";

  // Train + Encode quantized DBs (same as bench_chamfer_pq_fastscan).
  const uint32_t PQ_S = 20;

  MultiVecQuantizer<pq::Model<Metric>, Metric> pq_model;
  pq_model.train(db, pq_block, pq_k, PQ_S);
  auto pq_db = pq_model.encode(db);

  MultiVecQuantizer<fastscan::Model<Metric>, Metric> fs_model;
  fs_model.train(db, fs_block);
  auto fs_db = fs_model.encode(db);

  MultiVecQuantizer<rabitq::Model<Metric>, Metric> rq_model;
  rq_model.train(db, rbits);
  auto rq_db = rq_model.encode(db);

  MultiVecQuantizer<one_to_many_turboquant::Model<Metric>, Metric> tq_model;
  tq_model.train(db);
  auto tq_db = tq_model.encode(db);

  MultiVecQuantizer<byte_turboquant::Model<Metric>, Metric> btq_model;
  btq_model.train(db);
  auto btq_db = btq_model.encode(db);

  std::vector<std::pair<uint32_t, float>> exact_scores(Nclouds);
  std::vector<std::pair<uint32_t, float>> approx_scores(Nclouds);

  // Optionally load pre-computed ground truth.
  std::vector<std::vector<std::pair<uint32_t, float>>> gt_data;
  if (gt_file) {
    gt_data = load_ground_truth(gt_file, Qclouds, Kmax);
  }

  enum Method {
    PQ = 0,
    FASTSCAN = 1,
    RABITQ = 2,
    TURBOQUANT_4BIT = 3,
    BYTETQ = 4,
    TQ_SCALAR = 5,
    NUM_METHODS = 6
  };
  const char* method_names[NUM_METHODS] = {"PQ",     "FastScan", "RaBitQ", "TurboQuant-4bit",
                                           "ByteTQ", "TQ-Scalar"};

  std::vector<double> sum_M(NUM_METHODS * Kgrid.size(), 0.0);
  auto idx2 = [&](Method m, size_t k_i) { return static_cast<size_t>(m) * Kgrid.size() + k_i; };

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

    std::vector<std::unordered_set<uint32_t>> exact_sets;
    exact_sets.reserve(Kgrid.size());
    for (uint32_t K : Kgrid) {
      std::unordered_set<uint32_t> s;
      s.reserve(static_cast<size_t>(K) * 2);
      for (uint32_t j = 0; j < K; ++j)
        s.insert(sorted_exact[j].first);
      exact_sets.emplace_back(std::move(s));
    }

    auto eval_method = [&](Method meth) {
      std::sort(approx_scores.begin(), approx_scores.end(),
                [](const auto& a, const auto& b) { return a.second < b.second; });
      std::vector<uint32_t> approx_ids;
      approx_ids.reserve(Nclouds);
      for (const auto& p : approx_scores)
        approx_ids.push_back(p.first);

      for (size_t k_i = 0; k_i < Kgrid.size(); ++k_i) {
        const size_t K = static_cast<size_t>(Kgrid[k_i]);
        const size_t need = static_cast<size_t>(std::ceil(rec99 * static_cast<float>(K)));
        const size_t M = min_M_for_recall(approx_ids, exact_sets[k_i], need);
        sum_M[idx2(meth, k_i)] += static_cast<double>(M);
      }
    };

    {
      auto qq = pq_model.quantize_query(queries[qi]);
      pq_db.distances_all(qq, approx_scores.data());
      eval_method(PQ);
    }
    {
      auto qq = fs_model.quantize_query(queries[qi]);
      fs_db.distances_all(qq, approx_scores.data());
      eval_method(FASTSCAN);
    }
    {
      auto qq = rq_model.quantize_query(queries[qi]);
      rq_db.distances_all(qq, approx_scores.data());
      eval_method(RABITQ);
    }

    {
      auto qq = tq_model.quantize_query(queries[qi]);
      tq_db.distances_all(qq, approx_scores.data());
      eval_method(TURBOQUANT_4BIT);
    }
    {
      auto qq = btq_model.quantize_query(queries[qi]);
      btq_db.distances_all(qq, approx_scores.data());
      eval_method(BYTETQ);
    }

    std::cout << "Running TQ scalar." << std::endl;
    // TQ-Scalar: same encoding as TQ4 but using per-point scalar distance
    // instead of VNNI GEMM batch distance. Isolates GEMM kernel issues.
    {
      auto qq = tq_model.quantize_query(queries[qi]);
      parlay::parallel_for(0, Nclouds, [&](size_t cid) {
        float d = qq.distance_perpoint(tq_db[cid]);
        approx_scores[cid] = {static_cast<uint32_t>(cid), d};
      });
      eval_method(TQ_SCALAR);
    }
  }

  std::cout << "\n=== Number of candidates (M) to reach recall@K ===\n";
  std::cout << "Averages over Q=" << Qclouds << " query clouds.\n";

  for (int meth = 0; meth < NUM_METHODS; ++meth) {
    std::cout << "\n" << method_names[meth] << ":\n";
    for (size_t k_i = 0; k_i < Kgrid.size(); ++k_i) {
      const uint32_t K = Kgrid[k_i];
      const double avg_M =
          sum_M[idx2(static_cast<Method>(meth), k_i)] / std::max<size_t>(1, Qclouds);
      std::cout << "  K=" << std::setw(4) << K << "  rec" << int(rec99 * 100 + 0.5f)
                << "%: " << std::setw(10) << std::fixed << std::setprecision(1) << avg_M << "\n";
    }
  }

  return 0;
}

template<typename ChPoint>
static int run_synth(uint32_t N_db, uint32_t N_q, uint32_t K_db, uint32_t D, uint64_t seed_db,
                     uint64_t seed_q, uint32_t pq_block, uint32_t pq_k, uint32_t fs_block,
                     uint32_t rbits, uint32_t Kmax, float rec99) {
  constexpr bool Metric = ChPoint::is_metric();
  using PC = PointCloudSet<ChPoint>;

  const uint32_t K_q = 32;
  PC db(N_db, K_db, D);
  PC queries(N_q, K_q, D);

  const bool l2_normalize_vectors = !Metric;
  fill_random_point_cloud_set(db, seed_db, l2_normalize_vectors);
  fill_random_point_cloud_set(queries, seed_q, l2_normalize_vectors);

  std::cout << "Mode: synthetic (K_q fixed to 32)\n";
  return run_from_sets<ChPoint>(db, queries, pq_block, pq_k, fs_block, rbits, Kmax, rec99,
                                /*gt_file=*/nullptr);
}

template<typename ChPoint>
static int run_files(commandLine& P, uint32_t pq_block, uint32_t pq_k, uint32_t fs_block,
                     uint32_t rbits, uint32_t Kmax, float rec99, const char* gt_file) {
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
  return run_from_sets<ChPoint>(db, queries, pq_block, pq_k, fs_block, rbits, Kmax, rec99, gt_file);
}

int main(int argc, char** argv) {
  commandLine P(argc, argv,
                "[-i <dbFile>] [-q <qFile>] [-mm] [-gt <gtFile>] "
                "[-N_db <n>] [-N_q <n>] [-K_db <k>] [-D <d>] [-seed_db <s>] [-seed_q <s>] "
                "[-dist_func <L2|IP>] [-pq_block <b>] [-pq_k <k>] [-fs_block <b>] [-rbits <b>] "
                "[-Kmax <k>] [-rec99 <f>]");

  std::string df = P.getOptionValue("-dist_func", "L2");
  uint32_t pq_block = static_cast<uint32_t>(P.getOptionIntValue("-pq_block", 8));
  uint32_t pq_k = static_cast<uint32_t>(P.getOptionIntValue("-pq_k", 16));
  uint32_t fs_block = static_cast<uint32_t>(P.getOptionIntValue("-fs_block", 8));
  uint32_t rbits = static_cast<uint32_t>(P.getOptionIntValue("-rbits", 2));
  uint32_t Kmax = static_cast<uint32_t>(P.getOptionIntValue("-Kmax", 100));
  float rec99 = std::stof(P.getOptionValue("-rec99", "0.99"));
  const char* gt_file = P.getOptionValue("-gt");

  const bool file_mode = (P.getOptionValue("-i") != nullptr) || (P.getOptionValue("-q") != nullptr);
  if (file_mode) {
    if (P.getOptionValue("-i") == nullptr || P.getOptionValue("-q") == nullptr) {
      std::cerr << "ERROR: Provide both -i <dbFile> and -q <qFile> for file mode.\n";
      return 1;
    }
    if (df == "IP" || df == "ip")
      return run_files<ChamferIP_Point>(P, pq_block, pq_k, fs_block, rbits, Kmax, rec99, gt_file);
    return run_files<ChamferL2_Point>(P, pq_block, pq_k, fs_block, rbits, Kmax, rec99, gt_file);
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
                                      rbits, Kmax, rec99);
  }
  return run_synth<ChamferL2_Point>(N_db, N_q, K_db, D, seed_db, seed_q, pq_block, pq_k, fs_block,
                                    rbits, Kmax, rec99);
}
