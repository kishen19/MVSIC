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

#include <Eigen/Dense>

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

// --------------- PCA helpers ---------------

// Compute PCA projection matrix from the flattened DB vectors.
// Returns {W (D x pca_dim), mean (D)}.
template<typename ChPoint>
static std::pair<Eigen::MatrixXf, Eigen::VectorXf>
compute_pca(const PointCloudSet<ChPoint>& pcs, uint32_t pca_dim) {
  const uint32_t D = pcs.get_dims();
  const size_t N = pcs.total_size();
  pca_dim = std::min(pca_dim, D);

  // Map all vectors into an Eigen matrix (N x D).
  Eigen::MatrixXf X(N, D);
  const float* base = pcs.data();
  for (size_t i = 0; i < N; ++i)
    for (uint32_t j = 0; j < D; ++j)
      X(i, j) = base[i * D + j];

  // Center.
  Eigen::VectorXf mean = X.colwise().mean();          // (D,)
  X.rowwise() -= mean.transpose();

  // Covariance-based PCA (D x D covariance, D is typically 128).
  Eigen::MatrixXf cov = (X.transpose() * X) / float(N); // (D x D)
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> eig(cov);
  // Eigenvalues are ascending; take the last pca_dim columns.
  Eigen::MatrixXf W = eig.eigenvectors().rightCols(pca_dim); // (D x pca_dim)

  std::cout << "PCA: " << D << "d -> " << pca_dim << "d  (" << N << " training vectors)\n";
  return {W, mean};
}

// Project a PointCloudSet through PCA, returning a new set with dim=pca_dim.
template<typename ChPoint>
static PointCloudSet<ChPoint>
project_pcs(const PointCloudSet<ChPoint>& pcs, const Eigen::MatrixXf& W,
            const Eigen::VectorXf& mean, uint32_t pca_dim) {
  const uint32_t D_in  = pcs.get_dims();
  const size_t Nclouds = pcs.size();

  // Build offsets in the projected space (in float counts, not byte counts).
  std::vector<size_t> new_offsets(Nclouds + 1);
  new_offsets[0] = 0;
  for (size_t i = 0; i < Nclouds; ++i)
    new_offsets[i + 1] = new_offsets[i] + size_t(pcs.get_size(i)) * pca_dim;

  const size_t total_coords = new_offsets[Nclouds];
  std::vector<float> proj_vals(total_coords);

  // Project every vector: v_proj = (v - mean)^T * W
  parlay::parallel_for(0, Nclouds, [&](size_t ci) {
    const uint32_t K_ci = pcs.get_size(ci);
    const float* src = pcs.data(ci);
    float* dst = proj_vals.data() + new_offsets[ci];
    Eigen::Map<const Eigen::MatrixXf> Vsrc(src, K_ci, D_in);  // row-major natural
    // Manually center + project row by row (avoids temp allocation).
    for (uint32_t r = 0; r < K_ci; ++r) {
      Eigen::Map<const Eigen::VectorXf> v(src + size_t(r) * D_in, D_in);
      Eigen::Map<Eigen::VectorXf> out(dst + size_t(r) * pca_dim, pca_dim);
      out.noalias() = W.transpose() * (v - mean);
    }
  });

  // Build ids.
  std::vector<uint32_t> new_ids(Nclouds);
  for (size_t i = 0; i < Nclouds; ++i) new_ids[i] = pcs.get_id(i);

  return PointCloudSet<ChPoint>(static_cast<uint32_t>(Nclouds), pca_dim,
                                proj_vals.data(), new_offsets.data(), new_ids.data());
}

// Low-bit TurboQuant: shared header for 1-bit and 2-bit quality methods.
#include "mvsic/core/quantization/low_bit_turboquant.h"

using LowBitTQ_Vec = mvsic::low_bit_turboquant::EncodedVec;
using LowBitTQ_Query = mvsic::low_bit_turboquant::PreparedQuery;

// Chamfer distance using low-bit TQ.
// db_vecs: pre-encoded flat vector array, indexed by cloud_offsets.
template<typename ChPoint>
static float lowbit_chamfer(
    const std::vector<LowBitTQ_Vec>& db_vecs,
    size_t db_start, size_t db_count,
    const std::vector<LowBitTQ_Query>& q_vecs,
    size_t q_start, size_t q_count,
    size_t pdim, bool metric,
    bool use_2bit) {
  float total = 0.0f;
  for (size_t qi = 0; qi < q_count; ++qi) {
    float best = std::numeric_limits<float>::max();
    for (size_t di = 0; di < db_count; ++di) {
      float d = use_2bit
          ? mvsic::low_bit_turboquant::distance_2bit(db_vecs[db_start + di], q_vecs[q_start + qi], pdim, metric)
          : mvsic::low_bit_turboquant::distance_1bit(db_vecs[db_start + di], q_vecs[q_start + qi], pdim, metric);
      if (d < best) best = d;
    }
    total += best;
  }
  return total / static_cast<float>(q_count);
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
                         uint32_t Kmax, float rec99, uint32_t pca_dim = 40,
                         const char* gt_file = nullptr) {
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

  // PCA: fit on DB, project both DB and queries.
  auto [pca_W, pca_mean] = compute_pca(db, pca_dim);
  auto pca_db = project_pcs(db, pca_W, pca_mean, pca_dim);
  auto pca_queries = project_pcs(queries, pca_W, pca_mean, pca_dim);

  // Low-bit TQ: encode all DB and query vectors (flattened).
  // We access the inner TQ model to get the rotator.
  const auto& tq_inner = tq_model.vec_model;
  const size_t pdim = tq_inner.padded_dim;

  // Encode all DB vectors (flattened).
  const size_t total_db_vecs = db.total_size();
  std::vector<LowBitTQ_Vec> db_1bit(total_db_vecs), db_2bit(total_db_vecs);
  {
    std::cout << "Encoding DB for 1-bit and 2-bit TQ (" << total_db_vecs << " vecs)..." << std::endl;
    parlay::parallel_for(0, total_db_vecs, [&](size_t vi) {
      static thread_local std::vector<float> ws;
      const float* p = db.data() + vi * D;
      db_1bit[vi] = mvsic::low_bit_turboquant::encode_1bit(tq_inner, p, ws);
      db_2bit[vi] = mvsic::low_bit_turboquant::encode_2bit(tq_inner, p, ws);
    });
  }

  // Build per-cloud offsets into the flattened vector.
  std::vector<size_t> db_cloud_starts(Nclouds + 1);
  db_cloud_starts[0] = 0;
  for (size_t i = 0; i < Nclouds; ++i)
    db_cloud_starts[i + 1] = db_cloud_starts[i] + db.get_size(i);

  std::vector<std::pair<uint32_t, float>> exact_scores(Nclouds);
  std::vector<std::pair<uint32_t, float>> approx_scores(Nclouds);

  // Optionally load pre-computed ground truth.
  std::vector<std::vector<std::pair<uint32_t, float>>> gt_data;
  if (gt_file) {
    gt_data = load_ground_truth(gt_file, Qclouds, Kmax);
  }

  enum Method { PQ = 0, FASTSCAN = 1, RABITQ = 2, TURBOQUANT_4BIT = 3, BYTETQ = 4, TQ_SCALAR = 5, PCA_METHOD = 6, TQ_1BIT = 7, TQ_2BIT = 8, NUM_METHODS = 9 };
  std::string pca_label = "PCA-" + std::to_string(pca_dim) + "d";
  const char* method_names[NUM_METHODS] = {"PQ", "FastScan", "RaBitQ", "TurboQuant-4bit", "ByteTQ", "TQ-Scalar",
                                           pca_label.c_str(), "TQ-1bit", "TQ-2bit"};

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
//    {
//      auto qq = tq_model.quantize_query(queries[qi]);
//      parlay::parallel_for(0, Nclouds, [&](size_t cid) {
//        float d = qq.distance_perpoint(tq_db[cid]);
//        approx_scores[cid] = {static_cast<uint32_t>(cid), d};
//      });
//      eval_method(TQ_SCALAR);
//    }

    // PCA: exact Chamfer distance in reduced dimensionality.
    {
      pca_db.distances(pca_queries[qi], approx_scores.data());
      eval_method(PCA_METHOD);
    }

    // Low-bit TQ: encode query cloud, compute chamfer distances.
    {
      const auto& qcloud = queries[qi];
      const size_t nq = qcloud.size();
      std::vector<LowBitTQ_Query> q_lowbit(nq);
      for (size_t v = 0; v < nq; ++v) {
        q_lowbit[v] = mvsic::low_bit_turboquant::prepare_query(tq_inner, qcloud.data(v));
      }

      // 1-bit chamfer distance to each DB cloud.
      parlay::parallel_for(0, Nclouds, [&](size_t cid) {
        float d = lowbit_chamfer<ChPoint>(
            db_1bit, db_cloud_starts[cid], db.get_size(cid),
            q_lowbit, 0, nq, pdim, Metric, /*use_2bit=*/false);
        approx_scores[cid] = {static_cast<uint32_t>(cid), d};
      });
      eval_method(TQ_1BIT);

      // 2-bit chamfer distance to each DB cloud.
      parlay::parallel_for(0, Nclouds, [&](size_t cid) {
        float d = lowbit_chamfer<ChPoint>(
            db_2bit, db_cloud_starts[cid], db.get_size(cid),
            q_lowbit, 0, nq, pdim, Metric, /*use_2bit=*/true);
        approx_scores[cid] = {static_cast<uint32_t>(cid), d};
      });
      eval_method(TQ_2BIT);
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
                     uint32_t rbits, uint32_t Kmax, float rec99, uint32_t pca_dim) {
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
                                pca_dim, /*gt_file=*/nullptr);
}

template<typename ChPoint>
static int run_files(commandLine& P, uint32_t pq_block, uint32_t pq_k, uint32_t fs_block,
                     uint32_t rbits, uint32_t Kmax, float rec99, uint32_t pca_dim,
                     const char* gt_file) {
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
  return run_from_sets<ChPoint>(db, queries, pq_block, pq_k, fs_block, rbits, Kmax, rec99,
                                pca_dim, gt_file);
}

int main(int argc, char** argv) {
  commandLine P(argc, argv,
                "[-i <dbFile>] [-q <qFile>] [-mm] [-gt <gtFile>] "
                "[-N_db <n>] [-N_q <n>] [-K_db <k>] [-D <d>] [-seed_db <s>] [-seed_q <s>] "
                "[-dist_func <L2|IP>] [-pq_block <b>] [-pq_k <k>] [-fs_block <b>] [-rbits <b>] "
                "[-Kmax <k>] [-rec99 <f>] [-pca_dim <d>]");

  std::string df = P.getOptionValue("-dist_func", "L2");
  uint32_t pq_block = static_cast<uint32_t>(P.getOptionIntValue("-pq_block", 8));
  uint32_t pq_k = static_cast<uint32_t>(P.getOptionIntValue("-pq_k", 16));
  uint32_t fs_block = static_cast<uint32_t>(P.getOptionIntValue("-fs_block", 8));
  uint32_t rbits = static_cast<uint32_t>(P.getOptionIntValue("-rbits", 2));
  uint32_t Kmax = static_cast<uint32_t>(P.getOptionIntValue("-Kmax", 100));
  float rec99 = std::stof(P.getOptionValue("-rec99", "0.99"));
  uint32_t pca_dim = static_cast<uint32_t>(P.getOptionIntValue("-pca_dim", 40));
  const char* gt_file = P.getOptionValue("-gt");

  const bool file_mode = (P.getOptionValue("-i") != nullptr) || (P.getOptionValue("-q") != nullptr);
  if (file_mode) {
    if (P.getOptionValue("-i") == nullptr || P.getOptionValue("-q") == nullptr) {
      std::cerr << "ERROR: Provide both -i <dbFile> and -q <qFile> for file mode.\n";
      return 1;
    }
    if (df == "IP" || df == "ip")
      return run_files<ChamferIP_Point>(P, pq_block, pq_k, fs_block, rbits, Kmax, rec99, pca_dim, gt_file);
    return run_files<ChamferL2_Point>(P, pq_block, pq_k, fs_block, rbits, Kmax, rec99, pca_dim, gt_file);
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
    return run_synth<ChamferIP_Point>(N_db, N_q, K_db, D, seed_db, seed_q, pq_block, pq_k,
                                      fs_block, rbits, Kmax, rec99, pca_dim);
  }
  return run_synth<ChamferL2_Point>(N_db, N_q, K_db, D, seed_db, seed_q, pq_block, pq_k, fs_block,
                                    rbits, Kmax, rec99, pca_dim);
}
