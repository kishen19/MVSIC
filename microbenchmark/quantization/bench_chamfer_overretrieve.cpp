// bench_chamfer_overretrieve.cpp
//
// Quality microbenchmark for multi-vector methods:
// TurboQuant-4bit, TurboQuant-8bit, FastScan-b{2,4,8}, RaBitQ-{1,4,8},
// 1BTQ-mv (sym 1-bit) and 1BTQAsym-mv (asym 1-bit / int4 query).
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
//     -pq_block <u32>    (ignored; kept for CLI compatibility)
//     -fs_block <u32>    (ignored; FastScan-b2/b4/b8 all run when valid)
//     -rbits <u32>       (ignored; RaBitQ-1/4/8 all run)
//     -k <u32>           (default 10)    // report recall@k
//
// Notes:
// - k' grid: {k, 2k, 4k, 8k, ...} capped at min(Nclouds, 64k).
// - This benchmark is meant for *quality* not speed; it sorts full Nclouds lists.

#include <algorithm>
#include <cmath>
#include <filesystem>
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
#include "mvsic/core/quantization/rabitq_mv.h"
#include "mvsic/core/quantization/turboquant_mv.h"
#include "mvsic/core/quantization/turboquant_8bit_mv.h"
#include "mvsic/core/quantization/turboquant_1bit_mv.h"
#include "mvsic/core/quantization/turboquant_1bit_asym_mv.h"

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

// Single-vector-style k' schedule: geometric progression from k, capped by both
// dataset size and user max_k_prime.
static std::vector<uint32_t> make_k_primes(uint32_t k, uint32_t n_base, uint32_t max_k_prime,
                                           double growth) {
  std::vector<uint32_t> out;
  if (k == 0) return out;
  double cur = static_cast<double>(k);
  while (true) {
    uint32_t kp = static_cast<uint32_t>(
        std::min<uint64_t>(std::min<uint64_t>(static_cast<uint64_t>(cur), n_base), max_k_prime));
    if (out.empty() || kp > out.back()) out.push_back(kp);
    if (kp >= n_base || kp >= max_k_prime) break;
    double nxt = cur * growth;
    if (nxt < cur + 1.0) nxt = cur + 1.0;
    cur = nxt;
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
    std::cerr << "ERROR: ground truth k (" << k_gt << ") < k (" << k
              << "); not enough neighbors.\n";
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

// Load GT rows for an arbitrary sorted subset of query indices.
// `query_rows` are 0-based row ids into the full GT file.
static std::vector<std::vector<std::pair<uint32_t, float>>> load_ground_truth_subset(
    const char* path, const std::vector<uint32_t>& query_rows, uint32_t k) {
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    std::cerr << "ERROR: cannot open ground truth file: " << path << "\n";
    std::exit(1);
  }

  int k_gt = 0;
  in.read(reinterpret_cast<char*>(&k_gt), sizeof(int));
  if (k_gt < static_cast<int>(k)) {
    std::cerr << "ERROR: ground truth k (" << k_gt << ") < k (" << k
              << "); not enough neighbors.\n";
    std::exit(1);
  }
  if (query_rows.empty()) return {};

  std::vector<std::vector<std::pair<uint32_t, float>>> gt(query_rows.size());
  const size_t rec_bytes = static_cast<size_t>(k_gt) * (sizeof(float) + sizeof(uint32_t));

  size_t out_i = 0;
  uint32_t target = query_rows[out_i];
  for (uint32_t qi = 0; out_i < query_rows.size() && in; ++qi) {
    if (qi == target) {
      gt[out_i].resize(k);
      for (uint32_t j = 0; j < k; ++j) {
        float dist;
        uint32_t id;
        in.read(reinterpret_cast<char*>(&dist), sizeof(float));
        in.read(reinterpret_cast<char*>(&id), sizeof(uint32_t));
        gt[out_i][j] = {id, dist};
      }
      for (int j = static_cast<int>(k); j < k_gt; ++j) {
        float dist;
        uint32_t id;
        in.read(reinterpret_cast<char*>(&dist), sizeof(float));
        in.read(reinterpret_cast<char*>(&id), sizeof(uint32_t));
      }
      ++out_i;
      if (out_i < query_rows.size()) target = query_rows[out_i];
    } else {
      in.seekg(static_cast<std::streamoff>(rec_bytes), std::ios::cur);
    }
  }

  if (out_i != query_rows.size() || !in) {
    std::cerr << "ERROR: ground truth file too short for requested query subset.\n";
    std::exit(1);
  }

  std::cout << "Loaded ground truth subset from " << path << " (k_gt=" << k_gt
            << ", queries=" << query_rows.size() << ")\n";
  return gt;
}

template<typename ChPoint>
static int run_from_sets(const PointCloudSet<ChPoint>& db, const PointCloudSet<ChPoint>& queries,
                         uint32_t pq_block, uint32_t pq_k, uint32_t fs_block, uint32_t rbits,
                         uint32_t k, const char* gt_file = nullptr,
                         bool gt_neighbor_indices = false, const std::string& db_cache_key = "",
                         const std::string& q_cache_key = "",
                         const std::string& gt_cache_suffix = "",
                         uint32_t max_k_prime = std::numeric_limits<uint32_t>::max(),
                         double k_growth = 2.0,
                         const std::vector<uint32_t>* gt_query_rows = nullptr) {
  constexpr bool Metric = ChPoint::is_metric();

  const uint32_t D = db.get_dims();
  if (queries.get_dims() != D) {
    std::cerr << "ERROR: DB dims (" << D << ") != Query dims (" << queries.get_dims() << ")\n";
    return 1;
  }
  (void)pq_block;
  (void)pq_k;
  (void)fs_block;
  (void)rbits;

  const size_t Nclouds = db.size();
  const size_t Qclouds = queries.size();
  if (k == 0 || k > Nclouds) {
    std::cerr << "ERROR: invalid -k (k=" << k << ", Nclouds=" << Nclouds << ")\n";
    return 1;
  }
  const uint32_t kp_cap = std::min<uint32_t>(max_k_prime, static_cast<uint32_t>(Nclouds));
  const auto kprime_grid = make_k_primes(k, static_cast<uint32_t>(Nclouds), kp_cap, k_growth);

  std::cout << "DB: clouds=" << Nclouds << " dims=" << D << " total_vecs=" << db.total_size()
            << " avg_k=" << std::fixed << std::setprecision(2) << db.average_size() << "\n";
  std::cout << "Q : clouds=" << Qclouds << " dims=" << D << " total_vecs=" << queries.total_size()
            << " avg_k=" << std::fixed << std::setprecision(2) << queries.average_size() << "\n";
  std::cout << "dist=" << (Metric ? "L2" : "IP") << "\n";
  std::cout << "k=" << k << "  max_k_prime=" << kp_cap << "  k_growth=" << k_growth
            << "  k' grid: ";
  for (auto kp : kprime_grid)
    std::cout << kp << " ";
  std::cout << "\n";
  if (gt_file) {
    std::cout << "Ground truth: " << gt_file;
    if (gt_neighbor_indices) std::cout << " (neighbor ids = DB row indices, mapped with get_id)";
    std::cout << "\n";
  }

  // Match the single-vector benchmark method family:
  // TurboQuant-4bit, TurboQuant-8bit, FastScan-b{2,4,8}, RaBitQ-{1,4,8}.
  fastscan_mv::Model<Metric> fs2_model, fs4_model, fs8_model;
  const bool has_fs2 = (D % 2 == 0);
  const bool has_fs4 = (D % 4 == 0);
  const bool has_fs8 = (D % 8 == 0);
  if (has_fs2) fs2_model.train(db, 2);
  if (has_fs4) fs4_model.train(db, 4);
  if (has_fs8) fs8_model.train(db, 8);
  auto fs2_db = has_fs2 ? fs2_model.encode(db) : fastscan_mv::Quantized_Point_Cloud_Set<Metric>{};
  auto fs4_db = has_fs4 ? fs4_model.encode(db) : fastscan_mv::Quantized_Point_Cloud_Set<Metric>{};
  auto fs8_db = has_fs8 ? fs8_model.encode(db) : fastscan_mv::Quantized_Point_Cloud_Set<Metric>{};

  rabitq_mv::Model<Metric> rq1_model, rq4_model, rq8_model;
  rq1_model.train(db, 1);
  rq4_model.train(db, 4);
  rq8_model.train(db, 8);
  auto rq1_db = rq1_model.encode(db);
  auto rq4_db = rq4_model.encode(db);
  auto rq8_db = rq8_model.encode(db);

  turboquant_mv::Model<Metric> tq_model;
  tq_model.train(db);
  auto tq_db = tq_model.encode(db);

  turboquant_8bit_mv::Model<Metric> tq8_model;
  tq8_model.train(db);
  auto tq8_db = tq8_model.encode(db);

  // 1BTQ-mv (sym 1-bit) and 1BTQAsym-mv (asym 1-bit DB × int4 query) — the
  // multi-vector counterparts to 1BTQ / 1BTQAsym in the singlevector bench.
  turboquant_1bit_mv::Model<Metric> tq1_model;
  tq1_model.train(db);
  auto tq1_db = tq1_model.encode(db);

  turboquant_1bit_asym_mv::Model<Metric> tq1a_model;
  tq1a_model.train(db);
  auto tq1a_db = tq1a_model.encode(db);

  enum Method {
    TURBOQUANT_MV = 0,
    TURBOQUANT_8BIT_MV = 1,
    FASTSCAN_B2 = 2,
    FASTSCAN_B4 = 3,
    FASTSCAN_B8 = 4,
    RABITQ_B1 = 5,
    RABITQ_B4 = 6,
    RABITQ_B8 = 7,
    ONEBITTQ_MV = 8,
    ONEBITTQ_ASYM_MV = 9,
    NUM_METHODS = 10
  };
  const char* method_names[NUM_METHODS] = {
      "TurboQuant_mv (4bit)", "TurboQuant_mv (8bit)", "FastScan-b2", "FastScan-b4", "FastScan-b8",
      "RaBitQ-1bit",          "RaBitQ-4bit",          "RaBitQ-8bit", "1BTQ-mv",     "1BTQAsym-mv",
  };

  std::vector<double> sum_recall(NUM_METHODS * kprime_grid.size(), 0.0);
  auto idx2 = [&](Method m, size_t i) { return static_cast<size_t>(m) * kprime_grid.size() + i; };

  // ---------------------------------------------------------------------
  // Phase 1: Build per-query exact-top-k id sets (the "ground truth" used
  // for recall denominators).
  //   - If -gt was given: load that file (with optional row-index mapping).
  //   - Else if a /tmp cache hit exists for (db, q, N, Q, k, metric): load it.
  //   - Else: per-query brute force (db.distances + sort), then write cache.
  // The cache stores top-k logical ids only (we never need exact distances
  // for the recall denominator).
  // ---------------------------------------------------------------------
  std::vector<std::unordered_set<uint32_t>> exact_sets(Qclouds);

  // Cache path is only set in file mode (synth mode passes empty keys).
  auto make_gt_cache_path = [&]() -> std::string {
    if (db_cache_key.empty() || q_cache_key.empty()) return "";
    std::string p = "/tmp/chamfer_gt_" + db_cache_key + "_" + q_cache_key + "_n" +
                    std::to_string(Nclouds) + "_q" + std::to_string(Qclouds) + "_k" +
                    std::to_string(k) + "_" + (Metric ? "L2" : "IP");
    if (!gt_cache_suffix.empty()) p += "_" + gt_cache_suffix;
    return p + ".bin";
  };

  // Cache file format: int32 nq, int32 k, then nq*k uint32 logical ids
  // (one row per query, sorted ascending by distance). Read returns true on
  // success and populates `exact_sets` directly.
  auto try_load_cached_gt = [&](const std::string& path) -> bool {
    if (path.empty() || !std::filesystem::exists(path)) return false;
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) return false;
    int32_t nq_i32 = 0, k_i32 = 0;
    in.read(reinterpret_cast<char*>(&nq_i32), sizeof(int32_t));
    in.read(reinterpret_cast<char*>(&k_i32), sizeof(int32_t));
    if (!in || static_cast<size_t>(nq_i32) != Qclouds || static_cast<uint32_t>(k_i32) != k) {
      std::cerr << "WARNING: cached GT header mismatch at " << path << " (got nq=" << nq_i32
                << " k=" << k_i32 << "); recomputing.\n";
      return false;
    }
    std::vector<uint32_t> ids(static_cast<size_t>(nq_i32) * static_cast<size_t>(k_i32));
    in.read(reinterpret_cast<char*>(ids.data()), ids.size() * sizeof(uint32_t));
    if (!in) {
      std::cerr << "WARNING: cached GT file truncated at " << path << "; recomputing.\n";
      return false;
    }
    parlay::parallel_for(0, Qclouds, [&](size_t qi) {
      auto& es = exact_sets[qi];
      es.reserve(static_cast<size_t>(k) * 2);
      for (uint32_t j = 0; j < k; ++j)
        es.insert(ids[qi * static_cast<size_t>(k) + static_cast<size_t>(j)]);
    });
    std::cout << "Loaded cached GT from " << path << "\n";
    return true;
  };

  auto save_cached_gt = [&](const std::string& path) {
    if (path.empty()) return;
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
      std::cerr << "WARNING: could not open " << path << " for GT cache write.\n";
      return;
    }
    int32_t nq_i32 = static_cast<int32_t>(Qclouds);
    int32_t k_i32 = static_cast<int32_t>(k);
    out.write(reinterpret_cast<const char*>(&nq_i32), sizeof(int32_t));
    out.write(reinterpret_cast<const char*>(&k_i32), sizeof(int32_t));
    std::vector<uint32_t> flat(Qclouds * static_cast<size_t>(k));
    for (size_t qi = 0; qi < Qclouds; ++qi) {
      // Set iteration order is unstable, but we only check membership on read,
      // so order in the cache doesn't matter as long as the same ids are stored.
      size_t j = 0;
      for (uint32_t id : exact_sets[qi]) {
        flat[qi * static_cast<size_t>(k) + j++] = id;
        if (j >= static_cast<size_t>(k)) break;
      }
    }
    out.write(reinterpret_cast<const char*>(flat.data()), flat.size() * sizeof(uint32_t));
    if (!out) {
      std::cerr << "WARNING: short write to GT cache " << path << "\n";
      return;
    }
    std::cout << "  cached GT to " << path << "\n";
  };

  if (gt_file) {
    std::vector<std::vector<std::pair<uint32_t, float>>> gt_data;
    if (gt_query_rows != nullptr && !gt_query_rows->empty()) {
      gt_data = load_ground_truth_subset(gt_file, *gt_query_rows, k);
    } else {
      gt_data = load_ground_truth(gt_file, Qclouds, k);
    }
    parlay::parallel_for(0, Qclouds, [&](size_t qi) {
      auto& es = exact_sets[qi];
      es.reserve(static_cast<size_t>(k) * 2);
      for (uint32_t j = 0; j < k; ++j) {
        uint32_t nid = gt_data[qi][j].first;
        if (gt_neighbor_indices) {
          if (nid >= Nclouds) {
            std::cerr << "ERROR: ground-truth neighbor index " << nid << " >= Nclouds " << Nclouds
                      << " (query " << qi << ")\n";
            std::abort();
          }
          nid = db.get_id(static_cast<size_t>(nid));
        }
        es.insert(nid);
      }
    });
  } else {
    const std::string cache_path = make_gt_cache_path();
    if (!try_load_cached_gt(cache_path)) {
      std::cout << "Computing exact GT (brute force)..." << std::endl;
      parlay::internal::timer t;
      t.start();
      std::vector<std::pair<uint32_t, float>> scratch(Nclouds);
      for (size_t qi = 0; qi < Qclouds; ++qi) {
        db.distances(queries[qi], scratch.data());
        std::sort(scratch.begin(), scratch.end(),
                  [](const auto& a, const auto& b) { return a.second < b.second; });
        auto& es = exact_sets[qi];
        es.reserve(static_cast<size_t>(k) * 2);
        for (uint32_t j = 0; j < k; ++j)
          es.insert(scratch[j].first);
      }
      std::cout << "  GT: " << t.stop() << "s" << std::endl;
      save_cached_gt(cache_path);
    }
  }

  // ---------------------------------------------------------------------
  // Phase 2: Pre-quantize queries for all methods.
  // Some quantizers use mutable model internals during quantize_query (e.g.,
  // temporary rotator/model buffers), so doing this once up front avoids
  // potential cross-thread races and keeps phase-3 purely distance-eval.
  // ---------------------------------------------------------------------
  using fs_qq_t = decltype(fs2_model.quantize_query(queries[0]));
  std::vector<fs_qq_t> fs2_qs, fs4_qs, fs8_qs;
  if (has_fs2) {
    fs2_qs.reserve(Qclouds);
    for (size_t qi = 0; qi < Qclouds; ++qi)
      fs2_qs.push_back(fs2_model.quantize_query(queries[qi]));
  }
  if (has_fs4) {
    fs4_qs.reserve(Qclouds);
    for (size_t qi = 0; qi < Qclouds; ++qi)
      fs4_qs.push_back(fs4_model.quantize_query(queries[qi]));
  }
  if (has_fs8) {
    fs8_qs.reserve(Qclouds);
    for (size_t qi = 0; qi < Qclouds; ++qi)
      fs8_qs.push_back(fs8_model.quantize_query(queries[qi]));
  }

  using rq_qq_t = decltype(rq1_model.quantize_query(queries[0]));
  std::vector<rq_qq_t> rq1_qs, rq4_qs, rq8_qs;
  rq1_qs.reserve(Qclouds);
  rq4_qs.reserve(Qclouds);
  rq8_qs.reserve(Qclouds);
  for (size_t qi = 0; qi < Qclouds; ++qi) {
    rq1_qs.push_back(rq1_model.quantize_query(queries[qi]));
    rq4_qs.push_back(rq4_model.quantize_query(queries[qi]));
    rq8_qs.push_back(rq8_model.quantize_query(queries[qi]));
  }

  using tq_qq_t = decltype(tq_model.quantize_query(queries[0]));
  std::vector<tq_qq_t> tq_qs;
  tq_qs.reserve(Qclouds);
  for (size_t qi = 0; qi < Qclouds; ++qi)
    tq_qs.push_back(tq_model.quantize_query(queries[qi]));

  using tq8_qq_t = decltype(tq8_model.quantize_query(queries[0]));
  std::vector<tq8_qq_t> tq8_qs;
  tq8_qs.reserve(Qclouds);
  for (size_t qi = 0; qi < Qclouds; ++qi)
    tq8_qs.push_back(tq8_model.quantize_query(queries[qi]));

  using tq1_qq_t = decltype(tq1_model.quantize_query(queries[0]));
  std::vector<tq1_qq_t> tq1_qs;
  tq1_qs.reserve(Qclouds);
  for (size_t qi = 0; qi < Qclouds; ++qi)
    tq1_qs.push_back(tq1_model.quantize_query(queries[qi]));

  using tq1a_qq_t = decltype(tq1a_model.quantize_query(queries[0]));
  std::vector<tq1a_qq_t> tq1a_qs;
  tq1a_qs.reserve(Qclouds);
  for (size_t qi = 0; qi < Qclouds; ++qi)
    tq1a_qs.push_back(tq1a_model.quantize_query(queries[qi]));

  // ---------------------------------------------------------------------
  // Phase 3: Serial outer loop over queries. Each MV `distances_all` is
  // already internally parallelized over Nclouds (Nclouds=8674 >> Qclouds
  // for typical BEIR datasets, so the DB-axis is the better parallelism
  // boundary anyway). Nesting our own parallel_for here on top of those
  // inner parallel_fors causes parlay scheduler trouble (heavy worker
  // stack pressure -> heap corruption / SIGSEGV at large Q). Mirrors the
  // way `bench_singlevector_overretrieve` calls into kernels that own
  // their own parallelism.
  // ---------------------------------------------------------------------
  std::vector<std::pair<uint32_t, float>> approx(Nclouds);
  std::cout << "Outer iterations: " << Qclouds << " (serial; inner kernels are parallel)"
            << std::endl;

  auto eval_method = [&](Method meth, size_t qi) {
    std::sort(approx.begin(), approx.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });
    // distances_all() stores logical cloud ids (same as db.get_id(cid)) in .first.
    const auto& exact_set = exact_sets[qi];
    for (size_t i = 0; i < kprime_grid.size(); ++i) {
      const size_t kp = static_cast<size_t>(kprime_grid[i]);
      size_t hits = 0;
      const size_t limit = std::min(kp, approx.size());
      for (size_t j = 0; j < limit; ++j) {
        if (exact_set.find(approx[j].first) != exact_set.end()) ++hits;
      }
      sum_recall[idx2(meth, i)] += static_cast<double>(hits) / static_cast<double>(k);
    }
  };

  for (size_t qi = 0; qi < Qclouds; ++qi) {
    if (has_fs2) {
      fs2_db.distances_all(fs2_qs[qi], approx.data());
      eval_method(FASTSCAN_B2, qi);
    }
    if (has_fs4) {
      fs4_db.distances_all(fs4_qs[qi], approx.data());
      eval_method(FASTSCAN_B4, qi);
    }
    if (has_fs8) {
      fs8_db.distances_all(fs8_qs[qi], approx.data());
      eval_method(FASTSCAN_B8, qi);
    }
    tq_db.distances_all(tq_qs[qi], approx.data());
    eval_method(TURBOQUANT_MV, qi);
    tq8_db.distances_all(tq8_qs[qi], approx.data());
    eval_method(TURBOQUANT_8BIT_MV, qi);
    rq1_db.distances_all(rq1_qs[qi], approx.data());
    eval_method(RABITQ_B1, qi);
    rq4_db.distances_all(rq4_qs[qi], approx.data());
    eval_method(RABITQ_B4, qi);
    rq8_db.distances_all(rq8_qs[qi], approx.data());
    eval_method(RABITQ_B8, qi);
    tq1_db.distances_all(tq1_qs[qi], approx.data());
    eval_method(ONEBITTQ_MV, qi);
    tq1a_db.distances_all(tq1a_qs[qi], approx.data());
    eval_method(ONEBITTQ_ASYM_MV, qi);
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
    if (meth == FASTSCAN_B2 && !has_fs2) continue;
    if (meth == FASTSCAN_B4 && !has_fs4) continue;
    if (meth == FASTSCAN_B8 && !has_fs8) continue;
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
                     uint32_t rbits, uint32_t k, uint32_t max_k_prime, double k_growth) {
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
                                /*gt_file=*/nullptr, /*gt_neighbor_indices=*/false, "", "", "",
                                max_k_prime, k_growth);
}

template<typename ChPoint>
static int run_files(commandLine& P, uint32_t pq_block, uint32_t pq_k, uint32_t fs_block,
                     uint32_t rbits, uint32_t k, uint32_t max_k_prime, double k_growth,
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
  auto queries_full = PC(qFile, /*is_mmap=*/false);
  const long num_query_raw = P.getOptionLongValue("-num_query", 1000);
  const uint64_t subsample_seed =
      static_cast<uint64_t>(P.getOptionLongValue("-query_subsample_seed", 42));
  const size_t q_full = queries_full.size();
  const size_t target_n = (num_query_raw <= 0) ? q_full : std::min<size_t>(q_full, num_query_raw);

  std::vector<uint32_t> selected_rows;
  selected_rows.reserve(target_n);
  if (target_n < q_full) {
    std::vector<uint32_t> perm(q_full);
    for (size_t i = 0; i < q_full; ++i)
      perm[i] = static_cast<uint32_t>(i);
    std::mt19937_64 rng(subsample_seed);
    std::shuffle(perm.begin(), perm.end(), rng);
    perm.resize(target_n);
    std::sort(perm.begin(), perm.end());
    selected_rows = std::move(perm);
  } else {
    selected_rows.resize(q_full);
    for (size_t i = 0; i < q_full; ++i)
      selected_rows[i] = static_cast<uint32_t>(i);
  }

  auto sample = parlay::delayed_tabulate(selected_rows.size(),
                                         [&](size_t i) { return queries_full[selected_rows[i]]; });
  auto queries = PC(sample, queries_full.get_dims());

  std::cout << "Mode: file\n";
  std::cout << "  db=" << dbFile << (mm ? " (mmap)\n" : "\n");
  std::cout << "  q =" << qFile << "\n";
  if (target_n < q_full) {
    std::cout << "  [subsample] queries " << q_full << " -> " << target_n
              << " (seed=" << subsample_seed << ")\n";
  }
  const bool gt_neighbor_indices = P.getOption("-gt_neighbor_indices");
  // File-mode cache keys for the GT cache (skipped in synth mode).
  const std::string db_cache_key = std::filesystem::path(dbFile).stem().string();
  const std::string q_cache_key = std::filesystem::path(qFile).stem().string();
  const std::string gt_cache_suffix =
      (target_n < q_full)
          ? ("subsample_n" + std::to_string(target_n) + "_seed" + std::to_string(subsample_seed))
          : std::string();
  return run_from_sets<ChPoint>(db, queries, pq_block, pq_k, fs_block, rbits, k, gt_file,
                                gt_neighbor_indices, db_cache_key, q_cache_key, gt_cache_suffix,
                                max_k_prime, k_growth, &selected_rows);
}

int main(int argc, char** argv) {
  commandLine P(argc, argv,
                "[-i <dbFile>] [-q <qFile>] [-mm] [-gt <gtFile>] [-gt_neighbor_indices] "
                "[-num_query <n>] [-query_subsample_seed <u64>] "
                "[-N_db <n>] [-N_q <n>] [-K_db <k>] [-D <d>] [-seed_db <s>] [-seed_q <s>] "
                "[-dist_func <L2|IP>] [-pq_block <b>] [-pq_k <k>] [-fs_block <b>] [-rbits <b>] "
                "[-k <k>] [-max_k_prime <n>] [-k_growth <r>]");

  std::string df = P.getOptionValue("-dist_func", "L2");
  uint32_t pq_block = static_cast<uint32_t>(P.getOptionIntValue("-pq_block", 8));
  uint32_t pq_k = static_cast<uint32_t>(P.getOptionIntValue("-pq_k", 16));
  uint32_t fs_block = static_cast<uint32_t>(P.getOptionIntValue("-fs_block", 8));
  uint32_t rbits = static_cast<uint32_t>(P.getOptionIntValue("-rbits", 2));
  uint32_t k = static_cast<uint32_t>(P.getOptionIntValue("-k", 10));
  uint32_t max_k_prime = static_cast<uint32_t>(P.getOptionLongValue("-max_k_prime", 20000));
  double k_growth = P.getOptionDoubleValue("-k_growth", 2.0);
  const char* gt_file = P.getOptionValue("-gt");

  const bool file_mode = (P.getOptionValue("-i") != nullptr) || (P.getOptionValue("-q") != nullptr);
  if (file_mode) {
    if (P.getOptionValue("-i") == nullptr || P.getOptionValue("-q") == nullptr) {
      std::cerr << "ERROR: Provide both -i <dbFile> and -q <qFile> for file mode.\n";
      return 1;
    }
    if (df == "IP" || df == "ip")
      return run_files<ChamferIP_Point>(P, pq_block, pq_k, fs_block, rbits, k, max_k_prime,
                                        k_growth, gt_file);
    return run_files<ChamferL2_Point>(P, pq_block, pq_k, fs_block, rbits, k, max_k_prime, k_growth,
                                      gt_file);
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
                                      rbits, k, max_k_prime, k_growth);
  }
  return run_synth<ChamferL2_Point>(N_db, N_q, K_db, D, seed_db, seed_q, pq_block, pq_k, fs_block,
                                    rbits, k, max_k_prime, k_growth);
}
