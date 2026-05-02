// bench_singlevector_overretrieve.cpp
// Single-vector analog of bench_chamfer_overretrieve for TurboQuant quality evaluation.
// Measures recall@k vs k' for TurboQuant, TQ-PQ-4bit, and RaBitQ on real datasets.
//
// Usage:
//   ./bench_singlevector_overretrieve -i <base_file> -q <query_file> [-gt <gt_file>]
//     [-k <k>] [-dist_func L2|IP]
//     [-pq_method TQ4|TurboQuant|TQ8|FastScan|1BTQ|1BTQAsym|Ref1BTQAsym|Ref1BTQSym|RabitQ|RabitQ1|RabitQ4|RabitQ8|All]
//     [-dataset_as_query] [-max_k_prime <N>] [-k_growth <rate>]
//     [-rabitq_bits <bits>] [-fs_block <bits>] [-output_gt_path <path>]
//     [-pcs]   # load -i/-q as chamfer .pcs files; the per-cloud structure is
//              # discarded and every individual embedding becomes a single
//              # base/query vector.

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <queue>
#include <limits>
#include <atomic>
#include <random>
#include <numeric>
#include <fstream>
#include <filesystem>

#include <Eigen/Dense>

#include "parlay/parallel.h"
#include "parlay/primitives.h"

#include "mvsic/core/types/point_range.h"
#include "mvsic/core/types/l2_point.h"
#include "mvsic/core/types/ip_point.h"
#include "mvsic/core/utils/mmap.h"
#include "mvsic/core/utils/parse_command_line.h"
#include <sys/mman.h>
#include "mvsic/core/quantization/fastscan.h"
#include "mvsic/core/quantization/rabitq.h"
#include "mvsic/core/quantization/turboquant.h"
#include "mvsic/core/quantization/turboquant_1bit.h"
#include "mvsic/core/quantization/turboquant_1bit_asym.h"
#include "mvsic/core/quantization/turboquant_1bit_ref.h"
#include "mvsic/core/quantization/turboquant_8bit_mv.h"
#include "mvsic/core/stats.h"

using namespace mvsic;

// ---- Ground truth: blocked Eigen GEMM ----
template<typename Point, bool Metric>
parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>> compute_ground_truth(
    const mvsic::PointRange<float, Point>& queries,
    const mvsic::PointRange<float, Point>& base_points, size_t k) {
  const size_t n_q = queries.size();
  const size_t n_b = base_points.size();
  const size_t dim = queries.get_dims();
  const Eigen::Index aligned_q = queries.get_aligned_dims();
  const Eigen::Index aligned_b = base_points.get_aligned_dims();
  const Eigen::Index D = static_cast<Eigen::Index>(dim);

  parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>> results(n_q);

  constexpr size_t BQ = 128, BB = 4096;

  parlay::parallel_for(0, (n_q + BQ - 1) / BQ, [&](size_t bq) {
    size_t sq = bq * BQ, eq = std::min(sq + BQ, n_q);
    Eigen::Index nq_block = eq - sq;

    using RowMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using S = Eigen::Stride<Eigen::Dynamic, 1>;
    Eigen::Map<const RowMatrix, 0, S> Q(reinterpret_cast<const float*>(queries.location(sq)),
                                        nq_block, D, S(aligned_q, 1));

    Eigen::VectorXf Qn;
    if constexpr (Metric) Qn = Q.rowwise().squaredNorm();

    std::vector<std::priority_queue<std::pair<float, uint32_t>>> heaps(nq_block);

    for (size_t bb = 0; bb < (n_b + BB - 1) / BB; ++bb) {
      size_t sb = bb * BB, eb = std::min(sb + BB, n_b);
      Eigen::Index nb_block = eb - sb;

      Eigen::Map<const RowMatrix, 0, S> B(reinterpret_cast<const float*>(base_points.location(sb)),
                                          nb_block, D, S(aligned_b, 1));

      Eigen::MatrixXf Dist = Q * B.transpose();
      if constexpr (Metric) {
        Eigen::VectorXf Bn = B.rowwise().squaredNorm();
        Dist *= -2.0f;
        Dist.colwise() += Qn;
        Dist.rowwise() += Bn.transpose();
      } else {
        Dist *= -1.0f;
      }

      for (Eigen::Index i = 0; i < nq_block; ++i) {
        auto& h = heaps[i];
        for (Eigen::Index j = 0; j < nb_block; ++j) {
          float d = Dist(i, j);
          uint32_t id = static_cast<uint32_t>(sb + j);
          if (h.size() < k)
            h.push({d, id});
          else if (d < h.top().first) {
            h.pop();
            h.push({d, id});
          }
        }
      }
    }

    for (Eigen::Index i = 0; i < nq_block; ++i) {
      auto& h = heaps[i];
      size_t sz = h.size();
      results[sq + i].resize(sz);
      for (size_t j = 0; j < sz; ++j) {
        results[sq + i][sz - 1 - j] = {h.top().second, h.top().first};
        h.pop();
      }
    }
  });
  return results;
}

// ---- Loader for chamfer .pcs files (explode pointclouds into a flat PR) ----
// .pcs header: dims (size_t), n_clouds (size_t), num_vectors (size_t),
// followed by num_vectors*dims floats and trailing offset metadata we don't need.
//
// We mmap the file and hand the PointRange a shared_ptr that points directly
// at the embedding region of the mapping, with a custom deleter that munmaps
// on destruction. No copy and no allocation beyond the kernel's page cache —
// pages are faulted in on first touch and reclaimable by the OS under pressure.
// Requires dims*sizeof(float) to be a multiple of 64B so PointRange's
// aligned_dims == dims (no per-row padding); all BEIR .pcs files have dims=128.
template<typename Point>
mvsic::PointRange<float, Point> load_pcs_as_pointrange(const char* filename) {
  std::cout << "Loading exploded PCS (mmap) from " << filename << "..." << std::endl;
  auto mapping = mmap_file(filename);
  char* addr = mapping.first;
  size_t length = mapping.second;
  const size_t* hdr = reinterpret_cast<const size_t*>(addr);
  const size_t dims = hdr[0];
  const size_t n_clouds = hdr[1];
  const size_t num_vec = hdr[2];
  std::cout << "  dims=" << dims << " clouds=" << n_clouds << " embeddings=" << num_vec
            << std::endl;

  if (mvsic::dim_round_up(static_cast<long>(dims), sizeof(float)) !=
      static_cast<long>(dims)) {
    munmap(addr, length);
    std::cerr << "PCS mmap loader requires dims (=" << dims
              << ") to be a multiple of 16 floats (64B cacheline)." << std::endl;
    std::exit(1);
  }

  mvsic::PointRange<float, Point> pr;
  float* values_start = reinterpret_cast<float*>(addr + 3 * sizeof(size_t));
  pr.values = std::shared_ptr<float[]>(
      values_start, [addr, length](float*) { munmap(addr, length); });
  pr.n = num_vec;
  pr.dims = static_cast<unsigned>(dims);
  pr.aligned_dims = static_cast<unsigned>(dims);
  return pr;
}

// ---- Subset wrapper for dataset-as-query ----
template<typename PR>
struct PointRangeSubsetWrapper {
  const PR& pr;
  const std::vector<size_t>& idxs;
  size_t size() const { return idxs.size(); }
  auto operator[](size_t i) const { return pr[static_cast<long>(idxs[i])]; }
};

// ---- Generate k' schedule (geometric progression) ----
std::vector<size_t> make_k_primes(size_t k, size_t n_base, size_t max_k_prime, double growth) {
  std::vector<size_t> kps;
  double cur = (double)k;
  while (true) {
    size_t kp = std::min({(size_t)cur, n_base, max_k_prime});
    if (kps.empty() || kp > kps.back()) kps.push_back(kp);
    if (kp >= n_base || kp >= max_k_prime) break;
    double nxt = cur * growth;
    if (nxt < cur + 1.0) nxt = cur + 1.0;
    cur = nxt;
  }
  return kps;
}

// ---- Recall curve: generic over distance function ----
//
// Query-blocked sweep. Naively this loop iterates n_q × n_b distance evals,
// and the obvious "parallel_for over queries, walk all base" structure
// re-streams the entire encoded base once per query — for n_q=20K, n_b=2.3M
// that's 20K independent sweeps over a multi-hundred-MB array, which is
// DRAM-bandwidth-bound on every method.
//
// Instead we tile queries into blocks of BQ. Each block sweeps the encoded
// base once; the inner loop iterates BQ queries so enc[j] stays hot in L1
// across BQ uses. This collapses base-side memory traffic by BQ× without
// changing the per-pair distance kernel — the dist_fn closure is unchanged
// across all methods.
//
// We also drop the per-query std::vector<pair>(n_b) buffer (28 MB at this
// problem size) and use a top-lim max-heap instead. For lim = k_primes.back()
// (typically ≤ 20K << n_b) this turns a 28 MB allocation per query into a
// ~240 KB heap, and turns nth_element + sort over n_b into a heap walk over
// ~lim insertions in expectation (most pairs after the heap fills are early-
// exited by `d < h.top()`).
template<typename DistFn>
void recall_curve(DistFn&& dist_fn, size_t n_q, size_t n_b,
                  const parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>>& gt,
                  size_t k, const std::vector<size_t>& k_primes, const std::string& label) {

  parlay::internal::timer search_t;
  search_t.start();

  std::vector<std::atomic<size_t>> tc(k_primes.size());
  for (auto& x : tc)
    x = 0;
  const size_t lim = k_primes.back();

  // BQ sized to keep the per-thread working set (BQ × per-query state +
  // BQ × heap of top-lim pairs) inside L2 while still amortizing base reads.
  // Quantized_Query / int8 query-code state is small (≤ a few hundred bytes),
  // and 32 × 12B × lim ≤ ~7.5 MB at lim=20K which fits comfortably in L3.
  constexpr size_t BQ = 32;
  const size_t num_blocks = (n_q + BQ - 1) / BQ;

  parlay::parallel_for(0, num_blocks, [&](size_t qb) {
    const size_t q_start = qb * BQ;
    const size_t q_end = std::min(q_start + BQ, n_q);
    const size_t cur = q_end - q_start;

    using Pair = std::pair<float, uint32_t>;  // (distance, base id) — heap by distance
    std::vector<std::priority_queue<Pair>> heaps(cur);

    // Streaming base sweep. The j loop is the cache-miss-driver; the inner
    // qi loop reuses enc[j] BQ times before it's evicted.
    for (size_t j = 0; j < n_b; ++j) {
      const uint32_t jid = static_cast<uint32_t>(j);
      for (size_t li = 0; li < cur; ++li) {
        const float d = dist_fn(q_start + li, j);
        auto& h = heaps[li];
        if (h.size() < lim) {
          h.emplace(d, jid);
        } else if (d < h.top().first) {
          h.pop();
          h.emplace(d, jid);
        }
      }
    }

    // Drain each heap into a sorted-ascending id list, then reuse the
    // existing per-query recall-counting logic.
    std::vector<uint32_t> ids;
    ids.reserve(lim);
    for (size_t li = 0; li < cur; ++li) {
      const size_t qi = q_start + li;
      auto& h = heaps[li];
      const size_t hsz = h.size();
      ids.assign(hsz, 0u);
      for (size_t r = 0; r < hsz; ++r) {
        ids[hsz - 1 - r] = h.top().second;
        h.pop();
      }

      const size_t ak = std::min(k, gt[qi].size());
      if (ak == 0) continue;

      size_t cc = 0, ki = 0;
      for (size_t r = 0; r < hsz && ki < k_primes.size(); ++r) {
        const uint32_t rid = ids[r];
        for (size_t g = 0; g < ak; ++g) {
          if (gt[qi][g].first == rid) {
            ++cc;
            break;
          }
        }
        while (ki < k_primes.size() && r + 1 == k_primes[ki]) {
          tc[ki].fetch_add(cc, std::memory_order_relaxed);
          ++ki;
        }
      }
    }
  });

  const double search_secs = search_t.stop();

  std::cout << "\n=== " << label << " ===" << std::endl;
  std::cout << "  search: " << std::fixed << std::setprecision(4) << search_secs << "s"
            << "  (n_q=" << n_q << " × n_b=" << n_b << " distance evals + recall counting)"
            << std::endl;
  std::cout << std::setw(10) << "k'" << std::setw(15) << "Recall@" << k << std::endl;
  std::cout << "----------------------------------------" << std::endl;
  // Effective k per query is limited by available ground-truth neighbors.
  const size_t ak0 = gt.empty() ? 0ul : std::min(k, gt[0].size());
  for (size_t i = 0; i < k_primes.size(); ++i) {
    double denom = static_cast<double>(n_q) * static_cast<double>(ak0 == 0 ? 1ul : ak0);
    double rec = denom > 0.0 ? static_cast<double>(tc[i]) / denom : 0.0;
    std::cout << std::setw(10) << k_primes[i] << std::setw(15) << std::fixed << std::setprecision(4)
              << rec << std::endl;
  }
}

// ---- Main benchmark ----
template<typename Point, bool Metric>
void run_benchmark(commandLine& P) {
  char* inFile = P.getOptionValue("-i");
  char* qFile = P.getOptionValue("-q");
  bool dataset_as_query = P.getOption("-dataset_as_query");
  std::string gtFile = P.getOptionValue("-gt", "");
  size_t k = P.getOptionLongValue("-k", 10);
  std::string method = P.getOptionValue("-pq_method", "All");
  size_t max_kp = P.getOptionLongValue("-max_k_prime", 20000);
  double growth = P.getOptionDoubleValue("-k_growth", 2.0);
  uint32_t rbits = P.getOptionIntValue("-rabitq_bits", 8);
  uint32_t fs_block = P.getOptionIntValue("-fs_block", 8);
  size_t num_query = P.getOptionLongValue("-num_query", 0);
  bool use_pcs = P.getOption("-pcs");

  if (!inFile || (!qFile && !dataset_as_query)) {
    std::cerr << "Usage: bench_singlevector_overretrieve -i <base> [-q <queries> | -dataset_as_query]\n"
              << "  [-gt <gt>] [-k <k>] [-dist_func L2|IP]\n"
              << "  [-pq_method "
                 "TQ4|TurboQuant|TQ8|FastScan|1BTQ|1BTQAsym|Ref1BTQAsym|Ref1BTQSym|RabitQ|RabitQ1|RabitQ4|RabitQ8|All]\n"
              << "  [-max_k_prime <N>] [-k_growth <r>]\n"
              << "  [-rabitq_bits <b>] [-fs_block <b>] [-num_query <N>]\n"
              << "  [-pcs]    # load -i/-q as chamfer .pcs files and explode all\n"
              << "            # individual embeddings into the single-vector dataset\n";
    return;
  }

  using PR = mvsic::PointRange<float, Point>;

  PR base = use_pcs ? load_pcs_as_pointrange<Point>(inFile)
                    : (std::cout << "Loading base from " << inFile << "..." << std::endl, PR(inFile));

  PR queries_obj;
  std::vector<size_t> q_idx;
  if (dataset_as_query) {
    size_t nq = std::min((size_t)1000, base.size());
    if (num_query > 0) nq = std::min(nq, num_query);
    std::vector<size_t> all(base.size());
    std::iota(all.begin(), all.end(), 0);
    std::mt19937 rng(42);
    std::shuffle(all.begin(), all.end(), rng);
    q_idx.assign(all.begin(), all.begin() + nq);
    PointRangeSubsetWrapper<PR> w{base, q_idx};
    queries_obj = PR(w, base.get_dims());
    if (gtFile != "") {
      std::cout << "WARNING: ignoring -gt with -dataset_as_query\n";
      gtFile = "";
    }
  } else {
    queries_obj = use_pcs
                      ? load_pcs_as_pointrange<Point>(qFile)
                      : (std::cout << "Loading queries from " << qFile << "..." << std::endl,
                         PR(qFile));
    if (num_query > 0 && num_query < queries_obj.size()) {
      std::vector<size_t> idx(num_query);
      std::iota(idx.begin(), idx.end(), 0);
      PointRangeSubsetWrapper<PR> w{queries_obj, idx};
      PR trimmed(w, queries_obj.get_dims());
      queries_obj = std::move(trimmed);
      std::cout << "  using first " << num_query << " queries" << std::endl;
    }
  }
  PR& queries = queries_obj;

  // Ground truth: auto-cache to /tmp.
  parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>> gt;
  const size_t n_q = queries.size(), n_b = base.size(), D = queries.get_dims();

  // Build a deterministic cache path from dataset basename + query/base counts + metric.
  auto make_gt_cache_path = [&]() -> std::string {
    std::string base_name = std::filesystem::path(inFile).stem().string();
    return "/tmp/gt_cache_" + base_name + "_q" + std::to_string(n_q) + "_n" + std::to_string(n_b) +
           "_" + (Metric ? "L2" : "IP") + ".bin";
  };

  if (gtFile != "") {
    std::cout << "Loading GT from " << gtFile << "..." << std::endl;
    gt = ReadGT(gtFile, n_q);
  } else {
    std::string cache_path = make_gt_cache_path();
    if (std::filesystem::exists(cache_path)) {
      std::cout << "Loading cached GT from " << cache_path << "..." << std::endl;
      gt = ReadGT(cache_path, n_q);
    } else {
      std::cout << "Computing exact k-NN..." << std::endl;
      parlay::internal::timer tt;
      tt.start();
      gt = compute_ground_truth<Point, Metric>(queries, base, std::max(k, (size_t)100));
      std::cout << "GT: " << tt.stop() << "s" << std::endl;

      // Save to cache.
      size_t gt_k = gt.size() > 0 ? gt[0].size() : 0;
      std::ofstream out(cache_path, std::ios::binary);
      if (out.is_open()) {
        int32_t nn = static_cast<int32_t>(gt_k);
        out.write(reinterpret_cast<const char*>(&nn), sizeof(nn));
        for (size_t i = 0; i < n_q; ++i) {
          // ReadGT expects pair<float, uint32_t> on disk, flipped to <uint32_t, float> on read.
          for (size_t j = 0; j < gt_k; ++j) {
            float dist = gt[i][j].second;
            uint32_t id = gt[i][j].first;
            out.write(reinterpret_cast<const char*>(&dist), sizeof(dist));
            out.write(reinterpret_cast<const char*>(&id), sizeof(id));
          }
        }
        out.close();
        std::cout << "  cached GT to " << cache_path << " (" << n_q << " queries, k=" << gt_k << ")"
                  << std::endl;
      }
    }
  }

  std::cout << "\nN=" << n_b << " Q=" << n_q << " D=" << D << " dist=" << (Metric ? "L2" : "IP")
            << " k=" << k << std::endl;

  auto kps = make_k_primes(k, n_b, max_kp, growth);

  // ==== TurboQuant (4-bit codes; single-vector API) ====
  if (method == "TQ4" || method == "TurboQuant" || method == "All") {
    std::cout << "\n--- TurboQuant ---" << std::endl;
    parlay::internal::timer t;
    t.start();
    turboquant::Model<Metric> model;
    model.train(base);
    auto enc = model.encode(base);
    std::cout << "  encode: " << t.stop() << "s" << std::endl;

    std::vector<turboquant::Quantized_Query<Metric>> qqs(n_q);
    parlay::parallel_for(0, n_q, [&](size_t i) { qqs[i] = model.quantize_query(queries[i]); });

    recall_curve(
        [&](size_t qi, size_t j) { return enc[j].distance(qqs[qi]); }, n_q, n_b, gt, k, kps,
        "TurboQuant");
  }

  // ==== TurboQuant 8-bit (per-point adaptive max-abs scaling) ====
  // The new 8-bit TQ only ships a multi-vector chamfer kernel
  // (turboquant_8bit_mv). For single-vector quality evaluation we reuse the
  // public per-point encoder (Model::encode_single) on each base point and
  // inline the per-query int8 encoding (the MV quantize_query wraps it inside
  // a multi-vector ChPoint loop). The recall a quantization method achieves
  // is fully determined by encode_single + how the dot is reconstructed; the
  // panel/VPDPBUSD layout is irrelevant to quality.
  if (method == "TQ8" || method == "All") {
    std::cout << "\n--- TurboQuant-8bit ---" << std::endl;
    parlay::internal::timer t;
    t.start();

    turboquant_8bit_mv::Model<Metric> model;
    model.train(base);
    const size_t pdim = model.encoder.padded_dim;

    // Base: int8 codes (raw int8 stored in uint8 slot by encode_single) + nsf + sqn.
    std::vector<int8_t> b_codes(n_b * pdim, 0);
    std::vector<float> b_nsf(n_b, 0.0f);
    std::vector<float> b_sqn(n_b, 0.0f);
    parlay::parallel_for(0, n_b, [&](size_t i) {
      std::vector<float> ws(pdim);
      auto [sqn, nsf] = model.encode_single(
          reinterpret_cast<const float*>(base.location(i)),
          reinterpret_cast<uint8_t*>(b_codes.data() + i * pdim), ws);
      b_nsf[i] = nsf;
      b_sqn[i] = sqn;
    });

    // Queries: same per-point encoding (rotate -> max-abs scale to int8).
    std::vector<int8_t> q_codes(n_q * pdim, 0);
    std::vector<float> q_nsf(n_q, 0.0f);
    std::vector<float> q_sqn(n_q, 0.0f);
    parlay::parallel_for(0, n_q, [&](size_t qi) {
      std::vector<float> q_rot(pdim);
      model.encoder.rotator->rotate(reinterpret_cast<const float*>(queries.location(qi)),
                                    q_rot.data());

      float sqr_norm = 0.0f, max_value = 0.0f;
      for (size_t i = 0; i < pdim; ++i) {
        sqr_norm += q_rot[i] * q_rot[i];
        max_value = std::max(max_value, std::abs(q_rot[i]));
      }
      if (sqr_norm == 0.0f || !std::isfinite(sqr_norm) || max_value == 0.0f) return;

      const float norm = std::sqrt(sqr_norm);
      const float sf = 127.0f / max_value;
      int64_t quant_norm = 0;
      int8_t* q_out = q_codes.data() + qi * pdim;
      for (size_t i = 0; i < pdim; ++i) {
        const int snapped = static_cast<int>(std::lround(q_rot[i] * sf));
        const int8_t iv =
            static_cast<int8_t>(snapped < -127 ? -127 : (snapped > 127 ? 127 : snapped));
        q_out[i] = iv;
        quant_norm += static_cast<int64_t>(iv) * static_cast<int64_t>(iv);
      }
      q_nsf[qi] = quant_norm > 0 ? norm / std::sqrt(static_cast<float>(quant_norm)) : 0.0f;
      q_sqn[qi] = sqr_norm;
    });
    std::cout << "  encode: " << t.stop() << "s" << std::endl;

    recall_curve(
        [&](size_t qi, size_t j) {
          const int8_t* qp = q_codes.data() + qi * pdim;
          const int8_t* bp = b_codes.data() + j * pdim;
          int32_t dot = 0;
          size_t kk = 0;
#if defined(__AVX2__) || defined(__AVX512F__)
          // signed int8 * signed int8 via i8->i16 sign-extend + madd_epi16.
          __m256i acc = _mm256_setzero_si256();
          for (; kk + 32 <= pdim; kk += 32) {
            const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qp + kk));
            const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(bp + kk));
            const __m256i a_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(a));
            const __m256i a_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a, 1));
            const __m256i b_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(b));
            const __m256i b_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b, 1));
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(a_lo, b_lo));
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(a_hi, b_hi));
          }
          alignas(32) int32_t lanes[8];
          _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), acc);
          for (int x = 0; x < 8; ++x) dot += lanes[x];
#endif
          for (; kk < pdim; ++kk)
            dot += static_cast<int32_t>(qp[kk]) * static_cast<int32_t>(bp[kk]);

          const float neg_dot = -static_cast<float>(dot) * b_nsf[j] * q_nsf[qi];
          if constexpr (Metric)
            return b_sqn[j] + 2.0f * neg_dot + q_sqn[qi];
          else
            return neg_dot;
        },
        n_q, n_b, gt, k, kps, "TurboQuant-8bit");
  }

  // ==== FastScan (4-bit codes; LUT scan kernel) ====
  // Per-pair distance uses the scalar fallback in fastscan::Quantized_Query::distance,
  // which is slow but produces the same value as the SIMD distances_all kernel —
  // recall is identical and that's what this benchmark reports.
  auto run_fastscan = [&](uint32_t block) {
    if (D % block != 0) {
      std::cerr << "FastScan: D=" << D << " not divisible by block=" << block
                << "; skipping.\n";
      return;
    }
    std::cout << "\n--- FastScan (block=" << block << ") ---" << std::endl;
    parlay::internal::timer t;
    t.start();
    fastscan::Model<Metric> model;
    model.train(base, block);
    auto enc = model.encode(base);
    std::cout << "  encode: " << t.stop() << "s" << std::endl;

    std::vector<fastscan::Quantized_Query<Metric>> qqs(n_q);
    parlay::parallel_for(
        0, n_q, [&](size_t i) { qqs[i] = model.quantize_query(queries[i]); });

    recall_curve(
        [&](size_t qi, size_t j) { return enc[j].distance(qqs[qi]); }, n_q, n_b, gt, k, kps,
        "FastScan-b" + std::to_string(block));
  };

  if (method == "FastScan") {
    run_fastscan(fs_block);
  } else if (method == "All") {
    run_fastscan(2);
    run_fastscan(4);
    run_fastscan(8);
  }

  // ==== 1BTQ (single-vector 1-bit TurboQuant) ====
  if (method == "1BTQ" || method == "All") {
    std::cout << "\n--- 1BTQ ---" << std::endl;
    parlay::internal::timer t;
    t.start();
    turboquant_1bit::Model<Metric> model;
    model.train(base);
    auto enc = model.encode(base);
    std::cout << "  encode: " << t.stop() << "s" << std::endl;

    std::vector<turboquant_1bit::Quantized_Query<Metric>> qqs(n_q);
    parlay::parallel_for(0, n_q, [&](size_t i) { qqs[i] = model.quantize_query(queries[i]); });

    recall_curve(
        [&](size_t qi, size_t j) { return enc[j].distance(qqs[qi]); }, n_q, n_b, gt, k, kps,
        "1BTQ");
  }

  // ==== 1BTQAsym (1-bit DB × int4-quantized query, AVX-512 VPDPBUSD) ====
  // Same DB encoding as 1BTQ; query is rotated, max-abs scaled to [-7,+7]
  // and stored as int8 (sign-extended). Per-pair score uses one
  // VPDPBUSD per 64 dims with a 64-bit mask load from the DB sign bits.
  // Closes most of the recall gap to Ref1BTQAsym at ~2-3x the per-pair
  // cost of the symmetric 1BTQ kernel.
  if (method == "1BTQAsym" || method == "All") {
    std::cout << "\n--- 1BTQAsym ---" << std::endl;
    parlay::internal::timer t;
    t.start();
    turboquant_1bit_asym::Model<Metric> model;
    model.train(base);
    auto enc = model.encode(base);
    std::cout << "  encode: " << t.stop() << "s" << std::endl;

    std::vector<turboquant_1bit_asym::Quantized_Query<Metric>> qqs(n_q);
    parlay::parallel_for(0, n_q, [&](size_t i) { qqs[i] = model.quantize_query(queries[i]); });

    recall_curve(
        [&](size_t qi, size_t j) { return enc[j].distance(qqs[qi]); }, n_q, n_b, gt, k, kps,
        "1BTQAsym");
  }

  // ==== Ref1BTQAsym (scalar reference: 1-bit DB × full-float query) ====
  // DB encoding matches 1BTQ (rotate then sign-pack); the query is kept
  // as full-precision floats and scored against ±1 from the DB sign
  // bits. The scalar loop makes the math auditable. Use this to upper-
  // bound what the 1-bit-DB-asymmetric estimator family can achieve.
  if (method == "Ref1BTQAsym") {
    std::cout << "\n--- Ref1BTQAsym ---" << std::endl;
    parlay::internal::timer t;
    t.start();
    turboquant_1bit_ref::asym::Model<Metric> model;
    model.train(base);
    auto enc = model.encode(base);
    std::cout << "  encode: " << t.stop() << "s" << std::endl;

    std::vector<turboquant_1bit_ref::asym::Quantized_Query<Metric>> qqs(n_q);
    parlay::parallel_for(0, n_q, [&](size_t i) { qqs[i] = model.quantize_query(queries[i]); });

    recall_curve(
        [&](size_t qi, size_t j) { return enc[j].distance(qqs[qi]); }, n_q, n_b, gt, k, kps,
        "Ref1BTQAsym");
  }

  // ==== Ref1BTQSym (scalar reference: 1-bit DB × 1-bit query, Hamming) ====
  // Mirrors what production `turboquant_1bit.h` actually computes, but
  // as a plain scalar loop. Side-by-side with Ref1BTQAsym this isolates
  // the cost of sign-quantizing the query.
  if (method == "Ref1BTQSym") {
    std::cout << "\n--- Ref1BTQSym ---" << std::endl;
    parlay::internal::timer t;
    t.start();
    turboquant_1bit_ref::sym::Model<Metric> model;
    model.train(base);
    auto enc = model.encode(base);
    std::cout << "  encode: " << t.stop() << "s" << std::endl;

    std::vector<turboquant_1bit_ref::sym::Quantized_Query<Metric>> qqs(n_q);
    parlay::parallel_for(0, n_q, [&](size_t i) { qqs[i] = model.quantize_query(queries[i]); });

    recall_curve(
        [&](size_t qi, size_t j) { return enc[j].distance(qqs[qi]); }, n_q, n_b, gt, k, kps,
        "Ref1BTQSym");
  }

  // ==== RaBitQ ====
  // `RabitQ` honors -rabitq_bits (any width). `RabitQ1`/`RabitQ4`/`RabitQ8` are
  // explicit shortcuts; `All` runs the three explicit widths (skips the
  // configurable one to avoid duplicating an 8-bit run by default).
  auto run_rabitq = [&](uint32_t bits, const std::string& label) {
    std::cout << "\n--- " << label << " ---" << std::endl;
    parlay::internal::timer t;
    t.start();
    rabitq::Model<Metric> model;
    model.train(base, bits);
    auto enc = model.encode(base);
    std::cout << "  encode: " << t.stop() << "s" << std::endl;

    // RaBitQ quantize_query is not thread-safe (copies into internal state),
    // so we pre-compute per-query.
    std::vector<rabitq::Quantized_Query<Metric>> qqs;
    qqs.reserve(n_q);
    for (size_t i = 0; i < n_q; ++i) {
      qqs.push_back(model.quantize_query(queries[i]));
    }

    recall_curve(
        [&](size_t qi, size_t j) {
          auto pt = enc[j];
          return qqs[qi].distance(pt);
        },
        n_q, n_b, gt, k, kps, label);
  };

  if (method == "RabitQ") run_rabitq(rbits, "RaBitQ-" + std::to_string(rbits) + "bit");
  if (method == "RabitQ1" || method == "All") run_rabitq(1, "RaBitQ-1bit");
  if (method == "RabitQ4" || method == "All") run_rabitq(4, "RaBitQ-4bit");
  if (method == "RabitQ8" || method == "All") run_rabitq(8, "RaBitQ-8bit");

  if (method != "TQ4" && method != "TurboQuant" && method != "TQ8" && method != "FastScan" &&
      method != "1BTQ" && method != "1BTQAsym" &&
      method != "Ref1BTQAsym" && method != "Ref1BTQSym" &&
      method != "RabitQ" && method != "RabitQ1" && method != "RabitQ4" && method != "RabitQ8" &&
      method != "All") {
    std::cerr << "Unknown method: " << method
              << " (TQ4|TurboQuant|TQ8|FastScan|1BTQ|1BTQAsym|Ref1BTQAsym|Ref1BTQSym|RabitQ|RabitQ1|RabitQ4|RabitQ8|All)"
              << std::endl;
  }
}
int main(int argc, char* argv[]) {
  commandLine P(argc, argv,
                "-i <base> [-q <queries> | -dataset_as_query] [-gt <gt>] [-k <k>] "
                "[-dist_func L2|IP] "
                "[-pq_method "
                "TQ4|TurboQuant|TQ8|FastScan|1BTQ|1BTQAsym|Ref1BTQAsym|Ref1BTQSym|RabitQ|RabitQ1|RabitQ4|RabitQ8|All] "
                "[-max_k_prime <N>] [-k_growth <r>] [-rabitq_bits <b>] [-fs_block <b>] "
                "[-pcs]");
  std::string df = P.getOptionValue("-dist_func", "IP");

  if (df == "L2")
    run_benchmark<L2_Point<float>, true>(P);
  else if (df == "IP")
    run_benchmark<IP_Point<float>, false>(P);
  else {
    std::cerr << "Unknown dist_func: " << df << std::endl;
    return 1;
  }

  return 0;
}
