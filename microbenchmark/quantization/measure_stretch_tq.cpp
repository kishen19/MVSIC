// measure_stretch_tq.cpp
// Adapted from measure_stretch.cpp for TurboQuant quality evaluation.
// Measures recall@k vs k' for TurboQuant, TQ-PQ-4bit, and RaBitQ on real datasets.
//
// Usage:
//   ./measure_stretch_tq -i <base_file> -q <query_file> [-gt <gt_file>]
//     [-k <k>] [-dist_func L2|IP] [-pq_method TQ4|TurboQuant|RabitQ|TQPQ|All]
//     [-dataset_as_query] [-max_k_prime <N>] [-k_growth <rate>]
//     [-rabitq_bits <bits>] [-output_gt_path <path>]

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
#include "mvsic/core/utils/parse_command_line.h"
#include "mvsic/core/quantization/rabitq.h"
#include "mvsic/core/quantization/turboquant.h"
#include "mvsic/core/quantization/other_methods/turboquant_pq_4bit.h"
#include "mvsic/core/stats.h"

using namespace mvsic;

// ---- Ground truth: blocked Eigen GEMM (from measure_stretch.cpp) ----
template<typename Point, bool Metric>
parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>> compute_ground_truth(
    const PointRange<float, Point>& queries, const PointRange<float, Point>& base_points,
    size_t k) {
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
template<typename DistFn>
void recall_curve(DistFn&& dist_fn, size_t n_q, size_t n_b,
                  const parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>>& gt,
                  size_t k, const std::vector<size_t>& k_primes, const std::string& label) {

  std::vector<std::atomic<size_t>> tc(k_primes.size());
  for (auto& x : tc)
    x = 0;
  size_t lim = k_primes.back();

  parlay::parallel_for(0, n_q, [&](size_t qi) {
    std::vector<std::pair<float, uint32_t>> dists(n_b);
    for (size_t j = 0; j < n_b; ++j)
      dists[j] = {dist_fn(qi, j), (uint32_t)j};

    if (lim < dists.size()) {
      std::nth_element(dists.begin(), dists.begin() + lim, dists.end());
      std::sort(dists.begin(), dists.begin() + lim);
    } else {
      std::sort(dists.begin(), dists.end());
    }

    size_t ak = std::min(k, gt[qi].size());
    if (ak == 0) return;

    size_t cc = 0, ki = 0;
    for (size_t r = 0; r < lim && ki < k_primes.size(); ++r) {
      uint32_t rid = dists[r].second;
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
  });

  std::cout << "\n=== " << label << " ===" << std::endl;
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
  size_t num_query = P.getOptionLongValue("-num_query", 0);

  if (!inFile || (!qFile && !dataset_as_query)) {
    std::cerr << "Usage: measure_stretch_tq -i <base> [-q <queries> | -dataset_as_query]\n"
              << "  [-gt <gt>] [-k <k>] [-dist_func L2|IP] "
              << "[-pq_method TQ4|TurboQuant|RabitQ|TQPQ|All]\n"
              << "  [-max_k_prime <N>] [-k_growth <r>] [-rabitq_bits <b>] [-num_query <N>]\n";
    return;
  }

  using PR = PointRange<float, Point>;

  std::cout << "Loading base from " << inFile << "..." << std::endl;
  PR base(inFile);

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
    std::cout << "Loading queries from " << qFile << "..." << std::endl;
    queries_obj = PR(qFile);
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

  // ==== TQ-PQ-4bit (B = 1,2,4,8) ====
  if (method == "TQPQ" || method == "All") {
    auto run_tqpq = [&](auto block_tag, const std::string& label) {
      constexpr size_t B = decltype(block_tag)::value;
      std::cout << "\n--- TQ-PQ-4bit (B=" << B << ") ---" << std::endl;
      parlay::internal::timer t;
      t.start();
      turboquant_pq_4bit::Model<Metric, B> model;
      model.train(base);
      auto enc = model.encode(base);
      std::cout << "  encode: " << t.stop() << "s" << std::endl;

      std::vector<turboquant_pq_4bit::Quantized_Query<Metric, B>> qqs(n_q);
      parlay::parallel_for(0, n_q, [&](size_t i) {
        qqs[i] = model.quantize_query(reinterpret_cast<const float*>(queries.location(i)));
      });

      recall_curve(
          [&](size_t qi, size_t j) {
            auto pt = enc[j];
            return qqs[qi].distance(pt);
          },
          n_q, n_b, gt, k, kps, label);
    };

    run_tqpq(std::integral_constant<size_t, 1>{}, "TQ-PQ-4bit-B1");
    run_tqpq(std::integral_constant<size_t, 2>{}, "TQ-PQ-4bit-B2");
    run_tqpq(std::integral_constant<size_t, 4>{}, "TQ-PQ-4bit-B4");
    run_tqpq(std::integral_constant<size_t, 8>{}, "TQ-PQ-4bit-B8");
  }

  // ==== RaBitQ ====
  if (method == "RabitQ" || method == "All") {
    std::cout << "\n--- RaBitQ-" << rbits << "bit ---" << std::endl;
    parlay::internal::timer t;
    t.start();
    rabitq::Model<Metric> model;
    model.train(base, rbits);
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
        n_q, n_b, gt, k, kps, "RaBitQ-" + std::to_string(rbits) + "bit");
  }

  if (method != "TQ4" && method != "TurboQuant" && method != "RabitQ" && method != "TQPQ" &&
      method != "All") {
    std::cerr << "Unknown method: " << method << " (TQ4|TurboQuant|RabitQ|TQPQ|All)" << std::endl;
  }
}
int main(int argc, char* argv[]) {
  commandLine P(argc, argv,
                "-i <base> [-q <queries> | -dataset_as_query] [-gt <gt>] [-k <k>] "
                "[-dist_func L2|IP] [-pq_method TQ4|TurboQuant|RabitQ|TQPQ|All] "
                "[-max_k_prime <N>] [-k_growth <r>] [-rabitq_bits <b>]");
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
