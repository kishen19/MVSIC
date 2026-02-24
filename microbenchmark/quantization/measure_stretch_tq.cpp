// measure_stretch_tq.cpp
// Adapted from measure_stretch.cpp for TurboQuant quality evaluation.
// Measures recall@k vs k' for TQ-4bit, ByteTQ-8bit, and RaBitQ on real datasets.
//
// Usage:
//   ./measure_stretch_tq -i <base_file> -q <query_file> [-gt <gt_file>]
//     [-k <k>] [-dist_func L2|IP] [-pq_method TQ4|ByteTQ|RabitQ|All]
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

#include <Eigen/Dense>

#include "parlay/parallel.h"
#include "parlay/primitives.h"

#include "mvsic/core/types/point_range.h"
#include "mvsic/core/types/l2_point.h"
#include "mvsic/core/types/ip_point.h"
#include "mvsic/core/utils/parse_command_line.h"
#include "mvsic/core/quantization/rabitq.h"
#include "mvsic/core/quantization/one_to_many_turboquant.h"
#include "mvsic/core/quantization/byte_turboquant.h"
#include "mvsic/core/quantization/low_bit_turboquant.h"
#include "mvsic/core/quantization/centered_turboquant.h"
#include "mvsic/core/stats.h"

using namespace mvsic;

// ---- Ground truth: blocked Eigen GEMM (from measure_stretch.cpp) ----
template<typename Point, bool Metric>
parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>> compute_ground_truth(
    const PointRange<float, Point>& queries,
    const PointRange<float, Point>& base_points,
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
    Eigen::Map<const RowMatrix, 0, S> Q(
        reinterpret_cast<const float*>(queries.location(sq)), nq_block, D, S(aligned_q, 1));

    Eigen::VectorXf Qn;
    if constexpr (Metric) Qn = Q.rowwise().squaredNorm();

    std::vector<std::priority_queue<std::pair<float, uint32_t>>> heaps(nq_block);

    for (size_t bb = 0; bb < (n_b + BB - 1) / BB; ++bb) {
      size_t sb = bb * BB, eb = std::min(sb + BB, n_b);
      Eigen::Index nb_block = eb - sb;

      Eigen::Map<const RowMatrix, 0, S> B(
          reinterpret_cast<const float*>(base_points.location(sb)), nb_block, D, S(aligned_b, 1));

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
          if (h.size() < k) h.push({d, id});
          else if (d < h.top().first) { h.pop(); h.push({d, id}); }
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
void recall_curve(
    DistFn&& dist_fn,
    size_t n_q, size_t n_b,
    const parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>>& gt,
    size_t k,
    const std::vector<size_t>& k_primes,
    const std::string& label) {

  std::vector<std::atomic<size_t>> tc(k_primes.size());
  for (auto& x : tc) x = 0;
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
        if (gt[qi][g].first == rid) { ++cc; break; }
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
  for (size_t i = 0; i < k_primes.size(); ++i) {
    double rec = (double)tc[i] / (double)(n_q * k);
    std::cout << std::setw(10) << k_primes[i]
              << std::setw(15) << std::fixed << std::setprecision(4) << rec << std::endl;
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
              << "  [-gt <gt>] [-k <k>] [-dist_func L2|IP] [-pq_method TQ4|ByteTQ|RabitQ|All]\n"
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
    if (gtFile != "") { std::cout << "WARNING: ignoring -gt with -dataset_as_query\n"; gtFile = ""; }
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

  // Ground truth.
  parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>> gt;
  if (gtFile != "") {
    std::cout << "Loading GT from " << gtFile << "..." << std::endl;
    gt = ReadGT(gtFile, queries.size());
  } else {
    std::cout << "Computing exact k-NN..." << std::endl;
    parlay::internal::timer tt; tt.start();
    gt = compute_ground_truth<Point, Metric>(queries, base, std::max(k, (size_t)100));
    std::cout << "GT: " << tt.stop() << "s" << std::endl;
  }

  const size_t n_q = queries.size(), n_b = base.size(), D = queries.get_dims();
  std::cout << "\nN=" << n_b << " Q=" << n_q << " D=" << D
            << " dist=" << (Metric ? "L2" : "IP") << " k=" << k << std::endl;

  auto kps = make_k_primes(k, n_b, max_kp, growth);

  // ==== TurboQuant-4bit ====
  if (method == "TQ4" || method == "All") {
    std::cout << "\n--- TurboQuant-4bit ---" << std::endl;
    parlay::internal::timer t; t.start();
    one_to_many_turboquant::Model<Metric> model;
    model.train(base);
    auto enc = model.encode(base);
    std::cout << "  encode: " << t.stop() << "s" << std::endl;

    std::vector<one_to_many_turboquant::Quantized_Query<Metric>> qqs(n_q);
    parlay::parallel_for(0, n_q, [&](size_t i) {
      qqs[i] = model.quantize_query(queries[i]);
    });

    recall_curve(
        [&](size_t qi, size_t j) {
          auto pt = enc[j];
          return qqs[qi].distance(pt);
        },
        n_q, n_b, gt, k, kps, "TurboQuant-4bit");
  }



  // ==== RaBitQ ====
  if (method == "RabitQ" || method == "All") {
    std::cout << "\n--- RaBitQ-" << rbits << "bit ---" << std::endl;
    parlay::internal::timer t; t.start();
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

  // ==== TQ-1bit ====
  if (method == "TQ1" || method == "All") {
    std::cout << "\n--- TQ-1bit ---" << std::endl;
    parlay::internal::timer t; t.start();
    one_to_many_turboquant::Model<Metric> model;
    model.train(base);
    const size_t pdim = model.padded_dim;

    // Encode base points.
    std::vector<mvsic::low_bit_turboquant::EncodedVec> enc_1bit(n_b);
    parlay::parallel_for(0, n_b, [&](size_t i) {
      static thread_local std::vector<float> ws;
      enc_1bit[i] = mvsic::low_bit_turboquant::encode_1bit(model, reinterpret_cast<const float*>(base.location(i)), ws);
    });

    // Prepare queries.
    std::vector<mvsic::low_bit_turboquant::PreparedQuery> pqs_1bit(n_q);
    parlay::parallel_for(0, n_q, [&](size_t i) {
      pqs_1bit[i] = mvsic::low_bit_turboquant::prepare_query(model, reinterpret_cast<const float*>(queries.location(i)));
    });
    std::cout << "  encode: " << t.stop() << "s" << std::endl;

    recall_curve(
        [&](size_t qi, size_t j) {
          return mvsic::low_bit_turboquant::distance_1bit(enc_1bit[j], pqs_1bit[qi], pdim, Metric);
        },
        n_q, n_b, gt, k, kps, "TQ-1bit");
  }

  // ==== TQ-2bit ====
  if (method == "TQ2" || method == "All") {
    std::cout << "\n--- TQ-2bit ---" << std::endl;
    parlay::internal::timer t; t.start();
    one_to_many_turboquant::Model<Metric> model;
    model.train(base);
    const size_t pdim = model.padded_dim;

    // Encode base points.
    std::vector<mvsic::low_bit_turboquant::EncodedVec> enc_2bit(n_b);
    parlay::parallel_for(0, n_b, [&](size_t i) {
      static thread_local std::vector<float> ws;
      enc_2bit[i] = mvsic::low_bit_turboquant::encode_2bit(model, reinterpret_cast<const float*>(base.location(i)), ws);
    });

    // Prepare queries.
    std::vector<mvsic::low_bit_turboquant::PreparedQuery> pqs_2bit(n_q);
    parlay::parallel_for(0, n_q, [&](size_t i) {
      pqs_2bit[i] = mvsic::low_bit_turboquant::prepare_query(model, reinterpret_cast<const float*>(queries.location(i)));
    });
    std::cout << "  encode: " << t.stop() << "s" << std::endl;

    recall_curve(
        [&](size_t qi, size_t j) {
          return mvsic::low_bit_turboquant::distance_2bit(enc_2bit[j], pqs_2bit[qi], pdim, Metric);
        },
        n_q, n_b, gt, k, kps, "TQ-2bit");
  }

  // ==== Centered TQ (1-bit, 2-bit, 4-bit) ====
  // Train the centered model once (shared across bit depths).
  bool need_ctq = (method == "CTQ1" || method == "CTQ2" || method == "CTQ4" || method == "All");
  one_to_many_turboquant::Model<Metric> ctq_tq_model;
  mvsic::centered_turboquant::CenteredModel ctq_cm;
  size_t ctq_pdim = 0;
  if (need_ctq) {
    std::cout << "\n--- Training centered TQ model ---" << std::endl;
    parlay::internal::timer t; t.start();
    ctq_tq_model.train(base);
    ctq_cm = mvsic::centered_turboquant::train_centered(ctq_tq_model, base);
    ctq_pdim = ctq_cm.padded_dim;
    std::cout << "  train (centered): " << t.stop() << "s, mean_sq_norm=" << ctq_cm.mean_sq_norm << std::endl;
  }

  // ==== CTQ-1bit ====
  if (method == "CTQ1" || method == "All") {
    std::cout << "\n--- CTQ-1bit (centered) ---" << std::endl;
    parlay::internal::timer t; t.start();

    std::vector<mvsic::centered_turboquant::EncodedVec> enc(n_b);
    parlay::parallel_for(0, n_b, [&](size_t i) {
      static thread_local std::vector<float> ws;
      enc[i] = mvsic::centered_turboquant::encode_1bit_centered(
          ctq_tq_model, ctq_cm,
          reinterpret_cast<const float*>(base.location(i)), ws);
    });

    std::vector<mvsic::centered_turboquant::PreparedQuery> pqs(n_q);
    parlay::parallel_for(0, n_q, [&](size_t i) {
      pqs[i] = mvsic::centered_turboquant::prepare_query_centered(
          ctq_tq_model, ctq_cm,
          reinterpret_cast<const float*>(queries.location(i)));
    });
    std::cout << "  encode: " << t.stop() << "s" << std::endl;

    recall_curve(
        [&](size_t qi, size_t j) {
          return mvsic::centered_turboquant::distance_1bit_centered(
              enc[j], pqs[qi], ctq_pdim, Metric);
        },
        n_q, n_b, gt, k, kps, "CTQ-1bit");
  }

  // ==== CTQ-2bit ====
  if (method == "CTQ2" || method == "All") {
    std::cout << "\n--- CTQ-2bit (centered) ---" << std::endl;
    parlay::internal::timer t; t.start();

    std::vector<mvsic::centered_turboquant::EncodedVec> enc(n_b);
    parlay::parallel_for(0, n_b, [&](size_t i) {
      static thread_local std::vector<float> ws;
      enc[i] = mvsic::centered_turboquant::encode_2bit_centered(
          ctq_tq_model, ctq_cm,
          reinterpret_cast<const float*>(base.location(i)), ws);
    });

    std::vector<mvsic::centered_turboquant::PreparedQuery> pqs(n_q);
    parlay::parallel_for(0, n_q, [&](size_t i) {
      pqs[i] = mvsic::centered_turboquant::prepare_query_centered(
          ctq_tq_model, ctq_cm,
          reinterpret_cast<const float*>(queries.location(i)));
    });
    std::cout << "  encode: " << t.stop() << "s" << std::endl;

    recall_curve(
        [&](size_t qi, size_t j) {
          return mvsic::centered_turboquant::distance_2bit_centered(
              enc[j], pqs[qi], ctq_pdim, Metric);
        },
        n_q, n_b, gt, k, kps, "CTQ-2bit");
  }

  // ==== CTQ-4bit ====
  if (method == "CTQ4" || method == "All") {
    std::cout << "\n--- CTQ-4bit (centered) ---" << std::endl;
    parlay::internal::timer t; t.start();

    std::vector<mvsic::centered_turboquant::EncodedVec> enc(n_b);
    parlay::parallel_for(0, n_b, [&](size_t i) {
      static thread_local std::vector<float> ws;
      enc[i] = mvsic::centered_turboquant::encode_4bit_centered(
          ctq_tq_model, ctq_cm,
          reinterpret_cast<const float*>(base.location(i)), ws);
    });

    std::vector<mvsic::centered_turboquant::PreparedQuery> pqs(n_q);
    parlay::parallel_for(0, n_q, [&](size_t i) {
      pqs[i] = mvsic::centered_turboquant::prepare_query_centered(
          ctq_tq_model, ctq_cm,
          reinterpret_cast<const float*>(queries.location(i)));
    });
    std::cout << "  encode: " << t.stop() << "s" << std::endl;

    recall_curve(
        [&](size_t qi, size_t j) {
          return mvsic::centered_turboquant::distance_4bit_centered(
              enc[j], pqs[qi], ctq_pdim, Metric);
        },
        n_q, n_b, gt, k, kps, "CTQ-4bit");
  }

  // ==== ScalarRef: faithful copy of tq_reference/turboquant.h ====
  // Replicates the EXACT algorithm from the user's working implementation.
  // Order: rotate → normalize → scale(√padded_dim) → quantize.
  // Key: the √padded_dim scale factor converts from norm-preserving rotation
  // coordinates (~N(0, 1/√d)) to reference-scale coordinates (~N(0,1)).
  if (method == "ScalarRef" || method == "All") {
    std::cout << "\n--- ScalarRef-TQ4 (tq_reference copy) ---" << std::endl;

    // ---- Constants (from tq_reference/turboquant.h) ----
    constexpr std::array<int8_t, 16> kCentroidsInt8 = {
        6, 18, 31, 44, 58, 75, 96, 127,
        -6, -18, -31, -44, -58, -75, -96, -127};
    constexpr std::array<float, 8> kSqCentroids = {
        35.66f, 320.92f, 951.87f, 1917.61f,
        3332.04f, 5571.56f, 9128.44f, 15975.76f};
    constexpr std::array<float, 7> kBounds = {
        0.2581972f, 0.5271527f, 0.806866f, 1.097338f,
        1.430843f, 1.839655f, 2.399083f};
    constexpr float kValCap = 3.91724f;

    auto find_bucket = [&](float abs_x) -> uint8_t {
      uint8_t idx = kBounds.size();
      while (idx > 0 && abs_x < kBounds[idx - 1]) --idx;
      return idx;
    };
    auto four_bit_encode = [&](float x) -> uint8_t {
      return find_bucket(std::abs(x)) | (x > 0 ? 0 : 8);
    };

    // ---- Train rotator (using our FhtKacRotator, same as TQ4) ----
    parlay::internal::timer t; t.start();
    one_to_many_turboquant::Model<Metric> model;
    model.train(base);
    const size_t pdim = model.padded_dim;
    const size_t mdim = model.dim;
    const size_t stride = pdim / 2;  // bytes per encoded point
    const float hadamard_scale = std::sqrt(static_cast<float>(pdim));

    // ---- Encode base points: rotate → normalize → scale(√d) → quantize ----
    struct RefPt { std::vector<uint8_t> codes; float nsf; float sqn; };
    std::vector<RefPt> ref_pts(n_b);

    parlay::parallel_for(0, n_b, [&](size_t i) {
      auto& rp = ref_pts[i];
      rp.codes.resize(stride, 0);

      // Copy point data (dim elements only).
      std::vector<float> fpoint(mdim);
      auto pt = base[i];
      for (size_t d = 0; d < mdim; ++d) fpoint[d] = static_cast<float>(pt[d]);

      // Step 1: Rotate into rot_buf.
      std::vector<float> rot(pdim);
      model.rotator->rotate(fpoint.data(), rot.data());

      // Step 2: Compute norm of rotated vector.
      float norm_sq = 0.0f;
      for (size_t d = 0; d < pdim; ++d) norm_sq += rot[d] * rot[d];
      rp.sqn = norm_sq;
      float norm = std::sqrt(norm_sq);

      // Step 3: Normalize to unit norm.
      if (norm > 1e-9f) {
        float inv = 1.0f / norm;
        for (size_t d = 0; d < pdim; ++d) rot[d] *= inv;
      }

      // Step 4: Scale by sqrt(padded_dim), then quantize.
      float qsn = 0.0f;
      for (size_t b = 0; b < stride; ++b) {
        float val0 = rot[2*b] * hadamard_scale;
        float val1 = rot[2*b + 1] * hadamard_scale;
        uint8_t code0 = four_bit_encode(val0);
        uint8_t code1 = four_bit_encode(val1);
        rp.codes[b] = (code0 & 0x0F) | ((code1 << 4) & 0xF0);
        qsn += kSqCentroids[code0 & 7];
        qsn += kSqCentroids[code1 & 7];
      }

      float qnorm = std::sqrt(qsn);
      rp.nsf = (qnorm > 1e-9f) ? (norm / qnorm) : 0.0f;
    });

    // ---- Encode queries: rotate → normalize → scale(√d) → clamp → int8 ----
    struct RefQ { std::vector<int8_t> even, odd; float nsf; float sqn; };
    std::vector<RefQ> ref_qs(n_q);

    parlay::parallel_for(0, n_q, [&](size_t i) {
      auto& rq = ref_qs[i];
      rq.even.resize(stride, 0);
      rq.odd.resize(stride, 0);

      // Copy query data.
      std::vector<float> fq(mdim);
      auto pt = queries[i];
      for (size_t d = 0; d < mdim; ++d) fq[d] = static_cast<float>(pt[d]);

      // Step 1: Rotate.
      std::vector<float> rot(pdim);
      model.rotator->rotate(fq.data(), rot.data());

      // Step 2: Compute norm.
      float norm_sq = 0.0f;
      for (size_t d = 0; d < pdim; ++d) norm_sq += rot[d] * rot[d];
      rq.sqn = norm_sq;
      float norm = std::sqrt(norm_sq);

      // Step 3: Normalize to unit norm.
      if (norm > 1e-9f) {
        float inv = 1.0f / norm;
        for (size_t d = 0; d < pdim; ++d) rot[d] *= inv;
      }

      // Step 4: Scale by sqrt(padded_dim).
      for (size_t d = 0; d < pdim; ++d) rot[d] *= hadamard_scale;

      // Step 5: Clamp + find max abs.
      float max_abs = 0.0f;
      for (size_t d = 0; d < pdim; ++d) {
        rot[d] = std::max(-kValCap, std::min(kValCap, rot[d]));
        max_abs = std::max(max_abs, std::abs(rot[d]));
      }

      // Step 6: Scale to int8 + store deinterleaved.
      float sf = (max_abs > 1e-9f) ? (127.0f / max_abs) : 0.0f;
      float qn_sq = 0.0f;
      for (size_t b = 0; b < stride; ++b) {
        float v_even = rot[2*b];
        float v_odd  = rot[2*b + 1];
        int8_t ie = int8_t(std::max(-127.0f, std::min(127.0f, std::round(v_even * sf))));
        int8_t io = int8_t(std::max(-127.0f, std::min(127.0f, std::round(v_odd * sf))));
        rq.even[b] = ie;
        rq.odd[b] = io;
        qn_sq += float(ie) * ie + float(io) * io;
      }

      float qnorm = std::sqrt(qn_sq);
      rq.nsf = (qnorm > 1e-9f) ? (norm / qnorm) : 0.0f;
    });
    std::cout << "  encode: " << t.stop() << "s" << std::endl;

    // ---- Score: scalar reference (matches TurboQuantScoring4BitScalar) ----
    recall_curve(
        [&](size_t qi, size_t j) -> float {
          int32_t dot = 0;
          const auto& codes = ref_pts[j].codes;
          const auto& q_even = ref_qs[qi].even;
          const auto& q_odd  = ref_qs[qi].odd;
          for (size_t b = 0; b < stride; ++b) {
            uint8_t byte = codes[b];
            int8_t c0 = kCentroidsInt8[byte & 0x0F];
            int8_t c1 = kCentroidsInt8[(byte >> 4) & 0x0F];
            dot += int32_t(c0) * int32_t(q_even[b]);
            dot += int32_t(c1) * int32_t(q_odd[b]);
          }
          float neg_dp = -float(dot) * ref_pts[j].nsf * ref_qs[qi].nsf;
          if constexpr (Metric)
            return ref_pts[j].sqn + 2*neg_dp + ref_qs[qi].sqn;
          else
            return neg_dp;
        },
        n_q, n_b, gt, k, kps, "ScalarRef-TQ4 (tq_reference copy)");
  }


  if (method != "TQ4" && method != "RabitQ" &&
      method != "TQ1" && method != "TQ2" &&
      method != "CTQ1" && method != "CTQ2" && method != "CTQ4" &&
      method != "ScalarRef" && method != "All") {
    std::cerr << "Unknown method: " << method
              << " (TQ4|RabitQ|TQ1|TQ2|CTQ1|CTQ2|CTQ4|ScalarRef|All)" << std::endl;
  }
}
int main(int argc, char* argv[]) {
  commandLine P(argc, argv,
      "-i <base> [-q <queries> | -dataset_as_query] [-gt <gt>] [-k <k>] "
      "[-dist_func L2|IP] [-pq_method TQ4|ByteTQ|RabitQ|All] "
      "[-max_k_prime <N>] [-k_growth <r>] [-rabitq_bits <b>]");
  std::string df = P.getOptionValue("-dist_func", "IP");

  if (df == "L2")      run_benchmark<L2_Point<float>, true>(P);
  else if (df == "IP")  run_benchmark<IP_Point<float>, false>(P);
  else { std::cerr << "Unknown dist_func: " << df << std::endl; return 1; }

  return 0;
}
