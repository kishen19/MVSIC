#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "parlay/primitives.h"

#include "mvsic/core/query_compression.h"
#include "mvsic/core/stats.h"
#include "mvsic/core/types/point_cloud_set.h"
#include "mvsic/core/utils/parse_command_line.h"
#include "mvsic/flat/flat.h"

namespace mvsic {
namespace query_compression_sweep {

inline std::vector<float> parse_float_list(const std::string& s) {
  std::vector<float> out;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    if (tok.empty()) continue;
    out.push_back(static_cast<float>(std::stod(tok)));
  }
  return out;
}

inline std::vector<float> linspace_float(double lo, double hi, size_t n) {
  std::vector<float> out;
  if (n == 0) return out;
  if (n == 1) {
    out.push_back(static_cast<float>(lo));
    return out;
  }
  out.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    double t = static_cast<double>(i) / static_cast<double>(n - 1);
    out.push_back(static_cast<float>(lo + t * (hi - lo)));
  }
  return out;
}

// Same endpoints as linspace, but map t ∈ [0,1] through t^gamma so more samples lie near the
// **lo** endpoint (i = 0): for ball-carving IP defaults (lo=1, hi=0) that densifies τ near 1;
// for L2 ball-carving (lo=0, hi=4) near 0; for Ward (lo=0, hi=T) near 0 (full query, no merges).
inline std::vector<float> geom_toward_lo_float(double lo, double hi, size_t n, double gamma) {
  std::vector<float> out;
  if (n == 0) return out;
  if (gamma <= 0.0) gamma = 1.0;
  if (n == 1) {
    out.push_back(static_cast<float>(lo));
    return out;
  }
  out.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    double t = static_cast<double>(i) / static_cast<double>(n - 1);
    double tg = std::pow(t, gamma);
    out.push_back(static_cast<float>(lo + tg * (hi - lo)));
  }
  return out;
}

// Summary statistics over a per-query series. We keep both the absolute
// compressed-vector count and (when original sizes are provided) the ratio
// compressed/original — the ratio is invariant to query length, which is what
// you actually want when comparing across queries of very different sizes.
struct SizeStats {
  double min = 0.0;
  double mean = 0.0;
  double p50 = 0.0;
  double p90 = 0.0;
  double p99 = 0.0;
  double max = 0.0;
  double stddev = 0.0;
};

inline double percentile_sorted(const std::vector<double>& sorted, double q) {
  if (sorted.empty()) return 0.0;
  if (q <= 0.0) return sorted.front();
  if (q >= 1.0) return sorted.back();
  const double pos = q * static_cast<double>(sorted.size() - 1);
  const size_t lo = static_cast<size_t>(std::floor(pos));
  const size_t hi = static_cast<size_t>(std::ceil(pos));
  if (lo == hi) return sorted[lo];
  const double frac = pos - static_cast<double>(lo);
  return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

template <typename T>
inline SizeStats summarize(const std::vector<T>& values) {
  SizeStats s{};
  if (values.empty()) return s;
  std::vector<double> v;
  v.reserve(values.size());
  double sum = 0.0;
  for (const T& x : values) {
    const double xd = static_cast<double>(x);
    v.push_back(xd);
    sum += xd;
  }
  std::sort(v.begin(), v.end());
  s.min = v.front();
  s.max = v.back();
  s.mean = sum / static_cast<double>(v.size());
  s.p50 = percentile_sorted(v, 0.50);
  s.p90 = percentile_sorted(v, 0.90);
  s.p99 = percentile_sorted(v, 0.99);
  double sse = 0.0;
  for (double x : v) {
    const double d = x - s.mean;
    sse += d * d;
  }
  // Population stddev; with O(N) queries the unbiased correction is negligible.
  s.stddev = std::sqrt(sse / static_cast<double>(v.size()));
  return s;
}

// Deterministically select `n_max` distinct indices from [0, n_q) using
// std::mt19937_64(seed). Mirrors numpy.random.default_rng(seed).choice when
// the same seed is plumbed end-to-end (used by benchmarks/benchmark_search.py
// for the latency / multi_latency / batch sweeps; we replicate the *behavior*
// here, not the bit-exact indices, since this lives in C++ — the important
// invariant is "same seed in this binary => same sample across runs of this
// binary").
inline std::vector<size_t> sample_indices_sorted(size_t n_q, size_t n_max, uint64_t seed) {
  std::vector<size_t> idx(n_q);
  for (size_t i = 0; i < n_q; ++i) idx[i] = i;
  std::mt19937_64 rng(seed);
  // Partial Fisher–Yates: shuffle the first n_max positions.
  for (size_t i = 0; i < n_max; ++i) {
    std::uniform_int_distribution<size_t> dist(i, n_q - 1);
    const size_t j = dist(rng);
    std::swap(idx[i], idx[j]);
  }
  idx.resize(n_max);
  std::sort(idx.begin(), idx.end());
  return idx;
}

// Build a new PointCloudSet that contains only the queries at `indices`. The
// underlying float storage is *materialized* (not a view): we re-pack values
// + offsets into contiguous arrays, then hand them to the (n, dims, values,
// offsets, ids) constructor which deep-copies via parlay::p_malloc.
template<typename ChPoint>
inline PointCloudSet<ChPoint> subsample_pcs(const PointCloudSet<ChPoint>& src,
                                            const std::vector<size_t>& indices) {
  const uint32_t dims = src.get_dims();
  const uint32_t n_new = static_cast<uint32_t>(indices.size());

  std::vector<size_t> new_offsets(n_new + 1, 0);
  for (size_t i = 0; i < n_new; ++i) {
    const uint32_t cnt = src.get_size(indices[i]);
    new_offsets[i + 1] = new_offsets[i] + static_cast<size_t>(cnt) * dims;
  }
  std::vector<float> new_values(new_offsets[n_new]);
  std::vector<uint32_t> new_ids(n_new);
  for (size_t i = 0; i < n_new; ++i) {
    const size_t qi = indices[i];
    const float* src_row = src.data(qi);
    const size_t bytes = (new_offsets[i + 1] - new_offsets[i]) * sizeof(float);
    std::memcpy(new_values.data() + new_offsets[i], src_row, bytes);
    new_ids[i] = src.get_id(qi);
  }
  return PointCloudSet<ChPoint>(n_new, dims, new_values.data(),
                                new_offsets.data(), new_ids.data());
}

template<typename GTSeq>
inline GTSeq subsample_gt(const GTSeq& gt, const std::vector<size_t>& indices) {
  GTSeq out(indices.size());
  for (size_t i = 0; i < indices.size(); ++i) {
    out[i] = gt[indices[i]];
  }
  return out;
}

template<typename ChPoint, bool metric>
void run_sweep(commandLine& P, SearchParams::QueryCompression qc_mode, const char* method_label) {
  Eigen::setNbThreads(1);
  using PC = PointCloudSet<ChPoint>;

  std::string inFile = P.getOptionValue("-i", "");
  std::string qFile = P.getOptionValue("-q", "");
  std::string gtFile = P.getOptionValue("-gt", "");
  std::string outCsv = P.getOptionValue("-o", "");

  if (inFile.empty() || qFile.empty() || gtFile.empty()) {
    std::cerr << "Required: -i <points.pcs> -q <queries.pcs> -gt <gt>\n";
    std::exit(1);
  }

  bool is_mmap = P.getOption("-mm");
  size_t k = P.getOptionLongValue("-k", 10);

  std::string tau_grid = P.getOptionValue("-tau_grid", "");
  double tau_lo = P.getOptionDoubleValue("-tau_min", std::numeric_limits<double>::quiet_NaN());
  double tau_hi = P.getOptionDoubleValue("-tau_max", std::numeric_limits<double>::quiet_NaN());
  size_t tau_steps = static_cast<size_t>(P.getOptionLongValue("-tau_steps", 20));
  std::string tau_spacing = P.getOptionValue("-tau_spacing", "geom");
  double tau_geom_gamma = P.getOptionDoubleValue("-tau_geom_gamma", 3.0);

  // Match the latency / multi_latency / batch defaults plumbed through
  // benchmarks/benchmark_search.py: cap the queries at 1000 (when present),
  // seeded with 42 so the same subset is used across runs and modes.
  // Set -query_subsample 0 (or any non-positive value) to disable.
  const long subsample_n_raw = P.getOptionLongValue("-query_subsample", 1000);
  const uint64_t subsample_seed =
      static_cast<uint64_t>(P.getOptionLongValue("-query_subsample_seed", 42));
  const size_t subsample_n =
      (subsample_n_raw <= 0) ? 0 : static_cast<size_t>(subsample_n_raw);

  std::vector<float> taus;
  if (!tau_grid.empty()) {
    taus = parse_float_list(tau_grid);
  } else {
    if (std::isnan(tau_lo) || std::isnan(tau_hi)) {
      // Ward: merge while min Ward linkage distance <= τ — NOT cosine. Small τ => few merges
      // (full query); large τ => aggressive merges. Default sweep 0 -> τ_max (not ball-carving IP).
      if (qc_mode == SearchParams::QueryCompression::Wards) {
        if (std::isnan(tau_lo)) tau_lo = 0.0001;
        if (std::isnan(tau_hi)) tau_hi = 2.0;
      } else {
        if constexpr (metric) {
          if (std::isnan(tau_lo)) tau_lo = 0.0;
          if (std::isnan(tau_hi)) tau_hi = 4.0;
        } else {
          if (std::isnan(tau_lo)) tau_lo = 1.0;
          if (std::isnan(tau_hi)) tau_hi = 0.0001;
        }
      }
    }
    if (tau_spacing == "linear" || tau_spacing == "lin") {
      taus = linspace_float(tau_lo, tau_hi, tau_steps);
    } else if (tau_spacing == "geom" || tau_spacing == "geometric") {
      taus = geom_toward_lo_float(tau_lo, tau_hi, tau_steps, tau_geom_gamma);
    } else {
      std::cerr << "Unknown -tau_spacing: " << tau_spacing << " (use linear or geom)\n";
      std::exit(1);
    }
  }
  if (taus.empty()) {
    std::cerr << "No τ values (use -tau_grid or -tau_min/-tau_max/-tau_steps)\n";
    std::exit(1);
  }

  PC points(inFile.c_str(), is_mmap);
  PC queries(qFile.c_str());
  std::string gt_path = gtFile;
  auto gt = ReadGT(gt_path, queries.size());

  // Optionally subsample queries (and the matching GT rows) to a fixed cap.
  // Mirrors benchmarks/benchmark_search.py's --query_subsample flag so the
  // C++ sweep runs on the same volume of work as the Python harness.
  if (subsample_n > 0 && queries.size() > subsample_n) {
    const size_t n_q = queries.size();
    auto idx = sample_indices_sorted(n_q, subsample_n, subsample_seed);
    queries = subsample_pcs(queries, idx);
    gt = subsample_gt(gt, idx);
    std::cout << "  [subsample] queries " << n_q << " -> " << subsample_n
              << " (seed=" << subsample_seed << ")" << std::endl;
  }

  // Original per-query point counts, captured once. Used to compute per-query
  // ratios so we can compare across queries of very different sizes.
  std::vector<uint32_t> orig_counts(queries.size(), 0u);
  for (size_t qi = 0; qi < queries.size(); ++qi) {
    orig_counts[qi] = queries.get_size(qi);
  }
  const SizeStats orig_stats = summarize(orig_counts);
  std::cout << "  query sizes: min=" << orig_stats.min << " mean=" << orig_stats.mean
            << " p50=" << orig_stats.p50 << " p90=" << orig_stats.p90
            << " p99=" << orig_stats.p99 << " max=" << orig_stats.max
            << " std=" << std::fixed << std::setprecision(2) << orig_stats.stddev << "\n";

  IndexParams ip = IndexParams::flat();
  IndexFlat<metric, NoQuantizer<metric>> index(points.get_dims(), ip);
  index.build(points);

  SearchParams sp = SearchParams::flat(k, 0);
  sp.num_rerank = 0;
  sp.query_compression = SearchParams::QueryCompression::None;

  std::cout << method_label << " sweep: " << taus.size() << " τ values";
  if (tau_grid.empty()) {
    std::cout << " (" << tau_spacing;
    if (tau_spacing == "geom" || tau_spacing == "geometric") {
      std::cout << ", gamma=" << tau_geom_gamma;
    }
    std::cout << "), τ range [" << tau_lo << "," << tau_hi << "]";
  }
  std::cout << ", n=" << points.size() << " queries=" << queries.size() << " k=" << k << std::endl;
  const uint64_t dist_ops_per_tau =
      static_cast<uint64_t>(points.size()) * static_cast<uint64_t>(queries.size());
  if (dist_ops_per_tau >= 5'000'000ull) {
    std::cout
        << "Note: flat brute-force does ~" << points.size() << "×" << queries.size() << " ≈ "
        << dist_ops_per_tau
        << " Chamfer cloud comparisons per τ step; large corpora can take many minutes per τ.\n";
  }
  std::cout << std::flush;

  if (!outCsv.empty()) {
    std::ofstream hf(outCsv, std::ios::trunc);
    // Legacy `avg_compressed_vectors` stays first so older plot.py keeps
    // working. The remaining columns are appended at the end and consist of:
    //   q_count_{stat} : per-query absolute compressed-vector count stats.
    //   q_ratio_{stat} : per-query (compressed / original) ratio stats.
    hf << "tau,recall_1_" << k << ",recall_" << k << "_" << k
       << ",avg_compressed_vectors"
       << ",q_count_min,q_count_mean,q_count_p50,q_count_p90,q_count_p99,q_count_max,q_count_std"
       << ",q_ratio_min,q_ratio_mean,q_ratio_p50,q_ratio_p90,q_ratio_p99,q_ratio_max,q_ratio_std"
       << "\n";
    hf.close();
    std::cout << "Writing CSV (overwrite): " << outCsv << std::endl;
  }

  const uint32_t batch_align = 1u;

  for (size_t ti = 0; ti < taus.size(); ++ti) {
    const float tau = taus[ti];
    std::cout << "  [" << (ti + 1) << "/" << taus.size() << "] tau=" << std::fixed
              << std::setprecision(6) << tau << "  … " << std::flush;

    const auto t_step_start = std::chrono::steady_clock::now();

    // Inner seqs must be default-constructed: assign into uninitialized slots frees garbage.
    auto pred = parlay::tabulate(queries.size(), [](size_t) {
      return parlay::sequence<std::pair<uint32_t, float>>();
    });
    auto counts = parlay::sequence<uint32_t>::uninitialized(queries.size());
    parlay::parallel_for(0, queries.size(), [&](size_t qi) {
      auto compressed = compress_query<ChPoint>(queries[qi], qc_mode, tau, batch_align);
      counts[qi] = compressed.n;
      auto [results, bytes, stats] =
          index.search_with_stats_compressed(compressed.view(), points, sp);
      (void)bytes;
      (void)stats;
      pred[qi] = std::move(results);
    });
    // Materialize compressed counts + per-query ratios for the summary.
    std::vector<uint32_t> count_vec(queries.size(), 0u);
    std::vector<double> ratio_vec(queries.size(), 0.0);
    for (size_t qi = 0; qi < queries.size(); ++qi) {
      const uint32_t c = counts[qi];
      count_vec[qi] = c;
      const uint32_t orig = orig_counts[qi];
      // Guard against degenerate empty queries (orig==0). Treat as no
      // compression possible: ratio = 1 keeps the bucket from skewing low.
      ratio_vec[qi] =
          (orig == 0u) ? 1.0 : (static_cast<double>(c) / static_cast<double>(orig));
    }
    const SizeStats cstats = summarize(count_vec);
    const SizeStats rstats = summarize(ratio_vec);
    const double avg_q = cstats.mean;
    auto [r1, rk] = compute_scores(pred, gt, k);

    const double step_sec =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t_step_start).count();

    std::cout << "done in " << std::setprecision(1) << step_sec << "s\n";
    std::cout << std::fixed << std::setprecision(6)
              << "      recall@1@" << k << "=" << r1
              << " recall@" << k << "@" << k << "=" << rk
              << "\n      counts: min=" << std::setprecision(0) << cstats.min
              << " mean=" << std::setprecision(2) << cstats.mean
              << " p50=" << cstats.p50
              << " p90=" << cstats.p90
              << " p99=" << cstats.p99
              << " max=" << std::setprecision(0) << cstats.max
              << " std=" << std::setprecision(2) << cstats.stddev
              << "\n      ratio: min=" << std::setprecision(4) << rstats.min
              << " mean=" << rstats.mean
              << " p50=" << rstats.p50
              << " p90=" << rstats.p90
              << " p99=" << rstats.p99
              << " max=" << rstats.max
              << " std=" << rstats.stddev
              << std::endl;

    if (!outCsv.empty()) {
      std::ofstream f(outCsv, std::ios::app);
      f << std::fixed << std::setprecision(10) << tau << "," << r1 << "," << rk << ","
        << std::setprecision(6) << avg_q
        << "," << cstats.min
        << "," << cstats.mean
        << "," << cstats.p50
        << "," << cstats.p90
        << "," << cstats.p99
        << "," << cstats.max
        << "," << cstats.stddev
        << "," << std::setprecision(8) << rstats.min
        << "," << rstats.mean
        << "," << rstats.p50
        << "," << rstats.p90
        << "," << rstats.p99
        << "," << rstats.max
        << "," << rstats.stddev
        << "\n";
      f.close();
    }
  }
}

}  // namespace query_compression_sweep
}  // namespace mvsic
