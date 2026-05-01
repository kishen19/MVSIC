#pragma once

#include <chrono>
#include <cmath>
#include <fstream>
#include <limits>
#include <iomanip>
#include <iostream>
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
    hf << "tau,recall_1_" << k << ",recall_" << k << "_" << k << ",avg_compressed_vectors\n";
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
    double avg_q =
        static_cast<double>(parlay::reduce(counts)) / static_cast<double>(queries.size());
    auto [r1, rk] = compute_scores(pred, gt, k);

    const double step_sec =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t_step_start).count();

    std::cout << "done in " << std::setprecision(1) << step_sec << "s\n";
    std::cout << std::fixed << std::setprecision(6) << "      recall@1@" << k << "=" << r1
              << " recall@" << k << "@" << k << "=" << rk << " avg_q=" << std::setprecision(2)
              << avg_q << std::endl;

    if (!outCsv.empty()) {
      std::ofstream f(outCsv, std::ios::app);
      f << std::fixed << std::setprecision(10) << tau << "," << r1 << "," << rk << ","
        << std::setprecision(6) << avg_q << "\n";
      f.close();
    }
  }
}

}  // namespace query_compression_sweep
}  // namespace mvsic
