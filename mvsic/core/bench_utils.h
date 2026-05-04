#pragma once
// ========================================================================
// bench_utils.h — Shared utilities for C++ benchmarking.
//
// Provides:
//   1. Dataset path lookup by short name (e.g. "arguana", "nq500k")
//   2. Quantizer string → enum parsing
//   3. Common CLI argument parsing (quant, search, I/O, compression)
//   4. Pretty result printing
//   5. CSV output (append-friendly)
//   6. Generic build/load + search sweep (parallel & sequential)
//   7. Dist-func dispatch macro
// ========================================================================
#include <Eigen/Dense>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "mvsic/core/query_compression.h"
#include "mvsic/core/stats.h"
#include "mvsic/core/types/chamfer_ip_point.h"
#include "mvsic/core/types/chamfer_l2_point.h"
#include "mvsic/core/types/point_cloud_set.h"
#include "mvsic/core/utils/parse_command_line.h"

namespace mvsic {
namespace bench {

// ========================================================================
// 1. Dataset Paths — just pass "-d arguana" instead of full paths
// ========================================================================

struct DatasetPaths {
  std::string points;   // {name}_points.pcs
  std::string queries;  // {name}_queries.pcs
  std::string gt;       // {name}_chamfer_neighbors.gt
  std::string name;
};

inline std::string dataset_base_dir(const std::string& name) {
  static const std::vector<std::string> beir = {
      "arguana", "fiqa", "hotpotqa", "nfcorpus", "nq", "nq500k", "quora", "scidocs", "scifact"};
  for (const auto& d : beir) {
    if (name == d) return "data/beir/" + d;
  }
  static const std::vector<std::string> vidore = {
      "docvqa", "infovqa", "tatdqa", "arxivqa", "tabfquad", "chartqa"};
  for (const auto& d : vidore) {
    if (name == d) return "data/vidore/" + d;
  }
  return name;
}

inline DatasetPaths resolve_dataset(const std::string& name) {
  std::string base = dataset_base_dir(name);
  std::string file_prefix = name;
  auto slash = name.find_last_of('/');
  if (slash != std::string::npos) file_prefix = name.substr(slash + 1);

  DatasetPaths dp;
  dp.name = file_prefix;
  dp.points = base + "/" + file_prefix + "_points.pcs";
  dp.queries = base + "/" + file_prefix + "_queries.pcs";
  dp.gt = base + "/" + file_prefix + "_chamfer_neighbors.gt";
  return dp;
}

inline DatasetPaths parse_dataset(commandLine& P) {
  std::string dname = P.getOptionValue("-d", "");
  if (!dname.empty()) {
    return resolve_dataset(dname);
  }
  DatasetPaths dp;
  char* inFile = P.getOptionValue("-i");
  char* qFile = P.getOptionValue("-q");
  dp.points = inFile ? std::string(inFile) : "";
  dp.queries = qFile ? std::string(qFile) : "";
  dp.gt = P.getOptionValue("-gt", "");
  dp.name = "custom";
  return dp;
}

// ========================================================================
// 2. Quantizer Parsing
// ========================================================================

inline uint32_t parse_quant_method(const std::string& s) {
  if (s == "None" || s == "none")  return 0;
  if (s == "PQ"   || s == "pq")    return 1;
  if (s == "RQ"   || s == "rq")    return 2;
  if (s == "FS"   || s == "fs")    return 3;
  if (s == "TQ"   || s == "tq")    return 4;
  if (s == "SPQTQ" || s == "spqtq") return 5;
  if (s == "1BTQ"  || s == "1btq")  return 6;
  std::cerr << "Unknown quantizer: " << s
            << " (use None, PQ, RQ, FS, TQ, SPQTQ, 1BTQ)" << std::endl;
  std::exit(1);
}

struct QuantArgs {
  uint32_t pq_method;
  uint32_t block_size;
  uint32_t num_clusters_per_block;
  uint32_t num_points_per_cluster;
  uint32_t rabitq_bits;
};

inline QuantArgs parse_quant_args(commandLine& P, const std::string& default_method = "None") {
  // Defaults here must match the factory defaults in
  // mvsic/core/index_params.h (muvera_custom / mpool / vamana / svh_*):
  //   block_size                 = 8
  //   num_clusters_per_block     = 16
  //   num_points_per_cluster     = 100
  //   rabitq_bits                = 4
  QuantArgs qa;
  qa.pq_method = parse_quant_method(P.getOptionValue("-quant_method", default_method));
  qa.block_size = P.getOptionIntValue("-m", 8);
  qa.num_clusters_per_block = P.getOptionIntValue("-num_clusters_per_block", 16);
  qa.num_points_per_cluster = P.getOptionIntValue("-num_points_per_cluster", 100);
  qa.rabitq_bits = P.getOptionIntValue("-rbits", 4);
  return qa;
}

// ========================================================================
// 3. Common I/O & Search Parsing
// ========================================================================

struct IOArgs {
  std::string index_path;
  std::string save_path;
  std::string csv_path;
  bool is_mmap;
  uint32_t verbose;
  bool compress_input;
};

inline IOArgs parse_io_args(commandLine& P) {
  IOArgs io;
  io.index_path = P.getOptionValue("-index", "");
  io.save_path = P.getOptionValue("-o", "");
  io.csv_path = P.getOptionValue("-csv", "");
  io.is_mmap = P.getOption("-mm");
  io.verbose = P.getOptionIntValue("-v", 0);
  io.compress_input = P.getOption("-compress_input");
  return io;
}

inline void parse_compression_opts(SearchParams& sp, commandLine& P) {
  std::string qc = P.getOptionValue("-compress", "none");
  if (qc == "none" || qc == "off" || qc == "0") {
    sp.query_compression = SearchParams::QueryCompression::None;
  } else if (qc == "carve" || qc == "ball" || qc == "muvera") {
    sp.query_compression = SearchParams::QueryCompression::Carve;
  } else if (qc == "wards" || qc == "ward") {
    sp.query_compression = SearchParams::QueryCompression::Wards;
  } else {
    std::cerr << "Unknown -compress value: " << qc
              << " (use none, carve, wards)" << std::endl;
    std::exit(1);
  }
  sp.query_compression_threshold =
      static_cast<float>(P.getOptionDoubleValue("-compress_threshold", 0.7));
  sp.compress_rerank = P.getOption("-compress_rerank");
  sp.query_alignment =
      static_cast<uint32_t>(P.getOptionIntValue("-query_alignment", 0));
  sp.query_alignment_strict = P.getOption("-query_alignment_strict");
  sp.tq8_rerank = P.getOption("-tq8_rerank");
  sp.root_m2m = P.getOption("-root_m2m");
}

inline void print_compression_info(const SearchParams& sp) {
  if (sp.tq8_rerank) std::cout << "tq8_rerank=1" << std::endl;
  if (sp.root_m2m) std::cout << "root_m2m=1" << std::endl;
  if (sp.query_compression == SearchParams::QueryCompression::None) return;
  const char* mname = (sp.query_compression == SearchParams::QueryCompression::Carve)
                          ? "carve" : "wards";
  std::cout << "Query compression: " << mname
            << " threshold=" << sp.query_compression_threshold
            << " compress_rerank=" << (sp.compress_rerank ? 1 : 0)
            << " query_alignment=" << sp.query_alignment
            << " query_alignment_strict=" << (sp.query_alignment_strict ? 1 : 0) << std::endl;
}

// Compute and print average query point-cloud size pre/post compression.
// No-op if compression is disabled. `quantization_mode` determines batch
// alignment (matches what `search_with_stats` applies internally). Works with
// any container of ChPoint exposing .size() and operator[] (std::vector,
// PointCloudSet, etc.).
template <typename ChPoint, typename Queries>
inline void print_compression_stats(const Queries& queries, const SearchParams& sp,
                                    IndexParams::QuantizerType quantization_mode) {
  if (sp.query_compression == SearchParams::QueryCompression::None) return;
  const size_t nq = queries.size();
  if (nq == 0) return;
  const uint32_t ba = qc_internal::batch_alignment(quantization_mode);
  double sum_orig = 0.0;
  double sum_comp = 0.0;
  for (size_t i = 0; i < nq; ++i) {
    const ChPoint q = queries[i];
    sum_orig += static_cast<double>(q.size());
    auto c = compress_query<ChPoint>(q, sp.query_compression,
                                     sp.query_compression_threshold, ba,
                                     sp.query_alignment, sp.query_alignment_strict);
    sum_comp += static_cast<double>(c.n);
  }
  std::cout << std::fixed << std::setprecision(2)
            << "Avg query points (raw):        " << (sum_orig / nq) << "\n"
            << "Avg query points (compressed): " << (sum_comp / nq) << "\n"
            << "Compression ratio (raw/compr): "
            << (sum_comp > 0 ? sum_orig / sum_comp : 0.0) << std::endl;
}

inline std::vector<size_t> parse_csv_ints(const std::string& s) {
  std::vector<size_t> vals;
  std::istringstream ss(s);
  std::string token;
  while (std::getline(ss, token, ',')) {
    vals.push_back(std::stoul(token));
  }
  return vals;
}

// ========================================================================
// 4. Pretty Printing
// ========================================================================

inline void print_header(const std::string& method, const std::string& dataset,
                         size_t n_points, size_t n_queries) {
  std::cout << "\n"
            << "============================================================\n"
            << "  Method:   " << method << "\n"
            << "  Dataset:  " << dataset << "\n"
            << "  Points:   " << n_points << "\n"
            << "  Queries:  " << n_queries << "\n"
            << "============================================================\n"
            << std::endl;
}

inline void print_result(const Stats& r, size_t k) {
  std::cout << std::fixed << std::setprecision(4)
            << "  Recall 1@" << k << ":   " << r.recall_1_k << "\n"
            << "  Recall " << k << "@" << k << ":   " << r.recall_k_k << "\n"
            << std::setprecision(1)
            << "  QPS_seq:      " << r.QPS_seq << "\n"
            << "  QPS_par:      " << r.QPS_par << "\n"
            << std::setprecision(1)
            << "  Avg cmps:     " << r.avg_cmps << "\n"
            << std::flush;
}

inline void print_result_extended(const StatsExtended& r, size_t k,
                                   const std::vector<std::string>& timing_labels = {}) {
  std::cout << std::fixed << std::setprecision(4)
            << "  Recall 1@" << k << ":   " << r.recall_1_k << "\n"
            << "  Recall " << k << "@" << k << ":   " << r.recall_k_k << "\n"
            << std::setprecision(1)
            << "  QPS_seq:      " << r.QPS_seq << "\n"
            << "  QPS_par:      " << r.QPS_par << "\n"
            << std::setprecision(1)
            << "  Avg cmps:     " << r.avg_cmps << "\n";
  for (size_t i = 0; i < timing_labels.size() && i < r.avg_timings.size(); ++i) {
    std::cout << "  " << std::left << std::setw(16) << (timing_labels[i] + ":")
              << std::right << std::setprecision(4) << r.avg_timings[i] << " s\n";
  }
  std::cout << std::flush;
}

// ========================================================================
// 5. CSV Output (append-friendly)
// ========================================================================

inline void write_csv_row(const std::string& csv_path, const Stats& r, size_t k,
                          const std::string& variable_param_name, size_t variable_param_value,
                          size_t num_rerank = 0) {
  bool file_exists = std::ifstream(csv_path).good();
  std::ofstream f(csv_path, std::ios::app);
  if (!file_exists) {
    f << "k,recall_1_k,recall_k_k,QPS_seq,QPS_par,avg_cmps,"
      << variable_param_name << ",num_rerank\n";
  }
  f << k << ","
    << std::fixed << std::setprecision(6) << r.recall_1_k << ","
    << r.recall_k_k << ","
    << std::setprecision(1) << r.QPS_seq << ","
    << r.QPS_par << ","
    << std::setprecision(1) << r.avg_cmps << ","
    << variable_param_value << ","
    << num_rerank << "\n";
  f.close();
}

// ========================================================================
// 6. Generic Build/Load + Search Sweep
// ========================================================================

template <typename Index, typename ChPoint>
void build_or_load(Index& index, PointCloudSet<ChPoint>& points,
                   const std::string& index_path, const std::string& save_path = "") {
  if (!index_path.empty()) {
    std::cout << "Loading index from " << index_path << " ..." << std::endl;
    index.load(index_path, points);
    std::cout << "Index loaded." << std::endl;
  } else {
    std::cout << "Building index..." << std::endl;
    parlay::internal::timer t;
    t.start();
    index.build(points);
    t.stop();
    std::cout << "Index built in " << t.total_time() << " seconds." << std::endl;
    if (!save_path.empty()) {
      std::cout << "Saving index to " << save_path << " ..." << std::endl;
      index.save(save_path);
      std::cout << "Index saved." << std::endl;
    }
  }
}

template <typename Index, typename ChPoint, typename MakeSearchParams>
void run_search_sweep(Index& index,
                      PointCloudSet<ChPoint>& points,
                      PointCloudSet<ChPoint>& queries,
                      const parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>>& gt,
                      const std::string& variable_param_name,
                      const std::vector<size_t>& variable_values,
                      MakeSearchParams make_sp,
                      const std::string& csv_path = "",
                      const std::vector<std::string>& timing_labels = {}) {
  size_t k = 0;
  for (size_t val : variable_values) {
    SearchParams sp = make_sp(val);
    k = sp.k;

    std::cout << "\n--- " << variable_param_name << " = " << val << " ---" << std::endl;

    StatsExtended result = compute_stats_extended(index, points, queries, gt, sp);
    print_result_extended(result, k, timing_labels);

    if (!csv_path.empty()) {
      write_csv_row(csv_path, Stats(result.QPS_seq, result.QPS_par, result.avg_cmps,
                                    result.recall_1_k, result.recall_k_k),
                    k, variable_param_name, val, sp.num_rerank);
    }

    if (result.recall_k_k >= 1.0) {
      std::cout << "  Recall@k reached 1.0. Stopping." << std::endl;
      break;
    }
  }
}

template <typename Index, typename ChPoint, typename MakeSearchParams>
void run_search_sweep_seq(Index& index,
                          PointCloudSet<ChPoint>& points,
                          PointCloudSet<ChPoint>& queries,
                          const parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>>& gt,
                          const std::string& variable_param_name,
                          const std::vector<size_t>& variable_values,
                          MakeSearchParams make_sp,
                          const std::string& csv_path = "",
                          const std::vector<std::string>& timing_labels = {},
                          bool skip_warmup = false) {
  // Sweep-level warmup: run every query once at the first sweep value to fault
  // in mmap pages (raw points touched by rerank) and prime caches before any
  // timed measurement. Replaces the 10-query warmup that
  // compute_stats_extended_p_threaded used to do on every sweep value, which
  // dominated wall time at large nprobes.
  //
  // Runs under the *ambient* parlay scheduler (all cores), not pinned to one
  // worker like the timed loop. The warmup's only job is to prime pages as
  // fast as possible; running it serially would force mmap page faults to
  // happen one-at-a-time and dominate sweep wall time. The printed number is
  // therefore not directly comparable to the timed-regime QPS.
  if (!skip_warmup && !variable_values.empty() && queries.size() > 0) {
    SearchParams probe_sp = make_sp(variable_values.front());
    parlay::internal::timer t_warm;
    t_warm.start();
    for (size_t j = 0; j < queries.size(); ++j) {
      auto pair = index.search(queries[j], points, probe_sp);
      (void)pair;
    }
    t_warm.stop();
    std::cout << "[warmup] " << queries.size() << " queries at "
              << variable_param_name << "=" << variable_values.front()
              << " (parallel; not comparable to timed run): "
              << std::fixed << std::setprecision(3) << t_warm.total_time()
              << " s (excluded)" << std::endl;
  }

  size_t k = 0;
  for (size_t val : variable_values) {
    SearchParams sp = make_sp(val);
    k = sp.k;

    std::cout << "\n--- " << variable_param_name << " = " << val << " ---" << std::endl;

    parlay::internal::timer t_sweep;
    t_sweep.start();
    StatsExtended result = compute_stats_extended_p_threaded(
        index, points, queries, gt, sp, /*num_threads=*/1, /*num_warmup=*/0);
    t_sweep.stop();
    print_result_extended(result, k, timing_labels);
    std::cout << "  Total time:     " << std::fixed << std::setprecision(4)
              << t_sweep.total_time() << " s" << std::endl;

    if (!csv_path.empty()) {
      write_csv_row(csv_path, Stats(result.QPS_seq, result.QPS_par, result.avg_cmps,
                                    result.recall_1_k, result.recall_k_k),
                    k, variable_param_name, val, sp.num_rerank);
    }

    if (result.recall_k_k >= 1.0) {
      std::cout << "  Recall@k reached 1.0. Stopping." << std::endl;
      break;
    }
  }
}

template <typename Index, typename ChPoint, typename MakeSearchParams>
void run_search_all_sweep(Index& index,
                          PointCloudSet<ChPoint>& points,
                          PointCloudSet<ChPoint>& queries,
                          const parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>>& gt,
                          const std::string& variable_param_name,
                          const std::vector<size_t>& variable_values,
                          MakeSearchParams make_sp,
                          const std::string& csv_path = "") {
  size_t k = 0;
  constexpr size_t reps = 3;

  // Pre-warm the TQ8 rerank DB if any of the requested settings will use it,
  // so the first timed rep doesn't pay the one-time encode (~8 s on NQ).
  // The min-of-reps below would already exclude that cost from the reported
  // QPS, but pre-warming keeps the first rep representative — useful when
  // anyone is reading the per-rep stderr output for debugging.  Idempotent
  // via call_once.
  if (!variable_values.empty()) {
    SearchParams probe_sp = make_sp(variable_values.front());
    if (probe_sp.tq8_rerank) {
      parlay::internal::timer t_warm;
      t_warm.start();
      index.ensure_tq8_rerank_db_(points);
      t_warm.stop();
      std::cout << "  [warmup] TQ8 DB encode: " << t_warm.total_time() << " sec (excluded)"
                << std::endl;
    }
    // One discarded search_all to fault in encoded leaf data and warm CPU
    // caches.  Without this, rep 0 always pays a cold-page penalty
    // (~1-2 s on NQ on top of the encode).  We use the first nprobes value;
    // larger values touch a superset of leaves so warming the smallest one
    // is safe and slightly cheaper.
    parlay::internal::timer t_pre;
    t_pre.start();
    auto warmup_pair = index.search_all(queries, points, probe_sp);
    (void)warmup_pair;
    t_pre.stop();
    std::cout << "  [warmup] search_all (discarded): " << t_pre.total_time()
              << " sec (excluded)" << std::endl;
  }

  for (size_t val : variable_values) {
    SearchParams sp = make_sp(val);
    k = sp.k;

    std::cout << "\n--- " << variable_param_name << " = " << val << " ---" << std::endl;

    double best_time = 1e15;
    parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>> pred;
    for (size_t it = 0; it < reps; it++) {
      parlay::internal::timer t;
      t.start();
      auto [p, cmps] = index.search_all(queries, points, sp);
      t.stop();
      if (t.total_time() < best_time) {
        best_time = t.total_time();
        if (it == 0) pred = std::move(p);
      }
    }
    double QPS = queries.size() / best_time;
    double recall_1_k = compute_recall(pred, gt, k, 1);
    double recall_k_k = compute_recall(pred, gt, k, k);

    std::cout << std::fixed << std::setprecision(4)
              << "  Recall 1@" << k << ":   " << recall_1_k << "\n"
              << "  Recall " << k << "@" << k << ":   " << recall_k_k << "\n"
              << std::setprecision(1)
              << "  QPS_par:      " << QPS << "\n"
              << std::flush;

    if (!csv_path.empty()) {
      write_csv_row(csv_path, Stats(0.0, QPS, 0.0, recall_1_k, recall_k_k),
                    k, variable_param_name, val, sp.num_rerank);
    }

    if (recall_k_k >= 1.0) {
      std::cout << "  Recall@k reached 1.0. Stopping." << std::endl;
      break;
    }
  }
}

}  // namespace bench
}  // namespace mvsic

// ========================================================================
// 7. Dist-func dispatch macro
// ========================================================================
#define PARSE_DIST_FUNC_AND_RUN(run_fn, help_str)                                        \
  int main(int argc, char* argv[]) {                                                     \
    mvsic::commandLine P(argc, argv, help_str);                                          \
    std::string df = P.getOptionValue("-dist_func", "IP");                               \
    if (df == "L2") {                                                                    \
      run_fn<mvsic::ChamferL2_Point, true>(P);                                          \
    } else if (df == "IP") {                                                             \
      run_fn<mvsic::ChamferIP_Point, false>(P);                                         \
    } else {                                                                             \
      std::cerr << "Unknown -dist_func: " << df << " (use IP or L2)" << std::endl;      \
      return 1;                                                                          \
    }                                                                                    \
    return 0;                                                                            \
  }
