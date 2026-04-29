// Sequential MVIVF benchmark.
//
// Compiled with -DPARLAY_SEQUENTIAL so all parlay::parallel_for loops run
// sequentially. Only the (non-flat, non-spill) IndexMVIVF family is covered
// here; use bench_search_seq.cpp for the Flat/Spill variants. CLI flags
// -qc (compress centers) and -quant_method {None,PQ,FS,RQ,TQ,SPQTQ,1BTQ}
// select the concrete templated Index type at compile time.
#include <Eigen/Dense>
#include <iostream>

#include "mvivf.h"
#include "mvivf_flat.h"
#include "mvivf_spill.h"
#include "mvsic/core/stats.h"
#include "mvsic/core/types/chamfer_ip_point.h"
#include "mvsic/core/types/chamfer_l2_point.h"
#include "mvsic/core/types/point_cloud_set.h"
#include "mvsic/core/utils/parse_command_line.h"

using namespace mvsic;

// ----- CLI -> compile-time IndexMVIVF<metric, CompressCenters, LeafModel> dispatch. -----
namespace {
enum class MVIVFVariant { Regular, Flat, Spill };

inline MVIVFVariant parse_mvivf_variant(bool is_flat, bool is_spill) {
  if (is_flat && is_spill) {
    std::cerr << "-flat and -spill are mutually exclusive." << std::endl;
    std::exit(1);
  }
  return is_flat ? MVIVFVariant::Flat
       : is_spill ? MVIVFVariant::Spill
                  : MVIVFVariant::Regular;
}

inline const char* variant_name(MVIVFVariant v) {
  return v == MVIVFVariant::Flat  ? "MVIVF_Flat"
       : v == MVIVFVariant::Spill ? "MVIVF_Spill"
                                  : "MVIVF";
}

#define MVIVF_DISPATCH_LM(Fam, metric, C, qm, fn)                                                 \
  do {                                                                                            \
    if      (qm == "None"  || qm == "none")                                                       \
      fn.template operator()<Fam<metric, C, NoQuantizer<metric>>>();                              \
    else if (qm == "PQ"    || qm == "pq")                                                         \
      fn.template operator()<Fam<metric, C, pq_mv::Model<metric>>>();                             \
    else if (qm == "FS"    || qm == "fs")                                                         \
      fn.template operator()<Fam<metric, C, fastscan_mv::Model<metric>>>();                       \
    else if (qm == "RQ"    || qm == "rq")                                                         \
      fn.template operator()<Fam<metric, C, rabitq_mv::Model<metric>>>();                         \
    else if (qm == "TQ"    || qm == "tq")                                                         \
      fn.template operator()<Fam<metric, C, turboquant_mv::Model<metric>>>();                     \
    else if (qm == "SPQTQ" || qm == "spqtq")                                                      \
      fn.template operator()<Fam<metric, C, pqtq_mv::Model<metric>>>();                           \
    else if (qm == "1BTQ"  || qm == "1btq")                                                       \
      fn.template operator()<Fam<metric, C, turboquant_1bit_mv::Model<metric>>>();                \
    else {                                                                                        \
      std::cerr << "Unknown -quant_method: " << qm                                                \
                << " (use None, PQ, FS, RQ, TQ, SPQTQ, 1BTQ)" << std::endl;                       \
      std::exit(1);                                                                               \
    }                                                                                             \
  } while (0)

template <bool metric, class Fn>
void dispatch_mvivf(MVIVFVariant v, bool compress, const std::string& qm, Fn&& fn) {
  if (v == MVIVFVariant::Flat) {
    if (compress) MVIVF_DISPATCH_LM(IndexMVIVFFlat,  metric, true,  qm, fn);
    else          MVIVF_DISPATCH_LM(IndexMVIVFFlat,  metric, false, qm, fn);
  } else if (v == MVIVFVariant::Spill) {
    if (compress) MVIVF_DISPATCH_LM(IndexMVIVFSpill, metric, true,  qm, fn);
    else          MVIVF_DISPATCH_LM(IndexMVIVFSpill, metric, false, qm, fn);
  } else {
    if (compress) MVIVF_DISPATCH_LM(IndexMVIVF,      metric, true,  qm, fn);
    else          MVIVF_DISPATCH_LM(IndexMVIVF,      metric, false, qm, fn);
  }
}
#undef MVIVF_DISPATCH_LM
}  // namespace

static void apply_query_compression_opts(SearchParams& sp, mvsic::commandLine& P) {
  std::string qc = P.getOptionValue("-query_compress", "none");
  if (qc == "none" || qc == "off" || qc == "0") {
    sp.query_compression = SearchParams::QueryCompression::None;
  } else if (qc == "ball" || qc == "ballcarving" || qc == "muvera") {
    sp.query_compression = SearchParams::QueryCompression::Carve;
  } else if (qc == "wards" || qc == "ward") {
    sp.query_compression = SearchParams::QueryCompression::Wards;
  } else {
    std::cerr << "Unknown -query_compress: " << qc << " (none, ball, wards)" << std::endl;
    std::exit(1);
  }
  sp.query_compression_threshold =
      static_cast<float>(P.getOptionDoubleValue("-query_compress_threshold", 0.7));
  sp.compress_rerank = P.getOption("-compress_rerank");
}

template <typename ChPoint, bool metric>
void run(commandLine& P) {
  Eigen::setNbThreads(1);
  using PC = PointCloudSet<ChPoint>;

  char* inFile = P.getOptionValue("-i");
  char* qFile = P.getOptionValue("-q");
  std::string gtFile = P.getOptionValue("-gt", "");
  std::string indexFile = P.getOptionValue("-index", "");
  bool is_mmap = P.getOption("-mm");
  // Defaults mirror IndexParams::mvivf in mvsic/core/index_params.h.
  // Integer-valued bool flags (-qc, -wgh_kmeans) use `<flag> 0|1` form so we
  // can preserve factory defaults of `true`.
  bool compress_centers = P.getOptionIntValue("-qc", 0) != 0;
  bool compress_input = P.getOption("-compress_input");
  uint32_t verbose = P.getOptionIntValue("-v", 0);

  uint32_t k_per_level = P.getOptionIntValue("-k_per_level", 0);
  uint32_t max_leaf_size = P.getOptionIntValue("-max_leaf_size", 500);
  uint32_t max_depth = P.getOptionIntValue("-max_depth", 0);

  uint32_t niters = P.getOptionIntValue("-niters", 5);
  uint32_t mpcc = P.getOptionIntValue("-mpcc", 100);
  uint32_t mpcik = P.getOptionIntValue("-mpcik", 20);
  bool wgh_kmeans = P.getOptionIntValue("-wgh_kmeans", 1) != 0;

  std::string quant_method = P.getOptionValue("-quant_method", "None");
  [[maybe_unused]] uint32_t block_size = P.getOptionIntValue("-m", 8);
  [[maybe_unused]] uint32_t num_clusters_per_block =
      P.getOptionIntValue("-num_clusters_per_block", 16);
  [[maybe_unused]] uint32_t num_points_per_cluster =
      P.getOptionIntValue("-num_points_per_cluster", 100);
  [[maybe_unused]] uint32_t rabitq_bits = P.getOptionIntValue("-rbits", 4);

  size_t k = P.getOptionLongValue("-k", 10);
  size_t nprobes_single = P.getOptionLongValue("-nprobes", 32);
  size_t num_rerank = P.getOptionLongValue("-num_rerank", k);

  if (inFile == nullptr || qFile == nullptr || indexFile.empty() || gtFile.empty()) {
    std::cerr << "Required: -i <points> -q <queries> -gt <gt> -index <index>" << std::endl;
    std::exit(1);
  }

  std::cout << "Loading points..." << std::endl;
  auto points = PC(inFile, is_mmap);

  IndexParams index_params = IndexParams::mvivf(
      k_per_level, max_leaf_size, compress_input, verbose, niters, mpcc, mpcik, "Random", 0,
      wgh_kmeans, 0, max_depth);

  std::vector<size_t> nprobes_list = {nprobes_single};

  dispatch_mvivf<metric>(MVIVFVariant::Regular, compress_centers, quant_method,
      [&]<class IndexT>() {
        IndexT index(points.get_dims(), index_params);
        std::cout << "Loading index from " << indexFile << std::endl;
        index.load(indexFile, points);

        std::cout << "Loading queries..." << std::endl;
        auto queries = PC(qFile);
        auto gt = ReadGT(gtFile, queries.size());

        const size_t reps = 3;
        for (size_t nprobes : nprobes_list) {
          SearchParams search_params = SearchParams::mvivf(k, nprobes, num_rerank);
          apply_query_compression_opts(search_params, P);
          if (search_params.query_compression != SearchParams::QueryCompression::None) {
            const char* mname =
                (search_params.query_compression == SearchParams::QueryCompression::Carve)
                    ? "ball" : "wards";
            std::cout << "query_compress=" << mname
                      << " tau=" << search_params.query_compression_threshold
                      << " compress_rerank=" << (search_params.compress_rerank ? 1 : 0)
                      << std::endl;
          }
          std::cout << "\n=== nprobes=" << nprobes
                    << " (compress_centers=" << (compress_centers ? 1 : 0)
                    << ", leaf=" << quant_method << ") ===" << std::endl;

          parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>> pred(queries.size());
          parlay::sequence<size_t> cmps(queries.size());

          for (size_t j = 0; j < std::min<size_t>(10, queries.size()); j++) {
            (void)index.search(queries[j], points, search_params);
          }

          double best_time = 1e15;
          double tot_t_compress = 0.0;
          double tot_t_greedy = 0.0;
          double tot_t_distances = 0.0;
          double tot_t_quantize = 0.0;
          double tot_t_rerank = 0.0;
          double tot_t_rest = 0.0;
          for (size_t it = 0; it < reps; it++) {
            parlay::internal::timer t;
            double iter_time = 0.0;
            for (size_t j = 0; j < queries.size(); j++) {
              if (it == 0) {
                t.start();
                auto [p, b, stats] = index.search_with_stats(queries[j], points, search_params);
                t.stop();
                iter_time += t.total_time();
                t.reset();
                pred[j] = p;
                cmps[j] = static_cast<size_t>(stats[0]) + static_cast<size_t>(stats[1]);
                size_t n = stats.size();
                if (n < 6) {
                  std::cerr << "search_with_stats: expected >=6 trailing timings, got " << n
                            << std::endl;
                  std::exit(1);
                }
                tot_t_compress  += stats[n - 6];
                tot_t_quantize  += stats[n - 5];
                tot_t_distances += stats[n - 4];
                tot_t_rest      += stats[n - 3];
                tot_t_rerank    += stats[n - 2];
                tot_t_greedy    += stats[n - 1];
              } else {
                t.start();
                (void)index.search(queries[j], points, search_params);
                t.stop();
                iter_time += t.total_time();
                t.reset();
              }
            }
            best_time = std::min(best_time, iter_time);
          }

          double QPS = queries.size() / best_time;
          double avg_cmps = (double)parlay::reduce(cmps) / (double)cmps.size();
          double recall_1_k = compute_recall(pred, gt, k, 1);
          double recall_k_k = compute_recall(pred, gt, k, k);

          double t_score_total = tot_t_greedy + tot_t_distances;
          double frac_hier = t_score_total > 0 ? tot_t_greedy / t_score_total : 0.0;
          double frac_leaf = t_score_total > 0 ? tot_t_distances / t_score_total : 0.0;

          std::cout << "Number of Queries:    " << queries.size() << std::endl
                    << "QPS:                  " << QPS << std::endl
                    << "Avg pointcloud cmps:  " << avg_cmps << std::endl
                    << "Total pointclouds:    " << points.size() << std::endl
                    << "Recall 1@" << k << ":            " << recall_1_k << std::endl
                    << "Recall " << k << "@" << k << ":           " << recall_k_k << std::endl
                    << "t_hierarchy (greedy): " << tot_t_greedy << " s" << std::endl
                    << "t_leaves (distances): " << tot_t_distances << " s" << std::endl
                    << "t_compress:           " << tot_t_compress << " s" << std::endl
                    << "t_quantize:           " << tot_t_quantize << " s" << std::endl
                    << "t_rerank:             " << tot_t_rerank << " s" << std::endl
                    << "t_rest:               " << tot_t_rest << " s" << std::endl
                    << "frac scoring hier:    " << frac_hier << std::endl
                    << "frac scoring leaves:  " << frac_leaf << std::endl;
        }
      });
}

int main(int argc, char* argv[]) {
  mvsic::commandLine P(argc, argv,
      "MVIVF sequential combined build+search benchmark (PARLAY_SEQUENTIAL, latency-focused).\n\n"
      "Dataset / I/O:\n"
      "  -d <name> | -i <points> -q <queries> -gt <gt>\n"
      "  -index <path>                  Load pre-built index (skip build)\n"
      "  -o <save_path>                 Save built index (optional)\n"
      "  -mm  -v <level>  -compress_input  -dist_func IP|L2\n\n"
      "MVIVF variant:\n"
      "  -flat                          IndexMVIVFFlat\n"
      "  -spill <N>                     IndexMVIVFSpill\n"
      "  (no flag)                      IndexMVIVF\n\n"
      "Clustering structure (build; defaults mirror IndexParams::mvivf*):\n"
      "  -k_per_level <N> (0)   -max_leaf_size <N> (500)   -max_depth <N> (0)\n"
      "  -niters <N> (5)   -mpcc <N> (100)   -mpcik <N> (20)\n"
      "  -wgh_kmeans 0|1 (1)   -s <N> (0)\n\n"
      "Quantization:\n"
      "  -qc 0|1                        CompressCenters; default 0\n"
      "  -quant_method None|PQ|FS|RQ|TQ|SPQTQ|1BTQ    (default None)\n"
      "  -m <N> (8)   -num_clusters_per_block <N> (16)\n"
      "  -num_points_per_cluster <N> (100)   -rbits <N> (4)\n\n"
      "Search:\n"
      "  -k <N>                         Top-k (default 10)\n"
      "  -nprobes <csv>                 nprobes values to sweep\n"
      "  -num_rerank <N>                Rerank width\n\n"
      "Query compression:\n"
      "  -query_compress none|ball|wards\n"
      "  -query_compress_threshold <tau>\n"
      "  -compress_rerank\n");

  std::string df = P.getOptionValue("-dist_func", "IP");
  if (df == "L2") {
    run<ChamferL2_Point, true>(P);
  } else if (df == "IP") {
    run<ChamferIP_Point, false>(P);
  }
  return 0;
}
