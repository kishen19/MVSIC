#include <Eigen/Dense>
#include <iostream>
#include "mvsic/core/types/chamfer_ip_point.h"
#include "mvsic/core/types/chamfer_l2_point.h"
#include "mvsic/core/types/point_cloud_set.h"
#include "mvsic/core/utils/parse_command_line.h"
#include "mvsic/core/stats.h"
#include "mvivf.h"
#include "mvivf_flat.h"
#include "mvivf_spill.h"

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

// Parse -query_compress {none,ball,wards}, -query_compress_threshold, -compress_rerank
inline void apply_query_compression_opts(SearchParams& sp, mvsic::commandLine& P) {
  std::string qc = P.getOptionValue("-query_compress", "none");
  if (qc == "none" || qc == "off" || qc == "0") {
    sp.query_compression = SearchParams::QueryCompression::None;
  } else if (qc == "ball" || qc == "ballcarving" || qc == "muvera") {
    sp.query_compression = SearchParams::QueryCompression::Carve;
  } else if (qc == "wards" || qc == "ward") {
    sp.query_compression = SearchParams::QueryCompression::Wards;
  } else {
    std::cerr << "Unknown -query_compress value: " << qc << " (use none, ball, wards)" << std::endl;
    std::exit(1);
  }
  sp.query_compression_threshold =
      static_cast<float>(P.getOptionDoubleValue("-query_compress_threshold", 0.7));
  sp.compress_rerank = P.getOption("-compress_rerank");
}

template<typename ChPoint, bool metric>
void run_bench(mvsic::commandLine& P) {
  Eigen::setNbThreads(1);
  using PC = PointCloudSet<ChPoint>;

  char* inFile = P.getOptionValue("-i");
  char* qFile = P.getOptionValue("-q");
  std::string QFile = (qFile != nullptr) ? std::string(qFile) : "";
  std::string gtFile = P.getOptionValue("-gt", "");
  std::string outFile = P.getOptionValue("-o", "");
  std::string indexFile = P.getOptionValue("-index", "");
  bool is_mmap = P.getOption("-mm");
  // -qc selects the CompressCenters template variant (TQ-quantized centers).
  bool compress_centers = P.getOption("-qc");

  bool compress_input = P.getOption("-compress_input");
  uint32_t verbose = P.getOptionIntValue("-v", 0);

  uint32_t k_per_level = P.getOptionIntValue("-k_per_level", 0);
  uint32_t max_leaf_size = P.getOptionIntValue("-max_leaf_size", 200);
  uint32_t max_depth = P.getOptionIntValue("-max_depth", 0);

  uint32_t niters = P.getOptionIntValue("-niters", 5);
  uint32_t max_point_clouds_per_cluster = P.getOptionIntValue("-mpcc", 0);
  uint32_t max_points_per_centroid_inner_kmeans = P.getOptionIntValue("-mpcik", 20);
  bool use_weighted_inner_kmeans = P.getOption("-wgh_kmeans");

  bool is_flat = P.getOption("-flat");
  bool is_spill = P.getOption("-spill");
  uint32_t num_spill = static_cast<uint32_t>(P.getOptionIntValue("-spill", 2));
  uint32_t num_spill_l2 = static_cast<uint32_t>(P.getOptionIntValue("-spill_l2", 1));
  auto variant = parse_mvivf_variant(is_flat, is_spill);

  std::string quant_method = P.getOptionValue("-quant_method", "None");
  // Leaf-quantizer hyperparameters are currently passed via default
  // Model::Params; wire these into the dispatch lambda via a typed
  // LeafParams ctor to actually customize codebook shapes.
  [[maybe_unused]] uint32_t block_size = P.getOptionIntValue("-m", 8);
  [[maybe_unused]] uint32_t num_clusters_per_block =
      P.getOptionIntValue("-num_clusters_per_block", 256);
  [[maybe_unused]] uint32_t num_points_per_cluster =
      P.getOptionIntValue("-num_points_per_cluster", 20);
  [[maybe_unused]] uint32_t rabitq_bits = P.getOptionIntValue("-rbits", 8);

  size_t k = P.getOptionLongValue("-k", 10);
  size_t nprobes = P.getOptionLongValue("-nprobes", 2);
  size_t num_rerank = P.getOptionLongValue("-num_rerank", k);

  auto points = PC(inFile, is_mmap);
  std::cout << "Avg points per cloud: " << points.average_size() << std::endl;

  IndexParams index_params;
  SearchParams search_params;
  if (variant == MVIVFVariant::Flat) {
    index_params = IndexParams::mvivf_flat(
        k_per_level, compress_input, verbose, niters, max_point_clouds_per_cluster,
        max_points_per_centroid_inner_kmeans, "Random", 0, use_weighted_inner_kmeans, 0);
    search_params = SearchParams::mvivf_flat(k, nprobes, num_rerank);
  } else if (variant == MVIVFVariant::Spill) {
    index_params = IndexParams::mvivf_spill(
        k_per_level, max_leaf_size, num_spill, compress_input, verbose, niters,
        max_point_clouds_per_cluster, max_points_per_centroid_inner_kmeans, "Random", 0,
        use_weighted_inner_kmeans, 0, max_depth, num_spill_l2);
    search_params = SearchParams::mvivf(k, nprobes, num_rerank);
  } else {
    index_params = IndexParams::mvivf(
        k_per_level, max_leaf_size, compress_input, verbose, niters, max_point_clouds_per_cluster,
        max_points_per_centroid_inner_kmeans, "Random", 0, use_weighted_inner_kmeans, 0,
        max_depth);
    search_params = SearchParams::mvivf(k, nprobes, num_rerank);
  }
  apply_query_compression_opts(search_params, P);
  if (search_params.query_compression != SearchParams::QueryCompression::None) {
    const char* mname = (search_params.query_compression == SearchParams::QueryCompression::Carve)
                            ? "ball"
                            : "wards";
    std::cout << "Query compression: method=" << mname
              << " threshold=" << search_params.query_compression_threshold
              << " compress_rerank=" << (search_params.compress_rerank ? "1" : "0") << std::endl;
  }

  dispatch_mvivf<metric>(variant, compress_centers, quant_method,
      [&]<class IndexT>() {
        IndexT index(points.get_dims(), index_params);
        if (!indexFile.empty()) {
          std::cout << "Loading index from " << indexFile << std::endl;
          index.load(indexFile, points);
          std::cout << "Index loaded" << std::endl;
        } else {
          std::cout << "Building index (" << variant_name(variant)
                    << ", compress_centers=" << (compress_centers ? 1 : 0)
                    << ", leaf=" << quant_method << ")..." << std::endl;
          parlay::internal::timer it;
          it.start();
          index.build(points);
          it.stop();
          std::cout << "Index built in " << it.total_time() << " seconds." << std::endl;
        }
        if (!outFile.empty()) {
          std::cout << "Saving index to " << outFile << std::endl;
          index.save(outFile);
          std::cout << "Index saved." << std::endl;
        }

        if (QFile.empty()) return;
        auto queries = PC(qFile);
        if (search_params.query_compression != SearchParams::QueryCompression::None) {
          uint32_t ba = qc_internal::batch_alignment(index.quantization_mode);
          double sum_orig = 0.0;
          double sum_comp = 0.0;
          for (size_t j = 0; j < queries.size(); ++j) {
            sum_orig += static_cast<double>(queries[j].size());
            auto c = compress_query<ChPoint>(queries[j], search_params.query_compression,
                                             search_params.query_compression_threshold, ba);
            sum_comp += static_cast<double>(c.n);
          }
          const double nq = static_cast<double>(queries.size());
          std::cout << "Avg query points (raw):        " << (sum_orig / nq) << std::endl
                    << "Avg query points (compressed): " << (sum_comp / nq) << std::endl
                    << "Compression ratio (raw/compr): "
                    << (sum_comp > 0 ? sum_orig / sum_comp : 0.0) << std::endl;
        }
        auto gt = ReadGT(gtFile, queries.size());
        std::cout << "Computing stats..." << std::endl;
        Stats result = compute_stats(index, points, queries, gt, search_params);
        std::cout << "Number of Queries: " << queries.size() << std::endl
                  << "QPS_seq: " << result.QPS_seq << std::endl
                  << "QPS_par: " << result.QPS_par << std::endl
                  << "Average cmps: " << result.avg_cmps << std::endl
                  << "Average recall 1 @ " << k << ": " << result.recall_1_k << std::endl
                  << "Average recall " << k << " @ " << k << ": " << result.recall_k_k << std::endl;
      });
}

int main(int argc, char* argv[]) {
  mvsic::commandLine P(argc, argv,
      "MVIVF combined build+search benchmark (single-shot, parallel).\n\n"
      "Dataset / I/O:\n"
      "  -d <name> | -i <points> -q <queries> -gt <gt>\n"
      "  -o <save_path>                 Save built index (optional)\n"
      "  -index <path>                  Load pre-built index (skip build)\n"
      "  -mm                            Memory-map points file\n"
      "  -v <level>                     Verbosity (0..3)\n"
      "  -compress_input                Apply point-cloud input compression\n"
      "  -dist_func IP|L2               Distance metric (default IP)\n\n"
      "MVIVF variant:\n"
      "  -flat                          IndexMVIVFFlat\n"
      "  -spill <N>                     IndexMVIVFSpill with N spillover entries\n"
      "  (no flag)                      IndexMVIVF\n\n"
      "Clustering structure (build):\n"
      "  -k_per_level <N>               Branching factor (0 = auto)\n"
      "  -max_leaf_size <N>             Leaf size threshold (default 500)\n"
      "  -max_depth <N>                 Cap recursion depth (0 = unlimited)\n"
      "  -niters <N>                    K-means iters (default 5)\n"
      "  -mpcc <N>                      max_point_clouds_per_cluster\n"
      "  -mpcik <N>                     max_points_per_centroid_inner_kmeans\n"
      "  -wgh_kmeans                    Use weighted inner k-means\n"
      "  -s <N>                         Seeding strategy index\n\n"
      "Quantization (selects compile-time class):\n"
      "  -qc                            CompressCenters = true (TQ centers)\n"
      "  -quant_method None|PQ|FS|RQ|TQ|SPQTQ|1BTQ    Leaf quantizer\n"
      "  -m <N>  -num_clusters_per_block <N>  -num_points_per_cluster <N>  -rbits <N>\n\n"
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
    run_bench<ChamferL2_Point, true>(P);
  } else if (df == "IP") {
    run_bench<ChamferIP_Point, false>(P);
  }
  return 0;
}
