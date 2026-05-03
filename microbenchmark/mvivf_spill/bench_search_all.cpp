#include <filesystem>
#include <iostream>
#include <string>

#include "parlay/primitives.h"

#include "mvsic/core/bench_utils.h"
#include "mvsic/core/stats.h"
#include "mvsic/core/types/chamfer_ip_point.h"
#include "mvsic/core/types/chamfer_l2_point.h"
#include "mvsic/core/types/point_cloud_set.h"
#include "mvsic/core/utils/parse_command_line.h"
#include "mvsic/mvivf/mvivf_spill.h"
#include "microbenchmark/mvivf_spill/bench_search_all_inst.h"

using namespace mvsic;

// ----- CLI -> compile-time IndexMVIVFSpill<metric, CompressCenters, LeafModel>
//       dispatch.
//
// All heavy template specializations of run_one are instantiated in separate
// translation units under bench_search_all_inst/ — one .cc per
// (metric, quantizer) cell, each instantiating both compress=false and
// compress=true. Bazel compiles those TUs in parallel; this dispatch TU only
// sees `extern template` declarations and stays cheap.
namespace {

#define MVIVF_SPILL_DISPATCH_LM(metric, C, qm, fn)                                                \
  do {                                                                                            \
    if      (qm == "None"  || qm == "none")                                                       \
      fn.template operator()<IndexMVIVFSpill<metric, C, NoQuantizer<metric>>>();                  \
    else if (qm == "PQ"    || qm == "pq")                                                         \
      fn.template operator()<IndexMVIVFSpill<metric, C, pq_mv::Model<metric>>>();                 \
    else if (qm == "FS"    || qm == "fs")                                                         \
      fn.template operator()<IndexMVIVFSpill<metric, C, fastscan_mv::Model<metric>>>();           \
    else if (qm == "RQ"    || qm == "rq")                                                         \
      fn.template operator()<IndexMVIVFSpill<metric, C, rabitq_mv::Model<metric>>>();             \
    else if (qm == "TQ"    || qm == "tq")                                                         \
      fn.template operator()<IndexMVIVFSpill<metric, C, turboquant_mv::Model<metric>>>();         \
    else if (qm == "SPQTQ" || qm == "spqtq")                                                      \
      fn.template operator()<IndexMVIVFSpill<metric, C, pqtq_mv::Model<metric>>>();               \
    else if (qm == "1BTQ"  || qm == "1btq")                                                       \
      fn.template operator()<IndexMVIVFSpill<metric, C, turboquant_1bit_mv::Model<metric>>>();    \
    else if (qm == "1BTQA" || qm == "1btqa")                                                      \
      fn.template operator()<IndexMVIVFSpill<metric, C, turboquant_1bit_asym_mv::Model<metric>>>(); \
    else {                                                                                        \
      std::cerr << "Unknown -quant_method: " << qm                                                \
                << " (use None, PQ, FS, RQ, TQ, SPQTQ, 1BTQ, 1BTQA)" << std::endl;                \
      std::exit(1);                                                                               \
    }                                                                                             \
  } while (0)

template <bool metric, class Fn>
void dispatch_mvivf_spill(bool compress_centers, const std::string& quant_method, Fn&& fn) {
  if (compress_centers) MVIVF_SPILL_DISPATCH_LM(metric, true, quant_method, fn);
  else                  MVIVF_SPILL_DISPATCH_LM(metric, false, quant_method, fn);
}
#undef MVIVF_SPILL_DISPATCH_LM

void apply_query_compression_opts(SearchParams& sp, mvsic::commandLine& P) {
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
  sp.tq8_rerank = P.getOption("-tq8_rerank");
  sp.root_m2m = P.getOption("-root_m2m");
}

}  // namespace

template<typename ChPoint, bool metric>
void run_benchmark(mvsic::commandLine& P) {
  Eigen::setNbThreads(1);
  using PC = PointCloudSet<ChPoint>;

  std::string inFile = P.getOptionValue("-i", "");
  std::string qFile = P.getOptionValue("-q", "");
  std::string indexFile = P.getOptionValue("-index", "");
  std::string gtFile = P.getOptionValue("-gt", "");
  bool is_mmap = P.getOption("-mm");
  std::string mode = P.getOptionValue("-mode", "both");
  bool compress_centers = P.getOptionIntValue("-qc", 1) != 0;
  std::string quant_method = P.getOptionValue("-quant_method", "None");

  std::size_t k = P.getOptionLongValue("-k", 10);
  std::string nprobes_str = P.getOptionValue("-nprobes", "100");
  std::vector<std::size_t> nprobes_list = bench::parse_csv_ints(nprobes_str);
  if (nprobes_list.empty()) nprobes_list.push_back(100);
  std::size_t num_rerank = P.getOptionLongValue("-num_rerank", k);

  uint32_t k_per_level = static_cast<uint32_t>(P.getOptionIntValue("-k_per_level", 0));
  uint32_t max_leaf_size = static_cast<uint32_t>(P.getOptionIntValue("-max_leaf_size", 500));
  uint32_t max_depth = static_cast<uint32_t>(P.getOptionIntValue("-max_depth", 0));
  uint32_t niters = static_cast<uint32_t>(P.getOptionIntValue("-niters", 5));
  uint32_t mpcc = static_cast<uint32_t>(P.getOptionIntValue("-mpcc", 100));
  uint32_t mpcik = static_cast<uint32_t>(P.getOptionIntValue("-mpcik", 20));
  bool wgh_kmeans = P.getOptionIntValue("-wgh_kmeans", 1) != 0;
  uint32_t s_param = static_cast<uint32_t>(P.getOptionIntValue("-s", 0));
  uint32_t verbose = static_cast<uint32_t>(P.getOptionIntValue("-v", 0));

  // Spill-specific knobs.
  uint32_t num_spill = static_cast<uint32_t>(P.getOptionIntValue("-num_spill", 2));
  uint32_t num_spill_l2 = static_cast<uint32_t>(P.getOptionIntValue("-num_spill_l2", 1));

  if (inFile.empty() || qFile.empty() || indexFile.empty()) {
    std::cerr <<
        "MVIVF-Spill microbenchmark: compares search_all (per-query loop) vs\n"
        "search_all_new (batched, leaf-grouped) on a templated\n"
        "IndexMVIVFSpill<metric, /*CompressCenters=*/-qc, LeafModel>.\n"
        "If -index is missing, builds and saves a raw skeleton (CompressCenters=0,\n"
        "LeafModel=NoQuantizer) using the build-side flags below, then loads it\n"
        "under the requested template.\n\n"
        "Required:\n"
        "  -i <points.pcs>                Database point clouds\n"
        "  -q <queries.pcs>               Query point clouds\n"
        "  -index <path>                  Skeleton path (auto-built if missing)\n\n"
        "Search:\n"
        "  -gt <gt>                       Ground-truth file for recall reporting\n"
        "  -mm                            Memory-map points\n"
        "  -dist_func IP|L2               Distance metric (default IP)\n"
        "  -k <K>                         Top-k (default 10)\n"
        "  -nprobes <N|csv>               Probes; single value or CSV like \"32,128,256\" "
        "(default 100)\n"
        "  -num_rerank <N>                Rerank budget (default = k)\n"
        "  -mode old|new|both             Which kernel to run (default both)\n"
        "  -qc 0|1                        CompressCenters template (default 1)\n"
        "  -quant_method None|PQ|FS|RQ|TQ|SPQTQ|1BTQ|1BTQA   Leaf quantizer (default None)\n"
        "  -query_compress none|ball|wards  Query-side compression\n"
        "  -query_compress_threshold <t>  Threshold for ball/wards (default 0.7)\n"
        "  -compress_rerank               Use compressed query for rerank too\n"
        "  -tq8_rerank                    Rerank with 8-bit TurboQuant kernel\n"
        "  -root_m2m                      Share root-level greedy work across queries\n"
        "                                 via fused many-to-many chamfer\n\n"
        "Build (only used when the skeleton has to be built; defaults mirror\n"
        "IndexParams::mvivf_spill()):\n"
        "  -k_per_level <N> (0)           Branching factor per level (0 = 4*sqrt(n))\n"
        "  -max_leaf_size <N> (500)       Stop splitting when cluster <= this\n"
        "  -max_depth <N> (0)             Cap recursion depth (0 = unlimited)\n"
        "  -niters <N> (5)                K-means iterations\n"
        "  -mpcc <N> (100)                max_point_clouds_per_cluster\n"
        "  -mpcik <N> (20)                max_points_per_centroid_inner_kmeans\n"
        "  -wgh_kmeans 0|1 (1)            Use weighted inner k-means\n"
        "  -s <N> (0)                     Centroid point-cloud size (0 = avg)\n"
        "  -num_spill <N> (2)             Root-level spill factor (a)\n"
        "  -num_spill_l2 <N> (1)          Level-1 spill factor (b)\n"
        "  -build_8btq 0|1 (0)            Use 8BTQ panel kernel for k-means assignment\n"
        "  -v <level> (0)                 Build verbosity\n";
    exit(1);
  }

  std::cout << "Loading dataset..." << std::endl;
  auto points = PC(inFile.c_str(), is_mmap);

  std::cout << "Loading queries..." << std::endl;
  auto queries = PC(qFile.c_str());

  parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>> gt;
  bool has_gt = !gtFile.empty();
  if (has_gt) {
    std::cout << "Loading ground truth..." << std::endl;
    gt = ReadGT(gtFile, queries.size());
    std::cout << "GT loaded: " << gt.size() << " queries, " << gt[0].size()
              << " neighbors each" << std::endl;
  }

  IndexParams index_params = IndexParams::mvivf_spill(
      k_per_level, max_leaf_size, num_spill, /*compress_input=*/false, verbose, niters, mpcc,
      mpcik, "Random", /*seed=*/0, wgh_kmeans, s_param, max_depth, num_spill_l2);
  index_params.build_with_8btq = (P.getOptionIntValue("-build_8btq", 0) != 0);
  if (verbose >= 1) {
    std::cout << "[bench_search_all] num_spill=" << num_spill
              << " num_spill_l2=" << num_spill_l2
              << " build_with_8btq=" << (index_params.build_with_8btq ? "true" : "false")
              << std::endl;
  }

  build_skeleton_if_missing<ChPoint, metric>(points, index_params, indexFile);

  // Resolve query-compression options once into the ctx.
  SearchParams resolved_sp;
  apply_query_compression_opts(resolved_sp, P);

  MicroSpillSearchAllCtx<ChPoint> ctx{
      .points = &points,
      .queries = &queries,
      .index_params = index_params,
      .index_path = indexFile,
      .quant_method_name = quant_method,
      .mode = mode,
      .compress_centers = compress_centers,
      .nprobes_list = nprobes_list,
      .k = k,
      .num_rerank = num_rerank,
      .gt = std::move(gt),
      .has_gt = has_gt,
      .query_compression = static_cast<int>(resolved_sp.query_compression),
      .query_compression_threshold = resolved_sp.query_compression_threshold,
      .compress_rerank = resolved_sp.compress_rerank,
      .tq8_rerank = resolved_sp.tq8_rerank,
      .root_m2m = resolved_sp.root_m2m,
  };

  dispatch_mvivf_spill<metric>(compress_centers, quant_method, [&]<class IndexT>() {
    run_one<ChPoint, metric, IndexT>(ctx);
  });
}

int main(int argc, char* argv[]) {
  mvsic::commandLine P(argc, argv,
                       "-i <points> -q <queries> -index <skeleton> "
                       "[-gt <gt>] [-mm] [-dist_func IP|L2] "
                       "[-k <K>] [-nprobes <N>] [-num_rerank <N>] "
                       "[-mode old|new|both] [-qc 0|1] "
                       "[-quant_method None|PQ|FS|RQ|TQ|SPQTQ|1BTQ|1BTQA] "
                       "[-query_compress none|ball|wards] [-query_compress_threshold <tau>] "
                       "[-compress_rerank] [-tq8_rerank] [-root_m2m] "
                       "[-k_per_level <N>] [-max_leaf_size <N>] [-max_depth <N>] "
                       "[-niters <N>] [-mpcc <N>] [-mpcik <N>] [-wgh_kmeans 0|1] "
                       "[-s <N>] [-num_spill <N>] [-num_spill_l2 <N>] "
                       "[-build_8btq 0|1] [-v <level>]");
  std::string df = P.getOptionValue("-dist_func", "IP");

  if (df == "L2") {
    using ChPoint = ChamferL2_Point;
    run_benchmark<ChPoint, true>(P);
  } else if (df == "IP") {
    using ChPoint = ChamferIP_Point;
    run_benchmark<ChPoint, false>(P);
  } else {
    std::cerr << "Unknown distance function: " << df << std::endl;
    return 1;
  }
  return 0;
}
