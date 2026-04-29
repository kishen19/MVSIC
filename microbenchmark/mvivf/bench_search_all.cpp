#include <fstream>
#include <filesystem>
#include <iostream>
#include <string>

#include "parlay/primitives.h"

#include "mvsic/core/types/chamfer_ip_point.h"
#include "mvsic/core/types/chamfer_l2_point.h"
#include "mvsic/core/types/point_cloud_set.h"
#include "mvsic/core/utils/parse_command_line.h"
#include "mvsic/core/stats.h"
#include "mvsic/mvivf/mvivf.h"

using namespace mvsic;

namespace {
#define MVIVF_DISPATCH_LM(metric, C, qm, fn)                                                      \
  do {                                                                                            \
    if      (qm == "None"  || qm == "none")                                                       \
      fn.template operator()<IndexMVIVF<metric, C, NoQuantizer<metric>>>();                      \
    else if (qm == "PQ"    || qm == "pq")                                                         \
      fn.template operator()<IndexMVIVF<metric, C, pq_mv::Model<metric>>>();                     \
    else if (qm == "FS"    || qm == "fs")                                                         \
      fn.template operator()<IndexMVIVF<metric, C, fastscan_mv::Model<metric>>>();               \
    else if (qm == "RQ"    || qm == "rq")                                                         \
      fn.template operator()<IndexMVIVF<metric, C, rabitq_mv::Model<metric>>>();                 \
    else if (qm == "TQ"    || qm == "tq")                                                         \
      fn.template operator()<IndexMVIVF<metric, C, turboquant_mv::Model<metric>>>();             \
    else if (qm == "SPQTQ" || qm == "spqtq")                                                      \
      fn.template operator()<IndexMVIVF<metric, C, pqtq_mv::Model<metric>>>();                   \
    else if (qm == "1BTQ"  || qm == "1btq")                                                       \
      fn.template operator()<IndexMVIVF<metric, C, turboquant_1bit_mv::Model<metric>>>();        \
    else {                                                                                        \
      std::cerr << "Unknown -quant_method: " << qm                                                \
                << " (use None, PQ, FS, RQ, TQ, SPQTQ, 1BTQ)" << std::endl;                       \
      std::exit(1);                                                                               \
    }                                                                                             \
  } while (0)

template <bool metric, class Fn>
void dispatch_mvivf(bool compress_centers, const std::string& quant_method, Fn&& fn) {
  if (compress_centers) MVIVF_DISPATCH_LM(metric, true, quant_method, fn);
  else                  MVIVF_DISPATCH_LM(metric, false, quant_method, fn);
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

template<typename ChPoint, bool metric>
void run_benchmark(mvsic::commandLine& P) {
  Eigen::setNbThreads(1);
  using PC = PointCloudSet<ChPoint>;

  std::string inFile = P.getOptionValue("-i", "");
  std::string qFile = P.getOptionValue("-q", "");
  std::string indexFile = P.getOptionValue("-index", "");
  std::string gtFile = P.getOptionValue("-gt", "");
  bool is_mmap = P.getOption("-mm");
  // -mode: "both" (default), "old", "new" — lets you isolate each for perf stat
  std::string mode = P.getOptionValue("-mode", "both");
  // This microbenchmark always exercises the best center-compressed variant
  // for the load path.  The on-disk skeleton itself is variant-agnostic.
  constexpr bool compress_centers = true;
  std::string quant_method = P.getOptionValue("-quant_method", "None");

  size_t k = P.getOptionLongValue("-k", 10);
  size_t nprobes = P.getOptionLongValue("-nprobes", 100);
  size_t num_rerank = P.getOptionLongValue("-num_rerank", k);

  if (inFile.empty() || qFile.empty() || indexFile.empty()) {
    std::cerr <<
        "MVIVF microbenchmark: compares search_all (original) vs search_all_new\n"
        "on a templated IndexMVIVF<metric, /*CompressCenters=*/true, LeafModel>.\n"
        "If -index is missing, builds and saves a raw skeleton (CompressCenters=0,\n"
        "LeafModel=NoQuantizer) once, then loads it under the requested template.\n\n"
        "Required:\n"
        "  -i <points.pcs>                Database point clouds\n"
        "  -q <queries.pcs>               Query point clouds\n"
        "  -index <path>                  Skeleton path (auto-built if missing)\n\n"
        "Optional:\n"
        "  -gt <gt>                       Ground-truth file for recall reporting\n"
        "  -mm                            Memory-map points\n"
        "  -dist_func IP|L2               Distance metric (default IP)\n"
        "  -k <K>                         Top-k (default 10)\n"
        "  -nprobes <N>                   Number of probes (default 100)\n"
        "  -num_rerank <N>                Rerank budget (default = k)\n"
        "  -mode old|new|both             Which kernel to run (default both)\n"
        "  -quant_method None|PQ|FS|RQ|TQ|SPQTQ|1BTQ   Leaf quantizer (default None)\n"
        "  -query_compress none|ball|wards  Query-side compression\n"
        "  -query_compress_threshold <t>  Threshold for ball/wards (default 0.7)\n"
        "  -compress_rerank               Use compressed query for rerank too\n";
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

  IndexParams index_params = IndexParams::mvivf();
  const std::filesystem::path idx_path(indexFile);
  if (!std::filesystem::exists(idx_path)) {
    // Build and save a raw skeleton.  save() is only valid on the
    // <metric, false, NoQuantizer> variant; any templated variant can load
    // that skeleton and retrain its center / leaf quantizers on load.
    std::cout << "Index not found at " << indexFile
              << ". Building raw skeleton (compress_centers=0, leaf_quantizer=None)..."
              << std::endl;
    using SkeletonIndex = IndexMVIVF<metric, false, NoQuantizer<metric>>;
    SkeletonIndex skeleton(points.get_dims(), index_params);
    parlay::internal::timer tb;
    tb.start();
    skeleton.build(points);
    tb.stop();
    std::cout << "Skeleton built in " << tb.total_time() << " seconds." << std::endl;
    if (!idx_path.parent_path().empty()) {
      std::filesystem::create_directories(idx_path.parent_path());
    }
    std::cout << "Saving skeleton index to " << indexFile << std::endl;
    skeleton.save(indexFile);
  }

  dispatch_mvivf<metric>(compress_centers, quant_method, [&]<class IndexT>() {
    IndexT index(points.get_dims(), index_params);
    std::cout << "Loading index (compress_centers=" << (compress_centers ? 1 : 0)
              << ", leaf_quantizer=" << quant_method << ") from " << indexFile << " ..."
              << std::endl;
    index.load(indexFile, points);

    SearchParams search_params = SearchParams::mvivf(k, nprobes, num_rerank);
    apply_query_compression_opts(search_params, P);
    if (search_params.query_compression != SearchParams::QueryCompression::None) {
      const char* mname = (search_params.query_compression == SearchParams::QueryCompression::Carve)
                              ? "ball"
                              : "wards";
      std::cout << "Query compression: " << mname
                << " tau=" << search_params.query_compression_threshold
                << " compress_rerank=" << (search_params.compress_rerank ? 1 : 0) << std::endl;
    }

    bool run_old = (mode == "both" || mode == "old");
    bool run_new = (mode == "both" || mode == "new");

    parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>> pred1, pred2;
    size_t cmps1 = 0, cmps2 = 0;
    double time1 = 0, time2 = 0;

    if (run_old) {
      std::cout << "\n========================================" << std::endl;
      std::cout << "Running search_all (Original)..." << std::endl;
      parlay::internal::timer t1;
      t1.start();
      auto [p1, c1] = index.search_all(queries, points, search_params);
      t1.stop();
      pred1 = std::move(p1);
      cmps1 = c1;
      time1 = t1.total_time();
      std::cout << "search_all time: " << time1 << " seconds. Dist cmps: " << cmps1 << std::endl;
    }

    if (run_new) {
      std::cout << "\n========================================" << std::endl;
      std::cout << "Running search_all_new..." << std::endl;
      parlay::internal::timer t2;
      t2.start();
      auto [p2, c2] = index.search_all_new(queries, points, search_params);
      t2.stop();
      pred2 = std::move(p2);
      cmps2 = c2;
      time2 = t2.total_time();
      std::cout << "search_all_new time: " << time2 << " seconds. Dist cmps: " << cmps2 << std::endl;
    }

    if (run_old && run_new) {
      std::cout << "\n========================================" << std::endl;
      std::cout << "Speedup (old / new): " << time1 / time2 << "x" << std::endl;
      std::cout << "========================================" << std::endl;

      if (has_gt) {
        auto [r1_old, rk_old] = compute_scores(pred1, gt, k);
        auto [r1_new, rk_new] = compute_scores(pred2, gt, k);
        std::cout << "\nRecall vs Ground Truth:" << std::endl;
        std::cout << "  search_all:     recall@1 = " << r1_old << ", recall@" << k << " = " << rk_old
                  << std::endl;
        std::cout << "  search_all_new: recall@1 = " << r1_new << ", recall@" << k << " = " << rk_new
                  << std::endl;
      }

      double agreement = compute_recall(pred2, pred1, k, k);
      std::cout << "\nAgreement (new vs old): " << agreement << std::endl;
    } else if (has_gt) {
      auto& pred = run_old ? pred1 : pred2;
      auto [r1, rk] = compute_scores(pred, gt, k);
      std::cout << "\nRecall vs Ground Truth: recall@1 = " << r1
                << ", recall@" << k << " = " << rk << std::endl;
    }
  });
}

int main(int argc, char* argv[]) {
  mvsic::commandLine P(argc, argv,
                       "-i <points> -q <queries> -index <skeleton> "
                       "[-gt <gt>] [-mm] [-dist_func IP|L2] "
                       "[-k <K>] [-nprobes <N>] [-num_rerank <N>] "
                       "[-mode old|new|both] "
                       "[-quant_method None|PQ|FS|RQ|TQ|SPQTQ|1BTQ] "
                       "[-query_compress none|ball|wards] [-query_compress_threshold <tau>] "
                       "[-compress_rerank]");
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
