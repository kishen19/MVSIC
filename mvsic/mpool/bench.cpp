#include <Eigen/Dense>
#include <iostream>
#include "mvsic/core/types/chamfer_ip_point.h"
#include "mvsic/core/types/chamfer_l2_point.h"
#include "mvsic/core/types/point_cloud_set.h"
#include "mvsic/core/utils/parse_command_line.h"
#include "mvsic/core/stats.h"
#include "mpool.h"

using namespace mvsic;

namespace {
// CLI -> compile-time IndexMPool<metric, LeafModel> dispatch.  1BTQ is
// omitted because the SV-variant of turboquant_1bit hasn't been ported yet.
#define MPOOL_DISPATCH(metric, qm, fn)                                                        \
  do {                                                                                         \
    if      (qm == "None"  || qm == "none")                                                    \
      fn.template operator()<IndexMPool<metric, NoQuantizer<metric>>>();                       \
    else if (qm == "PQ"    || qm == "pq")                                                      \
      fn.template operator()<IndexMPool<metric, pq::Model<metric>>>();                         \
    else if (qm == "FS"    || qm == "fs")                                                      \
      fn.template operator()<IndexMPool<metric, fastscan::Model<metric>>>();                   \
    else if (qm == "RQ"    || qm == "rq")                                                      \
      fn.template operator()<IndexMPool<metric, rabitq::Model<metric>>>();                     \
    else if (qm == "TQ"    || qm == "tq")                                                      \
      fn.template operator()<IndexMPool<metric, turboquant::Model<metric>>>();                 \
    else if (qm == "SPQTQ" || qm == "spqtq")                                                   \
      fn.template operator()<IndexMPool<metric, pqtq::Model<metric>>>();                       \
    else {                                                                                     \
      std::cerr << "Unknown -quant_method: " << qm                                             \
                << " (use None, PQ, FS, RQ, TQ, SPQTQ)" << std::endl;                          \
      std::exit(1);                                                                            \
    }                                                                                          \
  } while (0)

template <bool metric, class Fn>
void dispatch_mpool(const std::string& qm, Fn&& fn) {
  MPOOL_DISPATCH(metric, qm, fn);
}
#undef MPOOL_DISPATCH
}  // namespace

template<typename ChPoint, bool metric>
void bench(mvsic::commandLine& P) {
  Eigen::setNbThreads(1);
  using PC = PointCloudSet<ChPoint>;

  char* inFile = P.getOptionValue("-i");
  char* qFile = P.getOptionValue("-q");
  std::string QFile;
  if (qFile != nullptr) {
    QFile = P.getOptionValue("-q");
  } else {
    QFile = "";
  }
  std::string gtFile = P.getOptionValue("-gt", "");
  std::string outFile = P.getOptionValue("-o", "");
  std::string indexFile = P.getOptionValue("-index", "");
  bool is_mmap = P.getOption("-mm");

  bool compress_input = P.getOption("-compress_input");
  uint32_t verbose = P.getOptionIntValue("-v", 0);
  bool not_normalized = P.getOption("-no_norm");

  // Vamana params
  uint32_t R = 200;
  uint32_t L_build = 600;
  double alpha = P.getOptionDoubleValue("-a", 1.2);
  int num_pass = P.getOptionIntValue("-np", 1);

  // Quantization params (determines which IndexMPool template variant to
  // instantiate; the legacy uint32_t method id is also passed into IndexParams
  // so the runtime variant-based fallback path keeps working).
  std::string quant_method = P.getOptionValue("-quant_method", "None");
  uint32_t quant_method_t = 0;
  if      (quant_method == "None"  || quant_method == "none")  quant_method_t = 0;
  else if (quant_method == "PQ"    || quant_method == "pq")    quant_method_t = 1;
  else if (quant_method == "RQ"    || quant_method == "rq")    quant_method_t = 2;
  else if (quant_method == "FS"    || quant_method == "fs")    quant_method_t = 3;
  else if (quant_method == "TQ"    || quant_method == "tq")    quant_method_t = 4;
  else if (quant_method == "SPQTQ" || quant_method == "spqtq") quant_method_t = 6;
  else {
    std::cerr << "Unknown -quant_method: " << quant_method
              << " (use None, PQ, FS, RQ, TQ, SPQTQ)" << std::endl;
    exit(1);
  }
  uint32_t block_size = P.getOptionIntValue("-m", 8);
  uint32_t num_clusters_per_block = P.getOptionIntValue("-num_clusters_per_block", 256);
  uint32_t num_points_per_cluster = P.getOptionIntValue("-num_points_per_cluster", 20);
  uint32_t rabitq_bits = P.getOptionIntValue("-rbits", 8);

  // Search Params
  size_t k = P.getOptionLongValue("-k", 10);
  size_t L = P.getOptionLongValue("-L", 16);
  double cut = 1.35;
  size_t num_rerank = P.getOptionLongValue("-num_rerank", k);
  bool norerank = P.getOption("-norerank");

  auto points = PC(inFile, is_mmap);
  IndexParams index_params = IndexParams::mpool(
      R, L_build, alpha, num_pass, !not_normalized, compress_input, verbose, quant_method_t,
      block_size, num_clusters_per_block, num_points_per_cluster, rabitq_bits);
  SearchParams search_params = SearchParams::mpool(k, L, num_rerank, cut, norerank);

  dispatch_mpool<metric>(quant_method, [&]<class IndexT>() {
    IndexT index(points.get_dims(), index_params);
    if (indexFile != "") {
      std::cout << "Loading index from " << indexFile << std::endl;
      index.load(indexFile, points);
      std::cout << "Index loaded" << std::endl;
    } else {
      std::cout << "Building index..." << std::endl;
      parlay::internal::timer it;
      it.start();
      index.build(points);
      it.stop();
      std::cout << "Index built in " << it.total_time() << " seconds." << std::endl;
    }
    if (outFile != "") {
      std::cout << "Saving index to " << outFile << std::endl;
      index.save(P.getOptionValue("-o"));
      std::cout << "Index saved." << std::endl;
    }

    if (QFile != "") {
      auto queries = PC(qFile);
      auto gt = ReadGT(gtFile, queries.size());
      std::cout << "Computing stats..." << std::endl;
      Stats result = compute_stats(index, points, queries, gt, search_params);
      std::cout << "Number of Queries: " << queries.size() << std::endl
                << "QPS_seq: " << result.QPS_seq << std::endl
                << "QPS_par: " << result.QPS_par << std::endl
                << "Average cmps: " << result.avg_cmps << std::endl
                << "Average recall 1 @ " << k << ": " << result.recall_1_k << std::endl
                << "Average recall " << k << " @ " << k << ": " << result.recall_k_k << std::endl;
    }
  });
}

int main(int argc, char* argv[]) {
  mvsic::commandLine P(argc, argv,
      "MPool combined build+search benchmark (single-point).\n\n"
      "Dataset / I/O:\n"
      "  -i <points>  -q <queries>  -gt <gt>\n"
      "  -o <save_path>  -index <path>\n"
      "  -mm  -v <level>  -compress_input  -dist_func IP|L2\n\n"
      "MPool (Vamana-based) build params:\n"
      "  -R <N>                         Graph out-degree (default 200)\n"
      "  -L_build <N>                   Build beam width (default 600)\n"
      "  -a <f>                         alpha (default 1.2)\n"
      "  -np <N>                        Num passes (default 1)\n"
      "  -no_norm                       Disable normalization\n\n"
      "Leaf quantization (selects the templated IndexMPool<metric, LeafModel>):\n"
      "  -quant_method None|PQ|FS|RQ|TQ|SPQTQ\n"
      "  -m <N>  -num_clusters_per_block <N>  -num_points_per_cluster <N>  -rbits <N>\n\n"
      "Search:\n"
      "  -k <N>                         Top-k (default 10)\n"
      "  -L <N>                         Search beam width (default 16)\n"
      "  -num_rerank <N>                Rerank width (default k)\n"
      "  -norerank                      Disable rerank stage\n");
  std::string df = P.getOptionValue("-dist_func", "IP");

  if (df == "L2") {
    using ChPoint = ChamferL2_Point;
    bench<ChPoint, true>(P);
  } else if (df == "IP") {
    using ChPoint = ChamferIP_Point;
    bench<ChPoint, false>(P);
  }
  return 0;
}
