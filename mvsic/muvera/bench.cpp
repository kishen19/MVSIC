#include <Eigen/Dense>
#include <iostream>
#include "mvsic/core/types/chamfer_ip_point.h"
#include "mvsic/core/types/chamfer_l2_point.h"
#include "mvsic/core/types/point_cloud_set.h"
#include "mvsic/core/utils/parse_command_line.h"
#include "mvsic/core/stats.h"
#include "muvera.h"

using namespace mvsic;

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

  // FDE params
  int32_t num_repetitions = P.getOptionIntValue("-num_reps", 20);
  int32_t num_simhash_projections = P.getOptionIntValue("-num_simhash", 4);
  int32_t seed = 1;
  int32_t projection_dimension = P.getOptionIntValue("-projd", 8);
  bool fill_empty_partitions = P.getOption("-fill_empty_partitions");
  int32_t final_projection_dimension = P.getOptionIntValue("-final_projd", 0);
  bool not_normalized = P.getOption("-no_norm");

  // Vamana params
  uint32_t R = 200;
  uint32_t L_build = 600;
  double alpha = P.getOptionDoubleValue("-a", 1.2);
  int num_pass = P.getOptionIntValue("-np", 1);

  // PQ params
  std::string quant_method = P.getOptionValue("-quant_method", "None");
  uint32_t quant_method_t = 0;
  if (quant_method == "None") {
    quant_method_t = 0;
  } else if (quant_method == "PQ") {
    quant_method_t = 1;
  } else if (quant_method == "RQ") {
    quant_method_t = 2;
  } else if (quant_method == "FS") {
    quant_method_t = 3;
  } else if (quant_method == "TQ") {
    quant_method_t = 4;
  } else {
    std::cerr << "Unknown PQ method: " << quant_method << std::endl;
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
  IndexParams index_params = IndexParams::muvera_custom(
      num_repetitions, num_simhash_projections, seed, projection_dimension, fill_empty_partitions,
      final_projection_dimension, !not_normalized, R, L_build, alpha, num_pass, compress_input,
      verbose, quant_method_t, block_size, num_clusters_per_block, num_points_per_cluster,
      rabitq_bits);
  SearchParams search_params = SearchParams::muvera(k, L, num_rerank, cut, norerank);
  IndexMUVERA<metric> index(points.get_dims(), index_params);
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
    double QPS_seq, QPS_par, avg_cmps, recall_1_k, recall_k_k;

    // Compute Stats:
    std::cout << "Computing stats..." << std::endl;
    Stats result = compute_stats(index, points, queries, gt, search_params);
    QPS_seq = result.QPS_seq;
    QPS_par = result.QPS_par;
    avg_cmps = result.avg_cmps;
    recall_1_k = result.recall_1_k;
    recall_k_k = result.recall_k_k;
    std::cout << "Number of Queries: " << queries.size() << std::endl
              << "QPS_seq: " << QPS_seq << std::endl
              << "QPS_par: " << QPS_par << std::endl
              << "Average cmps: " << avg_cmps << std::endl
              << "Average recall 1 @ " << k << ": " << recall_1_k << std::endl
              << "Average recall " << k << " @ " << k << ": " << recall_k_k << std::endl;
  }
}

int main(int argc, char* argv[]) {
  mvsic::commandLine P(argc, argv,
                       "[-i <inFile>] [-k <num_centers>] [-s <num_embeddings>]"
                       "[-data_type <tp>] [-dist_func <dist_func>]"
                       "[-seed <algorithm>] [-iters <num_iters>]"
                       // "[-kmeans_seed <algorithm>] [-kmeans_dist <algorithm>]"
  );
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
