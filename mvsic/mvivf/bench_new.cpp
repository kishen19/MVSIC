#include <Eigen/Dense>
#include <iostream>

#include "mvsic/core/types/chamfer_ip_point.h"
#include "mvsic/core/types/point_cloud_set.h"
#include "mvsic/core/utils/parse_command_line.h"
#include "mvsic/core/stats.h"
#include "mvsic/mvivf/mvivf_base.h"  // Updated absolute include path

using namespace mvsic;

void bench_ip(mvsic::commandLine& P) {
  Eigen::setNbThreads(1);
  using ChPoint = ChamferIP_Point;
  using PC = PointCloudSet<ChPoint>;

  // File Paths
  char* inFile = P.getOptionValue("-i");
  char* qFile = P.getOptionValue("-q");
  std::string QFile = (qFile != nullptr) ? std::string(qFile) : "";
  std::string gtFile = P.getOptionValue("-gt", "");
  std::string outFile = P.getOptionValue("-o", "");
  std::string indexFile = P.getOptionValue("-index", "");

  bool is_mmap = P.getOption("-mm");
  bool compress_input = P.getOption("-compress_input");
  uint32_t verbose = P.getOptionIntValue("-v", 0);

  // MVIVF Params
  uint32_t k_per_level = P.getOptionIntValue("-k_per_level", 0);
  uint32_t max_leaf_size = P.getOptionIntValue("-max_leaf_size", 200);

  // MVClus Params
  uint32_t niters = P.getOptionIntValue("-niters", 5);
  uint32_t max_point_clouds_per_cluster = P.getOptionIntValue("-mpcc", 0);
  uint32_t max_points_per_centroid_inner_kmeans = P.getOptionIntValue("-mpcik", 20);
  bool use_weighted_inner_kmeans = P.getOption("-wgh_kmeans");

  // Search Params
  size_t k = P.getOptionLongValue("-k", 10);
  size_t nprobes = P.getOptionLongValue("-nprobes", 2);
  size_t num_rerank = P.getOptionLongValue("-num_rerank", k);

  // Load Dataset
  auto points = PC(inFile, is_mmap);

  // Construct Index Params
  // Note: Since we stripped out quantization for this pure architecture test,
  // we pass dummy/default values (0/false) for the quantization-specific fields
  // at the end of your original mvivf_params struct.
  IndexParams index_params =
      IndexParams::mvivf(k_per_level, max_leaf_size, compress_input, verbose, niters,
                         max_point_clouds_per_cluster, max_points_per_centroid_inner_kmeans,
                         "Random", 0, use_weighted_inner_kmeans, 0, 0, 8, 256, 20, 8, false);

  SearchParams search_params = SearchParams::mvivf(k, nprobes, num_rerank);

  // Instantiate the unquantized IP index
  IndexMVIVFIP index(points.get_dims(), index_params);

  // Load or Build
  if (indexFile != "") {
    std::cout << "Loading index from " << indexFile << "..." << std::endl;
    index.load(indexFile, points);
    std::cout << "Index loaded." << std::endl;
  } else {
    std::cout << "Building index..." << std::endl;
    parlay::internal::timer it;
    it.start();
    index.build(points);
    it.stop();
    std::cout << "Index built in " << it.total_time() << " seconds." << std::endl;
  }

  // Save
  if (outFile != "") {
    std::cout << "Saving index to " << outFile << "..." << std::endl;
    index.save(outFile);
    std::cout << "Index saved." << std::endl;
  }

  // Search & Stats
  if (QFile != "") {
    auto queries = PC(QFile.c_str());
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
}

int main(int argc, char* argv[]) {
  mvsic::commandLine P(argc, argv,
                       "[-i <inFile>] [-q <queryFile>] [-gt <groundTruthFile>] "
                       "[-k <num_neighbors>] [-nprobes <num_probes>]");

  std::cout << "--- Running Simplified MVIVF Benchmark (IP Metric, No Quantization) ---"
            << std::endl;

  bench_ip(P);

  return 0;
}