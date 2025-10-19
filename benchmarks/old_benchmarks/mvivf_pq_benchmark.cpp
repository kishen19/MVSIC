#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <stdlib.h>

#include "mvsic/core/stats.h"
#include "mvsic/core/utils/chamfer_ip_point.h"
#include "mvsic/core/utils/chamfer_l2_point.h"
#include "mvsic/core/utils/parse_command_line.h"
#include "mvsic/core/utils/point_cloud_set.h"
#include "mvsic/mvivf/mvivf.h"

using namespace mvsic;

template<typename ChPoint, bool metric>
void grid_search_bench(mvsic::commandLine& P) {
  Eigen::setNbThreads(1);
  char* inFile = P.getOptionValue("-i");
  char* qFile = P.getOptionValue("-q");
  std::string QFile;
  if (qFile != nullptr) {
    QFile = P.getOptionValue("-q");
  } else {
    QFile = "";
  }
  std::string gtFile = P.getOptionValue("-gt", "");
  std::string index_dir = P.getOptionValue("-index_dir", "mvivf_indices");
  std::string results_dir = P.getOptionValue("-results_dir", "mvivf_pq_results");

  // PQ params
  const std::vector<uint32_t> num_blocks_values = {2, 4, 8, 16};
  const std::vector<uint32_t> num_clusters_per_block_values = {256};
  uint32_t sample_size = P.getOptionIntValue("-pq_ss", 100000);

  // Fixed params
  uint32_t max_leaf_size = P.getOptionIntValue("-mls", 500);
  uint32_t k_per_level = P.getOptionIntValue("-kpl", 0);

  const std::vector<uint32_t> nprobes_values = {1, 2, 4, 8, 16, 32, 64, 128, 256};
  const std::vector<uint32_t> k_values = {10, 100};

  using PC = PointCloudSet<ChPoint>;
  auto points = PC(inFile);
  auto queries = PC(qFile);
  auto gt = ReadGT(gtFile, queries.size());

  system(("mkdir -p " + index_dir).c_str());
  system(("mkdir -p " + results_dir).c_str());

  for (uint32_t num_blocks : num_blocks_values) {
    for (uint32_t num_clusters_per_block : num_clusters_per_block_values) {
      std::cout << "Building index for num_blocks=" << num_blocks
                << ", num_clusters_per_block=" << num_clusters_per_block << std::endl;
      IndexParams index_params =
          IndexParams::mvivf(k_per_level, max_leaf_size, false, 0, 5, 20, "Random", 0, false, 0,
                             true, num_blocks, num_clusters_per_block, sample_size);
      IndexMVIVF<metric> index(points.get_dims(), index_params);
      index.build(points);

      std::string index_path = index_dir + "/mvivf_mls" + std::to_string(max_leaf_size) + "_kpl" +
                               std::to_string(k_per_level) + "_pq_nb" + std::to_string(num_blocks) +
                               "_nc" + std::to_string(num_clusters_per_block) + ".index";
      index.save(index_path);
      std::cout << "Index saved to " << index_path << std::endl;

      std::string results_path = results_dir + "/results_mls" + std::to_string(max_leaf_size) +
                                 "_kpl" + std::to_string(k_per_level) + "_pq_nb" +
                                 std::to_string(num_blocks) + "_nc" +
                                 std::to_string(num_clusters_per_block) + ".csv";
      std::ofstream results_file(results_path);
      results_file << "k,nprobes,num_rerank,recall_1,recall_k,QPS,QPS_par,Avg_cmps" << std::endl;
      std::cout << "Writing results to " << results_path << std::endl;

      for (uint32_t k : k_values) {
        const std::vector<uint32_t> num_rerank_values = {0, 2 * k, 4 * k, 8 * k};
        for (uint32_t nprobes : nprobes_values) {
          for (uint32_t num_rerank : num_rerank_values) {
            SearchParams search_params = SearchParams::mvivf(k, nprobes, num_rerank);
            Stats result = compute_stats(index, points, queries, gt, search_params);
            results_file << k << "," << nprobes << "," << num_rerank << "," << result.recall_1_k << ","
                         << result.recall_k_k << "," << result.QPS_seq << "," << result.QPS_par << ","
                         << result.avg_cmps << std::endl;
          }
        }
      }
      results_file.close();
    }
  }
  std::cout << "Grid search complete." << std::endl;
}

int main(int argc, char* argv[]) {
  mvsic::commandLine P(argc, argv,
                       "[-i <inFile>] [-q <qFile>] [-gt <gtFile>] [-index_dir <dir>] "
                       "[-results_dir <dir>] [-dist_func <IP|L2>] [-pq_ss <sample_size>]"
                       "[-mls <max_leaf_size>] [-kpl <k_per_level>]");

  std::string df = P.getOptionValue("-dist_func", "IP");

  if (df == "L2") {
    using ChPoint = ChamferL2_Point;
    grid_search_bench<ChPoint, true>(P);
  } else if (df == "IP") {
    using ChPoint = ChamferIP_Point;
    grid_search_bench<ChPoint, false>(P);
  }
  return 0;
}
