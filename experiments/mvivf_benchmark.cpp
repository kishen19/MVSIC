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
  std::string results_dir = P.getOptionValue("-results_dir", "mvivf_results");

  std::vector<std::pair<uint32_t, uint32_t>> params;
#ifdef grid_search
  const std::vector<uint32_t> max_leaf_sizes = {200, 500, 1000, 2000};
  const std::vector<uint32_t> k_per_levels = {16, 32, 64};
  for (uint32_t mls : max_leaf_sizes) {
    for (uint32_t kpl : k_per_levels) {
      params.push_back({mls, kpl});
    }
  }
#else
  params.push_back({500, 32});
  params.push_back({1000, 32});
  params.push_back({500, 64});
  params.push_back({1000, 64});
#endif
  const std::vector<uint32_t> nprobes_values = {1, 2, 4, 8, 16, 32, 64, 128, 256};
  const uint32_t k = P.getOptionIntValue("-k", 10);

  using PC = PointCloudSet<ChPoint>;
  auto points = PC(inFile);
  auto queries = PC(qFile);
  auto gt = ReadGT(gtFile, queries.size());

  system(("mkdir -p " + index_dir).c_str());
  system(("mkdir -p " + results_dir).c_str());

  for (auto [max_leaf_size, k_per_level] : params) {
    std::cout << "Building index for max_leaf_size=" << max_leaf_size << ", k_per_level=" << k_per_level << std::endl;
    IndexParams index_params = IndexParams::mvivf(max_leaf_size, k_per_level);
    IndexMVIVF<metric> index(points.get_dims(), index_params);
    index.build(points);

    std::string index_path = index_dir + "/mvivf_mls" + std::to_string(max_leaf_size) + "_kpl" +
                             std::to_string(k_per_level) + ".index";
    index.save(index_path);
    std::cout << "Index saved to " << index_path << std::endl;

    std::string results_path = results_dir + "/results_mls" + std::to_string(max_leaf_size) + "_kpl" +
                               std::to_string(k_per_level) + ".csv";
    std::ofstream results_file(results_path);
    results_file << "nprobes,recall 1@" << k << ",recall " << k << "@" << k
                 << ",QPS,QPS_par,Avg_cmps" << std::endl;
    std::cout << "Writing results to " << results_path << std::endl;

    for (uint32_t nprobes : nprobes_values) {
        SearchParams search_params = SearchParams::mvivf(k, nprobes, 0);
        Stats result = compute_stats(index, points, queries, gt, search_params);
        results_file << nprobes << "," << result.recall_1_k << "," << result.recall_k_k
                     << "," << result.QPS_seq << "," << result.QPS_par << "," << result.avg_cmps
                     << std::endl;
    }
    results_file.close();
  }
  std::cout << "Grid search complete." << std::endl;
}

int main(int argc, char* argv[]) {
  mvsic::commandLine P(argc, argv,
                       "[-i <inFile>] [-q <qFile>] [-gt <gtFile>] [-k <k>] [-index_dir <dir>] "
                       "[-results_dir <dir>] [-dist_func <IP|L2>]");

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