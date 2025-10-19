#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "mvsic/core/stats.h"
#include "mvsic/core/utils/chamfer_ip_point.h"
#include "mvsic/core/utils/chamfer_l2_point.h"
#include "mvsic/core/utils/parse_command_line.h"
#include "mvsic/core/utils/point_cloud_set.h"
#include "mvsic/mpool/mpool.h"

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
  std::string index_dir = P.getOptionValue("-index_dir", "mpool_indices");
  std::string results_dir = P.getOptionValue("-results_dir", "mpool_results");

  // Grid search parameters
  const std::vector<uint32_t> num_cands = {1, 5, 10, 20, 50, 100, 200};
  const std::vector<uint32_t> L_values = {64, 128, 256, 512, 1024, 2048, 4096};
  const uint32_t k = P.getOptionIntValue("-k", 10);

  using PC = PointCloudSet<ChPoint>;
  auto points = PC(inFile);
  auto queries = PC(qFile);
  auto gt = ReadGT(gtFile, queries.size());

  system(("mkdir -p " + results_dir).c_str());

  std::cout << "Building index..." << std::endl;
  // Vamana params for mpool from mvsic/mpool/bench.cpp
  uint32_t R = 200;
  uint32_t L_build = 600;
  double alpha = 1.2;
  bool two_pass = false;
  uint32_t verbose = 0;
  IndexParams index_params =
      IndexParams::mpool(R, L_build, alpha, two_pass, true, verbose);  // normalize = true
  IndexMPool<metric> index(points.get_dims(), index_params);
  index.build(points);
  std::cout << "Index built." << std::endl;
  std::string index_path = index_dir + "/mpool.index";
  index.save(index_path);
  // index.load(index_path, points);
  std::cout << "Index saved to " << index_path << std::endl;
  // std::cout << "Index loaded from " << index_path << std::endl;

  std::string results_path = results_dir + "/results.csv";
  std::ofstream results_file(results_path);
  results_file << "num_cands,L,recall 1@" << k << ",recall " << k << "@" << k
               << ",QPS,QPS_par,Avg_cmps" << std::endl;
  std::cout << "Writing results to " << results_path << std::endl;

  for (uint32_t cands : num_cands) {
    for (uint32_t l : L_values) {
      if (l >= cands * k) {
        SearchParams search_params =
            SearchParams::mpool(k, l, 1.35, points.size(), R, true, k * cands);
        Stats result = compute_stats(index, points, queries, gt, search_params);
        results_file << cands << "," << l << "," << result.recall_1_k << "," << result.recall_k_k
                     << "," << result.QPS_seq << "," << result.QPS_par << "," << result.avg_cmps
                     << std::endl;
      }
    }
  }
  results_file.close();

  std::cout << "Grid search complete." << std::endl;
}

int main(int argc, char* argv[]) {
  mvsic::commandLine P(argc, argv,
                       "[-i <inFile>] [-q <qFile>] [-gt <gtFile>] [-k <k>]"
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
