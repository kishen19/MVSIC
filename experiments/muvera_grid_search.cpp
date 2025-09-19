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
#include "mvsic/muvera/muvera.h"

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
  std::string index_dir = P.getOptionValue("-index_dir", "muvera_indices");
  std::string results_dir = P.getOptionValue("-results_dir", "muvera_results");

  // {d_fde, num_reps, num_simhash, projd}
  std::vector<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> > params;
#ifdef grid_search
  // Grid index parameters
  const std::vector<uint32_t> num_reps = {1, 5, 10, 15, 20};
  const std::vector<uint32_t> num_simhash = {2, 3, 4, 5, 6};
  const std::vector<uint32_t> projd = {8, 16, 32, 64};
  for (uint32_t reps : num_reps) {
    for (uint32_t simhash : num_simhash) {
      for (uint32_t p_dim : projd) {
        params.push_back({reps, simhash, p_dim, reps * p_dim * (1 << simhash)});
      }
    }
  }
#else
  params.push_back({10, 3, 8, 10 * 8 * (1 << 3)});    // 640
  params.push_back({20, 3, 8, 20 * 8 * (1 << 3)});    // 1280
  params.push_back({15, 4, 8, 15 * 8 * (1 << 4)});    // 1920
  params.push_back({20, 4, 8, 20 * 8 * (1 << 4)});    // 2560
  params.push_back({20, 5, 8, 20 * 8 * (1 << 5)});    // 5120
  params.push_back({20, 5, 16, 20 * 16 * (1 << 5)});  // 10240
#endif
  const std::vector<uint32_t> num_cands = {1, 5, 10, 20};
  const std::vector<uint32_t> L_values = {64, 128, 256, 512, 1024};
  const uint32_t k = P.getOptionIntValue("-k", 10);

  using PC = PointCloudSet<ChPoint>;
  auto points = PC(inFile);
  auto queries = PC(qFile);
  auto gt = ReadGT(gtFile, queries.size());

  system(("mkdir -p " + index_dir).c_str());
  system(("mkdir -p " + results_dir).c_str());

  for (auto [reps, simhash, p_dim, d_fde] : params) {
    std::cout << "Building index for reps=" << reps << ", simhash=" << simhash
              << ", projd=" << p_dim << ", d_fde=" << d_fde << std::endl;
    IndexParams index_params =
        IndexParams::muvera(reps, simhash, 1, p_dim, false, 0, true, 200, 600, 1.2, false, 0);
    IndexMUVERA<metric> index(points.get_dims(), index_params);
    index.build(points);

    std::string index_path = index_dir + "/muvera_reps" + std::to_string(reps) + "_simhash" +
                             std::to_string(simhash) + "_projd" + std::to_string(p_dim) + ".index";
    index.save(index_path);
    std::cout << "Index saved to " << index_path << std::endl;

    std::string results_path = results_dir + "/results_reps" + std::to_string(reps) + "_simhash" +
                               std::to_string(simhash) + "_projd" + std::to_string(p_dim) + ".csv";
    std::ofstream results_file(results_path);
    results_file << "num_cands,L,recall 1@" << k << ",recall " << k << "@" << k
                 << ",QPS,QPS_par,Avg_cmps" << std::endl;
    std::cout << "Writing results to " << results_path << std::endl;

    for (uint32_t cands : num_cands) {
      for (uint32_t l : L_values) {
        if (l >= cands * k) {
          SearchParams search_params =
              SearchParams::muvera(k, l, 1.35, points.size(), 200, true, k * cands);
          Stats result = compute_stats(index, points, queries, gt, search_params);
          results_file << cands << "," << l << "," << result.recall_1_k << "," << result.recall_k_k
                       << "," << result.QPS_seq << "," << result.QPS_par << "," << result.avg_cmps
                       << std::endl;
        }
      }
    }
    results_file.close();
  }
  std::cout << "Grid search complete." << std::endl;
}

int main(int argc, char* argv[]) {
  mvsic::commandLine P(argc, argv,
                       "[-i <inFile>] [-q <qFile>] [-gt <gtFile>] [-k <k>] [-index_dir <dir>] "
                       "[-results_dir <dir>] [-dist_func <IP|L2>]");

  std::string df = P.getOptionValue("-dist_func", "L2");

  if (df == "L2") {
    using ChPoint = ChamferL2_Point;
    grid_search_bench<ChPoint, true>(P);
  } else if (df == "IP") {
    using ChPoint = ChamferIP_Point;
    grid_search_bench<ChPoint, false>(P);
  }
  return 0;
}
