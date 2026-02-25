#include "mvsic/core/utils/parse_command_line.h"
#include <iomanip>
#include "mvsic/core/mvclustering/mvclustering.h"
#include "evals.h"

parlay::sequence<uint32_t> ReadGT(const char* filename) {
  std::ifstream ifs(filename);

  parlay::sequence<uint32_t> values;
  std::string line;
  while (std::getline(ifs, line)) {
    std::istringstream iss(line);
    uint32_t value;
    if (!(iss >> value)) {
      throw std::runtime_error("Error reading value from file");
    }
    values.push_back(value);
  }
  ifs.close();
  return values;
}

int main(int argc, char* argv[]) {
  mvsic::commandLine P(argc, argv,
                       "[-i <inFile>] [-k <num_centers>] [-s <num_embeddings>]"
                       "[-data_type <tp>] [-dist_func <dist_func>]"
                       "[-seed <algorithm>] [-iters <num_iters>]"
                       "[-mpcc <max_point_clouds_per_cluster>]"
                       "[-use_weighted_inner_kmeans] [-v <verbose>] [-gt <gtFile>]");
  std::string df = P.getOptionValue("-dist_func", "IP");
  size_t k = P.getOptionLongValue("-k", 10);
  size_t s = P.getOptionLongValue("-s", 0);
  // auto seeding = P.getOptionValue("-seed", "Random");
  size_t iters = P.getOptionLongValue("-iters", 5);
  size_t max_point_clouds_per_cluster = P.getOptionLongValue("-mpcc", 0);
  bool use_weighted_inner_kmeans = P.getOption("-use_weighted_inner_kmeans");
  int verbose = P.getOptionIntValue("-v", 0);

  // MVClusteringParams params(iters, seeding, os_rate, verbose, lb);
  parlay::sequence<uint32_t> clustering;
  if (df == "L2") {
    auto points = mvsic::PointCloudSet<mvsic::ChamferL2_Point>(P.getOptionValue("-i"));
    mvsic::MVClustering<true> clus(points.get_dims(), k, s, iters, max_point_clouds_per_cluster, 20,
                                   verbose, "Random", 0, use_weighted_inner_kmeans);
    clus.train(points);
    clustering = clus.cluster_ids;
  } else if (df == "IP") {
    auto points = mvsic::PointCloudSet<mvsic::ChamferIP_Point>(P.getOptionValue("-i"));
    std::cout << points.size() << " point clouds read." << std::endl;
    std::cout << points.get_dims() << " dimensions per point cloud." << std::endl;
    std::cout << points.average_size() << " average number of points per point cloud." << std::endl;
    mvsic::MVClustering<false> clus(points.get_dims(), k, s, iters, max_point_clouds_per_cluster,
                                    20, verbose, "Random", 0, use_weighted_inner_kmeans);
    clus.train(points);
    clustering = clus.cluster_ids;
  }
  // Compute Stats
  if (P.getOption("-gt")) {
    auto gtFile = P.getOptionValue("-gt");
    auto gt = ReadGT(gtFile);
    double ari, nmi;
    std::tie(ari, nmi) = mvsic::computeMetrics(gt, clustering);

    std::cout << std::fixed << std::setprecision(4) << "ARI: " << ari << std::endl;
    std::cout << std::fixed << std::setprecision(4) << "NMI: " << nmi << std::endl;
  }
  return 0;
}
