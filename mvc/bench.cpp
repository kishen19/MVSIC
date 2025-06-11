#include "utils/parse_command_line.h"
// #include "utils/euclidian_point.h"
// #include "utils/mips_point.h"
// #include "utils/point_range.h"
#include "mvkmeans.h"
#include "utils/chamfer_ip_point.h"
#include "utils/chamfer_l2_point.h"
#include "utils/point_cloud_set.h"

int main(int argc, char *argv[]) {
  mvivf::commandLine P(argc, argv,
                       "[-i <inFile>] [-k <num_centers>] [-s <num_embeddings>]"
                       "[-data_type <tp>] [-dist_func <dist_func>]"
                       "[-seed <algorithm>] [-iters <num_iters>]"
                       "[-lb <bool>]");
  std::string df = P.getOptionValue("-dist_func", "IP");
  size_t k = P.getOptionLongValue("-k", 10);
  size_t s = P.getOptionLongValue("-s", 0);
  auto seeding = P.getOptionValue("-seed", "Random");
  size_t os_rate = P.getOptionLongValue("-osr", 20);
  size_t iters = P.getOptionLongValue("-iters", 5);
  bool lb = P.getOption("-lb");
  bool verbose = P.getOption("-v");

  mvivf::MVClusteringParams params(iters, seeding, os_rate, lb, verbose);
  if (df == "L2") {
    auto points = PointCloudSet<ChamferL2_Point>(P.getOptionValue("-i"));
    mvivf::MVClustering<mvivf::L2> clus(points.get_dims(), k, s, params);
    clus.train(points);
  } else if (df == "IP") {
    auto points = PointCloudSet<ChamferIP_Point>(P.getOptionValue("-i"));
    mvivf::MVClustering<mvivf::IP> clus(points.get_dims(), k, s, params);
    clus.train(points);
  }
  return 0;
}
