#include "mvsic/core/utils/parse_command_line.h"
#include "mvsic/core/mvclustering/mvclustering.h"

int main(int argc, char* argv[]) {
  mvsic::commandLine P(argc, argv,
                       "[-i <inFile>] [-k <num_centers>] [-s <num_embeddings>]"
                       "[-data_type <tp>] [-dist_func <dist_func>]"
                       "[-seed <algorithm>] [-iters <num_iters>]"
                       "[-lb <bool>]");
  std::string df = P.getOptionValue("-dist_func", "IP");
  size_t k = P.getOptionLongValue("-k", 10);
  size_t s = P.getOptionLongValue("-s", 0);
  // auto seeding = P.getOptionValue("-seed", "Random");
  size_t iters = P.getOptionLongValue("-iters", 5);
  // bool lb = P.getOption("-lb");
  int verbose = P.getOptionIntValue("-v", 0);

  // MVClusteringParams params(iters, seeding, os_rate, verbose, lb);
  if (df == "L2") {
    auto points = mvsic::PointCloudSet<mvsic::ChamferL2_Point>(P.getOptionValue("-i"));
    mvsic::MVClustering<true> clus(points.get_dims(), k, s, iters, 20, verbose);
    clus.train(points);
  } else if (df == "IP") {
    auto points = mvsic::PointCloudSet<mvsic::ChamferIP_Point>(P.getOptionValue("-i"));
    mvsic::MVClustering<false> clus(points.get_dims(), k, s, iters, 20, verbose);
    clus.train(points);
  }
  return 0;
}
