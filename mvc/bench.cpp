#include "utils/parse_command_line.h"
#include "utils/euclidian_point.h"
#include "utils/mips_point.h"
#include "utils/point_range.h"
#include "utils/chamfer_l2_point.h"
#include "utils/chamfer_ip_point.h"
#include "utils/point_cloud_set.h"
#include "mvkmeans.h"

int main(int argc, char* argv[]) {
  commandLine P(argc, argv,
    "[-i <inFile>] [-k <num_centers>] [-s <num_embeddings>]"
    "[-data_type <tp>] [-dist_func <dist_func>]"
    "[-seed <algorithm>] [-iters <num_iters>]"
    "[-lb <bool>]"
  );

  std::string tp = P.getOptionValue("-data_type", "float");
  std::string df = P.getOptionValue("-dist_func", "IP");

  if ((tp != "uint8") && (tp != "int8") && (tp != "float")) {
    std::cout << "Error: vector type not specified correctly, specify int8, "
      "uint8, or float"
      << std::endl;
    abort();
  }

  auto k = P.getOptionLongValue("-k", 10);
  auto s = P.getOptionLongValue("-s", 0);
  auto seeding = P.getOptionValue("-seed", "Random");
  auto iters = P.getOptionLongValue("-iters", 5);
  bool lb = P.getOption("-lb");

  mvivf::MVClusteringParams params;
  params.iters = iters;
  params.seeding = seeding;
  params.comp_lb = lb;
  params.verbose = true;

  if (df == "L2") {
    // using Point = Euclidian_Point<float>;
    // using Range = PointRange<float, Point>;
    auto points = PointCloudSet<ChamferL2_Point>(P.getOptionValue("-i"));

    mvivf::MVClustering<mvivf::L2> clus(points.get_dims(), k, s, params);
    clus.train(points);
  } else if (df == "IP") {
    // using Point = Euclidian_Point<float>;
    // using Range = PointRange<float, Point>;
    auto points = PointCloudSet<ChamferIP_Point>(P.getOptionValue("-i"));

    mvivf::MVClustering<mvivf::IP> clus(points.get_dims(), k, s, params);
    clus.train(points);
  }
  return 0;
}
