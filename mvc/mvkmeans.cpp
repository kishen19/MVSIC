#include "algorithms/bench/parse_command_line.h"
#include "algorithms/utils/euclidian_point.h"
#include "algorithms/utils/mips_point.h"
#include "algorithms/utils/point_range.h"
#include "utils/pointcloud.h"
#include "utils/chamferpoint.h"
#include "mvkmeans.h"

int main(int argc, char* argv[]) {
  commandLine P(argc, argv,
    "[-i <inFile>] [-k <num_centers>] [-s <num_embeddings>]"
    "[-data_type <tp>] [-dist_func <dist_func>]"
    "[-seed <algorithm>] [-iters <num_iters>]"
    // "[-kmeans_seed <algorithm>] [-kmeans_dist <algorithm>]"
  );

  std::string tp = P.getOptionValue("-data_type", "float");
  std::string df = P.getOptionValue("-dist_func", "Euclidian");

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
  // auto kmeans_seeding = P.getOptionValue("-kmeans_seed", "PrefixDoubling");
  // auto kmeans_dist_algo = P.getOptionValue("-kmeans_dist", "Pairwise");
  // auto kmeans_iters = P.getOptionLongValue("-kmeans_iters", 20);


  if (tp == "float") {
    if (df == "Euclidian") {
      using ChPoint = Chamfer_Euclidian_Point<float>;
      using Point = Euclidian_Point<float>;
      using Range = PointRange<float, Point>;
      auto points = PointCloud<ChPoint, Range>(P.getOptionValue("-i"));
      mvkmeans<Range>(points, k, s, iters, seeding);
      // ,kmeans_dist_algo, kmeans_seeding, kmeans_iters);
    } else if (df == "Mips") {
      using ChPoint = Chamfer_Mips_Point<float>;
      using Point = Mips_Point<float>;
      using Range = PointRange<float, Point>;
      auto points = PointCloud<ChPoint, Range>(P.getOptionValue("-i"));
      mvkmeans<Range>(points, k, s, iters, seeding);
      // ,kmeans_dist_algo, kmeans_seeding, kmeans_iters);
    }
  } else if (tp == "uint8") {
    if (df == "Euclidian") {
      using ChPoint = Chamfer_Euclidian_Point<uint8_t>;
      using Point = Euclidian_Point<uint8_t>;
      using Range = PointRange<uint8_t, Point>;
      auto points = PointCloud<ChPoint, Range>(P.getOptionValue("-i"));
      mvkmeans<Range>(points, k, s, iters, seeding);
      // ,kmeans_dist_algo, kmeans_seeding, kmeans_iters);
    } else if (df == "Mips") {
      using ChPoint = Chamfer_Mips_Point<uint8_t>;
      using Point = Mips_Point<uint8_t>;
      using Range = PointRange<uint8_t, Point>;
      auto points = PointCloud<ChPoint, Range>(P.getOptionValue("-i"));
      mvkmeans<Range>(points, k, s, iters, seeding);
      // ,kmeans_dist_algo, kmeans_seeding, kmeans_iters);
    }
  } else if (tp == "int8") {
    if (df == "Euclidian") {
      using ChPoint = Chamfer_Euclidian_Point<int8_t>;
      using Point = Euclidian_Point<int8_t>;
      using Range = PointRange<int8_t, Point>;
      auto points = PointCloud<ChPoint, Range>(P.getOptionValue("-i"));
      mvkmeans<Range>(points, k, s, iters, seeding);
      // ,kmeans_dist_algo, kmeans_seeding, kmeans_iters);
    } else if (df == "Mips") {
      using ChPoint = Chamfer_Mips_Point<int8_t>;
      using Point = Mips_Point<int8_t>;
      using Range = PointRange<int8_t, Point>;
      auto points = PointCloud<ChPoint, Range>(P.getOptionValue("-i"));
      mvkmeans<Range>(points, k, s, iters, seeding);
      // ,kmeans_dist_algo, kmeans_seeding, kmeans_iters);
    }
  }
  return 0;
}
