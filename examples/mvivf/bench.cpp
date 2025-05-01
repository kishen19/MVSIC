#include "algorithms/bench/parse_command_line.h"
#include "algorithms/utils/euclidian_point.h"
#include "algorithms/utils/mips_point.h"
#include "algorithms/utils/point_range.h"

#include "mvc/utils/pointcloud.h"
#include "mvc/utils/chamferpoint.h"

#include "index.h"

template <typename ChPoint, typename Range>
void bench(const char *inFile, size_t maxsize) {
  using T = typename ChPoint::distance_type;
  using PC = PointCloud<ChPoint, Range>;
  auto points = PC(inFile);
  auto index = mvivf::Index<T, PC>(points, maxsize);
  std::cout << "Index built with " << points.size() << " points." << std::endl;
  for(size_t i=0; i<10; i++) {
    std::cout << "Running query " << i << std::endl;
    auto query = points[i];
    auto k = 10;
    auto nprobes = 1;
    auto results = index.Search(query, k, nprobes);
    std::cout << "Query: " << i << ", Results: " << results.size() << std::endl;
    for(size_t j=0; j<results.size(); j++) {
      auto res = results[j];
      std::cout << "(" << res.first << ", " << res.second << ") ";
    }
    std::cout << std::endl;
  }
  std::cout << "Benchmark completed." << std::endl;
}

int main(int argc, char* argv[]) {
  commandLine P(argc, argv,
                "[-i <inFile>] [-k <num_centers>] [-s <num_embeddings>]"
                "[-data_type <tp>] [-dist_func <dist_func>]" 
                "[-seed <algorithm>] [-iters <num_iters>]" 
                "[-kmeans_seed <algorithm>] [-kmeans_dist <algorithm>]");

  std::string tp = P.getOptionValue("-data_type", "float");
  std::string df = P.getOptionValue("-dist_func", "Euclidian");

  if ((tp != "uint8") && (tp != "int8") && (tp != "float")) {
    std::cout << "Error: vector type not specified correctly, specify int8, "
                 "uint8, or float"
              << std::endl;
    abort();
  }

  auto inFile = P.getOptionValue("-i");
  auto maxsize = P.getOptionLongValue("-maxsize", 100);
  // auto seeding = P.getOptionValue("-seed", "Random");
  // auto iters = P.getOptionLongValue("-iters", 5);
  // auto kmeans_seeding = P.getOptionValue("-kmeans_seed", "PrefixDoubling");
  // auto kmeans_dist_algo = P.getOptionValue("-kmeans_dist", "ANNS");
  // auto kmeans_iters = P.getOptionLongValue("-kmeans_iters", 20);

  if (tp == "float") {
    if (df == "Euclidian"){
      using ChPoint = ChamferPoint<float>;
      using Point = Euclidian_Point<float>;
      using Range = PointRange<float, Point>;
      bench<ChPoint, Range>(inFile, maxsize);
    } else if (df == "Mips") {
      using ChPoint = ChamferPoint<float>;
      using Point = Mips_Point<float>;
      using Range = PointRange<float, Point>;
      bench<ChPoint, Range>(inFile, maxsize);
    }
  } else if (tp == "uint8") {
    if (df == "Euclidian"){
      using ChPoint = ChamferPoint<uint8_t>;
      using Point = Euclidian_Point<uint8_t>;
      using Range = PointRange<uint8_t, Point>;
      bench<ChPoint, Range>(inFile, maxsize);
    } else if (df == "Mips") {
      using ChPoint = ChamferPoint<uint8_t>;
      using Point = Mips_Point<uint8_t>;
      using Range = PointRange<uint8_t, Point>;
      bench<ChPoint, Range>(inFile, maxsize);
    }
  } else if (tp == "int8") {
    if (df == "Euclidian"){
      using ChPoint = ChamferPoint<int8_t>;
      using Point = Euclidian_Point<int8_t>;
      using Range = PointRange<int8_t, Point>;
      bench<ChPoint, Range>(inFile, maxsize);
    } else if (df == "Mips") {
      using ChPoint = ChamferPoint<int8_t>;
      using Point = Mips_Point<int8_t>;
      using Range = PointRange<int8_t, Point>;
      bench<ChPoint, Range>(inFile, maxsize);
    }
  }
  return 0;
}
