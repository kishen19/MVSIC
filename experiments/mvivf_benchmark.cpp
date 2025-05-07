#include "algorithms/bench/parse_command_line.h"
#include "algorithms/utils/euclidian_point.h"
#include "algorithms/utils/mips_point.h"
#include "algorithms/utils/point_range.h"

#include "mvc/utils/pointcloud.h"
#include "mvc/utils/chamferpoint.h"
#include "mvivf/utils/stats.h"
#include "mvivf/index.h"


template <typename ChPoint, typename Range>
void bench(commandLine& P) {
  using T = typename ChPoint::distance_type;
  using PC = PointCloud<ChPoint, Range>;

  char* inFile = P.getOptionValue("-i"); // Base Points
  char* resFile = P.getOptionValue("-r"); // CSV file to store stats
  std::string qFile = P.getOptionValue("-q", ""); // Query Points
  std::string gtFile = P.getOptionValue("-gt", ""); // Ground Truth
  std::string outFile = P.getOptionValue("-o", ""); // File to store index 
  std::string indexFile = P.getOptionValue("-index", ""); // File containing index
  long maxsize = P.getOptionLongValue("-maxsize", 100);
  long k = P.getOptionLongValue("-k", 10);
  long s = P.getOptionLongValue("-s", 0); // Deprecate after clustering evals
  auto seeding = P.getOptionValue("-seed", "Random"); // Deprecate after clustering evals
  auto iters = P.getOptionLongValue("-iters", 5);
  long rounds = P.getOptionLongValue("-rounds", 1);

  auto points = PC(inFile);
  if (indexFile != ""){ // Stats Benchmark
    mvivf::Index<T, PC> index;
    std::cout << "Loading index from " << indexFile << std::endl;
    index.Load(indexFile, points);
    std::cout << "Index loaded" << std::endl;
    
    auto queries = PC(P.getOptionValue("-q"));
    auto gt = ReadGT(gtFile, queries.size());
    // Compute Stats
    std::cout << "Computing stats..." << std::endl;
    search_and_parse(index, points, queries, gt, resFile, k);
    std::cout << "Stats computed and saved to " << resFile << std::endl;
  } else { // Indexing Benchmark
    std::cout << "Starting Indexing Benchmark..." << std::endl;
    parlay::internal::timer t;
    double index_time = 0.0;
    for (long it=0; it<=rounds; it++){
      t.start();
      mvivf::Index<T, PC> index;
      index.Build(points, maxsize, s, iters, seeding);
      t.stop();
      if (it!=0){
        index_time += t.total_time();
      } else{
        if (outFile != ""){
          std::cout << "Saving index to " << outFile << std::endl;
          index.Save(P.getOptionValue("-o"));
          std::cout << "Index saved." << std::endl;
        }
      }
      t.reset();
    }
    std::cout << "Average Indexing Time: " << index_time/rounds << " seconds." << std::endl;
  }
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

  // auto seeding = P.getOptionValue("-seed", "Random");
  // auto iters = P.getOptionLongValue("-iters", 5);
  // auto kmeans_seeding = P.getOptionValue("-kmeans_seed", "PrefixDoubling");
  // auto kmeans_dist_algo = P.getOptionValue("-kmeans_dist", "ANNS");
  // auto kmeans_iters = P.getOptionLongValue("-kmeans_iters", 20);

  if (tp == "float") {
    if (df == "Euclidian"){
      using ChPoint = Chamfer_Euclidian_Point<float>;
      using Point = Euclidian_Point<float>;
      using Range = PointRange<float, Point>;
      bench<ChPoint, Range>(P);
    } else if (df == "Mips") {
      using ChPoint = Chamfer_Mips_Point<float>;
      using Point = Mips_Point<float>;
      using Range = PointRange<float, Point>;
      bench<ChPoint, Range>(P);
    }
  } else if (tp == "uint8") {
    if (df == "Euclidian"){
      using ChPoint = Chamfer_Euclidian_Point<uint8_t>;
      using Point = Euclidian_Point<uint8_t>;
      using Range = PointRange<uint8_t, Point>;
      bench<ChPoint, Range>(P);
    } else if (df == "Mips") {
      using ChPoint = Chamfer_Mips_Point<uint8_t>;
      using Point = Mips_Point<uint8_t>;
      using Range = PointRange<uint8_t, Point>;
      bench<ChPoint, Range>(P);
    }
  } else if (tp == "int8") {
    if (df == "Euclidian"){
      using ChPoint = Chamfer_Euclidian_Point<int8_t>;
      using Point = Euclidian_Point<int8_t>;
      using Range = PointRange<int8_t, Point>;
      bench<ChPoint, Range>(P);
    } else if (df == "Mips") {
      using ChPoint = Chamfer_Mips_Point<int8_t>;
      using Point = Mips_Point<int8_t>;
      using Range = PointRange<int8_t, Point>;
      bench<ChPoint, Range>(P);
    }
  }
  return 0;
}
