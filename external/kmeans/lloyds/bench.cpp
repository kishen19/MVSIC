#include <iostream>
#include <string>
#include <vector>
#include <fstream>

#include "parlay/primitives.h"
#include "parlay/internal/get_time.h"

#include "algorithms/bench/parse_command_line.h"
#include "algorithms/utils/euclidian_point.h"
#include "algorithms/utils/mips_point.h"
#include "algorithms/utils/point_range.h"

#include "kmeans.h"

template<typename PointTy>
void run_kmeans_benchmark(commandLine& P) {
  char* inFile = P.getOptionValue("-i");
  if (inFile == nullptr) {
    std::cerr << "Error: Input file not specified. Use -i <filename>" << std::endl;
    exit(1);
  }

  size_t k = P.getOptionLongValue("-k", 10);
  size_t iters = P.getOptionLongValue("-iters", 5);
  std::string seed_algo = P.getOptionValue("-seed_algo", "UniformlyRandom");
  std::string oracle = P.getOptionValue("-oracle", "Pairwise");
  bool verbose = P.getOption("-v");

  // Use PointRange constructor to load data directly from file
  PointRange<float, PointTy> data_range(inFile);

  std::cout << "Running k-means with:" << std::endl;
  std::cout << "  Input file: " << inFile << std::endl;
  std::cout << "  Number of points: " << data_range.size() << std::endl;
  std::cout << "  Dimensions: " << data_range.get_dims() << std::endl;
  std::cout << "  K (clusters): " << k << std::endl;
  std::cout << "  Iterations: " << iters << std::endl;
  std::cout << "  Seeding Algorithm: " << seed_algo << std::endl;

  parlay::internal::timer t;
  t.start();
  auto centers = kmeans<float, PointTy>(data_range, k, seed_algo, oracle, iters, verbose);
  t.stop();

  std::cout << "K-means completed in " << t.total_time() << " seconds." << std::endl;
  std::cout << "Final Sum of Squared Cost: " << SumOfSquaredCost<PointTy>(data_range, centers)
            << std::endl;
}

int main(int argc, char* argv[]) {
  commandLine P(argc, argv,
                "[-i <inFile>] [-k <num_centers>] [-iters <num_iterations>] [-dist_func <L2|IP>] "
                "[-seed_algo <algorithm>] [-v]");

  std::string dist_func = P.getOptionValue("-dist_func", "L2");

  if (dist_func == "L2") {
    run_kmeans_benchmark<Euclidian_Point<float>>(P);
  } else if (dist_func == "IP") {
    run_kmeans_benchmark<Mips_Point<float>>(P);
  } else {
    std::cerr << "Error: Invalid distance function. Use L2 or IP." << std::endl;
    return 1;
  }

  return 0;
}
