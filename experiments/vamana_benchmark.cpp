#include <iostream>
#include "src/utils/chamfer_ip_point.h"
#include "src/utils/chamfer_l2_point.h"
#include "src/utils/parse_command_line.h"
#include "src/utils/point_cloud_set.h"
#include "src/utils/stats.h"
#include "src/vamana/vamana.h"

using namespace mvivf;

template<typename ChPoint, bool metric>
void bench(mvivf::commandLine& P) {
  using PC = PointCloudSet<ChPoint>;

  // Files
  char* inFile = P.getOptionValue("-i");                   // Base Points
  std::string qFile = P.getOptionValue("-q", "");          // Query Points
  std::string gtFile = P.getOptionValue("-gt", "");        // Ground Truth
  std::string outFile = P.getOptionValue("-o", "");        // File to store index
  std::string indexFile = P.getOptionValue("-index", "");  // File containing index
  char* resFile = P.getOptionValue("-r");                  // CSV file to store stats
  // Params
  size_t R = P.getOptionLongValue("-R", 32);
  size_t L_build = P.getOptionLongValue("-L", 2 * R);
  double alpha = P.getOptionDoubleValue("-alpha", 1.2);
  bool two_pass = P.getOption("-tp");
  double cut = P.getOptionDoubleValue("-cut", 1.35);
  size_t k = P.getOptionLongValue("-k", 10);
  size_t beamsizel = P.getOptionLongValue("-Ll", 16);
  size_t beamsizer = P.getOptionLongValue("-Lr", 128);
  size_t mp = P.getOptionLongValue("-Lmp", 2);
  int rounds = P.getOptionLongValue("-rounds", 1);
  bool verbose = P.getOption("-v");
  bool is_gold = P.getOption("-gold");

  auto points = PC(inFile);
  IndexVamanaParams index_params(R, L_build, alpha, two_pass, verbose);
  IndexVamana<metric> index(points.get_dims(), index_params);
  if (indexFile != "") {  // Stats Benchmark
    index.load(indexFile, points);
    std::cout << "Index loaded" << std::endl;
    auto queries = PC(P.getOptionValue("-q"));
    parlay::sequence<parlay::sequence<std::pair<float, uint32_t>>> gt;
    if (is_gold) {
      gt = ReadGoldGT(gtFile, queries.size());
    } else {
      gt = ReadGT(gtFile, queries.size());
    }
    parlay::sequence<SearchParams> search_params_list;
    size_t L = beamsizel;
    while (L <= beamsizer) {
      search_params_list.push_back(SearchParams(k, L, cut, points.size(), R));
      L *= mp;
    }
    // Compute Stats
    std::cout << "Computing stats..." << std::endl;
    if (is_gold) {
      search_all_gold(index, points, queries, gt, resFile, search_params_list);
    } else {
      search_all(index, points, queries, gt, resFile, search_params_list);
    }
    std::cout << "Stats computed and saved to " << resFile << std::endl;
  } else {  // Indexing Benchmark
    std::cout << "Starting Indexing Benchmark..." << std::endl;
    parlay::internal::timer t;
    double index_time = 0.0;
    for (long it = 0; it <= rounds; it++) {
      t.start();
      IndexVamana<metric> index(points.get_dims(), index_params);
      index.build(points);
      t.stop();
      if (it != 0) {
        index_time += t.total_time();
      } else {
        if (outFile != "") {
          std::cout << "Saving index to " << outFile << std::endl;
          index.save(P.getOptionValue("-o"));
          std::cout << "Index saved." << std::endl;
        }
        std::cout << "Warm up Time: " << t.total_time() << " seconds." << std::endl;
      }
      t.reset();
    }
    std::cout << "Average Indexing Time: " << index_time / rounds << " seconds." << std::endl;
  }
}

int main(int argc, char* argv[]) {
  mvivf::commandLine P(argc, argv,
                       "[-i <inFile>] [-k <num_centers>] [-s <num_embeddings>]"
                       "[-data_type <tp>] [-dist_func <dist_func>]"
                       "[-seed <algorithm>] [-iters <num_iters>]"
                       "[-kmeans_seed <algorithm>] [-kmeans_dist <algorithm>]");

  std::string df = P.getOptionValue("-dist_func", "IP");

  if (df == "L2") {
    using ChPoint = ChamferL2_Point;
    bench<ChPoint, true>(P);
  } else if (df == "IP") {
    using ChPoint = ChamferIP_Point;
    bench<ChPoint, false>(P);
  }
  return 0;
}
