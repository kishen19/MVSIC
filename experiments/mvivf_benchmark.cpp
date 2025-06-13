#include <iostream>
#include "src/utils/chamfer_ip_point.h"
#include "src/utils/chamfer_l2_point.h"
#include "src/utils/parse_command_line.h"
#include "src/utils/point_cloud_set.h"
#include "src/utils/stats.h"
#include "src/mvivf/mvivf.h"

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
  size_t minsize = P.getOptionLongValue("-minsize", 100);
  size_t maxsize = P.getOptionLongValue("-maxsize", 500);
  size_t nprobesl = P.getOptionLongValue("-npl", 1);
  size_t nprobesr = P.getOptionLongValue("-npr", 8);
  size_t nprobesmp = P.getOptionLongValue("-npmp", 2);
  size_t nprobesad = P.getOptionLongValue("-npad", 0);
  size_t k = P.getOptionLongValue("-k", 10);
  size_t s = P.getOptionLongValue("-s", 0);
  size_t os_rate = P.getOptionLongValue("-osr", 20);
  auto seeding = P.getOptionValue("-seed", "Random");
  auto iters = P.getOptionLongValue("-iters", 5);
  int rounds = P.getOptionLongValue("-rounds", 1);
  bool verbose = P.getOption("-v");
  bool is_gold = P.getOption("-gold");

  auto points = PC(inFile);
  IndexMVIVFParams index_params(minsize, maxsize, s, iters, seeding, os_rate, verbose);
  IndexMVIVF<metric> index(points.get_dims(), index_params);
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
    size_t nprobes = nprobesl;
    while (nprobes <= nprobesr) {
      search_params_list.push_back(SearchParams(k, nprobes, 0));
      nprobes = nprobesmp * nprobes + nprobesad;
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
      IndexMVIVF<metric> index(points.get_dims(), index_params);
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
