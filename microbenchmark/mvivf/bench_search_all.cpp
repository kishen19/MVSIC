#include <fstream>
#include <iostream>
#include <string>

#include "parlay/primitives.h"

#include "mvsic/core/types/chamfer_ip_point.h"
#include "mvsic/core/types/chamfer_l2_point.h"
#include "mvsic/core/types/point_cloud_set.h"
#include "mvsic/core/utils/parse_command_line.h"
#include "mvsic/mvivf/mvivf.h"

using namespace mvsic;

template<typename ChPoint, bool metric>
void run_benchmark(mvsic::commandLine& P) {
  Eigen::setNbThreads(1);
  using PC = PointCloudSet<ChPoint>;

  std::string inFile = P.getOptionValue("-i", "");
  std::string qFile = P.getOptionValue("-q", "");
  std::string indexFile = P.getOptionValue("-index", "");
  bool is_mmap = P.getOption("-mm");

  size_t k = P.getOptionLongValue("-k", 10);
  size_t nprobes = P.getOptionLongValue("-nprobes", 100);
  size_t num_rerank = P.getOptionLongValue("-num_rerank", k);

  if (inFile.empty() || qFile.empty() || indexFile.empty()) {
    std::cerr << "Usage: -i <inFile> -q <qFile> -index <indexFile>" << std::endl;
    exit(1);
  }

  std::cout << "Loading dataset..." << std::endl;
  auto points = PC(inFile.c_str(), is_mmap);

  std::cout << "Loading queries..." << std::endl;
  auto queries = PC(qFile.c_str());

  IndexParams index_params = IndexParams::mvivf();
  index_params.quantize_centers = true;
  index_params.pq.method = IndexParams::QuantizerType::FastScan;
  index_params.pq.block_size = 8;

  IndexMVIVF<metric> index(points.get_dims(), index_params);

  std::cout << "Loading index..." << std::endl;
  index.load(indexFile, points);

  SearchParams search_params = SearchParams::mvivf(k, nprobes, num_rerank);

  std::cout << "\n========================================" << std::endl;
  std::cout << "Running search_all (Original)..." << std::endl;
  parlay::internal::timer t1;
  t1.start();
  auto [pred1, cmps1] = index.search_all(queries, points, search_params);
  t1.stop();
  double time1 = t1.total_time();
  std::cout << "search_all time: " << time1 << " seconds. Dist cmps: " << cmps1 << std::endl;

  std::cout << "\n========================================" << std::endl;
  std::cout << "Running search_all_new..." << std::endl;
  parlay::internal::timer t2;
  t2.start();
  auto [pred2, cmps2] = index.search_all_new(queries, points, search_params);
  t2.stop();
  double time2 = t2.total_time();
  std::cout << "search_all_new time: " << time2 << " seconds. Dist cmps: " << cmps2 << std::endl;

  std::cout << "\n========================================" << std::endl;
  std::cout << "Speedup (old / new): " << time1 / time2 << "x" << std::endl;
  std::cout << "========================================" << std::endl;
}

int main(int argc, char* argv[]) {
  mvsic::commandLine P(argc, argv,
                       "[-i <inFile>] [-q <qFile>] [-index <indexFile>] [-dist_func <dist_func>]");
  std::string df = P.getOptionValue("-dist_func", "IP");

  if (df == "L2") {
    using ChPoint = ChamferL2_Point;
    run_benchmark<ChPoint, true>(P);
  } else if (df == "IP") {
    using ChPoint = ChamferIP_Point;
    run_benchmark<ChPoint, false>(P);
  } else {
    std::cerr << "Unknown distance function: " << df << std::endl;
    return 1;
  }
  return 0;
}
