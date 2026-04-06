#include <fstream>
#include <iostream>
#include <string>

#include "parlay/primitives.h"

#include "mvsic/core/types/chamfer_ip_point.h"
#include "mvsic/core/types/chamfer_l2_point.h"
#include "mvsic/core/types/point_cloud_set.h"
#include "mvsic/core/utils/parse_command_line.h"
#include "mvsic/core/stats.h"
#include "mvsic/mvivf/mvivf.h"

using namespace mvsic;

template<typename ChPoint, bool metric>
void run_benchmark(mvsic::commandLine& P) {
  Eigen::setNbThreads(1);
  using PC = PointCloudSet<ChPoint>;

  std::string inFile = P.getOptionValue("-i", "");
  std::string qFile = P.getOptionValue("-q", "");
  std::string indexFile = P.getOptionValue("-index", "");
  std::string gtFile = P.getOptionValue("-gt", "");
  bool is_mmap = P.getOption("-mm");
  // -mode: "both" (default), "old", "new" — lets you isolate each for perf stat
  std::string mode = P.getOptionValue("-mode", "both");

  size_t k = P.getOptionLongValue("-k", 10);
  size_t nprobes = P.getOptionLongValue("-nprobes", 100);
  size_t num_rerank = P.getOptionLongValue("-num_rerank", k);

  if (inFile.empty() || qFile.empty() || indexFile.empty()) {
    std::cerr << "Usage: -i <inFile> -q <qFile> -index <indexFile> [-gt <gtFile>] [-mode old|new|both]"
              << std::endl;
    exit(1);
  }

  std::cout << "Loading dataset..." << std::endl;
  auto points = PC(inFile.c_str(), is_mmap);

  std::cout << "Loading queries..." << std::endl;
  auto queries = PC(qFile.c_str());

  parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>> gt;
  bool has_gt = !gtFile.empty();
  if (has_gt) {
    std::cout << "Loading ground truth..." << std::endl;
    gt = ReadGT(gtFile, queries.size());
    std::cout << "GT loaded: " << gt.size() << " queries, " << gt[0].size()
              << " neighbors each" << std::endl;
  }

  IndexParams index_params = IndexParams::mvivf();
  index_params.quantize_centers = true;
  index_params.pq.method = IndexParams::QuantizerType::FastScan;
  index_params.pq.block_size = 8;

  IndexMVIVF<metric> index(points.get_dims(), index_params);

  std::cout << "Loading index..." << std::endl;
  index.load(indexFile, points);

  SearchParams search_params = SearchParams::mvivf(k, nprobes, num_rerank);

  bool run_old = (mode == "both" || mode == "old");
  bool run_new = (mode == "both" || mode == "new");

  parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>> pred1, pred2;
  size_t cmps1 = 0, cmps2 = 0;
  double time1 = 0, time2 = 0;

  if (run_old) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Running search_all (Original)..." << std::endl;
    parlay::internal::timer t1;
    t1.start();
    auto [p1, c1] = index.search_all(queries, points, search_params);
    t1.stop();
    pred1 = std::move(p1);
    cmps1 = c1;
    time1 = t1.total_time();
    std::cout << "search_all time: " << time1 << " seconds. Dist cmps: " << cmps1 << std::endl;
  }

  if (run_new) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Running search_all_new..." << std::endl;
    parlay::internal::timer t2;
    t2.start();
    auto [p2, c2] = index.search_all_new(queries, points, search_params);
    t2.stop();
    pred2 = std::move(p2);
    cmps2 = c2;
    time2 = t2.total_time();
    std::cout << "search_all_new time: " << time2 << " seconds. Dist cmps: " << cmps2 << std::endl;
  }

  if (run_old && run_new) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Speedup (old / new): " << time1 / time2 << "x" << std::endl;
    std::cout << "========================================" << std::endl;

    if (has_gt) {
      auto [r1_old, rk_old] = compute_scores(pred1, gt, k);
      auto [r1_new, rk_new] = compute_scores(pred2, gt, k);
      std::cout << "\nRecall vs Ground Truth:" << std::endl;
      std::cout << "  search_all:     recall@1 = " << r1_old << ", recall@" << k << " = " << rk_old
                << std::endl;
      std::cout << "  search_all_new: recall@1 = " << r1_new << ", recall@" << k << " = " << rk_new
                << std::endl;
    }

    double agreement = compute_recall(pred2, pred1, k, k);
    std::cout << "\nAgreement (new vs old): " << agreement << std::endl;
  } else if (has_gt) {
    auto& pred = run_old ? pred1 : pred2;
    auto [r1, rk] = compute_scores(pred, gt, k);
    std::cout << "\nRecall vs Ground Truth: recall@1 = " << r1
              << ", recall@" << k << " = " << rk << std::endl;
  }
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
