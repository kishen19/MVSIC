#include "src/utils/chamfer_ip_point.h"
#include "src/utils/chamfer_l2_point.h"
#include "src/utils/ip_point.h"
#include "src/utils/l2_point.h"
#include "src/utils/parse_command_line.h"
#include "src/utils/point_cloud_set.h"
#include "src/utils/point_range.h"
#include "src/utils/stats.h"
#include "svheuristic.h"

using namespace mvivf;

template<typename Point, typename ChPoint, bool metric>
void bench(mvivf::commandLine& P) {
  using PC = PointCloudSet<ChPoint>;
  using Range = mvivf::PointRange<float, Point>;

  char* inFile = P.getOptionValue("-i");
  char* qFile = P.getOptionValue("-q");
  std::string QFile;
  if (qFile != nullptr) {
    QFile = P.getOptionValue("-q");
  } else {
    QFile = "";
  }
  std::string gtFile = P.getOptionValue("-gt", "");
  std::string outFile = P.getOptionValue("-o", "");
  std::string indexFile = P.getOptionValue("-index", "");

  size_t minsize = P.getOptionLongValue("-minsize", 100);
  size_t maxsize = P.getOptionLongValue("-maxsize", 500);
  size_t nprobes = P.getOptionLongValue("-nprobes", 1);
  size_t beamsize = P.getOptionLongValue("-beamsize", 0);
  size_t k = P.getOptionLongValue("-k", 10);
  size_t cands = P.getOptionLongValue("-cands", 10 * k);
  size_t os_rate = P.getOptionLongValue("-osr", 20);
  bool verbose = P.getOption("-v");

  auto points = PC(inFile);
  IndexSVHParams index_params(minsize, maxsize, os_rate, verbose);
  SearchParams search_params(k, nprobes, beamsize, cands);
  IndexSVH<metric> index(points.get_dims(), index_params);
  if (indexFile != "") {
    std::cout << "Loading index from " << indexFile << std::endl;
    index.load(indexFile, points);
    std::cout << "Index loaded" << std::endl;
  } else {
    std::cout << "Building index..." << std::endl;
    parlay::internal::timer it;
    it.start();
    index.build(points);
    it.stop();
    std::cout << "Index built in " << it.total_time() << " seconds." << std::endl;
  }
  if (outFile != "") {
    std::cout << "Saving index to " << outFile << std::endl;
    index.save(P.getOptionValue("-o"));
    std::cout << "Index saved." << std::endl;
  }

  if (QFile != "") {
    auto queries = PC(qFile);
    auto gt = ReadGT(gtFile, queries.size());
    double QPS_seq, QPS_par, avg_cmps, recall_1_k, recall_k_k;

    // Compute Stats:
    std::cout << "Computing stats..." << std::endl;
    Stats result = compute_stats(index, points, queries, gt, search_params);
    QPS_seq = result.QPS_seq;
    QPS_par = result.QPS_par;
    avg_cmps = result.avg_cmps;
    recall_1_k = result.recall_1_k;
    recall_k_k = result.recall_k_k;
    std::cout << "Number of Queries: " << queries.size() << std::endl
              << "QPS_seq: " << QPS_seq << std::endl
              << "QPS_par: " << QPS_par << std::endl
              << "Average cmps: " << avg_cmps << std::endl
              << "Average recall 1 @ " << k << ": " << recall_1_k << std::endl
              << "Average recall " << k << " @ " << k << ": " << recall_k_k << std::endl;
  }
}

int main(int argc, char* argv[]) {
  mvivf::commandLine P(argc, argv,
                       "[-i <inFile>] [-k <num_centers>] [-s <num_embeddings>]"
                       "[-data_type <tp>] [-dist_func <dist_func>]"
                       "[-seed <algorithm>] [-iters <num_iters>]"
                       // "[-kmeans_seed <algorithm>] [-kmeans_dist <algorithm>]"
  );

  std::string df = P.getOptionValue("-dist_func", "IP");

  if (df == "L2") {
    using Point = mvivf::L2_Point<float>;
    using ChPoint = ChamferL2_Point;
    bench<Point, ChPoint, true>(P);
  } else if (df == "IP") {
    using Point = mvivf::IP_Point<float>;
    using ChPoint = ChamferIP_Point;
    bench<Point, ChPoint, false>(P);
  }
  return 0;
}
