#include "src/utils/chamfer_ip_point.h"
#include "src/utils/chamfer_l2_point.h"
#include "src/utils/parse_command_line.h"
#include "src/utils/point_cloud_set.h"
#include "src/utils/stats.h"
#include "vamana.h"

using namespace mvivf;

template<typename ChPoint, bool metric>
void bench(mvivf::commandLine& P) {
  using PC = PointCloudSet<ChPoint>;

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

  size_t R = P.getOptionLongValue("-R", 16);
  size_t L = P.getOptionLongValue("-L", 16);
  double alpha = P.getOptionDoubleValue("-a", 1.2);
  bool two_pass = P.getOption("-tp");
  size_t k = P.getOptionLongValue("-k", 10);
  bool verbose = P.getOption("-v");

  auto points = PC(inFile);
  IndexVamanaParams index_params(R, L, alpha, two_pass, verbose);
  SearchParams search_params(k, L, 1.35, points.size(), R);
  IndexVamana<metric> index(points.get_dims(), index_params);
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
    using ChPoint = ChamferL2_Point;
    bench<ChPoint, true>(P);
  } else if (df == "IP") {
    using ChPoint = ChamferIP_Point;
    bench<ChPoint, false>(P);
  }
  return 0;
}
