#include <Eigen/Dense>
#include <iostream>
#include "mvivf_flat.h"
#include "mvivf_flat_MVQ.h"
#include "src/utils/chamfer_ip_point.h"
#include "src/utils/chamfer_l2_point.h"
#include "src/utils/parse_command_line.h"
#include "src/utils/point_cloud_set.h"
#include "src/utils/stats.h"

using namespace mvivf;

template<typename ChPoint, bool metric>
void bench(mvivf::commandLine &P) {
  Eigen::setNbThreads(1);
  using PC = PointCloudSet<ChPoint>;

  char *inFile = P.getOptionValue("-i");
  char *qFile = P.getOptionValue("-q");
  std::string QFile;
  if (qFile != nullptr) {
    QFile = P.getOptionValue("-q");
  } else {
    QFile = "";
  }
  std::string gtFile = P.getOptionValue("-gt", "");
  std::string outFile = P.getOptionValue("-o", "");
  std::string indexFile = P.getOptionValue("-index", "");

  size_t num_clusters = P.getOptionLongValue("-num_clusters", 300);
  size_t nprobes = P.getOptionLongValue("-nprobes", 1);
  size_t k = P.getOptionLongValue("-k", 10);
  double s = P.getOptionDoubleValue("-s", 1.0);
  size_t os_rate = P.getOptionLongValue("-osr", 20);
  size_t iters = P.getOptionLongValue("-iters", 5);
  bool is_mmap = P.getOption("-mm");
  bool verbose = P.getOption("-v");

  // MVQ
  bool use_MVQ = P.getOption("-mvq");
  size_t num_leaf_centroids = P.getOptionLongValue("-nlc", 32);
  size_t cands = P.getOptionLongValue("-cands", 2 * k);

  auto points = PC(inFile, is_mmap);
  IndexParams index_params;
  SearchParams search_params;
  if (use_MVQ) {
    index_params =
        IndexParams::mvivf_flat_mvq(num_clusters, num_leaf_centroids, s, iters, os_rate, verbose);
    search_params = SearchParams::mvivf_flat_mvq(k, nprobes, cands);
  } else {
    index_params = IndexParams::mvivf_flat(num_clusters, s, iters, os_rate, verbose);
    search_params = SearchParams::mvivf_flat(k, nprobes);
  }

  auto run_bench = [&](auto &index) {
    if (indexFile != "") {
      std::cout << "Loading index from " << indexFile << std::endl;
      index.load(indexFile, points);
      std::cout << "Index loaded" << std::endl;
      std::cout << "Mean cluster size: " << index.mean_cluster_size() << std::endl;
    } else {
      std::cout << "Building index..." << std::endl;
      parlay::internal::timer it;
      it.start();
      index.build(points);
      it.stop();
      std::cout << "Index built in " << it.total_time() << " seconds." << std::endl;
      std::cout << "Mean cluster size: " << index.mean_cluster_size() << std::endl;
    }

    if (outFile != "") {
      std::cout << "Saving index to " << outFile << std::endl;
      index.save(P.getOptionValue("-o"));
      std::cout << "Index saved." << std::endl;
    }

    if (QFile != "") {
      auto queries = PC(qFile, is_mmap);
      auto gt = ReadGT(gtFile, queries.size());
      double QPS_seq, QPS_par, avg_cmps, recall_1_k, recall_k_k;

      // The large commented out block is preserved here.

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
  };

  if (use_MVQ) {
    IndexMVIVFFlatMVQ<metric> index(points.get_dims(), index_params);
    run_bench(index);
  } else {
    IndexMVIVFFlat<metric> index(points.get_dims(), index_params);
    run_bench(index);
  }
}

int main(int argc, char *argv[]) {
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
