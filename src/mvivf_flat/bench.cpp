#include <Eigen/Dense>
#include <iostream>
#include "src/utils/chamfer_ip_point.h"
#include "src/utils/chamfer_l2_point.h"
#include "src/utils/parse_command_line.h"
#include "src/utils/point_cloud_set.h"
#include "src/utils/stats.h"
#include "mvivf_flat.h"

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

  size_t maxsize = P.getOptionLongValue("-maxsize", 500);
  size_t nprobes = P.getOptionLongValue("-nprobes", 1);
  size_t k = P.getOptionLongValue("-k", 10);
  double s = P.getOptionDoubleValue("-s", 1.0);
  size_t os_rate = P.getOptionLongValue("-osr", 20);
  size_t iters = P.getOptionLongValue("-iters", 5);
  bool verbose = P.getOption("-v");

  auto points = PC(inFile);
  IndexParams index_params = IndexParams::mvivf_flat(maxsize, s, iters, os_rate, verbose);
  SearchParams search_params = SearchParams::mvivf_flat(k, nprobes);
  IndexMVIVFFlat<metric> index(points.get_dims(), index_params);

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
    parlay::internal::timer t;
    recall_1_k = 0.0;
    recall_k_k = 0.0;
    double query_time = 0.0;
    for (size_t i = 0; i < queries.size(); i++) {
      if (i % 100 == 0) {
        std::cout << queries.size() - i << " queries left" << std::endl;
      }
      std::unordered_set<size_t> out_set;
      // Run Index search
      t.start();
      auto [results_new, _cmps] = index.search(queries[i], points, search_params);
      t.stop();
      query_time += t.total_time();
      t.reset();
      for (const auto &[id, dist] : results_new) {
        out_set.insert(id);
      }

      // Run Brute-force search
      auto [bf_results, dist_cmps] = mvivf::get_knn(queries[i], points, 2 * k);

      // Calculate recall
      size_t correct = 0;
      for (size_t j = 0; j < k; j++) {
        auto [id, dist] = bf_results[j];
        if (out_set.find(id) != out_set.end()) {
          correct++;
        }
      }
      // Dealing with duplicates and near duplicates: fine to return
      // any of the (near) duplicates of the last point
      float last_dist = bf_results[k - 1].second;
      for (size_t j = k; j < bf_results.size(); j++) {
        auto [id, dist] = bf_results[j];
        if (std::abs(dist - last_dist) < 1e-6) {
          if (out_set.find(id) != out_set.end()) {
            correct++;
          }
        } else {
          break;
        }
      }
      recall_k_k += static_cast<double>(correct) / k;
      if (out_set.find(bf_results[0].first) != out_set.end()) {
        recall_1_k += 1.0;
      }
    }
    recall_1_k /= queries.size();
    recall_k_k /= queries.size();
    double QPS = queries.size() / query_time;
    double avg_query_time = 1 / QPS;
    std::cout << "Number of Queries: " << queries.size() << std::endl;
    std::cout << "Average recall 1 @ " << k << ": " << recall_1_k << std::endl;
    std::cout << "Average recall " << k << " @ " << k << ": " << recall_k_k << std::endl;
    std::cout << "QPS: " << QPS << std::endl;
    std::cout << "Average time per query: " << avg_query_time << " seconds" << std::endl;
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
