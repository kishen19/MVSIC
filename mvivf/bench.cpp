#include <iostream>
#include "mvc/utils/chamfer_ip_point.h"
#include "mvc/utils/chamfer_l2_point.h"
#include "mvc/utils/parse_command_line.h"
#include "mvc/utils/point_cloud_set.h"
#include "mvivf.h"
#include "utils/stats.h"

template<typename ChPoint, bool metric>
void bench(mvivf::commandLine &P) {
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

  size_t minsize = P.getOptionLongValue("-minsize", 100);
  size_t maxsize = P.getOptionLongValue("-maxsize", 500);
  size_t nprobes = P.getOptionLongValue("-nprobes", 1);
  size_t beamsize = P.getOptionLongValue("-beamsize", 0);
  size_t k = P.getOptionLongValue("-k", 10);
  size_t s = P.getOptionLongValue("-s", 0);
  size_t os_rate = P.getOptionLongValue("-osr", 20);
  auto seeding = P.getOptionValue("-seed", "Random");
  auto iters = P.getOptionLongValue("-iters", 5);
  bool verbose = P.getOption("-v");

  auto points = PC(inFile);
  mvivf::IndexMVIVFParams index_params(minsize, maxsize, s, iters, seeding, verbose, os_rate);
  mvivf::SearchParams search_params(k, nprobes, beamsize);
  mvivf::IndexMVIVF<metric> index(points.get_dims(), index_params);
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
      auto [bf_results, dist_cmps] = mvivf::get_knn(queries[i], points, k);

      // Calculate recall
      float correct = 0.0;
      for (const auto &[id, dist] : bf_results) {
        if (out_set.find(id) != out_set.end()) {
          correct += 1.0;
        }
      }
      recall_k_k += correct / k;
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
    mvivf::Stats result = compute_stats(index, points, queries, gt, search_params);
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
    bench<ChPoint, mvivf::L2>(P);
  } else if (df == "IP") {
    using ChPoint = ChamferIP_Point;
    bench<ChPoint, mvivf::IP>(P);
  }
  return 0;
}
