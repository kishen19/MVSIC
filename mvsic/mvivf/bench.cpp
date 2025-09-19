#include <Eigen/Dense>
#include <iostream>
#include "mvsic/core/utils/chamfer_ip_point.h"
#include "mvsic/core/utils/chamfer_l2_point.h"
#include "mvsic/core/utils/parse_command_line.h"
#include "mvsic/core/utils/point_cloud_set.h"
#include "mvsic/core/stats.h"
#include "mvivf.h"
#include "mvivf_flat.h"

using namespace mvsic;

template<typename ChPoint, bool metric>
void bench(mvsic::commandLine &P) {
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
  bool is_mmap = P.getOption("-mm");

  // MVIVF params
  uint32_t k_per_level = P.getOptionIntValue("-k_per_level", 0);
  uint32_t max_leaf_size = P.getOptionIntValue("-max_leaf_size", 200);
  uint32_t verbose = P.getOptionIntValue("-v", 0);
  bool compress_input = P.getOption("-compress_input");
  bool use_PQ = P.getOption("-pq");

  // Search Params
  size_t k = P.getOptionLongValue("-k", 10);
  size_t nprobes = P.getOptionLongValue("-nprobes", 2);
  size_t num_rerank = P.getOptionLongValue("-num_rerank", k);

  // Flat params
  bool is_flat = P.getOption("-flat");

  auto points = PC(inFile, is_mmap);
  IndexParams index_params;
  SearchParams search_params;
  if (is_flat) {
    index_params = IndexParams::mvivf_flat(k_per_level, compress_input, use_PQ, verbose);
    search_params = SearchParams::mvivf_flat(k, nprobes, num_rerank);
  } else {
    index_params = IndexParams::mvivf(k_per_level, max_leaf_size, compress_input, use_PQ, verbose);
    search_params = SearchParams::mvivf(k, nprobes, num_rerank);
  }

  auto run_bench = [&](auto &index) {
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
        auto [bf_results, dist_cmps] = mvsic::get_knn(queries[i], points, 2 * k);

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
  };

  if (is_flat) {
    IndexMVIVFFlat<metric> index(points.get_dims(), index_params);
    run_bench(index);
  } else {
    IndexMVIVF<metric> index(points.get_dims(), index_params);
    run_bench(index);
  }
}

int main(int argc, char *argv[]) {
  mvsic::commandLine P(argc, argv,
                       "[-i <inFile>] [-k <num_centers>] [-s <num_embeddings>]"
                       "[-data_type <tp>] [-dist_func <dist_func>]"
                       "[-seed <algorithm>] [-iters <num_iters>]");
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
