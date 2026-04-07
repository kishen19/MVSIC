// Sequential MVIVF benchmark.
//
// Compiled with -DPARLAY_SEQUENTIAL so all parlay::parallel_for loops run
// sequentially. Trimmed to the IP + IndexMVIVF<false> path only -- no flat,
// no spill, no L2, no Python bindings.
#include <Eigen/Dense>
#include <iostream>

#include "mvivf.h"
#include "mvsic/core/stats.h"
#include "mvsic/core/types/chamfer_ip_point.h"
#include "mvsic/core/types/point_cloud_set.h"
#include "mvsic/core/utils/parse_command_line.h"

using namespace mvsic;

int main(int argc, char* argv[]) {
  mvsic::commandLine P(argc, argv,
                       "[-i <points>] [-q <queries>] [-gt <gt>] [-index <index>] "
                       "[-max_leaf_size N] [-k_per_level N] [-quant_method TQ|FS|PQ|RQ|None] "
                       "[-qc] [-m N] [-num_clusters_per_block N] [-num_points_per_cluster N] "
                       "[-rbits N] [-k N] [-nprobes N] [-num_rerank N]");

  Eigen::setNbThreads(1);
  using ChPoint = ChamferIP_Point;
  using PC = PointCloudSet<ChPoint>;

  char* inFile = P.getOptionValue("-i");
  char* qFile = P.getOptionValue("-q");
  std::string gtFile = P.getOptionValue("-gt", "");
  std::string indexFile = P.getOptionValue("-index", "");
  bool is_mmap = P.getOption("-mm");
  bool quantize_centers = P.getOption("-qc");
  bool compress_input = P.getOption("-compress_input");
  uint32_t verbose = P.getOptionIntValue("-v", 0);

  uint32_t k_per_level = P.getOptionIntValue("-k_per_level", 0);
  uint32_t max_leaf_size = P.getOptionIntValue("-max_leaf_size", 500);

  uint32_t niters = P.getOptionIntValue("-niters", 5);
  uint32_t max_point_clouds_per_cluster = P.getOptionIntValue("-mpcc", 0);
  uint32_t max_points_per_centroid_inner_kmeans = P.getOptionIntValue("-mpcik", 20);
  bool use_weighted_inner_kmeans = P.getOption("-wgh_kmeans");

  std::string quant_method = P.getOptionValue("-quant_method", "FS");
  uint32_t quant_method_t = 0;
  if (quant_method == "None") quant_method_t = 0;
  else if (quant_method == "PQ") quant_method_t = 1;
  else if (quant_method == "RQ") quant_method_t = 2;
  else if (quant_method == "FS") quant_method_t = 3;
  else if (quant_method == "TQ") quant_method_t = 4;
  else { std::cerr << "Unknown PQ method: " << quant_method << std::endl; return 1; }

  uint32_t block_size = P.getOptionIntValue("-m", 8);
  uint32_t num_clusters_per_block = P.getOptionIntValue("-num_clusters_per_block", 256);
  uint32_t num_points_per_cluster = P.getOptionIntValue("-num_points_per_cluster", 20);
  uint32_t rabitq_bits = P.getOptionIntValue("-rbits", 8);

  size_t k = P.getOptionLongValue("-k", 10);
  bool nprobes_set = (P.getOptionValue("-nprobes") != nullptr);
  size_t nprobes_single = P.getOptionLongValue("-nprobes", 16);
  size_t num_rerank = P.getOptionLongValue("-num_rerank", k);

  std::vector<size_t> nprobes_list;
  if (nprobes_set) {
    nprobes_list = {nprobes_single};
  } else {
    nprobes_list = {1, 2, 4, 8, 16, 32, 64, 128, 256};
  }

  if (inFile == nullptr || qFile == nullptr || indexFile.empty() || gtFile.empty()) {
    std::cerr << "Required: -i <points> -q <queries> -gt <gt> -index <index>" << std::endl;
    return 1;
  }

  std::cout << "Loading points..." << std::endl;
  auto points = PC(inFile, is_mmap);

  IndexParams index_params = IndexParams::mvivf(
      k_per_level, max_leaf_size, compress_input, verbose, niters, max_point_clouds_per_cluster,
      max_points_per_centroid_inner_kmeans, "Random", 0, use_weighted_inner_kmeans, 0,
      quant_method_t, block_size, num_clusters_per_block, num_points_per_cluster, rabitq_bits,
      quantize_centers);
  IndexMVIVF<false> index(points.get_dims(), index_params);
  std::cout << "Loading index from " << indexFile << std::endl;
  index.load(indexFile, points);

  std::cout << "Loading queries..." << std::endl;
  auto queries = PC(qFile);
  auto gt = ReadGT(gtFile, queries.size());

  const size_t reps = 3;
  for (size_t nprobes : nprobes_list) {
    SearchParams search_params = SearchParams::mvivf(k, nprobes, num_rerank);
    std::cout << "\n=== nprobes=" << nprobes << " ===" << std::endl;

    parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>> pred(queries.size());
    parlay::sequence<size_t> cmps(queries.size());

    // Warmup
    for (size_t j = 0; j < std::min<size_t>(10, queries.size()); j++) {
      auto [p, b] = index.search(queries[j], points, search_params);
    }

    double best_time = 1e15;
    for (size_t it = 0; it < reps; it++) {
      parlay::internal::timer t;
      double iter_time = 0.0;
      for (size_t j = 0; j < queries.size(); j++) {
        if (it == 0) {
          // Use search_with_stats once to capture pointcloud cmps; stats[0]+stats[1].
          t.start();
          auto [p, b, stats] = index.search_with_stats(queries[j], points, search_params);
          t.stop();
          iter_time += t.total_time();
          t.reset();
          pred[j] = p;
          cmps[j] = static_cast<size_t>(stats[0]) + static_cast<size_t>(stats[1]);
        } else {
          t.start();
          auto [p, b] = index.search(queries[j], points, search_params);
          t.stop();
          iter_time += t.total_time();
          t.reset();
        }
      }
      best_time = std::min(best_time, iter_time);
    }

    double QPS = queries.size() / best_time;
    double avg_cmps = (double)parlay::reduce(cmps) / (double)cmps.size();
    double recall_1_k = compute_recall(pred, gt, k, 1);
    double recall_k_k = compute_recall(pred, gt, k, k);

    std::cout << "Number of Queries:    " << queries.size() << std::endl
              << "QPS:                  " << QPS << std::endl
              << "Avg pointcloud cmps:  " << avg_cmps << std::endl
              << "Total pointclouds:    " << points.size() << std::endl
              << "Recall 1@" << k << ":            " << recall_1_k << std::endl
              << "Recall " << k << "@" << k << ":           " << recall_k_k << std::endl;
  }
  return 0;
}
