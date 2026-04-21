#include <Eigen/Dense>
#include <iostream>
#include "mvsic/core/types/chamfer_ip_point.h"
#include "mvsic/core/types/chamfer_l2_point.h"
#include "mvsic/core/types/point_cloud_set.h"
#include "mvsic/core/utils/parse_command_line.h"
#include "mvsic/core/stats.h"
#include "mvivf.h"
#include "mvivf_spill.h"
#include "mvivf_flat.h"

using namespace mvsic;

// Parse -query_compress {none,ball,wards}, -query_compress_threshold, -compress_rerank
inline void apply_query_compression_opts(SearchParams& sp, mvsic::commandLine& P) {
  std::string qc = P.getOptionValue("-query_compress", "none");
  if (qc == "none" || qc == "off" || qc == "0") {
    sp.query_compression = SearchParams::QueryCompression::None;
  } else if (qc == "ball" || qc == "ballcarving" || qc == "muvera") {
    sp.query_compression = SearchParams::QueryCompression::Carve;
  } else if (qc == "wards" || qc == "ward") {
    sp.query_compression = SearchParams::QueryCompression::Wards;
  } else {
    std::cerr << "Unknown -query_compress value: " << qc << " (use none, ball, wards)" << std::endl;
    std::exit(1);
  }
  sp.query_compression_threshold =
      static_cast<float>(P.getOptionDoubleValue("-query_compress_threshold", 0.7));
  sp.compress_rerank = P.getOption("-compress_rerank");
}

template<typename ChPoint, bool metric>
void bench(mvsic::commandLine& P) {
  Eigen::setNbThreads(1);
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
  bool is_mmap = P.getOption("-mm");
  bool quantize_centers = P.getOption("-qc");

  bool compress_input = P.getOption("-compress_input");
  uint32_t verbose = P.getOptionIntValue("-v", 0);

  // MVIVF params
  uint32_t k_per_level = P.getOptionIntValue("-k_per_level", 0);
  uint32_t max_leaf_size = P.getOptionIntValue("-max_leaf_size", 200);

  // MVClus params
  uint32_t niters = P.getOptionIntValue("-niters", 5);
  uint32_t max_point_clouds_per_cluster = P.getOptionIntValue("-mpcc", 0);
  uint32_t max_points_per_centroid_inner_kmeans = P.getOptionIntValue("-mpcik", 20);
  bool use_weighted_inner_kmeans = P.getOption("-wgh_kmeans");

  // Flat params
  bool is_flat = P.getOption("-flat");
  // Spill params
  bool is_spill = P.getOption("-spill");
  uint32_t num_spill = static_cast<uint32_t>(P.getOptionIntValue("-spill", 2));

  // PQ params
  std::string quant_method = P.getOptionValue("-quant_method", "None");
  uint32_t quant_method_t = 0;
  if (quant_method == "None") {
    quant_method_t = 0;
  } else if (quant_method == "PQ") {
    quant_method_t = 1;
  } else if (quant_method == "RQ") {
    quant_method_t = 2;
  } else if (quant_method == "FS") {
    quant_method_t = 3;
  } else if (quant_method == "TQ") {
    quant_method_t = 4;
  } else if (quant_method == "SPQTQ") {
    quant_method_t = 5;
  } else if (quant_method == "1BTQ") {
    quant_method_t = 6;
  } else {
    std::cerr << "Unknown PQ method: " << quant_method << std::endl;
    exit(1);
  }
  uint32_t block_size = P.getOptionIntValue("-m", 8);
  uint32_t num_clusters_per_block = P.getOptionIntValue("-num_clusters_per_block", 256);
  uint32_t num_points_per_cluster = P.getOptionIntValue("-num_points_per_cluster", 20);
  uint32_t rabitq_bits = P.getOptionIntValue("-rbits", 8);

  // Search Params
  size_t k = P.getOptionLongValue("-k", 10);
  size_t nprobes = P.getOptionLongValue("-nprobes", 2);
  size_t num_rerank = P.getOptionLongValue("-num_rerank", k);

  auto points = PC(inFile, is_mmap);
  std::cout << "Avg points per cloud: " << points.average_size() << std::endl;
  IndexParams index_params;
  SearchParams search_params;
  if (is_flat) {
    index_params = IndexParams::mvivf_flat(
        k_per_level, compress_input, verbose, niters, max_point_clouds_per_cluster,
        max_points_per_centroid_inner_kmeans, "Random", 0, use_weighted_inner_kmeans, 0,
        quant_method_t, block_size, num_clusters_per_block, num_points_per_cluster, rabitq_bits,
        quantize_centers);
    search_params = SearchParams::mvivf_flat(k, nprobes, num_rerank);
  } else if (is_spill) {
    index_params = IndexParams::mvivf_spill(
        k_per_level, max_leaf_size, num_spill, compress_input, verbose, niters,
        max_point_clouds_per_cluster, max_points_per_centroid_inner_kmeans, "Random", 0,
        use_weighted_inner_kmeans, 0, quant_method_t, block_size, num_clusters_per_block,
        num_points_per_cluster, rabitq_bits, quantize_centers);
    search_params = SearchParams::mvivf(k, nprobes, num_rerank);
  } else {
    index_params = IndexParams::mvivf(
        k_per_level, max_leaf_size, compress_input, verbose, niters, max_point_clouds_per_cluster,
        max_points_per_centroid_inner_kmeans, "Random", 0, use_weighted_inner_kmeans, 0,
        quant_method_t, block_size, num_clusters_per_block, num_points_per_cluster, rabitq_bits,
        quantize_centers);
    search_params = SearchParams::mvivf(k, nprobes, num_rerank);
  }
  apply_query_compression_opts(search_params, P);
  if (search_params.query_compression != SearchParams::QueryCompression::None) {
    const char* mname = (search_params.query_compression == SearchParams::QueryCompression::Carve)
                            ? "ball"
                            : "wards";
    std::cout << "Query compression: method=" << mname
              << " threshold=" << search_params.query_compression_threshold
              << " compress_rerank=" << (search_params.compress_rerank ? "1" : "0") << std::endl;
  }

  auto run_bench = [&](auto& index) {
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
      if (search_params.query_compression != SearchParams::QueryCompression::None) {
        uint32_t ba = qc_internal::batch_alignment(index.quantization_mode);
        double sum_orig = 0.0;
        double sum_comp = 0.0;
        for (size_t j = 0; j < queries.size(); ++j) {
          sum_orig += static_cast<double>(queries[j].size());
          auto c = compress_query<ChPoint>(queries[j], search_params.query_compression,
                                           search_params.query_compression_threshold, ba);
          sum_comp += static_cast<double>(c.n);
        }
        const double nq = static_cast<double>(queries.size());
        std::cout << "Avg query points (raw):        " << (sum_orig / nq) << std::endl
                  << "Avg query points (compressed): " << (sum_comp / nq) << std::endl
                  << "Compression ratio (raw/compr): " << (sum_comp > 0 ? sum_orig / sum_comp : 0.0)
                  << std::endl;
      }
      auto gt = ReadGT(gtFile, queries.size());
      double QPS_seq, QPS_par, avg_cmps, recall_1_k, recall_k_k;
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
  } else if (is_spill) {
    IndexMVIVFSpill<metric> index(points.get_dims(), index_params);
    run_bench(index);
  } else {
    IndexMVIVF<metric> index(points.get_dims(), index_params);
    run_bench(index);
  }
}

int main(int argc, char* argv[]) {
  mvsic::commandLine P(argc, argv,
                       "[-i <inFile>] [-q <queries>] [-dist_func IP|L2] "
                       "[-query_compress none|ball|wards] [-query_compress_threshold <tau>] "
                       "[-compress_rerank] ...");
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
