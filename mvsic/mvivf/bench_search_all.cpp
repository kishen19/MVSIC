#include "mvsic/core/bench_utils.h"
#include "mvivf.h"
#include "mvivf_flat.h"
#include "mvivf_spill.h"

using namespace mvsic;

template <typename ChPoint, bool metric>
void run(commandLine& P) {
  Eigen::setNbThreads(1);
  using PC = PointCloudSet<ChPoint>;

  auto ds = bench::parse_dataset(P);
  if (ds.points.empty()) { std::cerr << "Error: specify -d <dataset>" << std::endl; std::exit(1); }

  auto io = bench::parse_io_args(P);
  auto qa = bench::parse_quant_args(P, "None");

  uint32_t k_per_level = P.getOptionIntValue("-k_per_level", 0);
  uint32_t max_leaf_size = P.getOptionIntValue("-max_leaf_size", 500);
  bool quantize_centers = P.getOption("-qc");
  uint32_t niters = P.getOptionIntValue("-niters", 5);
  uint32_t mpcc = P.getOptionIntValue("-mpcc", 0);
  uint32_t mpcik = P.getOptionIntValue("-mpcik", 20);
  bool wgh_kmeans = P.getOption("-wgh_kmeans");
  uint32_t s = P.getOptionIntValue("-s", 0);
  bool is_flat = P.getOption("-flat");
  int num_spill = P.getOptionIntValue("-spill", 0);
  bool is_spill = (num_spill > 0);

  size_t k = P.getOptionLongValue("-k", 10);
  size_t num_rerank = P.getOptionLongValue("-num_rerank", 0);
  std::string nprobes_str = P.getOptionValue("-nprobes", "1,2,4,8,16,32,64,128,256,512,1024");
  auto nprobes_list = bench::parse_csv_ints(nprobes_str);

  SearchParams sp_base;
  bench::parse_compression_opts(sp_base, P);
  bench::print_compression_info(sp_base);

  auto points = PC(ds.points.c_str(), io.is_mmap);

  auto do_search_all = [&](auto& index, const std::string& method_name) {
    bench::build_or_load(index, points, io.index_path);

    if (ds.queries.empty() || ds.gt.empty()) {
      std::cout << "No queries/GT specified. Done." << std::endl;
      return;
    }
    auto queries = PC(ds.queries.c_str());
    auto gt = ReadGT(ds.gt, queries.size());
    bench::print_header(method_name + " (search_all)", ds.name, points.size(), queries.size());
    bench::print_compression_stats<ChPoint>(queries, sp_base, index.quantization_mode);

    auto make_sp = [&](size_t np) {
      SearchParams sp;
      if (is_flat) sp = SearchParams::mvivf_flat(k, np, num_rerank);
      else if (is_spill) sp = SearchParams::mvivf_spill(k, np, num_rerank);
      else sp = SearchParams::mvivf(k, np, num_rerank);
      sp.query_compression = sp_base.query_compression;
      sp.query_compression_threshold = sp_base.query_compression_threshold;
      sp.compress_rerank = sp_base.compress_rerank;
      return sp;
    };

    bench::run_search_all_sweep(index, points, queries, gt,
        "nprobes", nprobes_list, make_sp, io.csv_path);
  };

  if (is_flat) {
    IndexParams ip = IndexParams::mvivf_flat(k_per_level, io.compress_input, io.verbose, niters,
        mpcc, mpcik, "Random", 0, wgh_kmeans, s, qa.pq_method, qa.block_size,
        qa.num_clusters_per_block, qa.num_points_per_cluster, qa.rabitq_bits, quantize_centers);
    IndexMVIVFFlat<metric> index(points.get_dims(), ip);
    do_search_all(index, "MVIVF_Flat");
  } else if (is_spill) {
    IndexParams ip = IndexParams::mvivf_spill(k_per_level, max_leaf_size, num_spill,
        io.compress_input, io.verbose, niters, mpcc, mpcik, "Random", 0, wgh_kmeans, s,
        qa.pq_method, qa.block_size, qa.num_clusters_per_block, qa.num_points_per_cluster,
        qa.rabitq_bits, quantize_centers);
    IndexMVIVFSpill<metric> index(points.get_dims(), ip);
    do_search_all(index, "MVIVF_Spill");
  } else {
    IndexParams ip = IndexParams::mvivf(k_per_level, max_leaf_size, io.compress_input, io.verbose,
        niters, mpcc, mpcik, "Random", 0, wgh_kmeans, s, qa.pq_method, qa.block_size,
        qa.num_clusters_per_block, qa.num_points_per_cluster, qa.rabitq_bits, quantize_centers);
    IndexMVIVF<metric> index(points.get_dims(), ip);
    do_search_all(index, "MVIVF");
  }
}

PARSE_DIST_FUNC_AND_RUN(run,
    "-d <dataset> -index <path> [-nprobes 1,2,...] [-k N] [-num_rerank N] "
    "[-compress none|carve|wards] [-csv <path>]")
