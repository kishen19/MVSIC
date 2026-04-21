#include "mvsic/core/bench_utils.h"
#include "vamana.h"

using namespace mvsic;

template <typename ChPoint, bool metric>
void run(commandLine& P) {
  Eigen::setNbThreads(1);
  using PC = PointCloudSet<ChPoint>;

  auto ds = bench::parse_dataset(P);
  if (ds.points.empty()) { std::cerr << "Error: specify -d <dataset>" << std::endl; std::exit(1); }

  auto io = bench::parse_io_args(P);
  auto qa = bench::parse_quant_args(P, "None");

  uint32_t R = P.getOptionIntValue("-R", 200);
  uint32_t L_build = P.getOptionIntValue("-L_build", 600);
  double alpha = P.getOptionDoubleValue("-a", 1.2);
  bool two_pass = P.getOption("-tp");

  size_t k = P.getOptionLongValue("-k", 10);
  size_t num_rerank = P.getOptionLongValue("-num_rerank", 0);
  double cut = 1.35;
  std::string L_str = P.getOptionValue("-L", "16,32,64,128,256,512,1024,2048");
  auto L_list = bench::parse_csv_ints(L_str);

  SearchParams sp_base;
  bench::parse_compression_opts(sp_base, P);

  auto points = PC(ds.points.c_str(), io.is_mmap);
  IndexParams ip = IndexParams::vamana(R, L_build, alpha, two_pass, io.compress_input, io.verbose,
                                        qa.pq_method, qa.block_size, qa.num_clusters_per_block,
                                        qa.num_points_per_cluster, qa.rabitq_bits);
  IndexVamana<metric> index(points.get_dims(), ip);
  bench::build_or_load(index, points, io.index_path);

  if (ds.queries.empty() || ds.gt.empty()) return;
  auto queries = PC(ds.queries.c_str());
  auto gt = ReadGT(ds.gt, queries.size());
  bench::print_header("Vamana (search_all)", ds.name, points.size(), queries.size());
  bench::print_compression_stats<ChPoint>(queries, sp_base, index.quantization_mode);

  bench::run_search_all_sweep(
      index, points, queries, gt, "L", L_list,
      [&](size_t L) {
        SearchParams sp = SearchParams::vamana(k, L, cut, num_rerank);
        sp.query_compression = sp_base.query_compression;
        sp.query_compression_threshold = sp_base.query_compression_threshold;
        sp.compress_rerank = sp_base.compress_rerank;
        return sp;
      },
      io.csv_path);
}

PARSE_DIST_FUNC_AND_RUN(run,
    "-d <dataset> -index <path> [-L 16,32,...] [-k N] [-num_rerank N] "
    "[-compress none|carve|wards] [-csv <path>]")
