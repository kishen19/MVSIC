#include <iomanip>
#include <iostream>

#include "mvsic/core/bench_utils.h"
#include "mvsic/core/stats.h"
#include "mvsic/flat/flat.h"

using namespace mvsic;

namespace {

template<typename ChPoint, bool metric>
void run(commandLine& P) {
  Eigen::setNbThreads(1);
  using PC = PointCloudSet<ChPoint>;

  auto ds = bench::parse_dataset(P);
  if (ds.points.empty()) {
    std::cerr << "Error: specify -d <dataset> or -i <points>\n";
    std::exit(1);
  }

  auto io = bench::parse_io_args(P);
  auto qa = bench::parse_quant_args(P, "None");

  size_t k = P.getOptionLongValue("-k", 10);
  size_t num_rerank = P.getOptionLongValue("-num_rerank", k);

  SearchParams sp_base;
  bench::parse_compression_opts(sp_base, P);

  auto points = PC(ds.points.c_str(), io.is_mmap);
  IndexParams ip =
      IndexParams::flat(io.compress_input, io.verbose, qa.pq_method, qa.block_size,
                        qa.num_clusters_per_block, qa.num_points_per_cluster, qa.rabitq_bits);

  IndexFlat<metric, NoQuantizer<metric>> index(points.get_dims(), ip);
  bench::build_or_load(index, points, io.index_path, io.save_path);

  if (ds.queries.empty() || ds.gt.empty()) {
    std::cout << "No queries/gt; index built/loaded only.\n";
    return;
  }

  auto queries = PC(ds.queries.c_str());
  auto gt = ReadGT(ds.gt, queries.size());

  bench::print_header("IndexFlat (brute force)", ds.name, points.size(), queries.size());
  bench::print_compression_stats<ChPoint>(queries, sp_base, index.quantization_mode);

  SearchParams sp = SearchParams::flat(k, num_rerank);
  sp.query_compression = sp_base.query_compression;
  sp.query_compression_threshold = sp_base.query_compression_threshold;
  sp.compress_rerank = sp_base.compress_rerank;

  auto [pred, cmps] = index.search_all(queries, points, sp);
  auto [r1, rk] = compute_scores(pred, gt, k);

  std::cout << std::fixed << std::setprecision(4) << "Total distance cmps (reported): " << cmps
            << "\nRecall 1@" << k << ": " << r1 << "\nRecall " << k << "@" << k << ": " << rk << "\n";
}

}  // namespace

PARSE_DIST_FUNC_AND_RUN(run,
                        "IndexFlat smoke benchmark (build/load + search_all).\n\n"
                        "Dataset:\n"
                        "  -d <name> | -i <points.pcs> [-q] [-gt]\n"
                        "  -mm  -index <path>  -o <save_path>  -v <verbose>\n"
                        "  -compress_input\n\n"
                        "Search:\n"
                        "  -dist_func IP|L2   -k <K> (10)   -num_rerank <N> (=k)\n"
                        "  -compress none|carve|wards  -compress_threshold <tau>  -compress_rerank\n\n"
                        "Quantizer flags (unused for NoQuantizer template):\n"
                        "  -quant_method …   -m …   -num_clusters_per_block …  …\n")
