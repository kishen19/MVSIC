#include "mvsic/core/bench_utils.h"
#include "vamana.h"

using namespace mvsic;

namespace {

// CLI -> compile-time IndexVamana<metric, LeafModel> dispatch.  Vamana
// reuses the multi-vector quantizer Models (`*_mv`) which include 1BTQ and
// 8BTQ, so the full menu is available here.
#define VAMANA_DISPATCH(metric, qm, fn)                                                       \
  do {                                                                                         \
    if      (qm == "None"  || qm == "none")                                                    \
      fn.template operator()<IndexVamana<metric, NoQuantizer<metric>>>();                      \
    else if (qm == "PQ"    || qm == "pq")                                                      \
      fn.template operator()<IndexVamana<metric, pq_mv::Model<metric>>>();                     \
    else if (qm == "FS"    || qm == "fs")                                                      \
      fn.template operator()<IndexVamana<metric, fastscan_mv::Model<metric>>>();               \
    else if (qm == "RQ"    || qm == "rq")                                                      \
      fn.template operator()<IndexVamana<metric, rabitq_mv::Model<metric>>>();                 \
    else if (qm == "TQ"    || qm == "tq")                                                      \
      fn.template operator()<IndexVamana<metric, turboquant_mv::Model<metric>>>();             \
    else if (qm == "SPQTQ" || qm == "spqtq")                                                   \
      fn.template operator()<IndexVamana<metric, pqtq_mv::Model<metric>>>();                   \
    else if (qm == "1BTQ"  || qm == "1btq")                                                    \
      fn.template operator()<IndexVamana<metric, turboquant_1bit_mv::Model<metric>>>();        \
    else if (qm == "8BTQ"  || qm == "8btq")                                                    \
      fn.template operator()<IndexVamana<metric, turboquant_8bit_mv::Model<metric>>>();        \
    else {                                                                                     \
      std::cerr << "Unknown -quant_method: " << qm                                             \
                << " (use None, PQ, FS, RQ, TQ, SPQTQ, 1BTQ, 8BTQ)" << std::endl;              \
      std::exit(1);                                                                            \
    }                                                                                          \
  } while (0)

template <bool metric, class Fn>
void dispatch_vamana(const std::string& qm, Fn&& fn) {
  VAMANA_DISPATCH(metric, qm, fn);
}
#undef VAMANA_DISPATCH
}  // namespace

template <typename ChPoint, bool metric>
void run(commandLine& P) {
  Eigen::setNbThreads(1);
  using PC = PointCloudSet<ChPoint>;

  auto ds = bench::parse_dataset(P);
  if (ds.points.empty()) { std::cerr << "Error: specify -d <dataset>" << std::endl; std::exit(1); }

  auto io = bench::parse_io_args(P);
  auto qa = bench::parse_quant_args(P, "None");
  std::string quant_method = P.getOptionValue("-quant_method", "None");

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

  dispatch_vamana<metric>(quant_method, [&]<class IndexT>() {
    IndexT index(points.get_dims(), ip);
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
  });
}

PARSE_DIST_FUNC_AND_RUN(run,
    "Vamana batched many-to-many search benchmark (search_all path).\n\n"
    "Dataset / I/O:\n"
    "  -d <name> | -i <points> -q <queries> -gt <gt>\n"
    "  -index <path>                  Pre-built index to load\n"
    "  -csv <path>                    Per-sweep-point output CSV\n"
    "  -mm  -v <level>  -compress_input  -dist_func IP|L2\n\n"
    "Vamana build params (build only; defaults mirror IndexParams::vamana):\n"
    "  -R <N> (200)   -L_build <N> (600)   -a <f> (1.2)   -tp (off)\n\n"
    "Leaf quantization (selects the templated IndexVamana<metric, LeafModel>):\n"
    "  -quant_method None|PQ|FS|RQ|TQ|SPQTQ|1BTQ|8BTQ    (default None)\n"
    "  -m <N> (8)   -num_clusters_per_block <N> (16)\n"
    "  -num_points_per_cluster <N> (100)   -rbits <N> (4)\n\n"
    "Search sweep:\n"
    "  -k <N>  -L <csv>  -num_rerank <N>\n\n"
    "Query compression:\n"
    "  -compress none|carve|wards  -compress_threshold <tau>  -compress_rerank\n")
