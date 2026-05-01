#include "mvsic/core/bench_utils.h"
#include "muvera.h"

using namespace mvsic;

namespace {

// CLI -> compile-time IndexMUVERA<metric, LeafModel> dispatch.  Mirrors the
// pattern used by mvsic/mvivf/bench_search_all.cpp so callers can exercise
// each templated quantizer variant from a single binary.  1BTQ is omitted
// because the SV-variant of turboquant_1bit hasn't been ported yet (only
// turboquant_1bit_mv exists).
#define MUVERA_DISPATCH(metric, qm, fn)                                                       \
  do {                                                                                         \
    if      (qm == "None"  || qm == "none")                                                    \
      fn.template operator()<IndexMUVERA<metric, NoQuantizer<metric>>>();                      \
    else if (qm == "PQ"    || qm == "pq")                                                      \
      fn.template operator()<IndexMUVERA<metric, pq::Model<metric>>>();                        \
    else if (qm == "FS"    || qm == "fs")                                                      \
      fn.template operator()<IndexMUVERA<metric, fastscan::Model<metric>>>();                  \
    else if (qm == "RQ"    || qm == "rq")                                                      \
      fn.template operator()<IndexMUVERA<metric, rabitq::Model<metric>>>();                    \
    else if (qm == "TQ"    || qm == "tq")                                                      \
      fn.template operator()<IndexMUVERA<metric, turboquant::Model<metric>>>();                \
    else if (qm == "SPQTQ" || qm == "spqtq")                                                   \
      fn.template operator()<IndexMUVERA<metric, pqtq::Model<metric>>>();                      \
    else {                                                                                     \
      std::cerr << "Unknown -quant_method: " << qm                                             \
                << " (use None, PQ, FS, RQ, TQ, SPQTQ)" << std::endl;                          \
      std::exit(1);                                                                            \
    }                                                                                          \
  } while (0)

template <bool metric, class Fn>
void dispatch_muvera(const std::string& qm, Fn&& fn) {
  MUVERA_DISPATCH(metric, qm, fn);
}
#undef MUVERA_DISPATCH
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

  // Defaults mirror IndexParams::muvera_custom.
  int32_t num_reps = P.getOptionIntValue("-num_reps", 20);
  int32_t num_simhash = P.getOptionIntValue("-num_simhash", 4);
  int32_t projd = P.getOptionIntValue("-projd", 8);
  bool fill_empty = P.getOption("-fill_empty_partitions");
  int32_t final_projd = P.getOptionIntValue("-final_projd", 0);
  bool no_norm = P.getOption("-no_norm");
  uint32_t R = P.getOptionIntValue("-R", 200);
  uint32_t L_build = P.getOptionIntValue("-L_build", 600);
  double alpha = P.getOptionDoubleValue("-a", 1.1);
  int num_pass = P.getOptionIntValue("-np", 1);

  size_t k = P.getOptionLongValue("-k", 10);
  size_t num_rerank = P.getOptionLongValue("-num_rerank", k);
  double cut = 1.35;
  bool norerank = P.getOption("-norerank");
  std::string L_str = P.getOptionValue("-L", "16,32,64,128,256,512,1024,2048");
  auto L_list = bench::parse_csv_ints(L_str);

  SearchParams sp_base;
  bench::parse_compression_opts(sp_base, P);

  auto points = PC(ds.points.c_str(), io.is_mmap);
  IndexParams ip = IndexParams::muvera_custom(
      num_reps, num_simhash, 1, projd, fill_empty, final_projd, !no_norm,
      R, L_build, alpha, num_pass, io.compress_input, io.verbose,
      qa.pq_method, qa.block_size, qa.num_clusters_per_block,
      qa.num_points_per_cluster, qa.rabitq_bits);

  dispatch_muvera<metric>(quant_method, [&]<class IndexT>() {
    IndexT index(points.get_dims(), ip);
    bench::build_or_load(index, points, io.index_path);

    if (ds.queries.empty() || ds.gt.empty()) return;
    auto queries = PC(ds.queries.c_str());
    auto gt = ReadGT(ds.gt, queries.size());
    bench::print_header("MUVERA (search_all)", ds.name, points.size(), queries.size());
    bench::print_compression_stats<ChPoint>(queries, sp_base, index.quantization_mode);

    bench::run_search_all_sweep(
        index, points, queries, gt, "L", L_list,
        [&](size_t L) {
          SearchParams sp = SearchParams::muvera(k, L, num_rerank, cut, norerank);
          sp.query_compression = sp_base.query_compression;
          sp.query_compression_threshold = sp_base.query_compression_threshold;
          sp.compress_rerank = sp_base.compress_rerank;
          return sp;
        },
        io.csv_path);
  });
}

PARSE_DIST_FUNC_AND_RUN(run,
    "MUVERA batched many-to-many search benchmark (search_all path).\n\n"
    "Dataset / I/O:\n"
    "  -d <name> | -i <points> -q <queries> -gt <gt>\n"
    "  -index <path>  -o <save_path>  -csv <path>\n"
    "  -mm  -v <level>  -compress_input  -dist_func IP|L2\n\n"
    "MUVERA FDE params (build; defaults mirror IndexParams::muvera_custom):\n"
    "  -num_reps <N> (20)   -num_simhash <N> (4)   -projd <N> (8)\n"
    "  -fill_empty_partitions (off)   -final_projd <N> (0)   -no_norm (normalized)\n\n"
    "Underlying Vamana params (build):\n"
    "  -R <N> (200)   -L_build <N> (600)   -a <f> (1.1)   -np <N> (1)\n\n"
    "Leaf quantization (selects the templated IndexMUVERA<metric, LeafModel>):\n"
    "  -quant_method None|PQ|FS|RQ|TQ|SPQTQ    (default None)\n"
    "  -m <N> (8)   -num_clusters_per_block <N> (16)\n"
    "  -num_points_per_cluster <N> (100)   -rbits <N> (4)\n\n"
    "Search sweep:\n"
    "  -k <N>  -L <csv>  -num_rerank <N>  -norerank\n\n"
    "Query compression:\n"
    "  -compress none|carve|wards  -compress_threshold <tau>  -compress_rerank\n")
