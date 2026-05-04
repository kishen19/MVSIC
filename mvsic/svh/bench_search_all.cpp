#include "mvsic/core/bench_utils.h"
#include "svh_graph.h"
#include "svh_ivf.h"

using namespace mvsic;

namespace {

// CLI -> compile-time dispatch for the SVH families.  Mirrors the pattern
// used by mvsic/mvivf/bench_search_all.cpp.
//
//  - SVHGraph<metric, LeafModel>: 6 quantizer variants (1BTQ omitted; the
//    SV-variant of turboquant_1bit hasn't been ported yet).
//  - SVHIVF<metric, CompressCenters, LeafModel>: same 6 quantizers, each
//    available with -qc 0 (raw centers) or -qc 1 (TQ-encoded centers).
#define SVH_GRAPH_DISPATCH(metric, qm, fn)                                                    \
  do {                                                                                         \
    if      (qm == "None"  || qm == "none")                                                    \
      fn.template operator()<IndexSVHGraph<metric, NoQuantizer<metric>>>();                    \
    else if (qm == "PQ"    || qm == "pq")                                                      \
      fn.template operator()<IndexSVHGraph<metric, pq::Model<metric>>>();                      \
    else if (qm == "FS"    || qm == "fs")                                                      \
      fn.template operator()<IndexSVHGraph<metric, fastscan::Model<metric>>>();                \
    else if (qm == "RQ"    || qm == "rq")                                                      \
      fn.template operator()<IndexSVHGraph<metric, rabitq::Model<metric>>>();                  \
    else if (qm == "TQ"    || qm == "tq")                                                      \
      fn.template operator()<IndexSVHGraph<metric, turboquant::Model<metric>>>();              \
    else if (qm == "SPQTQ" || qm == "spqtq")                                                   \
      fn.template operator()<IndexSVHGraph<metric, pqtq::Model<metric>>>();                    \
    else {                                                                                     \
      std::cerr << "Unknown -quant_method: " << qm                                             \
                << " (use None, PQ, FS, RQ, TQ, SPQTQ)" << std::endl;                          \
      std::exit(1);                                                                            \
    }                                                                                          \
  } while (0)

#define SVH_IVF_DISPATCH_LM(metric, C, qm, fn)                                                \
  do {                                                                                         \
    if      (qm == "None"  || qm == "none")                                                    \
      fn.template operator()<IndexSVHIVF<metric, C, NoQuantizer<metric>>>();                   \
    else if (qm == "PQ"    || qm == "pq")                                                      \
      fn.template operator()<IndexSVHIVF<metric, C, pq::Model<metric>>>();                     \
    else if (qm == "FS"    || qm == "fs")                                                      \
      fn.template operator()<IndexSVHIVF<metric, C, fastscan::Model<metric>>>();               \
    else if (qm == "RQ"    || qm == "rq")                                                      \
      fn.template operator()<IndexSVHIVF<metric, C, rabitq::Model<metric>>>();                 \
    else if (qm == "TQ"    || qm == "tq")                                                      \
      fn.template operator()<IndexSVHIVF<metric, C, turboquant::Model<metric>>>();             \
    else if (qm == "SPQTQ" || qm == "spqtq")                                                   \
      fn.template operator()<IndexSVHIVF<metric, C, pqtq::Model<metric>>>();                   \
    else {                                                                                     \
      std::cerr << "Unknown -quant_method: " << qm                                             \
                << " (use None, PQ, FS, RQ, TQ, SPQTQ)" << std::endl;                          \
      std::exit(1);                                                                            \
    }                                                                                          \
  } while (0)

template <bool metric, class Fn>
void dispatch_svh_graph(const std::string& qm, Fn&& fn) {
  SVH_GRAPH_DISPATCH(metric, qm, fn);
}

template <bool metric, class Fn>
void dispatch_svh_ivf(bool compress_centers, const std::string& qm, Fn&& fn) {
  if (compress_centers) SVH_IVF_DISPATCH_LM(metric, true,  qm, fn);
  else                  SVH_IVF_DISPATCH_LM(metric, false, qm, fn);
}
#undef SVH_GRAPH_DISPATCH
#undef SVH_IVF_DISPATCH_LM
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
  bool is_graph = P.getOption("-graph");

  size_t k = P.getOptionLongValue("-k", 10);
  size_t num_rerank = P.getOptionLongValue("-num_rerank", k);
  bool no_rerank = P.getOption("-no_rerank");
  double cut = 1.35;

  SearchParams sp_base;
  bench::parse_compression_opts(sp_base, P);

  auto points = PC(ds.points.c_str(), io.is_mmap);

  if (is_graph) {
    uint32_t R = P.getOptionIntValue("-R", 200);
    uint32_t L_build = P.getOptionIntValue("-L_build", 600);
    double alpha = P.getOptionDoubleValue("-a", 1.2);
    int num_pass = P.getOptionIntValue("-np", 1);
    IndexParams ip = IndexParams::svh_graph(R, L_build, alpha, num_pass, io.compress_input,
        io.verbose, qa.pq_method, qa.block_size, qa.num_clusters_per_block,
        qa.num_points_per_cluster, qa.rabitq_bits);

    std::string L_str = P.getOptionValue("-L", "16,32,64,128,256,512,1024,2048");
    auto L_list = bench::parse_csv_ints(L_str);

    dispatch_svh_graph<metric>(quant_method, [&]<class IndexT>() {
      IndexT index(points.get_dims(), ip);
      bench::build_or_load(index, points, io.index_path);

      if (ds.queries.empty() || ds.gt.empty()) return;
      auto queries = PC(ds.queries.c_str());
      auto gt = ReadGT(ds.gt, queries.size());
      bench::print_header("SVH_Graph (search_all)", ds.name, points.size(), queries.size());
      bench::print_compression_stats<ChPoint>(queries, sp_base, index.quantization_mode);

      bench::run_search_all_sweep(
          index, points, queries, gt, "L", L_list,
          [&](size_t L) {
            SearchParams sp = SearchParams::svh_graph(k, L, num_rerank, cut, no_rerank);
            sp.query_compression = sp_base.query_compression;
            sp.query_compression_threshold = sp_base.query_compression_threshold;
            sp.compress_rerank = sp_base.compress_rerank;
            sp.query_alignment = sp_base.query_alignment;
            sp.query_alignment_strict = sp_base.query_alignment_strict;
            return sp;
          },
          io.csv_path);
    });
  } else {
    uint32_t k_per_level = P.getOptionIntValue("-k_per_level", 0);
    uint32_t max_leaf_size = P.getOptionIntValue("-max_leaf_size", 500);
    uint32_t max_ppc = P.getOptionIntValue("-max_points_per_centroid", 100);
    // Factory default for quantize_centers = true; use -qc 0 to disable.
    bool compress_centers = P.getOptionIntValue("-qc", 1) != 0;
    IndexParams ip = IndexParams::svh_ivf(k_per_level, max_leaf_size, io.compress_input,
        io.verbose, max_ppc, qa.pq_method, qa.block_size, qa.num_clusters_per_block,
        qa.num_points_per_cluster, qa.rabitq_bits, compress_centers);

    std::string np_str = P.getOptionValue("-nprobes", "1,2,4,8,16,32,64,128,256,512,1024");
    auto np_list = bench::parse_csv_ints(np_str);

    dispatch_svh_ivf<metric>(compress_centers, quant_method, [&]<class IndexT>() {
      IndexT index(points.get_dims(), ip);
      bench::build_or_load(index, points, io.index_path);

      if (ds.queries.empty() || ds.gt.empty()) return;
      auto queries = PC(ds.queries.c_str());
      auto gt = ReadGT(ds.gt, queries.size());
      bench::print_header("SVH_IVF (search_all)", ds.name, points.size(), queries.size());
      bench::print_compression_stats<ChPoint>(queries, sp_base, index.quantization_mode);

      bench::run_search_all_sweep(
          index, points, queries, gt, "nprobes", np_list,
          [&](size_t np) {
            SearchParams sp = SearchParams::svh_ivf(k, np, num_rerank, no_rerank);
            sp.query_compression = sp_base.query_compression;
            sp.query_compression_threshold = sp_base.query_compression_threshold;
            sp.compress_rerank = sp_base.compress_rerank;
            sp.query_alignment = sp_base.query_alignment;
            sp.query_alignment_strict = sp_base.query_alignment_strict;
            return sp;
          },
          io.csv_path);
    });
  }
}

PARSE_DIST_FUNC_AND_RUN(run,
    "SVH batched many-to-many search benchmark (search_all path).\n\n"
    "Dataset / I/O:\n"
    "  -d <name> | -i <points> -q <queries> -gt <gt>\n"
    "  -index <path>  -o <save_path>  -csv <path>\n"
    "  -mm  -v <level>  -compress_input  -dist_func IP|L2\n\n"
    "SVH variant:\n"
    "  -graph                         IndexSVHGraph\n"
    "  (no flag)                      IndexSVHIVF\n\n"
    "SVHGraph build params (defaults mirror IndexParams::svh_graph):\n"
    "  -R <N> (200)   -L_build <N> (600)   -a <f> (1.2)   -np <N> (1)\n\n"
    "SVHIVF build params (defaults mirror IndexParams::svh_ivf):\n"
    "  -k_per_level <N> (0)   -max_leaf_size <N> (500)\n"
    "  -max_points_per_centroid <N> (100)\n"
    "  -qc 0|1 (1)                    CompressCenters template (TQ-quantized centers)\n\n"
    "Leaf quantization (selects the templated IndexSVHGraph/IVF<metric, ..., LeafModel>):\n"
    "  -quant_method None|PQ|FS|RQ|TQ|SPQTQ    (default None)\n"
    "  -m <N> (8)   -num_clusters_per_block <N> (16)\n"
    "  -num_points_per_cluster <N> (100)   -rbits <N> (4)\n\n"
    "Search sweep:\n"
    "  -k <N>  -L <csv>  -nprobes <csv>  -num_rerank <N>\n\n"
    "Query compression:\n"
    "  -compress none|carve|wards  -compress_threshold <tau>  -compress_rerank\n")
