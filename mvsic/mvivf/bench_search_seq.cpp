#include "mvsic/core/bench_utils.h"
#include "mvsic/mvivf/bench_search_seq_inst.h"

using namespace mvsic;
using namespace mvsic::bench_seq;

// ----- CLI -> compile-time IndexMVIVF<metric, CompressCenters, LeafModel> dispatch. -----
//
// All heavy template specializations of run_one_seq are instantiated in
// separate translation units under bench_search_seq_inst/ — one .cc per
// (metric, variant, quantizer) cell, each instantiating both compress=false
// and compress=true. Bazel compiles those TUs in parallel; this dispatch TU
// only sees `extern template` declarations and stays cheap.
namespace {

inline MVIVFVariant parse_mvivf_variant(bool is_flat, bool is_spill) {
  if (is_flat && is_spill) {
    std::cerr << "-flat and -spill are mutually exclusive." << std::endl;
    std::exit(1);
  }
  return is_flat ? MVIVFVariant::Flat
       : is_spill ? MVIVFVariant::Spill
                  : MVIVFVariant::Regular;
}

#define MVIVF_DISPATCH_LM(Fam, metric, C, qm, fn)                                                 \
  do {                                                                                            \
    if      (qm == "None"  || qm == "none")                                                       \
      fn.template operator()<Fam<metric, C, NoQuantizer<metric>>>();                              \
    else if (qm == "PQ"    || qm == "pq")                                                         \
      fn.template operator()<Fam<metric, C, pq_mv::Model<metric>>>();                             \
    else if (qm == "FS"    || qm == "fs")                                                         \
      fn.template operator()<Fam<metric, C, fastscan_mv::Model<metric>>>();                       \
    else if (qm == "RQ"    || qm == "rq")                                                         \
      fn.template operator()<Fam<metric, C, rabitq_mv::Model<metric>>>();                         \
    else if (qm == "TQ"    || qm == "tq")                                                         \
      fn.template operator()<Fam<metric, C, turboquant_mv::Model<metric>>>();                     \
    else if (qm == "SPQTQ" || qm == "spqtq")                                                      \
      fn.template operator()<Fam<metric, C, pqtq_mv::Model<metric>>>();                           \
    else if (qm == "1BTQ"  || qm == "1btq")                                                       \
      fn.template operator()<Fam<metric, C, turboquant_1bit_mv::Model<metric>>>();                \
    else if (qm == "1BTQA" || qm == "1btqa")                                                      \
      fn.template operator()<Fam<metric, C, turboquant_1bit_asym_mv::Model<metric>>>();           \
    else if (qm == "8BTQ"  || qm == "8btq")                                                       \
      fn.template operator()<Fam<metric, C, turboquant_8bit_mv::Model<metric>>>();                \
    else {                                                                                        \
      std::cerr << "Unknown -quant_method: " << qm                                                \
                << " (use None, PQ, FS, RQ, TQ, SPQTQ, 1BTQ, 1BTQA, 8BTQ)" << std::endl;          \
      std::exit(1);                                                                               \
    }                                                                                             \
  } while (0)

template <bool metric, class Fn>
void dispatch_mvivf(MVIVFVariant v, bool compress, const std::string& qm, Fn&& fn) {
  if (v == MVIVFVariant::Flat) {
    if (compress) MVIVF_DISPATCH_LM(IndexMVIVFFlat,  metric, true,  qm, fn);
    else          MVIVF_DISPATCH_LM(IndexMVIVFFlat,  metric, false, qm, fn);
  } else if (v == MVIVFVariant::Spill) {
    if (compress) MVIVF_DISPATCH_LM(IndexMVIVFSpill, metric, true,  qm, fn);
    else          MVIVF_DISPATCH_LM(IndexMVIVFSpill, metric, false, qm, fn);
  } else {
    if (compress) MVIVF_DISPATCH_LM(IndexMVIVF,      metric, true,  qm, fn);
    else          MVIVF_DISPATCH_LM(IndexMVIVF,      metric, false, qm, fn);
  }
}
#undef MVIVF_DISPATCH_LM
}  // namespace

template <typename ChPoint, bool metric>
void run(commandLine& P) {
  Eigen::setNbThreads(1);
  using PC = PointCloudSet<ChPoint>;

  auto ds = bench::parse_dataset(P);
  if (ds.points.empty()) {
    std::cerr << "Error: specify -d <dataset>" << std::endl;
    std::exit(1);
  }

  auto io = bench::parse_io_args(P);
  auto qa = bench::parse_quant_args(P, "None");
  std::string quant_method = P.getOptionValue("-quant_method", "None");

  uint32_t k_per_level = P.getOptionIntValue("-k_per_level", 0);
  uint32_t max_leaf_size = P.getOptionIntValue("-max_leaf_size", 500);
  uint32_t max_depth = P.getOptionIntValue("-max_depth", 0);
  bool compress_centers = P.getOptionIntValue("-qc", 0) != 0;
  uint32_t niters = P.getOptionIntValue("-niters", 5);
  uint32_t mpcc = P.getOptionIntValue("-mpcc", 100);
  uint32_t mpcik = P.getOptionIntValue("-mpcik", 20);
  bool wgh_kmeans = P.getOptionIntValue("-wgh_kmeans", 1) != 0;
  uint32_t s = P.getOptionIntValue("-s", 0);
  bool is_flat = P.getOption("-flat");
  int num_spill = P.getOptionIntValue("-spill", 0);
  int num_spill_l2 = P.getOptionIntValue("-spill_l2", 1);
  bool is_spill = (num_spill > 0);
  auto variant = parse_mvivf_variant(is_flat, is_spill);

  size_t k = P.getOptionLongValue("-k", 10);
  size_t num_rerank = P.getOptionLongValue("-num_rerank", 0);
  size_t num_queries = P.getOptionLongValue("-num_queries", 0);
  uint64_t query_sample_seed = static_cast<uint64_t>(
      P.getOptionLongValue("-query_sample_seed", 42));
  bool skip_warmup = P.getOption("-no_warmup");
  std::string nprobes_str = P.getOptionValue("-nprobes", "1,2,4,8,16,32,64,128,256,512,1024");
  auto nprobes_list = bench::parse_csv_ints(nprobes_str);

  SearchParams sp_base;
  bench::parse_compression_opts(sp_base, P);
  // Override parse_compression_opts's presence-flag parse of -tq8_rerank with
  // an int-style 0|1 so the default stays on (matching SearchParams::mvivf's
  // default) and -tq8_rerank 0 can disable it.
  sp_base.tq8_rerank = P.getOptionIntValue("-tq8_rerank", 1) != 0;
  bench::print_compression_info(sp_base);

  auto points = PC(ds.points.c_str(), io.is_mmap);

  IndexParams ip;
  if (variant == MVIVFVariant::Flat) {
    ip = IndexParams::mvivf_flat(k_per_level, io.compress_input, io.verbose, niters, mpcc, mpcik,
                                 "Random", 0, wgh_kmeans, s);
  } else if (variant == MVIVFVariant::Spill) {
    ip = IndexParams::mvivf_spill(k_per_level, max_leaf_size, num_spill, io.compress_input,
                                  io.verbose, niters, mpcc, mpcik, "Random", 0, wgh_kmeans, s,
                                  max_depth, static_cast<uint32_t>(num_spill_l2));
  } else {
    ip = IndexParams::mvivf(k_per_level, max_leaf_size, io.compress_input, io.verbose, niters,
                            mpcc, mpcik, "Random", 0, wgh_kmeans, s, max_depth);
  }

  RunOneCtxSeq<ChPoint> ctx{
      .points = &points,
      .ip = ip,
      .sp_base = sp_base,
      .ds = ds,
      .io = io,
      .variant = variant,
      .k = k,
      .num_rerank = num_rerank,
      .nprobes_list = nprobes_list,
      .num_queries = num_queries,
      .query_sample_seed = query_sample_seed,
      .skip_warmup = skip_warmup,
  };

  dispatch_mvivf<metric>(variant, compress_centers, quant_method,
      [&]<class IndexT>() { run_one_seq<ChPoint, metric, IndexT>(ctx); });
}

PARSE_DIST_FUNC_AND_RUN(run,
    "MVIVF sequential search benchmark (search_with_stats path; per-query "
    "loop pinned to 1 worker via parlay::execute_with_scheduler). "
    "Index loading runs with full parlay parallelism.\n\n"
    "Dataset / I/O:\n"
    "  -d <name> | -i <points> -q <queries> -gt <gt>\n"
    "  -index <path>                  Pre-built index to load\n"
    "  -csv <path>                    Append per-sweep-point result rows to CSV\n"
    "  -mm  -v <level>  -compress_input  -dist_func IP|L2\n\n"
    "MVIVF variant:\n"
    "  -flat                          IndexMVIVFFlat\n"
    "  -spill <N>                     IndexMVIVFSpill\n"
    "  (no flag)                      IndexMVIVF\n\n"
    "Clustering structure (only used if no -index; defaults mirror IndexParams::mvivf*):\n"
    "  -k_per_level <N> (0)   -max_leaf_size <N> (500)   -max_depth <N> (0)\n"
    "  -niters <N> (5)   -mpcc <N> (100)   -mpcik <N> (20)\n"
    "  -wgh_kmeans 0|1 (1)   -s <N> (0)\n\n"
    "Quantization (picks the concrete templated class):\n"
    "  -qc 0|1                        CompressCenters; default 0\n"
    "  -quant_method None|PQ|FS|RQ|TQ|SPQTQ|1BTQ|1BTQA|8BTQ    (default None)\n"
    "  -m <N> (8)   -num_clusters_per_block <N> (16)\n"
    "  -num_points_per_cluster <N> (100)   -rbits <N> (4)\n\n"
    "Search sweep:\n"
    "  -k <N>                         Top-k to retrieve (default 10)\n"
    "  -nprobes <csv>                 nprobes values to sweep\n"
    "  -num_rerank <N>                Rerank width\n"
    "  -num_queries <N>               If <N> < #queries, randomly subsample (0 = use all)\n"
    "  -query_sample_seed <N>         RNG seed for -num_queries subsampling (default 42)\n"
    "  -no_warmup                     Skip the sweep-level warmup pass\n\n"
    "Query compression:\n"
    "  -compress none|carve|wards  -compress_threshold <tau>  -compress_rerank\n\n"
    "Rerank:\n"
    "  -tq8_rerank 0|1                Score rerank candidates with 8-bit TQ kernel\n"
    "                                 instead of float chamfer (default 1)\n")
