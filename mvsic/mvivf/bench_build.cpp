#include "mvsic/core/bench_utils.h"
#include "mvivf.h"
#include "mvivf_flat.h"
#include "mvivf_spill.h"

using namespace mvsic;

// ----- CLI -> compile-time IndexMVIVF<metric, CompressCenters, LeafModel> dispatch. -----
namespace {
enum class MVIVFVariant { Regular, Flat, Spill };

inline MVIVFVariant parse_mvivf_variant(bool is_flat, bool is_spill) {
  if (is_flat && is_spill) {
    std::cerr << "-flat and -spill are mutually exclusive." << std::endl;
    std::exit(1);
  }
  return is_flat ? MVIVFVariant::Flat
       : is_spill ? MVIVFVariant::Spill
                  : MVIVFVariant::Regular;
}

inline const char* variant_name(MVIVFVariant v) {
  return v == MVIVFVariant::Flat  ? "MVIVF_Flat"
       : v == MVIVFVariant::Spill ? "MVIVF_Spill"
                                  : "MVIVF";
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
    else {                                                                                        \
      std::cerr << "Unknown -quant_method: " << qm                                                \
                << " (use None, PQ, FS, RQ, TQ, SPQTQ, 1BTQ)" << std::endl;                       \
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

  // Defaults below mirror IndexParams::mvivf / mvivf_flat / mvivf_spill in
  // mvsic/core/index_params.h.  Integer-valued bool flags (-qc, -wgh_kmeans)
  // use `<flag> 0|1` form so that we can preserve factory defaults of `true`.
  uint32_t k_per_level = P.getOptionIntValue("-k_per_level", 0);
  uint32_t max_leaf_size = P.getOptionIntValue("-max_leaf_size", 500);
  uint32_t max_depth = P.getOptionIntValue("-max_depth", 0);
  // -qc selects the CompressCenters template variant (TQ-quantized centers).
  // Factory default = false (raw float centers).  Use -qc 1 to enable.
  bool compress_centers = P.getOptionIntValue("-qc", 0) != 0;
  uint32_t niters = P.getOptionIntValue("-niters", 5);
  uint32_t mpcc = P.getOptionIntValue("-mpcc", 100);
  uint32_t mpcik = P.getOptionIntValue("-mpcik", 20);
  // Factory default for use_weighted_inner_kmeans = true.  Use -wgh_kmeans 0
  // to disable.
  bool wgh_kmeans = P.getOptionIntValue("-wgh_kmeans", 1) != 0;
  uint32_t s = P.getOptionIntValue("-s", 0);
  bool is_flat = P.getOption("-flat");
  int num_spill = P.getOptionIntValue("-spill", 0);
  int num_spill_l2 = P.getOptionIntValue("-spill_l2", 1);
  bool is_spill = (num_spill > 0);
  auto variant = parse_mvivf_variant(is_flat, is_spill);

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

  dispatch_mvivf<metric>(variant, compress_centers, quant_method,
      [&]<class IndexT>() {
        IndexT index(points.get_dims(), ip);
        std::cout << "Building index (" << variant_name(variant)
                  << ", compress_centers=" << (compress_centers ? 1 : 0)
                  << ", leaf_quantizer=" << quant_method << ")..." << std::endl;
        parlay::internal::timer t;
        t.start();
        index.build(points);
        t.stop();
        std::cout << "Index built in " << t.total_time() << " seconds." << std::endl;
        if (!io.save_path.empty()) {
          std::cout << "Saving index to " << io.save_path << " ..." << std::endl;
          index.save(io.save_path);
          std::cout << "Index saved." << std::endl;
        }
      });
}

PARSE_DIST_FUNC_AND_RUN(run,
    "MVIVF build-only benchmark. Builds IndexMVIVF / IndexMVIVFFlat / IndexMVIVFSpill.\n\n"
    "Dataset / I/O:\n"
    "  -d <name>                      Short dataset name (e.g. arguana, nq500k)\n"
    "  -i <points>                    Points .pcs file (alternative to -d)\n"
    "  -o <save_path>                 Where to save the built index\n"
    "  -mm                            Memory-map the points file\n"
    "  -v <level>                     Verbosity (0..3)\n"
    "  -compress_input                Apply point-cloud input compression\n"
    "  -dist_func IP|L2               Distance metric (default IP)\n\n"
    "MVIVF variant:\n"
    "  -flat                          Use IndexMVIVFFlat (root-level clustering)\n"
    "  -spill <a>                     Use IndexMVIVFSpill with top-a spill at root\n"
    "  -spill_l2 <b>                  Second-level spill factor (default 1; no spill)\n"
    "  (no flag)                      Default hierarchical IndexMVIVF\n\n"
    "Clustering structure (defaults mirror IndexParams::mvivf*):\n"
    "  -k_per_level <N>               Branching factor per level (default 0 = 4*sqrt(n))\n"
    "  -max_leaf_size <N>             Stop splitting when cluster <= this (default 500)\n"
    "  -max_depth <N>                 Cap recursion depth (default 0 = unlimited)\n"
    "  -niters <N>                    K-means iterations (default 5)\n"
    "  -mpcc <N>                      max_point_clouds_per_cluster (default 100)\n"
    "  -mpcik <N>                     max_points_per_centroid_inner_kmeans (default 20)\n"
    "  -wgh_kmeans 0|1                Use weighted inner k-means (default 1)\n"
    "  -s <N>                         Seeding strategy index (default 0 = Random)\n\n"
    "Quantization (selects concrete templated class at compile time):\n"
    "  -qc 0|1                        CompressCenters (TQ-quantize internal centers); default 0\n"
    "  -quant_method None|PQ|FS|RQ|TQ|SPQTQ|1BTQ    Leaf quantizer (default None)\n"
    "  -m <N>                         Block size for PQ/FS (default 8)\n"
    "  -num_clusters_per_block <N>    Codebook size per block (default 16)\n"
    "  -num_points_per_cluster <N>    K-means points per centroid (default 100)\n"
    "  -rbits <N>                     RaBitQ bit-width (default 4)\n")
