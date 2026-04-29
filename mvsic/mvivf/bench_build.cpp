#include "mvsic/core/bench_utils.h"
#include "mvivf.h"
#include "mvivf_flat.h"
#include "mvivf_spill.h"

using namespace mvsic;

// ----- Skeleton-only dispatch.  Build always uses the raw skeleton variant -----
// (CompressCenters=false, LeafModel=NoQuantizer); any quantized variant can load
// that skeleton later and retrain its quantizer on load.  We therefore do not
// expose -qc / -quant_method here, since the on-disk skeleton is variant
// agnostic.
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

template <bool metric, class Fn>
void dispatch_skeleton(MVIVFVariant v, Fn&& fn) {
  if (v == MVIVFVariant::Flat) {
    fn.template operator()<IndexMVIVFFlat<metric, false, NoQuantizer<metric>>>();
  } else if (v == MVIVFVariant::Spill) {
    fn.template operator()<IndexMVIVFSpill<metric, false, NoQuantizer<metric>>>();
  } else {
    fn.template operator()<IndexMVIVF<metric, false, NoQuantizer<metric>>>();
  }
}
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

  // Defaults below mirror IndexParams::mvivf / mvivf_flat / mvivf_spill in
  // mvsic/core/index_params.h.  -wgh_kmeans uses the integer 0|1 form so we
  // can preserve the factory default of true.
  uint32_t k_per_level = P.getOptionIntValue("-k_per_level", 0);
  uint32_t max_leaf_size = P.getOptionIntValue("-max_leaf_size", 500);
  uint32_t max_depth = P.getOptionIntValue("-max_depth", 0);
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

  dispatch_skeleton<metric>(variant,
      [&]<class IndexT>() {
        IndexT index(points.get_dims(), ip);
        std::cout << "Building skeleton (" << variant_name(variant)
                  << ", raw centers, raw leaves) ..." << std::endl;
        parlay::internal::timer t;
        t.start();
        index.build(points);
        t.stop();
        std::cout << "Index built in " << t.total_time() << " seconds." << std::endl;
        if (!io.save_path.empty()) {
          std::cout << "Saving skeleton to " << io.save_path << " ..." << std::endl;
          index.save(io.save_path);
          std::cout << "Skeleton saved." << std::endl;
        }
      });
}

PARSE_DIST_FUNC_AND_RUN(run,
    "MVIVF skeleton build benchmark. Builds the raw skeleton (no quantization)\n"
    "for IndexMVIVF / IndexMVIVFFlat / IndexMVIVFSpill.  Quantized variants are\n"
    "instantiated on load by the search benches; the on-disk file is variant\n"
    "agnostic.\n\n"
    "Dataset / I/O:\n"
    "  -d <name>                      Short dataset name (e.g. arguana, nq500k)\n"
    "  -i <points>                    Points .pcs file (alternative to -d)\n"
    "  -o <save_path>                 Where to save the built skeleton\n"
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
    "  -s <N>                         Seeding strategy index (default 0 = Random)\n")
