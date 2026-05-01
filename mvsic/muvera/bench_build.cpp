#include "mvsic/core/bench_utils.h"
#include "muvera.h"

using namespace mvsic;

// ----- Skeleton-only dispatch.  Build always uses the raw skeleton variant -----
// (LeafModel=NoQuantizer); any quantized variant can load that skeleton later
// and retrain its quantizer on load.  We therefore do not expose -quant_method
// here, since the on-disk skeleton is variant agnostic.
namespace {
template <bool metric, class Fn>
void dispatch_skeleton(Fn&& fn) {
  fn.template operator()<IndexMUVERA<metric, NoQuantizer<metric>>>();
}
}  // namespace

template <typename ChPoint, bool metric>
void run(commandLine& P) {
  Eigen::setNbThreads(1);
  using PC = PointCloudSet<ChPoint>;

  auto ds = bench::parse_dataset(P);
  if (ds.points.empty()) { std::cerr << "Error: specify -d <dataset>" << std::endl; std::exit(1); }

  auto io = bench::parse_io_args(P);

  // Defaults mirror IndexParams::muvera_custom in mvsic/core/index_params.h.
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

  auto points = PC(ds.points.c_str(), io.is_mmap);
  IndexParams ip = IndexParams::muvera_custom(
      num_reps, num_simhash, 1, projd, fill_empty, final_projd, !no_norm,
      R, L_build, alpha, num_pass, io.compress_input, io.verbose,
      /*pq_method=*/0, /*block_size=*/8, /*num_clusters_per_block=*/16,
      /*num_points_per_cluster=*/100, /*rabitq_bits=*/4);

  dispatch_skeleton<metric>(
      [&]<class IndexT>() {
        IndexT index(points.get_dims(), ip);
        std::cout << "Building MUVERA skeleton (raw FDEs, no leaf quantization)..." << std::endl;
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
    "MUVERA skeleton build benchmark. Builds the raw skeleton (no quantization)\n"
    "for IndexMUVERA<metric>.  Quantized variants are instantiated on load by\n"
    "the search benches; the on-disk file is variant-agnostic.\n\n"
    "Dataset / I/O:\n"
    "  -d <name> | -i <points>\n"
    "  -o <save_path>                 Where to save the built skeleton\n"
    "  -mm                            Memory-map the points file\n"
    "  -v <level>                     Verbosity (0..3)\n"
    "  -compress_input                Apply point-cloud input compression\n"
    "  -dist_func IP|L2               Distance metric (default IP)\n\n"
    "MUVERA FDE parameters (defaults mirror IndexParams::muvera_custom):\n"
    "  -num_reps <N>                  Number of repetitions (default 20)\n"
    "  -num_simhash <N>               SimHash bits per partition (default 4)\n"
    "  -projd <N>                     Projected dim per partition (default 8)\n"
    "  -fill_empty_partitions         Fill empty partitions with nearest (default off)\n"
    "  -final_projd <N>               Final projection dim (default 0 = off)\n"
    "  -no_norm                       Disable FDE normalization (default: normalized)\n\n"
    "Underlying Vamana params:\n"
    "  -R <N> (200)   -L_build <N> (600)   -a <f> (1.1)   -np <N> (1)\n")
