#include "mvsic/core/bench_utils.h"
#include "vamana.h"

using namespace mvsic;

// ----- Skeleton-only dispatch.  Build always uses the raw skeleton variant -----
// (LeafModel=NoQuantizer); any quantized variant can load that skeleton later
// and retrain its quantizer on load.  We therefore do not expose -quant_method
// here, since the on-disk skeleton is variant agnostic.
namespace {
template <bool metric, class Fn>
void dispatch_skeleton(Fn&& fn) {
  fn.template operator()<IndexVamana<metric, NoQuantizer<metric>>>();
}
}  // namespace

template <typename ChPoint, bool metric>
void run(commandLine& P) {
  Eigen::setNbThreads(1);
  using PC = PointCloudSet<ChPoint>;

  auto ds = bench::parse_dataset(P);
  if (ds.points.empty()) { std::cerr << "Error: specify -d <dataset>" << std::endl; std::exit(1); }

  auto io = bench::parse_io_args(P);

  uint32_t R = P.getOptionIntValue("-R", 200);
  uint32_t L_build = P.getOptionIntValue("-L_build", 600);
  double alpha = P.getOptionDoubleValue("-a", 1.2);
  bool two_pass = P.getOption("-tp");

  auto points = PC(ds.points.c_str(), io.is_mmap);
  IndexParams ip = IndexParams::vamana(R, L_build, alpha, two_pass, io.compress_input, io.verbose,
                                        /*pq_method=*/0, /*block_size=*/8,
                                        /*num_clusters_per_block=*/16,
                                        /*num_points_per_cluster=*/100, /*rabitq_bits=*/4);

  dispatch_skeleton<metric>(
      [&]<class IndexT>() {
        IndexT index(points.get_dims(), ip);
        std::cout << "Building Vamana skeleton (graph + start_point, no leaf quantization)..."
                  << std::endl;
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
    "Vamana skeleton build benchmark. Builds the raw skeleton (no quantization)\n"
    "for IndexVamana<metric>.  Quantized variants are instantiated on load by\n"
    "the search benches; the on-disk file is variant-agnostic.\n\n"
    "Dataset / I/O:\n"
    "  -d <name> | -i <points>\n"
    "  -o <save_path>                 Where to save the built skeleton\n"
    "  -mm                            Memory-map the points file\n"
    "  -v <level>                     Verbosity (0..3)\n"
    "  -compress_input                Apply point-cloud input compression\n"
    "  -dist_func IP|L2               Distance metric (default IP)\n\n"
    "Vamana build parameters (defaults mirror IndexParams::vamana):\n"
    "  -R <N>                         Graph out-degree (default 200)\n"
    "  -L_build <N>                   Beam width during build (default 600)\n"
    "  -a <f>                         alpha / prune threshold (default 1.2)\n"
    "  -tp                            Two-pass build (default off)\n")
