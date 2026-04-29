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

  auto points = PC(ds.points.c_str(), io.is_mmap);
  IndexParams ip = IndexParams::vamana(R, L_build, alpha, two_pass, io.compress_input, io.verbose,
                                        qa.pq_method, qa.block_size, qa.num_clusters_per_block,
                                        qa.num_points_per_cluster, qa.rabitq_bits);
  IndexVamana<metric> index(points.get_dims(), ip);

  std::cout << "Building index..." << std::endl;
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
}

PARSE_DIST_FUNC_AND_RUN(run,
    "Vamana build-only benchmark. Builds IndexVamana<metric> and optionally saves it.\n\n"
    "Dataset / I/O:\n"
    "  -d <name> | -i <points>\n"
    "  -o <save_path>                 Where to save the built index\n"
    "  -mm                            Memory-map the points file\n"
    "  -v <level>                     Verbosity (0..3)\n"
    "  -compress_input                Apply point-cloud input compression\n"
    "  -dist_func IP|L2               Distance metric (default IP)\n\n"
    "Vamana build parameters (defaults mirror IndexParams::vamana):\n"
    "  -R <N>                         Graph out-degree (default 200)\n"
    "  -L_build <N>                   Beam width during build (default 600)\n"
    "  -a <f>                         alpha / prune threshold (default 1.2)\n"
    "  -tp                            Two-pass build (default off)\n\n"
    "Leaf quantization (passed through IndexParams):\n"
    "  -quant_method None|PQ|FS|RQ|TQ           Quantizer family (default None)\n"
    "  -m <N>                         Block size (default 8)\n"
    "  -num_clusters_per_block <N>    Codebook size (default 16)\n"
    "  -num_points_per_cluster <N>    K-means points per centroid (default 100)\n"
    "  -rbits <N>                     RaBitQ bit-width (default 4)\n")
