#include "mvsic/core/bench_utils.h"
#include "muvera.h"

using namespace mvsic;

template <typename ChPoint, bool metric>
void run(commandLine& P) {
  Eigen::setNbThreads(1);
  using PC = PointCloudSet<ChPoint>;

  auto ds = bench::parse_dataset(P);
  if (ds.points.empty()) { std::cerr << "Error: specify -d <dataset>" << std::endl; std::exit(1); }

  auto io = bench::parse_io_args(P);
  auto qa = bench::parse_quant_args(P, "None");

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
      qa.pq_method, qa.block_size, qa.num_clusters_per_block,
      qa.num_points_per_cluster, qa.rabitq_bits);
  IndexMUVERA<metric> index(points.get_dims(), ip);

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
    "MUVERA build-only benchmark. Builds IndexMUVERA<metric> (MUVERA over Vamana).\n\n"
    "Dataset / I/O:\n"
    "  -d <name> | -i <points>\n"
    "  -o <save_path>                 Where to save the built index\n"
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
    "  -R <N> (200)   -L_build <N> (600)   -a <f> (1.1)   -np <N> (1)\n\n"
    "Leaf quantization (passed via IndexParams):\n"
    "  -quant_method None|PQ|FS|RQ|TQ    (default None)\n"
    "  -m <N> (8)   -num_clusters_per_block <N> (16)\n"
    "  -num_points_per_cluster <N> (100)   -rbits <N> (4)\n")
