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

PARSE_DIST_FUNC_AND_RUN(run, "-d <dataset> -o <save_path> [-R N] [-L_build N] ...")
