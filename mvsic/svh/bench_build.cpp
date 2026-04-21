#include "mvsic/core/bench_utils.h"
#include "svh_graph.h"
#include "svh_ivf.h"

using namespace mvsic;

template <typename ChPoint, bool metric>
void run(commandLine& P) {
  Eigen::setNbThreads(1);
  using PC = PointCloudSet<ChPoint>;

  auto ds = bench::parse_dataset(P);
  if (ds.points.empty()) { std::cerr << "Error: specify -d <dataset>" << std::endl; std::exit(1); }

  auto io = bench::parse_io_args(P);
  auto qa = bench::parse_quant_args(P, "None");
  bool is_graph = P.getOption("-graph");

  auto points = PC(ds.points.c_str(), io.is_mmap);

  auto do_build = [&](auto& index) {
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
  };

  if (is_graph) {
    uint32_t R = P.getOptionIntValue("-R", 200);
    uint32_t L_build = P.getOptionIntValue("-L_build", 600);
    double alpha = P.getOptionDoubleValue("-a", 1.2);
    int num_pass = P.getOptionIntValue("-np", 1);
    IndexParams ip = IndexParams::svh_graph(R, L_build, alpha, num_pass, io.compress_input,
        io.verbose, qa.pq_method, qa.block_size, qa.num_clusters_per_block,
        qa.num_points_per_cluster, qa.rabitq_bits);
    IndexSVHGraph<metric> index(points.get_dims(), ip);
    do_build(index);
  } else {
    uint32_t k_per_level = P.getOptionIntValue("-k_per_level", 0);
    uint32_t max_leaf_size = P.getOptionIntValue("-max_leaf_size", 500);
    uint32_t max_ppc = P.getOptionIntValue("-max_points_per_centroid", 100);
    bool quantize_centers = P.getOption("-qc");
    IndexParams ip = IndexParams::svh_ivf(k_per_level, max_leaf_size, io.compress_input,
        io.verbose, max_ppc, qa.pq_method, qa.block_size, qa.num_clusters_per_block,
        qa.num_points_per_cluster, qa.rabitq_bits, quantize_centers);
    IndexSVHIVF<metric> index(points.get_dims(), ip);
    do_build(index);
  }
}

PARSE_DIST_FUNC_AND_RUN(run, "-d <dataset> -o <save_path> [-graph] ...")
