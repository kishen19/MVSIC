#include "mvsic/core/bench_utils.h"
#include "mvivf.h"
#include "mvivf_flat.h"
#include "mvivf_spill.h"

using namespace mvsic;

template <typename ChPoint, bool metric>
void run(commandLine& P) {
  Eigen::setNbThreads(1);
  using PC = PointCloudSet<ChPoint>;

  auto ds = bench::parse_dataset(P);
  if (ds.points.empty()) { std::cerr << "Error: specify -d <dataset>" << std::endl; std::exit(1); }

  auto io = bench::parse_io_args(P);
  auto qa = bench::parse_quant_args(P, "None");

  uint32_t k_per_level = P.getOptionIntValue("-k_per_level", 0);
  uint32_t max_leaf_size = P.getOptionIntValue("-max_leaf_size", 500);
  bool quantize_centers = P.getOption("-qc");
  uint32_t niters = P.getOptionIntValue("-niters", 5);
  uint32_t mpcc = P.getOptionIntValue("-mpcc", 0);
  uint32_t mpcik = P.getOptionIntValue("-mpcik", 20);
  bool wgh_kmeans = P.getOption("-wgh_kmeans");
  uint32_t s = P.getOptionIntValue("-s", 0);
  bool is_flat = P.getOption("-flat");
  int num_spill = P.getOptionIntValue("-spill", 0);
  bool is_spill = (num_spill > 0);

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

  if (is_flat) {
    IndexParams ip = IndexParams::mvivf_flat(k_per_level, io.compress_input, io.verbose, niters,
        mpcc, mpcik, "Random", 0, wgh_kmeans, s, qa.pq_method, qa.block_size,
        qa.num_clusters_per_block, qa.num_points_per_cluster, qa.rabitq_bits, quantize_centers);
    IndexMVIVFFlat<metric> index(points.get_dims(), ip);
    do_build(index);
  } else if (is_spill) {
    IndexParams ip = IndexParams::mvivf_spill(k_per_level, max_leaf_size, num_spill,
        io.compress_input, io.verbose, niters, mpcc, mpcik, "Random", 0, wgh_kmeans, s,
        qa.pq_method, qa.block_size, qa.num_clusters_per_block, qa.num_points_per_cluster,
        qa.rabitq_bits, quantize_centers);
    IndexMVIVFSpill<metric> index(points.get_dims(), ip);
    do_build(index);
  } else {
    IndexParams ip = IndexParams::mvivf(k_per_level, max_leaf_size, io.compress_input, io.verbose,
        niters, mpcc, mpcik, "Random", 0, wgh_kmeans, s, qa.pq_method, qa.block_size,
        qa.num_clusters_per_block, qa.num_points_per_cluster, qa.rabitq_bits, quantize_centers);
    IndexMVIVF<metric> index(points.get_dims(), ip);
    do_build(index);
  }
}

PARSE_DIST_FUNC_AND_RUN(run, "-d <dataset> -o <save_path> [-flat] [-spill N] ...")
