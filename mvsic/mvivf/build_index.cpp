// Parallel MVIVF index builder.
//
// Builds an IndexMVIVF<false> (IP, ChamferIP_Point) with the supplied
// parameters, then saves it to <out_dir>/<name>.bin where <name> encodes
// the build-time parameters. A sidecar <name>.params text file records the
// same parameters so bench_seq can verify that a loaded index matches the
// command-line flags it was launched with.
#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <sstream>

#include "mvivf.h"
#include "mvsic/core/types/chamfer_ip_point.h"
#include "mvsic/core/types/point_cloud_set.h"
#include "mvsic/core/utils/parse_command_line.h"

using namespace mvsic;

// Map -quant_method string -> (enum_int, short_name). Keep in sync with
// bench_seq.cpp and IndexParams::QuantizerType.
static bool parse_quant_method(const std::string& s, uint32_t& out) {
  if (s == "None")  { out = 0; return true; }
  if (s == "PQ")    { out = 1; return true; }
  if (s == "RQ")    { out = 2; return true; }
  if (s == "FS")    { out = 3; return true; }
  if (s == "TQ")    { out = 4; return true; }
  if (s == "SPQTQ") { out = 5; return true; }
  if (s == "1BTQ")  { out = 6; return true; }
  if (s == "8BTQ")  { out = 7; return true; }
  return false;
}

struct BuildKey {
  std::string quant_method;
  uint32_t block_size;
  uint32_t max_leaf_size;
  uint32_t k_per_level;
  uint32_t quantize_centers;  // 0/1
  uint32_t num_clusters_per_block;
  uint32_t num_points_per_cluster;
  uint32_t rabitq_bits;

  std::string filename_stem() const {
    std::ostringstream o;
    o << "mvivf"
      << "_qm" << quant_method
      << "_m" << block_size
      << "_ml" << max_leaf_size
      << "_kpl" << k_per_level
      << "_qc" << quantize_centers
      << "_ncpb" << num_clusters_per_block
      << "_nppc" << num_points_per_cluster
      << "_rb" << rabitq_bits;
    return o.str();
  }

  void write_params(const std::string& path) const {
    std::ofstream f(path);
    f << "quant_method=" << quant_method << "\n"
      << "block_size=" << block_size << "\n"
      << "max_leaf_size=" << max_leaf_size << "\n"
      << "k_per_level=" << k_per_level << "\n"
      << "quantize_centers=" << quantize_centers << "\n"
      << "num_clusters_per_block=" << num_clusters_per_block << "\n"
      << "num_points_per_cluster=" << num_points_per_cluster << "\n"
      << "rabitq_bits=" << rabitq_bits << "\n";
  }
};

int main(int argc, char* argv[]) {
  mvsic::commandLine P(argc, argv,
      "-i <points> -out_dir <dir> "
      "[-max_leaf_size N] [-k_per_level N] "
      "[-quant_method TQ|FS|PQ|RQ|SPQTQ|1BTQ|8BTQ|None] [-m N] [-qc] "
      "[-num_clusters_per_block N] [-num_points_per_cluster N] [-rbits N]");

  Eigen::setNbThreads(1);
  using ChPoint = ChamferIP_Point;
  using PC = PointCloudSet<ChPoint>;

  char* inFile = P.getOptionValue("-i");
  std::string outDir = P.getOptionValue("-out_dir", "");
  if (inFile == nullptr || outDir.empty()) {
    std::cerr << "Required: -i <points> -out_dir <dir>" << std::endl;
    return 1;
  }

  bool is_mmap = P.getOption("-mm");
  bool compress_input = P.getOption("-compress_input");
  uint32_t verbose = P.getOptionIntValue("-v", 0);
  uint32_t niters = P.getOptionIntValue("-niters", 5);
  uint32_t max_point_clouds_per_cluster = P.getOptionIntValue("-mpcc", 0);
  uint32_t max_points_per_centroid_inner_kmeans = P.getOptionIntValue("-mpcik", 20);
  bool use_weighted_inner_kmeans = P.getOption("-wgh_kmeans");

  BuildKey key;
  key.quant_method = P.getOptionValue("-quant_method", "FS");
  uint32_t quant_method_t = 0;
  if (!parse_quant_method(key.quant_method, quant_method_t)) {
    std::cerr << "Unknown quant method: " << key.quant_method << std::endl;
    return 1;
  }
  key.block_size = P.getOptionIntValue("-m", 8);
  key.max_leaf_size = P.getOptionIntValue("-max_leaf_size", 500);
  key.k_per_level = P.getOptionIntValue("-k_per_level", 0);
  key.quantize_centers = P.getOption("-qc") ? 1u : 0u;
  key.num_clusters_per_block = P.getOptionIntValue("-num_clusters_per_block", 256);
  key.num_points_per_cluster = P.getOptionIntValue("-num_points_per_cluster", 20);
  key.rabitq_bits = P.getOptionIntValue("-rbits", 8);

  std::cout << "Loading points..." << std::endl;
  auto points = PC(inFile, is_mmap);

  IndexParams index_params = IndexParams::mvivf(
      key.k_per_level, key.max_leaf_size, compress_input, verbose, niters,
      max_point_clouds_per_cluster, max_points_per_centroid_inner_kmeans, "Random", 0,
      use_weighted_inner_kmeans, 0, quant_method_t, key.block_size,
      key.num_clusters_per_block, key.num_points_per_cluster, key.rabitq_bits,
      key.quantize_centers != 0);
  IndexMVIVF<false> index(points.get_dims(), index_params);

  std::cout << "Building index (parallel)..." << std::endl;
  parlay::internal::timer it;
  it.start();
  index.build(points);
  it.stop();
  std::cout << "Index built in " << it.total_time() << " seconds." << std::endl;

  std::string stem = key.filename_stem();
  std::string bin_path = outDir + "/" + stem + ".bin";
  std::string params_path = outDir + "/" + stem + ".params";

  std::cout << "Saving index to " << bin_path << std::endl;
  index.save(bin_path);
  key.write_params(params_path);
  std::cout << "Wrote params sidecar: " << params_path << std::endl;
  return 0;
}
