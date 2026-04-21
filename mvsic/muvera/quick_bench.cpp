// quick_bench.cpp — Lightweight MUVERA benchmarking.
//
// Usage:
//   bazel run //mvsic/muvera:quick_bench -- -d arguana -quant_method TQ
//   bazel run //mvsic/muvera:quick_bench -- -d nq500k -L 16,32,64,128,256,512,1024 -csv /tmp/out.csv
#include "mvsic/core/bench_utils.h"
#include "muvera.h"

using namespace mvsic;

static std::vector<size_t> parse_csv_ints(const std::string& s) {
  std::vector<size_t> result;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) result.push_back(std::stoul(item));
  return result;
}

template <typename ChPoint, bool metric>
void run(commandLine& P) {
  Eigen::setNbThreads(1);
  using PC = PointCloudSet<ChPoint>;

  auto ds = bench::parse_dataset(P);
  if (ds.points.empty()) { std::cerr << "Error: specify -d <dataset>" << std::endl; std::exit(1); }

  bool is_mmap = P.getOption("-mm");
  std::string index_file = P.getOptionValue("-index", "");
  std::string save_file = P.getOptionValue("-o", "");
  std::string csv_path = P.getOptionValue("-csv", "");
  uint32_t verbose = P.getOptionIntValue("-v", 0);
  bool compress_input = P.getOption("-compress_input");

  // FDE params
  int32_t num_reps = P.getOptionIntValue("-num_reps", 20);
  int32_t num_simhash = P.getOptionIntValue("-num_simhash", 4);
  int32_t projd = P.getOptionIntValue("-projd", 8);
  bool fill_empty = P.getOption("-fill_empty_partitions");
  int32_t final_projd = P.getOptionIntValue("-final_projd", 0);
  bool no_norm = P.getOption("-no_norm");

  // Vamana build params
  uint32_t R = P.getOptionIntValue("-R", 200);
  uint32_t L_build = P.getOptionIntValue("-L_build", 600);
  double alpha = P.getOptionDoubleValue("-a", 1.2);
  int num_pass = P.getOptionIntValue("-np", 1);

  auto qa = bench::parse_quant_args(P, "None");

  // Search
  size_t k = P.getOptionLongValue("-k", 10);
  size_t num_rerank = P.getOptionLongValue("-num_rerank", k);
  double cut = 1.35;
  bool norerank = P.getOption("-norerank");
  std::string L_str = P.getOptionValue("-L", "16,32,64,128,256,512,1024,2048");
  auto L_list = parse_csv_ints(L_str);

  auto points = PC(ds.points.c_str(), is_mmap);
  IndexParams ip = IndexParams::muvera_custom(
      num_reps, num_simhash, 1, projd, fill_empty, final_projd, !no_norm,
      R, L_build, alpha, num_pass, compress_input, verbose,
      qa.pq_method, qa.block_size, qa.num_clusters_per_block,
      qa.num_points_per_cluster, qa.rabitq_bits);
  IndexMUVERA<metric> index(points.get_dims(), ip);
  bench::build_or_load(index, points, index_file, save_file);

  if (ds.queries.empty() || ds.gt.empty()) return;
  auto queries = PC(ds.queries.c_str());
  auto gt = ReadGT(ds.gt, queries.size());
  bench::print_header("MUVERA", ds.name, points.size(), queries.size());

  bench::run_search_sweep(
      index, points, queries, gt, "L", L_list,
      [&](size_t L) { return SearchParams::muvera(k, L, num_rerank, cut, norerank); },
      csv_path, {"search_cmps", "t_fde", "t_quant", "t_search", "t_rerank"});
}

int main(int argc, char* argv[]) {
  commandLine P(argc, argv,
      "[-d <dataset>] [-quant_method None|PQ|RQ|FS|TQ] [-L 16,32,...] "
      "[-k N] [-num_rerank N] [-csv <path>] [-dist_func IP|L2]");
  std::string df = P.getOptionValue("-dist_func", "IP");
  if (df == "L2") run<ChamferL2_Point, true>(P);
  else run<ChamferIP_Point, false>(P);
  return 0;
}
