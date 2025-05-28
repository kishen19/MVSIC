#include "mvc/utils/chamfer_ip_point.h"
#include "mvc/utils/chamfer_l2_point.h"
#include "mvc/utils/parse_command_line.h"
#include "mvc/utils/point_cloud_set.h"
#include "mvivf/utils/stats.h"
#include "mvivf/mvivf.h"


template <typename ChPoint, bool metric>
void bench(commandLine& P) {
  using PC = PointCloudSet<ChPoint>;

  // Files
  char* inFile = P.getOptionValue("-i"); // Base Points
  std::string qFile = P.getOptionValue("-q", ""); // Query Points
  std::string gtFile = P.getOptionValue("-gt", ""); // Ground Truth
  std::string outFile = P.getOptionValue("-o", ""); // File to store index 
  std::string indexFile = P.getOptionValue("-index", ""); // File containing index
  char* resFile = P.getOptionValue("-r"); // CSV file to store stats
  // Params
  size_t minsize = P.getOptionLongValue("-minsize", 100);
  size_t maxsize = P.getOptionLongValue("-maxsize", 500);
  size_t nprobesl = P.getOptionLongValue("-npl", 1);
  size_t nprobesr = P.getOptionLongValue("-npr", 8);
  size_t nprobesmp = P.getOptionLongValue("-npmp", 2);
  size_t nprobesad = P.getOptionLongValue("-npad", 0);
  // size_t beamsize = P.getOptionLongValue("-beamsize", 0);
  size_t k = P.getOptionLongValue("-k", 10);
  size_t s = P.getOptionLongValue("-s", 0);
  auto seeding = P.getOptionValue("-seed", "Random");
  auto iters = P.getOptionLongValue("-iters", 5);
  int rounds = P.getOptionLongValue("-rounds", 1);
  bool verbose = P.getOption("-v");

  auto points = PC(inFile);
  mvivf::IndexMVIVFParams index_params(minsize, maxsize, s, iters, seeding, verbose);
  mvivf::IndexMVIVF<metric> index(points.get_dims(), index_params);
  if (indexFile != ""){ // Stats Benchmark
    index.load(indexFile, points);
    std::cout << "Index loaded" << std::endl;
    auto queries = PC(P.getOptionValue("-q"));
    auto gt = ReadGT(gtFile, queries.size());
    parlay::sequence<mvivf::SearchParams> search_params_list;
    size_t nprobes = nprobesl;
    while (nprobes<=nprobesr) {
      search_params_list.push_back(mvivf::SearchParams(k,s,nprobes,0));
      nprobes = nprobesmp * nprobes + nprobesad;
    }
    // Compute Stats
    std::cout << "Computing stats..." << std::endl;
    search_all(index, points, queries, gt, resFile, search_params_list);
    std::cout << "Stats computed and saved to " << resFile << std::endl;
  } else { // Indexing Benchmark
    std::cout << "Starting Indexing Benchmark..." << std::endl;
    parlay::internal::timer t;
    double index_time = 0.0;
    for (long it=0; it<=rounds; it++){
      t.start();
      mvivf::IndexMVIVF<metric> index(points.get_dims(), index_params);
      index.build(points);
      t.stop();
      if (it!=0){
        index_time += t.total_time();
      } else{
        if (outFile != ""){
          std::cout << "Saving index to " << outFile << std::endl;
          index.save(P.getOptionValue("-o"));
          std::cout << "Index saved." << std::endl;
        }
        std::cout << "Warm up Time: " << t.total_time() << " seconds." << std::endl;
      }
      t.reset();
    }
    std::cout << "Average Indexing Time: " << index_time/rounds << " seconds." << std::endl;
  }
}

int main(int argc, char* argv[]) {
  commandLine P(argc, argv,
                "[-i <inFile>] [-k <num_centers>] [-s <num_embeddings>]"
                "[-data_type <tp>] [-dist_func <dist_func>]" 
                "[-seed <algorithm>] [-iters <num_iters>]" 
                "[-kmeans_seed <algorithm>] [-kmeans_dist <algorithm>]");

  std::string df = P.getOptionValue("-dist_func", "IP");

  if (df == "L2"){
    using ChPoint = ChamferL2_Point;
    bench<ChPoint, mvivf::L2>(P);
  } else if (df == "IP") {
    using ChPoint = ChamferIP_Point;
    bench<ChPoint, mvivf::IP>(P);
  }
  return 0;
}
