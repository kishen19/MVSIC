#include "algorithms/bench/parse_command_line.h"
#include "algorithms/utils/euclidian_point.h"
#include "algorithms/utils/mips_point.h"
#include "algorithms/utils/point_range.h"

#include "mvc/utils/pointcloud.h"
#include "mvc/utils/chamferpoint.h"

#include "index.h"

template <typename ChPoint, typename Range>
void bench(commandLine& P) {
  using T = typename ChPoint::distance_type;
  using PC = PointCloud<ChPoint, Range>;

  auto inFile = P.getOptionValue("-i");
  auto qFile = P.getOptionValue("-q");
  long maxsize = P.getOptionLongValue("-maxsize", 100);
  long nprobes = P.getOptionLongValue("-nprobes", 1);
  long k = P.getOptionLongValue("-k", 10);
  auto seeding = P.getOptionValue("-seed", "Random");
  auto iters = P.getOptionLongValue("-iters", 5);
  auto kmeans_seeding = P.getOptionValue("-kmeans_seed", "PrefixDoubling");
  auto kmeans_dist_algo = P.getOptionValue("-kmeans_dist", "ANNS");
  auto kmeans_iters = P.getOptionLongValue("-kmeans_iters", 20);
  // long num = P.getOptionLongValue("-num", 1000);

  auto points = PC(inFile);
  auto queries = PC(qFile);
  parlay::internal::timer it;
  it.start();
  auto index = mvivf::Index<T, PC>(points, maxsize, iters, seeding, 
      kmeans_dist_algo, kmeans_seeding, kmeans_iters);
  it.stop();
  std::cout << "Index built in " << it.total_time() << " seconds." << std::endl;
  
  parlay::internal::timer t;
  double recall = 0.0;
  double query_time = 0.0;
  for(size_t i = 0; i < queries.size(); i++) {
    // Run Brute-force search
    auto bf_results = mvivf::get_knn(queries[i], points, k);
    std::unordered_set<uint32_t> bf_set;
    for (const auto& [id, dist] : bf_results) {
      bf_set.insert(id);
    }

    // Run Index search
    t.start();
    auto results = index.Search(queries[i], k, nprobes);
    t.stop();
    query_time += t.total_time();
    t.reset();

    // Calculate recall
    size_t correct = 0;
    for (const auto& [id, dist] : results) {
      if (bf_set.find(id) != bf_set.end()) {
        correct++;
      }
    }
    recall += static_cast<float>(correct)/k;
  }
  recall /= queries.size();
  double QPS = queries.size() / query_time;
  double avg_query_time = 1/QPS;
  std::cout << "Average recall over " << queries.size() << " queries: " << recall << std::endl;
  std::cout << "QPS: " << QPS << std::endl;
  std::cout << "Average time per query: " << avg_query_time << " seconds" << std::endl;
}

int main(int argc, char* argv[]) {
  commandLine P(argc, argv,
                "[-i <inFile>] [-k <num_centers>] [-s <num_embeddings>]"
                "[-data_type <tp>] [-dist_func <dist_func>]" 
                "[-seed <algorithm>] [-iters <num_iters>]" 
                "[-kmeans_seed <algorithm>] [-kmeans_dist <algorithm>]");

  std::string tp = P.getOptionValue("-data_type", "float");
  std::string df = P.getOptionValue("-dist_func", "Euclidian");

  if ((tp != "uint8") && (tp != "int8") && (tp != "float")) {
    std::cout << "Error: vector type not specified correctly, specify int8, "
                 "uint8, or float"
              << std::endl;
    abort();
  }

  // auto seeding = P.getOptionValue("-seed", "Random");
  // auto iters = P.getOptionLongValue("-iters", 5);
  // auto kmeans_seeding = P.getOptionValue("-kmeans_seed", "PrefixDoubling");
  // auto kmeans_dist_algo = P.getOptionValue("-kmeans_dist", "ANNS");
  // auto kmeans_iters = P.getOptionLongValue("-kmeans_iters", 20);

  if (tp == "float") {
    if (df == "Euclidian"){
      using ChPoint = Chamfer_Euclidian_Point<float>;
      using Point = Euclidian_Point<float>;
      using Range = PointRange<float, Point>;
      bench<ChPoint, Range>(P);
    } else if (df == "Mips") {
      using ChPoint = Chamfer_Mips_Point<float>;
      using Point = Mips_Point<float>;
      using Range = PointRange<float, Point>;
      bench<ChPoint, Range>(P);
    }
  } else if (tp == "uint8") {
    if (df == "Euclidian"){
      using ChPoint = Chamfer_Euclidian_Point<uint8_t>;
      using Point = Euclidian_Point<uint8_t>;
      using Range = PointRange<uint8_t, Point>;
      bench<ChPoint, Range>(P);
    } else if (df == "Mips") {
      using ChPoint = Chamfer_Mips_Point<uint8_t>;
      using Point = Mips_Point<uint8_t>;
      using Range = PointRange<uint8_t, Point>;
      bench<ChPoint, Range>(P);
    }
  } else if (tp == "int8") {
    if (df == "Euclidian"){
      using ChPoint = Chamfer_Euclidian_Point<int8_t>;
      using Point = Euclidian_Point<int8_t>;
      using Range = PointRange<int8_t, Point>;
      bench<ChPoint, Range>(P);
    } else if (df == "Mips") {
      using ChPoint = Chamfer_Mips_Point<int8_t>;
      using Point = Mips_Point<int8_t>;
      using Range = PointRange<int8_t, Point>;
      bench<ChPoint, Range>(P);
    }
  }
  return 0;
}
