#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include "mvsic/core/utils/point_cloud_set.h"
#include "mvsic/core/utils/parse_command_line.h"
#include "mvsic/core/utils/top_neighbors.h"
#include "mvsic/core/utils/chamfer_l2_point.h"
#include "mvsic/core/utils/chamfer_ip_point.h"
#include "parlay/primitives.h"

template<typename ChPoint>
void compute_ground_truth(mvsic::commandLine& P) {
  char* db_file = P.getOptionValue("-i");
  char* query_file = P.getOptionValue("-q");
  char* out_file = P.getOptionValue("-o");
  int k = P.getOptionIntValue("-k", 2000);

  mvsic::PointCloudSet<ChPoint> database(db_file);
  mvsic::PointCloudSet<ChPoint> queries(query_file);

  std::cout << "Database size: " << database.size() << std::endl;
  std::cout << "Query size: " << queries.size() << std::endl;
  std::cout << "k: " << k << std::endl;

  parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>> all_neighbors(queries.size());

  parlay::parallel_for(0, queries.size(), [&](size_t i) {
    if (i % 100 == 0) {
      std::cout << "Processing query " << i << std::endl;
    }
    auto [neighbors, dist_cmps] = mvsic::get_knn(queries[i], database, k);
    all_neighbors[i] = neighbors;
  });

  std::cout << "Finished computing all neighbors. Writing to file." << std::endl;

  std::ofstream out(out_file, std::ios::binary);
  if (!out.is_open()) {
    std::cerr << "Error opening output file " << out_file << std::endl;
    exit(1);
  }

  out.write(reinterpret_cast<const char*>(&k), sizeof(int));
  for (size_t i = 0; i < queries.size(); ++i) {
    for (const auto& neighbor : all_neighbors[i]) {
      float distance = neighbor.second;
      uint32_t id = neighbor.first;
      out.write(reinterpret_cast<const char*>(&distance), sizeof(float));
      out.write(reinterpret_cast<const char*>(&id), sizeof(uint32_t));
    }
  }

  out.close();
  std::cout << "Ground truth written to " << out_file << std::endl;
}

int main(int argc, char* argv[]) {
  mvsic::commandLine P(
      argc, argv,
      "{-i <database_file> -q <query_file> -o <output_file> [-k <k>] [-dist_func <L2|IP>]}");

  std::string dist_func = P.getOptionValue("-dist_func", "IP");

  if (dist_func == "L2") {
    compute_ground_truth<mvsic::ChamferL2_Point>(P);
  } else if (dist_func == "IP") {
    compute_ground_truth<mvsic::ChamferIP_Point>(P);
  } else {
    std::cerr << "Invalid distance function: " << dist_func << std::endl;
    exit(1);
  }

  return 0;
}
