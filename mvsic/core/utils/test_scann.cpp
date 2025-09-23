#include <iostream>
#include "mvsic/core/utils/parse_command_line.h"
#include "mvsic/core/utils/point_cloud_set.h"
#include "mvsic/core/utils/quantized_point_cloud_set.h"
#include "mvsic/core/utils/chamfer_ip_point.h"
#include "scann/proto/scann.pb.h"

using namespace mvsic;

void test_distances(const PointCloudSet<ChamferIP_Point>& pcs,
                        uint32_t num_blocks, uint32_t num_clusters_per_block, uint32_t sample_size) {
      QuantizedPointCloudSet<ChamferIP_Point> qpcs(pcs, num_blocks, num_clusters_per_block, sample_size);

  // Select a query point cloud (e.g., the first one)
  if (pcs.size() == 0) {
    std::cout << "No point clouds in the dataset to query." << std::endl;
    return;
  }
  auto query_pc = pcs[0];

  std::cout << "Querying for approximated distances from the first point cloud (PQ)..."
            << std::endl;
  auto result_approx = qpcs.distances(query_pc);
  auto distances_approx = result_approx.first;

  std::cout << "Top 10 approximated distances (PQ):" << std::endl;
  for (size_t i = 0; i < std::min((size_t)10, (size_t)distances_approx.size()); ++i) {
    std::cout << "  ID: " << distances_approx[i].first
              << ", Distance: " << distances_approx[i].second << std::endl;
  }

  // Save and load the quantized point cloud set
  const std::string pq_filename = "test_pq.bin";
  std::cout << "\nSaving quantized point cloud set to " << pq_filename << "..." << std::endl;
  qpcs.save(pq_filename);

  std::cout << "Loading quantized point cloud set from " << pq_filename << "..." << std::endl;
  QuantizedPointCloudSet<ChamferIP_Point> qpcs_loaded(pcs, pq_filename);

  std::cout << "Querying for approximated distances from the loaded data..."
            << std::endl;
  auto result_loaded = qpcs_loaded.distances(query_pc);
  auto distances_loaded = result_loaded.first;

  std::cout << "Top 10 approximated distances (from loaded PQ):" << std::endl;
  for (size_t i = 0; i < std::min((size_t)10, (size_t)distances_loaded.size()); ++i) {
    std::cout << "  ID: " << distances_loaded[i].first
              << ", Distance: " << distances_loaded[i].second << std::endl;
  }

  std::cout << "\nQuerying for exact distances from the first point cloud..." << std::endl;
  auto result_exact = pcs.distances(query_pc);
  auto distances_exact = result_exact.first;

  std::cout << "Top 10 exact distances:" << std::endl;
  for (size_t i = 0; i < std::min((size_t)10, (size_t)distances_exact.size()); ++i) {
    std::cout << "  ID: " << distances_exact[i].first << ", Distance: " << distances_exact[i].second
              << std::endl;
  }
}

int main(int argc, char* argv[]) {
  mvsic::commandLine P(argc, argv, "[-i <inFile>] [-num_blocks <int>]");

  char* inFile = P.getOptionValue("-i");
  if (!inFile) {
    std::cerr << "Input file not specified. Use -i <inFile>" << std::endl;
    return 1;
  }

  int num_blocks = P.getOptionIntValue("-num_blocks", 8);
  int num_clusters = P.getOptionIntValue("-num_clusters", 256);

  std::cout << "Loading PointCloudSet from " << inFile << "..." << std::endl;
  PointCloudSet<ChamferIP_Point> pcs(inFile);
  std::cout << "PointCloudSet loaded with " << pcs.size() << " point clouds." << std::endl;
  std::cout << "Total number of vectors: " << pcs.total_size() << std::endl;
  std::cout << "Dimensions: " << pcs.get_dims() << std::endl;

  if (pcs.get_dims() % num_blocks != 0) {
    std::cerr << "Error: Vector dimension (" << pcs.get_dims()
              << ") is not divisible by num_blocks (" << num_blocks << ")." << std::endl;
    return 1;
  }
  if (num_clusters > 256) {
    std::cerr << "Warning: num_clusters is set to " << num_clusters
              << ". Only <= 256 is supported for uint8_t encoding." << std::endl;
    return 1;
  }

  uint32_t sample_size = 100000; // Default value

  test_distances(pcs, num_blocks, num_clusters, sample_size);

  return 0;
}
