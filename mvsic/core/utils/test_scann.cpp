#include <iostream>
#include <chrono>
#include "mvsic/core/utils/parse_command_line.h"
#include "mvsic/core/utils/point_cloud_set.h"
#include "mvsic/core/utils/quantized_point_cloud_set.h"
#include "mvsic/core/utils/chamfer_ip_point.h"
#include "mvsic/core/utils/pq_helper.h"
#include "scann/proto/scann.pb.h"

using namespace mvsic;

void test_distances(const PointCloudSet<ChamferIP_Point>& pcs,
                    const QuantizedPointCloudSet<ChamferIP_Point>& qpcs) {
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
  QuantizedPointCloudSet<ChamferIP_Point> qpcs_loaded(pq_filename);

  std::cout << "Querying for approximated distances from the loaded data..." << std::endl;
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

void test_distance_performance(const PointCloudSet<ChamferIP_Point>& pcs,
                               const QuantizedPointCloudSet<ChamferIP_Point>& qpcs) {
  std::cout << "\n--- Testing Distance Performance ---" << std::endl;

  if (pcs.size() == 0) {
    std::cout << "No point clouds in the dataset to query." << std::endl;
    return;
  }
  auto query_pc = pcs[0];

  // Time distances_old()
  auto start_dist1 = std::chrono::high_resolution_clock::now();
  auto result1 = qpcs.distances_old(query_pc);
  auto end_dist1 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration1 = end_dist1 - start_dist1;
  std::cout << "distances_old() time: " << duration1.count() << " seconds" << std::endl;

  // Time distances()
  auto start_dist2 = std::chrono::high_resolution_clock::now();
  auto result2 = qpcs.distances(query_pc);
  auto end_dist2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration2 = end_dist2 - start_dist2;
  std::cout << "distances() (Eigen) time: " << duration2.count() << " seconds" << std::endl;

  // Compare results
  std::cout << "\nComparing results..." << std::endl;
  auto distances1 = result1.first;
  auto distances2 = result2.first;

  if (distances1.size() != distances2.size()) {
    std::cerr << "Result sizes differ!" << std::endl;
    return;
  }

  // Sort results by ID to compare them
  parlay::sort_inplace(distances1);
  parlay::sort_inplace(distances2);

  bool mismatch = false;
  for (size_t i = 0; i < distances1.size(); ++i) {
    if (distances1[i].first != distances2[i].first) {
      std::cerr << "ID mismatch at index " << i << std::endl;
      mismatch = true;
      break;
    }
    if (std::abs(distances1[i].second - distances2[i].second) > 1e-5) {
      std::cerr << "Distance mismatch for ID " << distances1[i].first
                << ": distances() = " << distances1[i].second
                << ", distances2() = " << distances2[i].second << std::endl;
      mismatch = true;
      break;
    }
  }

  if (!mismatch) {
    std::cout << "Results match!" << std::endl;
  }

  std::cout << "\n--- End of Distance Performance Test ---" << std::endl;
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

  uint32_t sample_size = 100000;  // Default value

  QuantizedPointCloudSet<ChamferIP_Point> qpcs(pcs, num_blocks, num_clusters, sample_size);

  // test_distances(pcs, qpcs);
  test_distance_performance(pcs, qpcs);

  return 0;
}