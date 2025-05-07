#pragma once




// template <typename PointCloud>
// auto checkRecall(const PointCloud& query_points, Index& index, 
//     ){
//   parlay::internal::timer t;
//     double recall_1_k = 0.0;
//     double recall_k_k = 0.0;
//     double query_time = 0.0;
//     for(size_t i = 0; i < queries.size(); i++) {
//       if (i % 100 == 0) {
//         std::cout << queries.size()-i << " queries left" << std::endl;
//       }
//       // Run Brute-force search
//       auto bf_results = mvivf::get_knn(queries[i], points, k);
//       std::unordered_set<uint32_t> bf_set;
//       for (const auto& [id, dist] : bf_results) {
//         bf_set.insert(id);
//       }

//       // Run Index search
//       t.start();
//       auto results = index.Search(queries[i], k, nprobes);
//       t.stop();
//       query_time += t.total_time();
//       t.reset();

//       // Calculate recall
//       size_t correct = 0;
//       for (const auto& [id, dist] : results) {
//         if (bf_set.find(id) != bf_set.end()) {
//           correct++;
//         }
//         if (id == bf_results[0].first) {
//           recall_1_k += 1.0;
//         }
//       }
//       recall_k_k += static_cast<float>(correct)/k;
//     }
//     recall_1_k /= queries.size();
//     recall_k_k /= queries.size();
//     double QPS = queries.size() / query_time;
//     double avg_query_time = 1/QPS;
//     std::cout << "Number of Queries: " << queries.size() << std::endl;
//     std::cout << "Average recall 1 @ " << k << ": " << recall_1_k << std::endl;
//     std::cout << "Average recall " << k << " @ " << k << ": " << recall_k_k << std::endl;
//     std::cout << "QPS: " << QPS << std::endl;
//     std::cout << "Average time per query: " << avg_query_time << " seconds" << std::endl;
// }


auto ReadGT(std::string& file_path, int num_points) {
  std::ifstream file(file_path, std::ios::binary | std::ios::in);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file for reading: " + file_path);
  }

  int num_neighbors = 0;
  file.read(reinterpret_cast<char*>(&num_neighbors), sizeof(num_neighbors));

  parlay::sequence<parlay::sequence<std::pair<float, uint32_t>>> result(num_points);
  for (int i = 0; i < num_points; ++i) {
    parlay::sequence<std::pair<float, uint32_t>> neighbors(num_neighbors);
    file.read(reinterpret_cast<char*>(neighbors.data()),
              num_neighbors * sizeof(std::pair<float, uint32_t>));
    result[i] = std::move(neighbors);
  }
  file.close();
  return result;
}