#pragma once

#include "csvfile.h"

template <typename T>
auto compute_recall(parlay::sequence<parlay::sequence<std::pair<uint32_t, T>>>& pred, parlay::sequence<parlay::sequence<std::pair<float, uint32_t>>>& gt, int k, int k_gt){
  if (k_gt > gt[0].size()){
    std::cout << "Not enough gt values" << std::endl;
    exit(-1);
  } else if (k > pred[0].size()){
    std::cout << "Not enough pred values" << std::endl;
    exit(-1);
  }
  std::cout << "Computing Recall " << k_gt << " @ " << k << std::endl;
  parlay::internal::timer t;
  auto ind_recall = parlay::sequence<double>::from_function(pred.size(), 
      [&](size_t i) {
    std::unordered_set<uint32_t> out_set;
    for (size_t j=0; j<k; j++) {
      auto [id, dist] = pred[i][j];
      out_set.insert(id);
    }
    size_t correct = 0;
    for (size_t j=0; j<k_gt; j++) {
      auto [dist, id] = gt[i][j];
      if (out_set.find(id) != out_set.end()) {
        correct++;
      }
    }
    return static_cast<double>(correct)/k_gt;
  });
  double recall = parlay::reduce(ind_recall)/ind_recall.size();
  std::cout << "Recall " << k_gt << " @ " << k << ": " << recall << std::endl;
  return recall;
}

template <typename Index, typename PointCloud, typename GT>
auto check_stats(Index& index, PointCloud& base_points, PointCloud& query_points, GT& gt, int k, int nprobes) {
  using ChPoint = typename PointCloud::point_type;
  using T = typename ChPoint::distance_type;
  parlay::internal::timer t;
  double query_time_seq = 0.0;
  auto pred = parlay::sequence<parlay::sequence<std::pair<uint32_t, T>>>(query_points.size());
  for (size_t i = 0; i < query_points.size(); i++) {
    t.start();
    auto results = index.Search(query_points[i], k, nprobes);
    t.stop();
    query_time_seq += t.total_time();
    t.reset();
    pred[i] = results;
  }
  t.start();
  parlay::parallel_for(0, query_points.size(), [&](size_t i) {
    auto results = index.Search(query_points[i], k, nprobes);
    pred[i] = results;
  });
  t.stop();
  double query_time_par = t.total_time();
  t.reset();
  double QPS_seq = query_points.size() / query_time_seq;
  double QPS_par = query_points.size() / query_time_par;
  double recall_1_k = compute_recall(pred, gt, k, 1);
  double recall_k_k = compute_recall(pred, gt, k, k);
  return std::make_tuple(QPS_seq, QPS_par, recall_1_k, recall_k_k);
}

// TODO: batch queries

template <typename Index, typename PointCloud, typename GT>
void search_and_parse(Index& index, PointCloud& base_points, PointCloud& query_points, GT& gt, const char* res_file, int k) {
  parlay::sequence<std::tuple<double, double, double, double>> results;
  std::vector<int> nprobes_vals = {1, 2, 4, 8, 16, 32, 64};

  for (int nprobes : nprobes_vals) {
    auto result = check_stats_seq(index, base_points, query_points, gt, k, nprobes);
    results.push_back(result);
  }
  write_to_csv(std::string(res_file), results, k, query_points.size());
}

inline void write_to_csv(std::string csv_filename,
                         parlay::sequence<std::tuple<double, double, double, double>>& results, int k, size_t num_queries) {
  csvfile csv(csv_filename);
  csv << "Num queries" << "k" << "nprobes" << "QPS_seq" << "QPS_par" << "Recall 1@k" << "Recall k@k" << endrow;
  int nprobes = 1;
  for (auto& result : results) {
    double QPS_seq, QPS_par, recall_1_k, recall_k_k;
    std::tie(QPS_seq, QPS_par, recall_1_k, recall_k_k) = result;
    csv << num_queries << k << nprobes << QPS_seq << QPS_par << recall_1_k << recall_k_k << endrow;
    nprobes *= 2;
  }
  csv << endrow;
  csv << endrow;
}

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