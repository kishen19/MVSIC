#pragma once

#include "parlay/primitives.h"
#include "mvc/utils/point_cloud_set.h"
#include "csvfile.h"
#include "search_params.h"

namespace mvivf{
struct Stats{
  double QPS_seq;
  double QPS_par;
  double avg_cmps;
  double recall_1_k;
  double recall_k_k;
};
} // namespace mvivf

double compute_recall(
    parlay::sequence<parlay::sequence<std::pair<size_t, float>>>& pred, 
    parlay::sequence<parlay::sequence<std::pair<float, size_t>>>& gt, 
    int k, int k_gt){
  if (k_gt > gt[0].size()){
    std::cerr << "Not enough gt values" << std::endl;
    exit(-1);
  } else if (k > pred[0].size()){
    std::cerr << "Not enough pred values" << std::endl;
    exit(-1);
  }
  parlay::internal::timer t;
  auto ind_recall = parlay::sequence<double>::from_function(pred.size(), 
      [&](size_t i) {
    std::unordered_set<size_t> out_set;
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
  return parlay::reduce(ind_recall)/ind_recall.size();
}

template <typename Index, typename ChPoint, typename GT>
auto compute_stats(const Index& index, const PointCloudSet<ChPoint>& points, 
    const PointCloudSet<ChPoint>& query_points, 
    const GT& gt, int k, const mvivf::SearchParams& params) {
  parlay::internal::timer t;
  double query_time_seq = 0.0;
  auto pred = parlay::sequence<parlay::sequence<std::pair<size_t, float>>>(
    query_points.size());
  auto cmps = parlay::sequence<size_t>::uninitialized(query_points.size());
  for (size_t i = 0; i < query_points.size(); i++) {
    t.start();
    auto [results, dist_cmps_i] = index.search(query_points[i], points, k, params);
    t.stop();
    query_time_seq += t.total_time();
    t.reset();
    pred[i] = results;
    cmps[i] = dist_cmps_i;
  }
  t.start();
  parlay::parallel_for(0, query_points.size(), [&](size_t i) {
    auto [results, dist_cmps_i] = index.search(query_points[i], points, k, params);
    pred[i] = results;
    assert(cmps[i] == dist_cmps_i);
    cmps[i] = dist_cmps_i;
  });
  t.stop();
  double query_time_par = t.total_time();
  t.reset();
  double QPS_seq = query_points.size() / query_time_seq;
  double QPS_par = query_points.size() / query_time_par;
  double avg_cmps = (double)parlay::reduce(cmps) / (double)cmps.size();
  double recall_1_k = compute_recall(pred, gt, k, 1);
  double recall_k_k = compute_recall(pred, gt, k, k);
  return mvivf::Stats(QPS_seq, QPS_par, avg_cmps, recall_1_k, recall_k_k);
}

// inline void write_to_csv(std::string csv_filename, mvivf::Stats& result, 
//     int k, size_t num_queries) {
//   csvfile csv(csv_filename);
//   csv << "Num queries" << "k" << "nprobes" << "QPS_seq" << "QPS_par" << "Avg Cmps" 
//       << "Recall 1@k" << "Recall k@k" << endrow;
//   int nprobes = 1;
//   csv << num_queries << k << nprobes << QPS_seq << QPS_par << avg_cmps 
//       << recall_1_k << recall_k_k << endrow;
//   csv << endrow;
// }

// TODO: batch queries

// template <typename Index, typename ChPoint, typename GT>
// void search_and_parse(Index& index, PointCloudSet<ChPoint>& base_points, 
//     PointCloudSet<ChPoint>& query_points, GT& gt, const char* res_file, int k) {
//   parlay::sequence<std::tuple<double, double, double, double, double>> results;
//   std::vector<int> nprobes_vals = {1, 2, 4, 8, 16, 32, 64};

//   for (int nprobes : nprobes_vals) {
//     auto result = check_stats(index, base_points, query_points, gt, k, nprobes);
//     results.push_back(result);
//   }
//   write_to_csv(std::string(res_file), results, k, query_points.size());
// }

auto ReadGT(std::string& file_path, size_t num_points) {
  std::ifstream file(file_path, std::ios::binary | std::ios::in);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file for reading: " + file_path);
  }

  int num_neighbors = 0;
  file.read(reinterpret_cast<char*>(&num_neighbors), sizeof(num_neighbors));

  parlay::sequence<parlay::sequence<std::pair<float, size_t>>> result(num_points);
  for (int i = 0; i < num_points; ++i) {
    parlay::sequence<std::pair<float, size_t>> neighbors(num_neighbors);
    file.read(reinterpret_cast<char*>(neighbors.data()),
              num_neighbors * sizeof(std::pair<float, size_t>));
    result[i] = std::move(neighbors);
  }
  file.close();
  return result;
}