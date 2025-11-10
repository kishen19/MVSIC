#pragma once

#include "parlay/primitives.h"
#include "search_params.h"

namespace mvsic {
struct Stats {
  double QPS_seq = 0.0;
  double QPS_par = 0.0;
  double avg_cmps = 0.0;
  double recall_1_k = 0.0;
  double recall_k_k = 0.0;

  Stats() {}

  Stats(double QPS_seq, double QPS_par, double avg_cmps, double recall_1_k, double recall_k_k) :
      QPS_seq(QPS_seq),
      QPS_par(QPS_par),
      avg_cmps(avg_cmps),
      recall_1_k(recall_1_k),
      recall_k_k(recall_k_k) {}
};

struct StatsExtended {
  double QPS_seq = 0.0;
  double QPS_par = 0.0;
  double avg_cmps = 0.0;
  double recall_1_k = 0.0;
  double recall_k_k = 0.0;
  parlay::sequence<double> avg_timings;

  StatsExtended() {}

  StatsExtended(double QPS_seq, double QPS_par, double avg_cmps, double recall_1_k,
                double recall_k_k, parlay::sequence<double> avg_timings) :
      QPS_seq(QPS_seq),
      QPS_par(QPS_par),
      avg_cmps(avg_cmps),
      recall_1_k(recall_1_k),
      recall_k_k(recall_k_k),
      avg_timings(avg_timings) {}
};
}  // namespace mvsic

double compute_recall(const parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>> &pred,
                      const parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>> &gt,
                      size_t k, size_t k_gt) {
  if (k_gt > gt[0].size()) {
    std::cerr << "Not enough gt values" << std::endl;
    exit(-1);
  } else if (k > pred[0].size()) {
    std::cout << "Not enough pred values: " << pred[0].size() << " out of " << k << std::endl;
  }
  parlay::internal::timer t;
  auto ind_recall = parlay::sequence<double>::from_function(pred.size(), [&](size_t i) {
    std::unordered_set<uint32_t> out_set;
    for (size_t j = 0; j < pred[i].size(); j++) {
      auto [id, dist] = pred[i][j];
      out_set.insert(id);
    }
    size_t correct = 0;
    for (size_t j = 0; j < k_gt; j++) {
      auto [id, dist] = gt[i][j];
      if (out_set.find(id) != out_set.end()) {
        correct++;
      }
    }
    // Dealing with duplicates and near duplicates: fine to return
    // any of the (near) duplicates of the last point
    float last_dist = gt[i][k_gt - 1].second;
    for (size_t j = k_gt; j < gt[i].size(); j++) {
      auto [id, dist] = gt[i][j];
      if (std::abs(dist - last_dist) < 1e-6) {
        if (out_set.find(id) != out_set.end()) {
          correct++;
        }
      } else {
        break;
      }
    }
    return static_cast<double>(correct) / static_cast<double>(k_gt);
  });
  auto val = parlay::reduce(ind_recall);
  return parlay::reduce(ind_recall) / static_cast<double>(ind_recall.size());
}

template<typename Index, typename PC>
parlay::sequence<mvsic::Stats> compute_stats(
    Index &index, const PC &points, const PC &query_points,
    const parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>> &gt,
    const parlay::sequence<mvsic::SearchParams> &params) {
  auto results = parlay::sequence<mvsic::Stats>(params.size());
  for (size_t i = 0; i < params.size(); i++) {
    parlay::internal::timer t;
    size_t k = params[i].k;
    double query_time_seq = 0.0;
    auto pred = parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>>(query_points.size());
    auto cmps = parlay::sequence<size_t>::uninitialized(query_points.size());
    for (size_t j = 0; j < query_points.size(); j++) {
      t.start();
      auto [p, c] = index.search(query_points[j], points, params[i]);
      t.stop();
      query_time_seq += t.total_time();
      t.reset();
      pred[j] = p;
      cmps[j] = c;
    }

    t.start();
    auto [pred_par, cmps_par] = index.search_all(query_points, points, params[i]);
    t.stop();
    double query_time_par = t.total_time();
    t.reset();

    double QPS_seq = query_points.size() / query_time_seq;
    double QPS_par = query_points.size() / query_time_par;
    double avg_cmps = (double)parlay::reduce(cmps) / (double)cmps.size();
    double recall_1_k = compute_recall(pred, gt, k, 1);
    double recall_k_k = compute_recall(pred, gt, k, k);
    results[i] = mvsic::Stats(QPS_seq, QPS_par, avg_cmps, recall_1_k, recall_k_k);
  }
  return results;
}

template<typename Index, typename PC>
parlay::sequence<mvsic::StatsExtended> compute_stats_extended(
    Index &index, const PC &points, const PC &query_points,
    const parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>> &gt,
    const parlay::sequence<mvsic::SearchParams> &params) {
  auto results = parlay::sequence<mvsic::StatsExtended>(params.size());
  for (size_t i = 0; i < params.size(); i++) {
    parlay::internal::timer t;
    size_t k = params[i].k;
    double query_time_seq = 0.0;
    auto pred = parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>>(query_points.size());
    auto cmps = parlay::sequence<size_t>::uninitialized(query_points.size());
    auto timings = parlay::sequence<std::vector<double>>(query_points.size());

    for (size_t j = 0; j < query_points.size(); j++) {
      t.start();
      auto [p, c, time] = index.search_with_stats(query_points[j], points, params[i]);
      t.stop();
      query_time_seq += t.total_time();
      t.reset();
      pred[j] = p;
      cmps[j] = c;
      timings[j] = time;
    }
    t.start();
    auto [pred_par, cmps_par] = index.search_all(query_points, points, params[i]);
    t.stop();
    double query_time_par = t.total_time();
    t.reset();

    double QPS_seq = query_points.size() / query_time_seq;
    double QPS_par = query_points.size() / query_time_par;
    double avg_cmps = (double)parlay::reduce(cmps) / (double)cmps.size();
    double recall_1_k = compute_recall(pred, gt, k, 1);
    double recall_k_k = compute_recall(pred, gt, k, k);

    size_t num_timings = 0;
    if (query_points.size() > 0) {
      num_timings = timings[0].size();
    }
    auto avg_timings = parlay::sequence<double>(num_timings);
    for (size_t j = 0; j < num_timings; j++) {
      double total_time = 0;
      for (size_t l = 0; l < timings.size(); l++) {
        total_time += timings[l][j];
      }
      avg_timings[j] = total_time / timings.size();
    }

    results[i] =
        mvsic::StatsExtended(QPS_seq, QPS_par, avg_cmps, recall_1_k, recall_k_k, avg_timings);
  }
  return results;
}

template<typename Index, typename PC>
mvsic::Stats compute_stats(Index &index, const PC &points, const PC &query_points,
                           const parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>> &gt,
                           const mvsic::SearchParams &params) {
  return compute_stats(index, points, query_points, gt,
                       parlay::sequence<mvsic::SearchParams>{params})[0];
}

template<typename Index, typename PC>
mvsic::StatsExtended compute_stats_extended(
    Index &index, const PC &points, const PC &query_points,
    const parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>> &gt,
    const mvsic::SearchParams &params) {
  return compute_stats_extended(index, points, query_points, gt,
                                parlay::sequence<mvsic::SearchParams>{params})[0];
}

std::pair<double, double> compute_scores(
    const parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>> &pred,
    const parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>> &gt, size_t k) {
  double recall_1_k = compute_recall(pred, gt, k, 1);
  double recall_k_k = compute_recall(pred, gt, k, k);
  return std::make_pair(recall_1_k, recall_k_k);
}

parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>> ReadGT(std::string &file_path,
                                                                      size_t num_points) {
  std::ifstream file(file_path, std::ios::binary | std::ios::in);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file for reading: " + file_path);
  }

  int num_neighbors = 0;
  file.read(reinterpret_cast<char *>(&num_neighbors), sizeof(num_neighbors));

  parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>> result(num_points);
  for (int i = 0; i < num_points; ++i) {
    parlay::sequence<std::pair<float, uint32_t>> neighbors(num_neighbors);
    file.read(reinterpret_cast<char *>(neighbors.data()),
              num_neighbors * sizeof(std::pair<float, uint32_t>));
    auto neighbors_flipped = parlay::sequence<std::pair<uint32_t, float>>::from_function(
        num_neighbors,
        [&](size_t j) { return std::make_pair(neighbors[j].second, neighbors[j].first); });
    result[i] = std::move(neighbors_flipped);
  }
  file.close();
  return result;
}

parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>> ReadGoldGT(std::string &file_path,
                                                                          size_t num_points) {
  std::ifstream file(file_path, std::ios::binary | std::ios::in);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file for reading: " + file_path);
  }
  size_t num_offsets;
  size_t num_gt_entries;
  // Read sizes
  file.read(reinterpret_cast<char *>(&num_offsets), sizeof(num_offsets));
  file.read(reinterpret_cast<char *>(&num_gt_entries), sizeof(num_gt_entries));
  std::vector<size_t> offsets(num_offsets);
  std::vector<uint32_t> ground_truth(num_gt_entries);
  // Read offset and ground truth data
  file.read(reinterpret_cast<char *>(offsets.data()), num_offsets * sizeof(size_t));
  file.read(reinterpret_cast<char *>(ground_truth.data()), num_gt_entries * sizeof(uint32_t));
  file.close();
  parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>> result(num_points);
  parlay::parallel_for(0, num_points, [&](size_t i) {
    size_t start_index = offsets[i];
    size_t end_index = offsets[i + 1];
    auto neighbors = parlay::tabulate(end_index - start_index, [&](size_t j) {
      return std::make_pair(ground_truth[start_index + j], static_cast<float>(j));
    });
    result[i] = std::move(neighbors);
  });
  return result;
}