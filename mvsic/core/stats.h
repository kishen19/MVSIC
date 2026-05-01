#pragma once

#include <algorithm>
#include <cmath>
#include <fstream>
#include <vector>
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
  // Per-query latency percentiles in seconds. Populated by compute_stats_latency;
  // left at 0 for paths that do not time individual queries (e.g. the existing
  // batch-only or multi-latency entry points).
  double latency_p50 = 0.0;
  double latency_p95 = 0.0;
  double latency_p99 = 0.0;

  StatsExtended() {}

  StatsExtended(double QPS_seq, double QPS_par, double avg_cmps, double recall_1_k,
                double recall_k_k, parlay::sequence<double> avg_timings) :
      QPS_seq(QPS_seq),
      QPS_par(QPS_par),
      avg_cmps(avg_cmps),
      recall_1_k(recall_1_k),
      recall_k_k(recall_k_k),
      avg_timings(avg_timings) {}

  StatsExtended(double QPS_seq, double QPS_par, double avg_cmps, double recall_1_k,
                double recall_k_k, parlay::sequence<double> avg_timings, double p50, double p95,
                double p99) :
      QPS_seq(QPS_seq),
      QPS_par(QPS_par),
      avg_cmps(avg_cmps),
      recall_1_k(recall_1_k),
      recall_k_k(recall_k_k),
      avg_timings(avg_timings),
      latency_p50(p50),
      latency_p95(p95),
      latency_p99(p99) {}
};
}  // namespace mvsic

inline double compute_recall(
    const parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>>& pred,
    const parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>>& gt, size_t k,
    size_t k_gt) {
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
  return parlay::reduce(ind_recall) / static_cast<double>(ind_recall.size());
}

template<typename Index, typename PC>
parlay::sequence<mvsic::Stats> compute_stats(
    Index& index, const PC& points, const PC& query_points,
    const parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>>& gt,
    const parlay::sequence<mvsic::SearchParams>& params) {
  auto results = parlay::sequence<mvsic::Stats>(params.size());
  size_t reps = 3;
  for (size_t i = 0; i < params.size(); i++) {
    parlay::internal::timer t;
    size_t k = params[i].k;
    double query_time_seq = 1e15;
    auto pred = parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>>(query_points.size());
    auto cmps = parlay::sequence<size_t>::uninitialized(query_points.size());
    // Warmup
    for (size_t j = 0; j < std::min(static_cast<size_t>(10), query_points.size()); j++) {
      auto [p, c] = index.search(query_points[j], points, params[i]);
    }
    // Single Batch Run
    for (size_t it = 0; it < reps; it++) {
      double query_time_seq_it = 0.0;
      for (size_t j = 0; j < query_points.size(); j++) {
        t.start();
        auto [p, c] = index.search(query_points[j], points, params[i]);
        t.stop();
        query_time_seq_it += t.total_time();
        t.reset();
        if (it == 0) {
          pred[j] = p;
          cmps[j] = c;
        }
      }
      query_time_seq = std::min(query_time_seq, query_time_seq_it);
    }

    // Batch Run (all queries)
    double query_time_par = 1e15;
    for (size_t it = 0; it < reps; it++) {
      t.start();
      auto [pred_par, cmps_par] = index.search_all(query_points, points, params[i]);
      t.stop();
      query_time_par = std::min(query_time_par, t.total_time());
      t.reset();
    }

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
    Index& index, const PC& points, const PC& query_points,
    const parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>>& gt,
    const parlay::sequence<mvsic::SearchParams>& params) {
  auto results = parlay::sequence<mvsic::StatsExtended>(params.size());
  size_t reps = 3;
  for (size_t i = 0; i < params.size(); i++) {
    parlay::internal::timer t;
    size_t k = params[i].k;
    double query_time_seq = 1e15;
    auto pred = parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>>(query_points.size());
    auto cmps = parlay::sequence<size_t>::uninitialized(query_points.size());
    auto timings = parlay::sequence<std::vector<double>>(query_points.size());

    // Warmup
    for (size_t j = 0; j < std::min(static_cast<size_t>(100), query_points.size()); j++) {
      auto [p, c] = index.search(query_points[j], points, params[i]);
    }
    // Single Batch Run
    for (size_t it = 0; it < reps; it++) {
      double query_time_seq_it = 0.0;
      for (size_t j = 0; j < query_points.size(); j++) {
        t.start();
        auto [p, c, time] = index.search_with_stats(query_points[j], points, params[i]);
        t.stop();
        query_time_seq_it += t.total_time();
        t.reset();
        if (it == 0) {
          pred[j] = p;
          cmps[j] = c;
          timings[j] = time;
        }
      }
      query_time_seq = std::min(query_time_seq, query_time_seq_it);
    }

    // Batch Run (all queries) for 3 reps
    double query_time_par = 1e15;
    for (size_t it = 0; it < reps; it++) {
      t.start();
      auto [pred_par, cmps_par] = index.search_all(query_points, points, params[i]);
      t.stop();
      query_time_par = std::min(query_time_par, t.total_time());
      t.reset();
    }

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
      avg_timings[j] = total_time;  // / timings.size();
    }

    results[i] =
        mvsic::StatsExtended(QPS_seq, QPS_par, avg_cmps, recall_1_k, recall_k_k, avg_timings);
  }
  return results;
}

// Percentile helper. Sorts a copy of `xs` and returns the q-th percentile in [0,100].
inline double _percentile(const std::vector<double>& xs, double q) {
  if (xs.empty()) return 0.0;
  std::vector<double> v = xs;
  std::sort(v.begin(), v.end());
  if (v.size() == 1) return v[0];
  double pos = (q / 100.0) * (v.size() - 1);
  size_t lo = static_cast<size_t>(std::floor(pos));
  size_t hi = static_cast<size_t>(std::ceil(pos));
  if (lo == hi) return v[lo];
  double frac = pos - static_cast<double>(lo);
  return v[lo] + frac * (v[hi] - v[lo]);
}

// ---------------------------------------------------------------------------
// Per-query, single-thread latency. Wraps the per-query loop in a parlay
// scheduler with num_threads=1 so kernels that internally fan out via parlay
// are forced to run sequentially. Reports QPS_seq plus p50/p95/p99 query
// latency in seconds.
// ---------------------------------------------------------------------------
template<typename Index, typename PC>
parlay::sequence<mvsic::StatsExtended> compute_stats_latency(
    Index& index, const PC& points, const PC& query_points,
    const parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>>& gt,
    const parlay::sequence<mvsic::SearchParams>& params) {
  auto results = parlay::sequence<mvsic::StatsExtended>(params.size());
  auto func = [&]() {
    for (size_t i = 0; i < params.size(); i++) {
      parlay::internal::timer t;
      size_t k = params[i].k;
      auto pred =
          parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>>(query_points.size());
      auto cmps = parlay::sequence<size_t>::uninitialized(query_points.size());
      auto timings = parlay::sequence<std::vector<double>>(query_points.size());
      std::vector<double> per_query_latencies(query_points.size(), 0.0);

      // Warmup
      for (size_t j = 0; j < std::min(static_cast<size_t>(10), query_points.size()); j++) {
        auto [p, c] = index.search(query_points[j], points, params[i]);
      }
      double query_time_seq = 0.0;
      for (size_t j = 0; j < query_points.size(); j++) {
        t.start();
        auto [p, c, time] = index.search_with_stats(query_points[j], points, params[i]);
        t.stop();
        double q_time = t.total_time();
        query_time_seq += q_time;
        per_query_latencies[j] = q_time;
        t.reset();
        pred[j] = p;
        cmps[j] = c;
        timings[j] = time;
      }

      double QPS_seq = query_points.size() / query_time_seq;
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
        avg_timings[j] = total_time;
      }

      double p50 = _percentile(per_query_latencies, 50.0);
      double p95 = _percentile(per_query_latencies, 95.0);
      double p99 = _percentile(per_query_latencies, 99.0);
      results[i] = mvsic::StatsExtended(QPS_seq, QPS_seq, avg_cmps, recall_1_k, recall_k_k,
                                        avg_timings, p50, p95, p99);
    }
  };
  parlay::execute_with_scheduler(1, func);
  return results;
}

// ---------------------------------------------------------------------------
// Per-query, multi-threaded latency (no execute_with_scheduler wrapper).
// Mirrors compute_stats_extended's per-query loop but skips the search_all
// run, so the per-query path uses the ambient parlay thread pool freely.
// Reports QPS_seq (effective per-query throughput at full concurrency) plus
// p50/p95/p99 query latency from per-query timing.
// ---------------------------------------------------------------------------
template<typename Index, typename PC>
parlay::sequence<mvsic::StatsExtended> compute_stats_multi_latency(
    Index& index, const PC& points, const PC& query_points,
    const parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>>& gt,
    const parlay::sequence<mvsic::SearchParams>& params) {
  auto results = parlay::sequence<mvsic::StatsExtended>(params.size());
  size_t reps = 3;
  for (size_t i = 0; i < params.size(); i++) {
    parlay::internal::timer t;
    size_t k = params[i].k;
    double query_time_seq = 1e15;
    auto pred = parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>>(query_points.size());
    auto cmps = parlay::sequence<size_t>::uninitialized(query_points.size());
    auto timings = parlay::sequence<std::vector<double>>(query_points.size());
    std::vector<double> per_query_latencies(query_points.size(), 0.0);

    // Warmup (mirrors compute_stats_extended).
    for (size_t j = 0; j < std::min(static_cast<size_t>(100), query_points.size()); j++) {
      auto [p, c] = index.search(query_points[j], points, params[i]);
    }
    // Best-of-`reps` per-query loop.
    for (size_t it = 0; it < reps; it++) {
      double query_time_seq_it = 0.0;
      std::vector<double> per_q_it(query_points.size(), 0.0);
      for (size_t j = 0; j < query_points.size(); j++) {
        t.start();
        auto [p, c, time] = index.search_with_stats(query_points[j], points, params[i]);
        t.stop();
        double q_time = t.total_time();
        per_q_it[j] = q_time;
        query_time_seq_it += q_time;
        t.reset();
        if (it == 0) {
          pred[j] = p;
          cmps[j] = c;
          timings[j] = time;
        }
      }
      if (query_time_seq_it < query_time_seq) {
        query_time_seq = query_time_seq_it;
        per_query_latencies = std::move(per_q_it);
      }
    }

    double QPS_seq = query_points.size() / query_time_seq;
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
      avg_timings[j] = total_time;
    }

    double p50 = _percentile(per_query_latencies, 50.0);
    double p95 = _percentile(per_query_latencies, 95.0);
    double p99 = _percentile(per_query_latencies, 99.0);
    results[i] = mvsic::StatsExtended(QPS_seq, QPS_seq, avg_cmps, recall_1_k, recall_k_k,
                                      avg_timings, p50, p95, p99);
  }
  return results;
}

// ---------------------------------------------------------------------------
// Batch throughput: only times index.search_all (no per-query loop). Reports
// QPS_par measured by best-of-3 wall time across the batch. Recall is
// computed from the batched output. avg_timings is empty for now (no
// per-query breakdown is available from search_all).
// ---------------------------------------------------------------------------
template<typename Index, typename PC>
parlay::sequence<mvsic::StatsExtended> compute_stats_batch(
    Index& index, const PC& points, const PC& query_points,
    const parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>>& gt,
    const parlay::sequence<mvsic::SearchParams>& params) {
  auto results = parlay::sequence<mvsic::StatsExtended>(params.size());
  size_t reps = 3;
  for (size_t i = 0; i < params.size(); i++) {
    parlay::internal::timer t;
    size_t k = params[i].k;

    // One warmup batch.
    {
      auto [pred_warm, cmps_warm] = index.search_all(query_points, points, params[i]);
      (void)pred_warm;
      (void)cmps_warm;
    }

    parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>> last_pred;
    size_t last_cmps = 0;
    double query_time_par = 1e15;
    for (size_t it = 0; it < reps; it++) {
      t.start();
      auto [pred_par, cmps_par] = index.search_all(query_points, points, params[i]);
      t.stop();
      double batch_time = t.total_time();
      t.reset();
      if (batch_time < query_time_par) {
        query_time_par = batch_time;
        last_pred = std::move(pred_par);
        last_cmps = cmps_par;
      }
    }

    double QPS_par = query_points.size() / query_time_par;
    double avg_cmps =
        query_points.size() == 0 ? 0.0 : static_cast<double>(last_cmps) / query_points.size();
    double recall_1_k = compute_recall(last_pred, gt, k, 1);
    double recall_k_k = compute_recall(last_pred, gt, k, k);

    parlay::sequence<double> avg_timings;
    results[i] = mvsic::StatsExtended(QPS_par, QPS_par, avg_cmps, recall_1_k, recall_k_k,
                                      avg_timings);
  }
  return results;
}

template<typename Index, typename PC>
parlay::sequence<mvsic::StatsExtended> compute_stats_extended_p_threaded(
    Index& index, const PC& points, const PC& query_points,
    const parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>>& gt,
    const parlay::sequence<mvsic::SearchParams>& params, size_t num_threads = 1) {
  // Single-Threaded Version
  auto results = parlay::sequence<mvsic::StatsExtended>(params.size());
  auto func = [&]() {
    for (size_t i = 0; i < params.size(); i++) {
      parlay::internal::timer t;
      size_t k = params[i].k;
      auto pred =
          parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>>(query_points.size());
      auto cmps = parlay::sequence<size_t>::uninitialized(query_points.size());
      auto timings = parlay::sequence<std::vector<double>>(query_points.size());

      // Warmup
      for (size_t j = 0; j < std::min(static_cast<size_t>(10), query_points.size()); j++) {
        auto [p, c] = index.search(query_points[j], points, params[i]);
      }
      double query_time_seq = 0.0;
      // Single Batch Run
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

      double QPS_seq = query_points.size() / query_time_seq;
      double QPS_par = QPS_seq;
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
        avg_timings[j] = total_time;  // / timings.size();
      }

      results[i] =
          mvsic::StatsExtended(QPS_seq, QPS_par, avg_cmps, recall_1_k, recall_k_k, avg_timings);
    }
  };
  parlay::execute_with_scheduler(num_threads, func);
  return results;
}

template<typename Index, typename PC>
mvsic::Stats compute_stats(Index& index, const PC& points, const PC& query_points,
                           const parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>>& gt,
                           const mvsic::SearchParams& params) {
  return compute_stats(index, points, query_points, gt,
                       parlay::sequence<mvsic::SearchParams>{params})[0];
}

template<typename Index, typename PC>
mvsic::StatsExtended compute_stats_extended(
    Index& index, const PC& points, const PC& query_points,
    const parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>>& gt,
    const mvsic::SearchParams& params) {
  return compute_stats_extended(index, points, query_points, gt,
                                parlay::sequence<mvsic::SearchParams>{params})[0];
}

template<typename Index, typename PC>
mvsic::StatsExtended compute_stats_extended_p_threaded(
    Index& index, const PC& points, const PC& query_points,
    const parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>>& gt,
    const mvsic::SearchParams& params, size_t num_threads = 1) {
  return compute_stats_extended_p_threaded(index, points, query_points, gt,
                                           parlay::sequence<mvsic::SearchParams>{params},
                                           num_threads)[0];
}

template<typename Index, typename PC>
mvsic::StatsExtended compute_stats_latency(
    Index& index, const PC& points, const PC& query_points,
    const parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>>& gt,
    const mvsic::SearchParams& params) {
  return compute_stats_latency(index, points, query_points, gt,
                               parlay::sequence<mvsic::SearchParams>{params})[0];
}

template<typename Index, typename PC>
mvsic::StatsExtended compute_stats_multi_latency(
    Index& index, const PC& points, const PC& query_points,
    const parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>>& gt,
    const mvsic::SearchParams& params) {
  return compute_stats_multi_latency(index, points, query_points, gt,
                                     parlay::sequence<mvsic::SearchParams>{params})[0];
}

template<typename Index, typename PC>
mvsic::StatsExtended compute_stats_batch(
    Index& index, const PC& points, const PC& query_points,
    const parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>>& gt,
    const mvsic::SearchParams& params) {
  return compute_stats_batch(index, points, query_points, gt,
                             parlay::sequence<mvsic::SearchParams>{params})[0];
}

inline std::pair<double, double> compute_scores(
    const parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>>& pred,
    const parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>>& gt, size_t k) {
  double recall_1_k = compute_recall(pred, gt, k, 1);
  double recall_k_k = compute_recall(pred, gt, k, k);
  return std::make_pair(recall_1_k, recall_k_k);
}

inline parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>> ReadGT(std::string& file_path,
                                                                             size_t num_points) {
  std::ifstream file(file_path, std::ios::binary | std::ios::in);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file for reading: " + file_path);
  }

  int num_neighbors = 0;
  file.read(reinterpret_cast<char*>(&num_neighbors), sizeof(num_neighbors));

  parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>> result(num_points);
  for (int i = 0; i < num_points; ++i) {
    parlay::sequence<std::pair<float, uint32_t>> neighbors(num_neighbors);
    file.read(reinterpret_cast<char*>(neighbors.data()),
              num_neighbors * sizeof(std::pair<float, uint32_t>));
    auto neighbors_flipped = parlay::sequence<std::pair<uint32_t, float>>::from_function(
        num_neighbors,
        [&](size_t j) { return std::make_pair(neighbors[j].second, neighbors[j].first); });
    result[i] = std::move(neighbors_flipped);
  }
  file.close();
  return result;
}

inline parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>> ReadGoldGT(
    std::string& file_path, size_t num_points) {
  std::ifstream file(file_path, std::ios::binary | std::ios::in);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file for reading: " + file_path);
  }
  size_t num_offsets;
  size_t num_gt_entries;
  // Read sizes
  file.read(reinterpret_cast<char*>(&num_offsets), sizeof(num_offsets));
  file.read(reinterpret_cast<char*>(&num_gt_entries), sizeof(num_gt_entries));
  std::vector<size_t> offsets(num_offsets);
  std::vector<uint32_t> ground_truth(num_gt_entries);
  // Read offset and ground truth data
  file.read(reinterpret_cast<char*>(offsets.data()), num_offsets * sizeof(size_t));
  file.read(reinterpret_cast<char*>(ground_truth.data()), num_gt_entries * sizeof(uint32_t));
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