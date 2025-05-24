#pragma once

#include "parlay/sequence.h"
#include "seeding/uniformlyrandom.h"
#include "utils/faiss_kmeans.h"
#include "lower_bounds.h"

template <typename T, typename PointCloudRange>
T sum_of_squared_cost(const PointCloudRange& points, const PointCloudRange& centers,
  const parlay::sequence<uint32_t>& clusters) {
  auto distances = parlay::delayed_seq<T>(points.size(), [&](size_t i) {
    return points[i].distance(centers[clusters[i]]);
    });
  return parlay::reduce(distances);
}

template <typename PointCloudRange>
parlay::sequence<uint32_t> compute_cluster_ids(const PointCloudRange& points,
  const PointCloudRange& centers) {
  size_t n = points.size();
  size_t k = centers.size();
  parlay::sequence<uint32_t> updated_cluster_ids(n);
  parlay::parallel_for(0, n, [&](size_t i) {
    auto dist = parlay::tabulate(
      k, [&](size_t j) { return points[i].distance(centers[j]); });
    updated_cluster_ids[i] = parlay::min_element(dist) - begin(dist);
    });
  return updated_cluster_ids;
}

template <typename T, typename Range, typename PointCloudRange>
auto mvkmeans(const PointCloudRange& points, size_t k,
    size_t s = 0, long iters = 5, std::string seeding = "Random",
    bool comp_lb = false) {
  uint32_t n = points.size();
  uint32_t d = points.get_dims();
  bool is_metric = points[0].is_metric();

  if (s == 0) {
    auto num_embeddings = parlay::delayed_seq<size_t>(points.size(),
      [&](size_t i) { return points.get_size(i); });
    s = parlay::reduce(num_embeddings) / n;
    std::cout << "Average number of embeddings per point: " << s << std::endl;
  }
  if (comp_lb){
    auto lb = lowerbound<Range>(points, k, s);
    std::cout << "Naive Lower Bound: " << lb << std::endl;
  }
  // Step 1: Initialization 
  parlay::internal::timer st;
  st.start();
  PointCloudRange centers;
  parlay::sequence<uint32_t> cluster_ids;
  if (seeding == "Random") {
    centers = UniformlyRandomMV(points, k);
  } else {
    std::cout << "Error: seeding algorithm not specified correctly"
      << std::endl;
    abort();
  }
  cluster_ids = compute_cluster_ids(points, centers);
  st.stop();

  std::vector<double> lloyds_times;
  std::vector<T> costs;
  T seed_cost = sum_of_squared_cost<T>(points, centers, cluster_ids);
  costs.push_back(seed_cost);
  lloyds_times.push_back(st.total_time());
  std::cout << "Seeding cost: " << seed_cost << std::endl;
  std::cout << "Seeding time: " << st.total_time() << " seconds" << std::endl;

  // Step 2: Lloyd's Iteration
  parlay::internal::timer it_timer;
  T cost;
  for (long it = 0; it < iters; it++) {
    it_timer.start();
    // Step 2A: Compute new centers
    auto id_pt = parlay::delayed_seq<std::pair<uint32_t, uint32_t>>(n,
      [&](size_t i) { return std::make_pair(cluster_ids[i], i); });
    auto grouped = parlay::group_by_index(id_pt, k);
    auto new_centers = parlay::sequence<Range>(k);
    parlay::parallel_for(0, k, [&](size_t i) {
      if (grouped[i].size() > 0) {
        auto data = points.filter_flattened(grouped[i]);
        if (s >= data.size()) {
          new_centers[i] = Range(data, d);
        } else {
          new_centers[i] = Range(faiss_kmeans(data, d, s, is_metric), d);
        }
      } else { // Empty Cluster, sample from input
        std::cout << "Cluster " << i << ": empty" << std::endl;
        std::cout << "Sampling from input" << std::endl;
        static uint32_t seed = 42;
        parlay::sequence<uint32_t> id = {parlay::hash32(seed++) % n};
        auto data = points.filter_flattened(id);
        if (s >= data.size()) {
          new_centers[i] = Range(data, d);
        } else {
          new_centers[i] = Range(faiss_kmeans(data, d, s, is_metric), d);
        }
      }
    });
    centers = PointCloudRange(new_centers, d, {});
    // Step 2B: Reassign points
    cluster_ids = compute_cluster_ids(points, centers);
    it_timer.stop();
    double round_time = it_timer.total_time();
    lloyds_times.push_back(round_time);
    it_timer.reset();
    cost = sum_of_squared_cost<T>(points, centers, cluster_ids);
    std::cout << "Lloyd's iteration " << it << ": cost = " << cost << ", time = " 
              << round_time << " seconds" << std::endl;
    costs.push_back(cost);
  }
  for(auto c: costs) {
    std::cout << -c << std::endl;
  }
  for (auto t : lloyds_times) {
    std::cout << t << std::endl;
  }
  return std::make_pair(centers, cluster_ids);
}