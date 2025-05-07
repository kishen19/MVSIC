#pragma once

#include "parlay/sequence.h"
#include "lloyds/kmeans.h"
#include "seeding/uniformlyrandom.h"
#include "faisskmeans.h"

template <typename T, typename PointCloud>
T sum_of_squared_cost(const PointCloud& points, const PointCloud& centers,
  const parlay::sequence<uint32_t>& clusters) {
  auto distances = parlay::delayed_seq<T>(points.size(), [&](size_t i) {
    return points[i].distance(centers[clusters[i]]);
    });
  return parlay::reduce(distances);
}

template <typename PointCloud>
parlay::sequence<uint32_t> compute_cluster_ids(const PointCloud& points,
  const PointCloud& centers) {
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

template <typename Range, typename PointCloud>
auto mvkmeans(const PointCloud& points, size_t k,
  size_t s = 0, long iters = 5, std::string seeding = "Random",
  std::string kmeans_dist_algo = "Pairwise", std::string kmeans_seeding = "PrefixDoubling",
  long kmeans_iters = 20) {
  using T = typename PointCloud::T;
  uint32_t n = points.size();
  uint32_t d = points.get_dims();
  bool is_metric = points[0].is_metric();

  if (s == 0) {
    auto num_embeddings = parlay::delayed_seq<size_t>(points.size(),
      [&](size_t i) { return points.get_size(i); });
    s = parlay::reduce(num_embeddings) / n;
    std::cout << "Average number of embeddings per point: " << s << std::endl;
  }
  // Step 1: Initialization 
  parlay::internal::timer st;
  st.start();
  PointCloud centers;
  parlay::sequence<uint32_t> cluster_ids;
  if (seeding == "Random") {
    centers = UniformlyRandomMV(points, k);
  } else {
    std::cout << "Error: seeding algorithm not specified correctly"
      << std::endl;
    abort();
  }
  st.stop();

  cluster_ids = compute_cluster_ids(points, centers);
  T seed_cost = sum_of_squared_cost<T>(points, centers, cluster_ids);
  std::cout << "Seeding cost: " << seed_cost << std::endl;
  std::cout << "Seeding time: " << st.total_time() << " seconds" << std::endl;

  // Step 2: Lloyd's Iteration
  parlay::internal::timer it_timer;
  T cost;
  it_timer.start();
  for (long it = 0; it < iters; it++) {
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
          // auto range_data = Range(data, d);
          // new_centers[i] = kmeans(range_data, s, kmeans_seeding, kmeans_dist_algo, kmeans_iters);
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
          // auto range_data = Range(data, d);
          // new_centers[i] = kmeans(range_data, s, kmeans_seeding, kmeans_dist_algo, kmeans_iters);
        }
      }
      });
    std::cout << "Here" << std::endl;
    centers = PointCloud(new_centers, d, {});
    // Step 2B: Reassign points
    cluster_ids = compute_cluster_ids(points, centers);
    cost = sum_of_squared_cost<T>(points, centers, cluster_ids);
    std::cout << "Lloyd's iteration " << it << ": cost = " << cost << std::endl;
  }
  it_timer.stop();
  std::cout << "Lloyd's iterations time: " << it_timer.total_time() << " seconds" << std::endl;
  return std::make_pair(centers, cluster_ids);
}