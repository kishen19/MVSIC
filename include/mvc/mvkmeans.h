#pragma once

#include "parlay/sequence.h"
#include "lloyds/kmeans.h"
#include "seeding/uniformlyrandom.h"

template <typename T, typename PointCloud>
T sum_of_squared_cost(const PointCloud& points, const PointCloud& centers,
                   const parlay::sequence<uint32_t>& clusters) {
  auto distances = parlay::delayed_seq<T>(points.size(), [&](size_t i) {
    return points[i].distance(centers[clusters[i]]);
  });
  return parlay::reduce(distances);
}

template <typename PointCloud>
parlay::sequence<uint32_t> compute_cluster_ids(const PointCloud &points, 
    const PointCloud &centers) {
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
auto mvkmeans(const PointCloud& points, size_t k, size_t s = 0,
    long iters = 5, std::string seeding="Random",
    std::string kmeans_dist_algo = "ANNS", std::string kmeans_seeding = "PrefixDoubling",
    long kmeans_iters = 10) {
  using T = typename PointCloud::T;
  uint32_t n = points.size();
  uint32_t d = points.dimension();
  
  if (s == 0){
    auto num_embeddings = parlay::delayed_seq<size_t>(points.size(), 
        [&](size_t i) { return points.NumEmb(i); });
    s = parlay::reduce(num_embeddings)/n;
  }
  // Step 1: Initialization 
  PointCloud centers;
  parlay::sequence<uint32_t> cluster_ids;
  if (seeding == "Random"){
    centers = UniformlyRandomMV(points, k);
  } else {
    std::cout << "Error: seeding algorithm not specified correctly"
              << std::endl;
    abort();
  }
  
  cluster_ids = compute_cluster_ids(points, centers);
  T seed_cost = sum_of_squared_cost<T>(points, centers, cluster_ids);
  std::cout << "Seeding cost: " << seed_cost << std::endl;

  // Step 2: Lloyd's Iteration
  T cost;
  for (long it = 0; it < iters; it++){
    // Step 2A: Compute new centers
    auto id_pt = parlay::delayed_seq<std::pair<uint32_t, uint32_t>>(n, 
        [&](size_t i) { return std::make_pair(cluster_ids[i], i); });
    auto grouped = parlay::group_by_index(id_pt, k);
    auto new_centers = parlay::sequence<Range>(k);
    parlay::parallel_for(0, k, [&](size_t i) {
      if (grouped[i].size() > 0){
        auto data = points.GetRange(grouped[i]);
        if (s >= data.size()) {
          new_centers[i] = data;
        } else {
          new_centers[i] = kmeans(data, std::min(s, data.size()), kmeans_seeding, 
                                kmeans_dist_algo, kmeans_iters);
        }
      } else { // Empty Cluster, sample from input
        static uint32_t seed = 42;
        uint32_t id = parlay::hash32(seed++) % n;
        auto data = Range(points[id], d);
        if (s >= data.size()) {
          new_centers[i] = data;
        } else {
          new_centers[i] = kmeans(data, std::min(s, data.size()), kmeans_seeding, 
                                kmeans_dist_algo, kmeans_iters);
        }
      }
    });
    std::cout << "Here" << std::endl;
    centers = PointCloud(new_centers, d);
    // Step 2B: Reassign points
    cluster_ids = compute_cluster_ids(points, centers);
    cost = sum_of_squared_cost<T>(points, centers, cluster_ids);
    std::cout << "Lloyd's iteration " << it << ": cost = " << cost << std::endl;
  }
  return std::make_pair(centers, cluster_ids);
}