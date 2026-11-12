// Source: https://github.com/ParAlg/Parallel-KMeans, lloyds_weighted/kmeans_weighted.h
// Revision: 6cdb5b8649ea254c31a329fcdbbf6537e01452b2 (mvsic branch).
// Paper: UltraKMeans: Fully Scalable and High Quality k-means, PPoPP 2027
// (to appear). https://kishen19.github.io/ (citation in ../README.md).
// See ../README.md for provenance and local modifications.

#pragma once

#include "algorithms/utils/types.h"
#include "lloyds/pairwise.h"
#include "lloyds/kmeans.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"
#include "seeding/uniformlyrandom.h"

// A weighted variant of Lloyd's k-means that:
//  - Uses the same PointRange / PointTy abstractions as the unweighted version,
//  - Supports only UniformlyRandom seeding and exact pairwise Lloyd's iterations,
//  - Incorporates per-point weights into the centroid update step.
//
// Intended use in MVSIC:
//  - Weights will typically be 1 / (point_cloud_size) for each flattened vector,
//    so that each original point cloud contributes equal total weight.

// Compute a weighted mean of a set of points.
// points: range-like object (supports size() and operator[] returning PointTy)
// weights: range-like object aligned with points (size(), operator[])
template<typename PointTy, typename SeqPoints, typename SeqWeights, typename Range>
void compute_weighted_mean(const SeqPoints& points, const SeqWeights& weights, Range& centers,
                           uint32_t id, uint32_t d) {
  using T = typename PointTy::distanceType;
  assert(points.size() != 0);
  assert(points.size() == weights.size());

  // Total weight for this cluster
  T total_w = parlay::reduce(weights);
  if (total_w == static_cast<T>(0)) {
    total_w = static_cast<T>(1);
  }

  // Weighted sum for each coordinate
  for (size_t dim = 0; dim < d; dim++) {
    auto coord_weighted = parlay::delayed_seq<T>(points.size(), [&](size_t j) {
      return static_cast<T>(points[j][dim]) * weights[j];
    });
    centers[id][dim] = parlay::reduce(coord_weighted) / total_w;
  }

  // For IP (non-metric) points, normalize to unit norm, mirroring unweighted compute_mean.
  if (!centers[0].is_metric()) {
    double sum_sqrs = 0.0;
    for (size_t dim = 0; dim < d; dim++) {
      double v = centers[id][dim];
      sum_sqrs += v * v;
    }
    if (sum_sqrs != 0.0) {
      float inv = 1.0f / std::sqrt(sum_sqrs);
      for (size_t dim = 0; dim < d; dim++) {
        centers[id][dim] *= inv;
      }
    }
  }
}

// Compute weighted centroids from cluster assignments.
// `weights_global` is a length-n array of per-point weights (same indexing as `points`).
template<typename PointTy, typename DistTy = typename PointTy::distanceType, typename Range>
auto compute_centroids_from_clusters_weighted(
    const Range& points, const parlay::sequence<DistTy>& weights_global,
    parlay::sequence<uint32_t>& cluster_ids, size_t k, bool verbose = false) {
  using IndexPair = std::pair<uint32_t, uint32_t>;

  size_t n = points.size();
  size_t d = points.get_dims();
  assert(weights_global.size() == n);

  auto min_distance_ids =
      parlay::sequence<IndexPair>::from_function(n, [&](size_t i) {
        return std::make_pair(cluster_ids[i], static_cast<uint32_t>(i));
      });
  auto grouped = parlay::group_by_index(min_distance_ids, k);

  Range updated_centers(k, d);
  static uint32_t seed = 0;
  auto sampled = parlay::sequence<uint32_t>(k, k);

  parlay::parallel_for(0, k, [&](size_t cid) {
    if (grouped[cid].size() > 0) {
      // Build per-cluster views over points and weights.
      auto subset_points = parlay::delayed_seq<typename Range::Point>(
          grouped[cid].size(),
          [&](size_t j) { return points[grouped[cid][j]]; });
      auto subset_weights = parlay::delayed_seq<DistTy>(
          grouped[cid].size(),
          [&](size_t j) { return weights_global[grouped[cid][j]]; });

      compute_weighted_mean<PointTy>(subset_points, subset_weights, updated_centers,
                                     static_cast<uint32_t>(cid), static_cast<uint32_t>(d));
    } else {
      // Empty cluster: sample a point from the data as in the unweighted version.
      uint32_t id = parlay::hash32(seed + cid) % n;
      updated_centers[cid] = points[id];
      cluster_ids[id] = static_cast<uint32_t>(cid);
      sampled[cid] = static_cast<uint32_t>(cid);
    }
  });

  auto filtered_samples =
      parlay::filter(sampled, [&](uint32_t i) { return i < k; });
  if (verbose) {
    std::cout << "Sampled " << filtered_samples.size()
              << " centers from data (for empty clusters)." << std::endl;
  }
  seed += k;
  return updated_centers;
}

// One Lloyds iteration with weighted centroid updates and exact pairwise assignments.
template<typename PointTy, typename DistTy = typename PointTy::distanceType, typename Range>
auto lloyds_pairwise_weighted(const Range& points,
                              const parlay::sequence<DistTy>& weights,
                              Range& centers,
                              parlay::sequence<uint32_t>& cluster_ids,
                              bool verbose = false) {
  size_t k = centers.size();

  // Step 1: Compute weighted centroids of each cluster.
  auto updated_centers =
      compute_centroids_from_clusters_weighted<PointTy>(points, weights, cluster_ids, k, verbose);

  // Step 2: Assign points to new clusters (unweighted distance oracle).
  auto updated_cluster_ids =
      compute_cluster_ids_pairwise_blocked<PointTy>(const_cast<Range&>(points), updated_centers);

  return std::make_pair(updated_centers, updated_cluster_ids);
}

// Top-level weighted k-means:
//  - points: Range of PointTy
//  - weights: per-point weights (same indexing as points)
//  - k: number of clusters
//  - seed_algo/dist_algo: only "UniformlyRandom" + "Pairwise" are supported here.
template<typename DistTy, typename PointTy, typename Range>
auto kmeans_weighted(Range& points,
                     const parlay::sequence<DistTy>& weights,
                     uint32_t k,
                     std::string seed_algo = "UniformlyRandom",
                     std::string dist_algo = "Pairwise",
                     size_t lloyds_iterations = 10,
                     bool verbose = false) {
  parlay::sequence<uint32_t> center_ids;
  parlay::sequence<uint32_t> cluster_ids;
  Range centers;

  if (weights.size() != points.size()) {
    std::cout << "[kmeans_weighted] Error: weights.size() must equal points.size()." << std::endl;
    abort();
  }

  // Seeding: we only support UniformlyRandom here.
  if (seed_algo == "UniformlyRandom") {
    center_ids = UniformlyRandom<DistTy>(points, k);
    centers = copyPoints(points, center_ids);
  } else {
    std::cout << "[kmeans_weighted] Error: only UniformlyRandom seeding is supported."
              << std::endl;
    abort();
  }

  // Initial assignments: exact pairwise.
  if (dist_algo == "Pairwise") {
    cluster_ids = compute_cluster_ids_pairwise_blocked<PointTy>(points, centers);
  } else {
    std::cout << "[kmeans_weighted] Error: only Pairwise distance oracle is supported."
              << std::endl;
    abort();
  }

  if (verbose) {
    auto init_cost = SumOfSquaredCost<DistTy>(points, centers, cluster_ids);
    std::cout << "[kmeans_weighted] cost (Seeding, before Lloyd's): "
              << init_cost << std::endl;
  }

  for (size_t j = 0; j < lloyds_iterations; j++) {
    parlay::internal::timer it_timer;
    it_timer.start();
    std::tie(centers, cluster_ids) =
        lloyds_pairwise_weighted<PointTy>(points, weights, centers, cluster_ids, verbose);
    it_timer.stop();

    if (verbose) {
      auto cost = SumOfSquaredCost<DistTy>(points, centers, cluster_ids);
      std::cout << "[kmeans_weighted] iteration " << (j + 1)
                << ": cost = " << cost
                << " time = " << it_timer.total_time() << std::endl;
    }
  }

  return centers;
}

