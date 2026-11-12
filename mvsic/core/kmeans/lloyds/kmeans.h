// Source: https://github.com/ParAlg/Parallel-KMeans, lloyds/kmeans.h
// Revision: 6cdb5b8649ea254c31a329fcdbbf6537e01452b2 (mvsic branch).
// Paper: UltraKMeans: Fully Scalable and High Quality k-means, PPoPP 2027
// (to appear). https://kishen19.github.io/ (citation in ../README.md).
// See ../README.md for provenance and local modifications.

#pragma once

#include "algorithms/utils/types.h"
#include "anns.h"
#include "multi-swap.h"
#include "pairwise.h"
#include "parlay/io.h"
#include "parlay/sequence.h"
#include "seeding/kmeansparallel.h"
#include "seeding/kmeansplusplus.h"
#include "seeding/ksetcover.h"
#include "seeding/prefixdoubling.h"
#include "seeding/uniformlyrandom.h"

template<typename PointTy, typename T = typename PointTy::distanceType, typename Range>
T SumOfSquaredCost(const Range& points, const parlay::sequence<uint32_t>& center_ids) {
  auto min_distances = parlay::delayed_seq<T>(points.size(), [&](size_t i) {
    auto new_distances = parlay::delayed_seq<T>(
        center_ids.size(), [&](size_t j) { return points[i].distance(points[center_ids[j]]); });
    T smallest_new_distance = reduce(new_distances, parlay::minm<T>());
    return smallest_new_distance;
  });
  return parlay::reduce(min_distances);
}

template<typename T, typename Range>
T SumOfSquaredCost(const Range& points, const Range& centers,
                   const parlay::sequence<uint32_t>& cluster_ids) {
  auto distances = parlay::delayed_seq<T>(
      points.size(), [&](size_t i) { return points[i].distance(centers[cluster_ids[i]]); });
  return parlay::reduce(distances);
}

template<typename PointTy, typename T = typename PointTy::distanceType, typename Range>
T SumOfSquaredCost(Range& points, Range& centers) {
  parlay::sequence<uint32_t> cluster_ids =
      compute_cluster_ids_pairwise_blocked<PointTy>(points, centers);
  return SumOfSquaredCost<T>(points, centers, cluster_ids);
}

template<typename DistTy, typename PointTy, typename Range>
auto kmeans(Range& points, uint32_t k, std::string seed_algo = "UniformlyRandom",
            std::string dist_algo = "Pairwise", size_t lloyds_iterations = 10, bool verbose = false,
            long R = 16, long L = 32, double alpha = 1.2, bool two_pass = false) {
  parlay::sequence<uint32_t> center_ids;
  parlay::sequence<uint32_t> cluster_ids;
  Range centers;
  BuildParams BP(R, L, alpha, two_pass);
  // Seeding
  if (seed_algo == "SequentialPlusPlus") {
    center_ids = SequentialPlusPlus<DistTy>(points, k);
    centers = copyPoints(points, center_ids);
  } else if (seed_algo == "PrefixDoubling") {
    center_ids = PrefixDoubling<DistTy>(points, k);
    centers = copyPoints(points, center_ids);
  } else if (seed_algo == "UniformlyRandom") {
    center_ids = UniformlyRandom<DistTy>(points, k);
    centers = copyPoints(points, center_ids);
  } else if (seed_algo == "ParallelPlusPlus") {
    center_ids = ParallelPlusPlus<DistTy>(points, k);
    centers = copyPoints(points, center_ids);
  } else if (seed_algo == "KSetCover") {
    center_ids = KSetCover<DistTy>(points, k, BP);
    centers = copyPoints(points, center_ids);
  } else {
    std::cout << "[kmeans] Error: seeding algorithm not specified correctly" << std::endl;
    abort();
  }

  if (dist_algo == "Pairwise") {
    cluster_ids = compute_cluster_ids_pairwise_blocked<PointTy>(points, centers);
  } else if (dist_algo == "ANNS") {
    cluster_ids = compute_cluster_ids_anns<PointTy>(points, centers, BP);
  } else {
    std::cout << "[kmeans] Error: distance oracle not specified correctly" << std::endl;
    abort();
  }

  if (verbose) {
    std::cout << "[kmeans] cost (Seeding): " << SumOfSquaredCost<PointTy>(points, centers)
              << std::endl;
  }

  for (size_t j = 0; j < lloyds_iterations; j++) {
    if (dist_algo == "Pairwise") {
      std::tie(centers, cluster_ids) =
            lloyds_pairwise<PointTy>(points, centers, cluster_ids, verbose);
    } else if (dist_algo == "ANNS") {
      std::tie(centers, cluster_ids) =
            lloyds_anns<PointTy>(points, centers, cluster_ids, BP);
    } else {
      std::cout << "[kmeans] Error: distance oracle not specified correctly" << std::endl;
      abort();
    }
    if (verbose) {
      std::cout << "[kmeans] cost (" << j + 1 << "): " << SumOfSquaredCost<PointTy>(points, centers)
                << std::endl;
    }
  }
  return centers;
}