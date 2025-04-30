#pragma once

#include "algorithms/utils/point_range.h"
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
// #include "seeding/wards.h"
#include "utils/evals.h"

template <typename T, typename Range>
T SumOfSquaredCost(const Range& points,
                   const parlay::sequence<uint32_t>& center_ids) {
  auto min_distances = parlay::delayed_seq<T>(points.size(), [&](size_t i) {
    auto new_distances = parlay::delayed_seq<T>(
        center_ids.size(),
        [&](size_t j) { return points[i].distance(points[center_ids[j]]); });
    T smallest_new_distance = reduce(new_distances, parlay::minm<T>());
    return smallest_new_distance;
  });
  return parlay::reduce(min_distances);
}

template <typename T, typename Range>
T SumOfSquaredCost(const Range& points, const Range& centers) {
  auto min_distances = parlay::delayed_seq<T>(points.size(), [&](size_t i) {
    auto new_distances = parlay::delayed_seq<T>(centers.size(), [&](size_t j) {
      return points[i].distance(centers[j]);
    });
    T smallest_new_distance = reduce(new_distances, parlay::minm<T>());
    return smallest_new_distance;
  });
  return parlay::reduce(min_distances);
}

template <typename T, typename Range>
T SumOfSquaredCost(const Range& points, const Range& centers,
                   const parlay::sequence<uint32_t>& clusters) {
  auto distances = parlay::delayed_seq<T>(points.size(), [&](size_t i) {
    return points[i].distance(centers[clusters[i]]);
  });
  return parlay::reduce(distances);
}

template <typename DistTy, typename PointTy>
auto kmeans(PointRange<DistTy, PointTy>& points, uint32_t k, 
    std::string seed_algo = "UniformlyRandom", std::string dist_algo = "ANNS", size_t lloyds_iterations = 10, 
    long R = 75, long L = 128, double alpha = 1.2, bool two_pass = true,
    double epsw = 0.8, double deltaw = 1.0, double samw = 20, bool wghw = false) {
  parlay::sequence<uint32_t> center_ids;
  parlay::sequence<uint32_t> cluster_ids;
  PointRange<DistTy, PointTy> centers;
  BuildParams BP(R, L, alpha, two_pass);
  // Seeding
  if (seed_algo == "SequentialPlusPlus") {
    center_ids = SequentialPlusPlus<DistTy>(points, k);
    centers = copyPoints<PointTy>(points, center_ids);
  } else if (seed_algo == "PrefixDoubling") {
    center_ids = PrefixDoubling<DistTy>(points, k);
    centers = copyPoints<PointTy>(points, center_ids);
  } else if (seed_algo == "UniformlyRandom") {
    center_ids = UniformlyRandom<DistTy>(points, k);
    centers = copyPoints<PointTy>(points, center_ids);
  } else if (seed_algo == "ParallelPlusPlus") {
    center_ids = ParallelPlusPlus<DistTy>(points, k);
    centers = copyPoints<PointTy>(points, center_ids);
  // } else if (seed_algo == "Wards") {
  //   centers = Wards<DistTy, PointTy>(points, k, BP, epsw, deltaw, samw, wghw);
  } else if (seed_algo == "KSetCover") {
    center_ids = KSetCover<DistTy, PointTy>(points, k, BP);
    centers = copyPoints<PointTy>(points, center_ids);
  } else {
    std::cout << "Error: seeding algorithm not specified correctly"
              << std::endl;
    abort();
  }
  
  if (dist_algo == "Pairwise") {
    cluster_ids = compute_cluster_ids_pairwise<PointTy>(points, centers);
  } else if (dist_algo == "ANNS") {
    cluster_ids = compute_cluster_ids_anns<PointTy>(points, centers, BP);
  } else {
    std::cout << "Error: distance oracle not specified correctly"
              << std::endl;
    abort();
  }

  for (size_t j = 0; j < lloyds_iterations; j++) {
    if (dist_algo == "Pairwise") {
      std::tie(centers, cluster_ids) =
          lloyds_pairwise<PointTy>(points, centers, cluster_ids);
    } else if (dist_algo == "ANNS") {
      std::tie(centers, cluster_ids) =
          lloyds_anns<PointTy>(points, centers, cluster_ids, BP);
    } else {
      std::cout << "Error: distance oracle not specified correctly"
                << std::endl;
      abort();
    }
  }
  return centers;
}