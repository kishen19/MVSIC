/*
  Ward's Agglomerative Clustering for seeding k-means
  1. Exact Implementation:
    Input: Points, k
      - k: Number of final clusters
    Running Time: O(n^2 * polylog(n))
  2. Approximate Implementation
    Input: Points, k, eps, delta
      - k: Number of final clusters
      - eps: (1+eps) merges performed
      - delta: clusters of size within (1+delta) grouped together
    Approximation Factor: (1+eps)*(1+delta)
    Running Time: O(n * polylog(n))
*/

#pragma once
#include "lloyds/pairwise.h"
#include "parlay/delayed_sequence.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"
#include "wards/dyn_point_range.h"
#include "wards/wards_hac.h"

template<typename PointTy, typename DistTy = typename PointTy::distanceType, typename RangeP,
         typename Range>
auto compute_weights_anns(RangeP &points, Range &sampled_points) {  //, BuildParams BP) {
  size_t n = points.size();
  size_t ns = sampled_points.size();
  size_t d = points.get_dims();
  // Build Index on sampled points
  // Graph<uint32_t> G = Graph<uint32_t>(BP.R, ns);
  BuildParams BP(64, 128, 1.2, false);
  Graph<uint32_t> G = Graph<uint32_t>(BP.R, ns);
  knn_index<PointTy, Range, uint32_t> I(BP);
  stats<uint32_t> BuildStats(G.size());
  I.build_index(G, sampled_points, BuildStats);
  // Compute nearest sampled point for each point
  double cut = 1.35;
  auto QP = QueryParams(1, BP.L, cut, (long)G.size(), (long)G.max_degree());
  auto cluster_ids = parlay::sequence<tuple<uint32_t, uint32_t>>::uninitialized(n);
  parlay::parallel_for(0, n, [&](size_t i) {
    parlay::sequence<uint32_t> start_points = {I.get_start()};
    auto [pairElts, dist_cmps] =
        beam_search<PointTy, Range, uint32_t>(points[i], G, sampled_points, start_points, QP);
    auto [beamElts, visitedElts] = pairElts;
    uint32_t min_id = beamElts[0].first;
    cluster_ids[i] = make_tuple(min_id, i);
  });
  auto groups = parlay::group_by_index(cluster_ids, ns);
  parlay::parallel_for(0, ns, [&](size_t i) {
    if (groups[i].size() == 0) {
      return;
    }
    auto subset_points = parlay::delayed_seq<PointTy>(
        groups[i].size(), [&](size_t j) { return points[groups[i][j]]; });
    compute_mean<PointTy>(subset_points, sampled_points, i, d);
  });
  auto weights =
      parlay::sequence<uint32_t>::from_function(ns, [&](size_t i) { return groups[i].size() + 1; });
  return weights;
}

template<typename T, typename PointTy, typename Range>
auto Wards(const Range &points, uint32_t k, BuildParams BP, double eps = 0.8, double delta = 1.0,
           double over_samples = 20.0, bool weighted = true) {
  size_t n = points.size();
  unsigned int d = points.get_dims();
  size_t num_samples = std::min((size_t)std::ceil(k * over_samples), n);
  if (num_samples == n) {  // Run on all points
    auto dyn_points = dyn::PointRange<T, PointTy>(points, d);
    auto weights = parlay::sequence<uint32_t>(dyn_points.size(), 1);
    if (eps == 0.0) {  // Exact Wards
      auto center_ids = WardsHAC<uint32_t, T>(dyn_points, k, weights);
      auto centers = parlay::delayed_seq<PointTy>(
          center_ids.size(), [&](size_t i) { return dyn_points[center_ids[i]]; });
      return Range(centers, d);
    } else {  // Approximate Wards
      auto center_ids = ApxWardsHAC<uint32_t, T>(dyn_points, k, eps, delta, weights, BP);
      auto centers = parlay::delayed_seq<PointTy>(
          center_ids.size(), [&](size_t i) { return dyn_points[center_ids[i]]; });
      return Range(centers, d);
    }
  } else {
    // Step 1: Compute a random sub-sample of points
    auto sampled_ids = parlay::sequence<uint32_t>::from_function(num_samples, [&](size_t i) {
      uint32_t random_pt = parlay::hash32(i) % n;
      return random_pt;
    });
    auto samples = parlay::delayed_seq<PointTy>(sampled_ids.size(),
                                                [&](size_t i) { return points[sampled_ids[i]]; });
    auto sampled_points = dyn::PointRange<T, PointTy>(samples, d);
    std::cout << "Sampled points size: " << sampled_points.size() << std::endl;
    // Compute Weights
    parlay::sequence<uint32_t> weights;
    if (weighted) {
      weights = compute_weights_anns<PointTy>(points, sampled_points);
    } else {
      weights = parlay::sequence<uint32_t>(sampled_points.size(), 1);
    }

    // Step 2: Run Ward's HAC on the sample (returns a list of ids)
    if (eps == 0.0) {  // Run Exact Ward's
      auto center_ids = WardsHAC<uint32_t, T>(sampled_points, k, weights);
      auto centers = parlay::delayed_seq<PointTy>(
          center_ids.size(), [&](size_t i) { return sampled_points[center_ids[i]]; });
      return Range(centers, d);
    } else {  // Run Approx Ward's
      auto center_ids = ApxWardsHAC<uint32_t, T>(sampled_points, k, eps, delta, weights, BP);
      auto centers = parlay::delayed_seq<PointTy>(
          center_ids.size(), [&](size_t i) { return sampled_points[center_ids[i]]; });
      // writeVectorsToBinaryFile<T>("centers.fbin", centers, d);
      return Range(centers, d);
    }
  }
}

// template <typename T, typename Seq>
// void writeVectorsToBinaryFile(const std::string& filename, const Seq data, unsigned int d) {
//   std::ofstream writer(filename, std::ios::binary);
//   assert(writer.is_open());

//   unsigned int n = static_cast<unsigned int>(data.size());
//   std::vector<std::vector<T>> vectors(n, std::vector<T>(d));
//   for (size_t i=0; i<n; i++) {
//     for (size_t j=0; j<d; j++) {
//       vectors[i][j] = data[i][j];
//     }
//   }

//   // Write the number of points (n) and the dimension (d) as unsigned integers
//   writer.write(reinterpret_cast<const char*>(&n), sizeof(unsigned int));
//   writer.write(reinterpret_cast<const char*>(&d), sizeof(unsigned int));

//   // Write the raw binary data of the vectors
//   for (size_t i=0; i<n; i++) {
//     for (size_t j=0; j<d; j++) {
//       writer.write(reinterpret_cast<const char*>(&vectors[i][j]), sizeof(T));
//     }
//   }
//   std::cout << "Successfully wrote " << n << " points with dimension " << d << " to "
//             << filename << " (binary format)" << std::endl;
//   writer.close();
// }

// template <typename T, typename PointTy, typename Range>
// auto Wards(const Range &points, uint32_t k, BuildParams BP, double eps = 0.8,
//            double delta = 1.0, double over_samples = 20.0, bool
//            weighted=true) {
//   // Step 1: Compute a random sub-sample of points
//   size_t n = points.size();
//   // auto num_points = std::ceil(min(n, (size_t)(k * over_samples)));
//   // auto sampled_ids =
//   //     parlay::sequence<uint32_t>::from_function(num_points, [&](size_t i)
//   {
//   //       uint32_t random_pt = parlay::hash32(i) % n;
//   //       return random_pt;
//   //     });
//   auto samples =
//       parlay::tabulate<PointTy>(n, [&](size_t i) { return points[i]; });
//   int d = points.get_dims();
//   // Remark: Below is a mutable PointRange struct
//   // (that will store updated centroids as HAC progresses)
//   auto sampled_points = dyn::PointRange<T, PointTy>(samples, d);

//   // TODO: create copies to hack compute_cluster_ids_anns. Can be removed
//   // using DistTy = typename PointTy::distanceType;
//   // auto sampled_points_copy = PointRange<DistTy, PointTy>(samples, d);
//   // auto cluster_ids =
//   //     compute_cluster_ids_anns<PointTy>(points, sampled_points_copy, BP);
//   // parlay::sequence<double> weights(sampled_points.size(), 0);
//   // parlay::parallel_for(
//   //     0, n, [&](size_t i) { write_add(&weights[cluster_ids[i]], 1.0); });

//   // Step 2: Run Ward's HAC on the sample (returns a list of ids)
//   // auto center_ids = ApxWardsHAC<uint32_t, T>(sampled_points, k, eps,
//   delta,
//   // 75, 100, 1.2, false, 1, weights);
//   auto center_ids = ApxWardsHAC<uint32_t, T>(sampled_points, k, eps, delta);
//   auto centers = parlay::delayed_seq<PointTy>(center_ids.size(), [&](size_t
//   i) {
//     return sampled_points[center_ids[i]];
//   });
//   return Range(centers, d);
// }
