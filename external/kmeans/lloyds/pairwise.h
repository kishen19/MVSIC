#include "parlay/delayed_sequence.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"

template <typename PointTy, typename DistTy = typename PointTy::distanceType,
          typename Range>
PointRange<DistTy, PointTy> copyPoints(
    const Range &points, const parlay::sequence<uint32_t> &center_ids) {
  auto centers = parlay::delayed_seq<PointTy>(
      center_ids.size(), [&](size_t i) { return points[center_ids[i]]; });
  int d = points.get_dims();
  return PointRange<DistTy, PointTy>(centers, d);
}

template <typename PointTy, typename Seq, typename Range>
void compute_mean(const Seq &points, Range &centers, uint32_t id, uint32_t d) {
  using T = typename PointTy::distanceType;
  assert(points.size() != 0);
  for (size_t i = 0; i < d; i++) {
    auto coords = parlay::delayed_seq<T>(
        points.size(), [&](size_t j) { return points[j][i]; });
    centers[id][i] = reduce(coords) / points.size();
  }
}

template <typename PointTy, typename DistTy = typename PointTy::distanceType,
          typename Range>
auto compute_centroids_from_clusters(
    const Range &points, const parlay::sequence<uint32_t> &cluster_ids,
    size_t k) {
  size_t n = points.size();
  size_t d = points.get_dims();

  auto min_distance_ids =
      parlay::sequence<std::pair<uint32_t, uint32_t>>::from_function(
          n, [&](size_t i) { return std::make_pair(cluster_ids[i], i); });
  parlay::sort_inplace(make_slice(min_distance_ids));
  auto cutoff_indices = parlay::delayed_seq<uint32_t>(n + 1, [&](size_t i) {
    return i == 0 || i == n ||
           min_distance_ids[i].first != min_distance_ids[i - 1].first;
  });
  auto offsets = parlay::pack_index(cutoff_indices);
  // assert(offsets.size() == k + 1);
  Range updated_centers(k, d);
  parlay::parallel_for(0, offsets.size() - 1, [&](size_t i) {
    size_t start_index = offsets[i];
    size_t end_index = offsets[i + 1];
    auto subset_points =
        parlay::delayed_seq<PointTy>(end_index - start_index, [&](size_t j) {
          return points[min_distance_ids[j + start_index].second];
        });
    compute_mean<PointTy>(subset_points, updated_centers, i, d);
  });
  size_t num_assigned = offsets.size() - 1;
  if (num_assigned != k) {
    printf("Needs %zu more centers\n", k - num_assigned);
  }
  static uint32_t seed = 0;
  parlay::parallel_for(num_assigned, k, [&](size_t i) {
    uint32_t id = parlay::hash32(seed + i - num_assigned) % n;
    updated_centers[i] = points[id];
  });
  seed += (k - num_assigned);
  return updated_centers;
}

template <typename PointTy, typename DistTy = typename PointTy::distanceType,
          typename Range>
auto compute_cluster_ids_pairwise(const Range &points, const Range &centers) {
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

template <typename PointTy, typename DistTy = typename PointTy::distanceType,
          typename Range>
auto lloyds_pairwise(const Range &points, const Range &centers,
                     const parlay::sequence<uint32_t> &cluster_ids) {
  size_t k = centers.size();

  // Step 1: Compute centroids of each cluster
  auto updated_centers =
      compute_centroids_from_clusters<PointTy>(points, cluster_ids, k);

  // Step 2: Assigning points to new clusters
  auto updated_cluster_ids =
      compute_cluster_ids_pairwise<PointTy>(points, updated_centers);

  return std::make_pair(updated_centers, updated_cluster_ids);
}
