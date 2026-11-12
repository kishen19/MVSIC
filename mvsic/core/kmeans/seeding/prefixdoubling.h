// Source: https://github.com/ParAlg/Parallel-KMeans, seeding/prefixdoubling.h
// Revision: 6cdb5b8649ea254c31a329fcdbbf6537e01452b2 (mvsic branch).
// Paper: UltraKMeans: Fully Scalable and High Quality k-means, PPoPP 2027
// (to appear). https://kishen19.github.io/ (citation in ../README.md).
// See ../README.md for provenance and local modifications.

#include "kmeans_utils.h"
#include "parlay/delayed_sequence.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"

template <typename T, typename Range>
parlay::sequence<uint32_t> PrefixDoubling(const Range &points, uint32_t k) {
  size_t n = points.size();
  parlay::sequence<T> distances(n, std::numeric_limits<T>::max());
  parlay::sequence<bool> centers_set(n);

  parlay::sequence<uint32_t> centers(k);
  T sum_squared_distances = 0;
  size_t selected_centers = 0;
  for (size_t round_id = 0; selected_centers < k; round_id++) {
    parlay::sequence<uint32_t> new_centers;
    if (round_id == 0) {
      uint32_t random_center = parlay::hash32(round_id) % n;
      new_centers.push_back(random_center);
    } else {
      // TODO: add command line options for alpha
      constexpr double alpha = 2;
      size_t num_centers_to_add =
          std::min(static_cast<size_t>(floor(pow(alpha, round_id))),
                   n - selected_centers);
      new_centers = PickRandomCenters(
          distances, round_id, sum_squared_distances, num_centers_to_add);
    }
    // resize if more than k centers are selected
    if (selected_centers + new_centers.size() > k) {
      new_centers.resize(k - selected_centers);
    }
    if (!new_centers.empty()) {
      parlay::parallel_for(0, new_centers.size(), [&](size_t i) {
        uint32_t c = new_centers[i];
        centers_set[c] = true;
        centers[selected_centers + i] = c;
      });
      selected_centers += new_centers.size();
      sum_squared_distances =
          BatchUpdateDistances(points, distances, new_centers);
    }
  }
  return centers;
}