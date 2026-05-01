#pragma once

#include "parlay/primitives.h"
#include "algorithms/utils/euclidian_point.h"
#include "algorithms/utils/mips_point.h"
#include "algorithms/utils/point_range.h"
#include "lloyds/kmeans.h"
#include "lloyds_weighted/kmeans_weighted.h"
#include "mvsic/core/utils/lloyds_kmeans.h"

namespace mvsic {

// Runs kmeans on a subsample of size max_points_per_centroid*k.
//
// The Lloyd's loop is now driven by ::mvsic::lloyds::lloyds_kmeans against a
// FloatLloydsBackend; the backend wraps a parlayANN PointRange<float> so the
// seeding, blocked-Eigen assignment, and per-cluster mean math match the
// previous parlayANN call path exactly.  Swapping the backend is the only
// thing needed to change the underlying point representation.
//
// `tq8_cache` and `tq8_vec_root_indices`, if provided alongside
// `use_tq8 = true`, let the TQ8 backend skip its rotate+quantize step and
// instead gather the per-row encoded state out of a shared build-top-level
// per-vector buffer.  `tq8_vec_root_indices[i]` must index the cache row that
// was produced by quantizing `data[i]` under `tq8_cache->model`'s rotator.
// When the inner loop subsamples to m < n rows, the same hash-map is applied
// to vec_root_indices so the alignment is preserved.
template<bool metric>
::mvsic::lloyds::CenterSet kmeans_subsample(
    const parlay::sequence<parlay::sequence<float>>& data, uint32_t k,
    uint32_t max_points_per_centroid, bool verbose = false, bool use_tq8 = false,
    const ::mvsic::lloyds::TQ8VectorCache<metric>* tq8_cache = nullptr,
    const parlay::sequence<uint32_t>* tq8_vec_root_indices = nullptr) {
  const char* seed_algo = metric ? "PrefixDoubling" : "UniformlyRandom";

  size_t n = data.size();
  uint32_t dims = static_cast<uint32_t>(data[0].size());
  const bool have_cache = use_tq8 && tq8_cache != nullptr && tq8_cache->model != nullptr &&
                          tq8_vec_root_indices != nullptr &&
                          tq8_vec_root_indices->size() == n;
  if (static_cast<size_t>(max_points_per_centroid) * k >= n) {
    if (use_tq8) {
      if (have_cache) {
        ::mvsic::lloyds::TQ8LloydsBackend<metric> B(data, dims, *tq8_cache,
                                                    *tq8_vec_root_indices);
        return ::mvsic::lloyds::lloyds_kmeans(B, k, seed_algo, /*niters=*/10, verbose);
      }
      ::mvsic::lloyds::TQ8LloydsBackend<metric> B(data, dims);
      return ::mvsic::lloyds::lloyds_kmeans(B, k, seed_algo, /*niters=*/10, verbose);
    }
    ::mvsic::lloyds::FloatLloydsBackend<metric> B(data, dims);
    return ::mvsic::lloyds::lloyds_kmeans(B, k, seed_algo, /*niters=*/10, verbose);
  }
  const size_t m = static_cast<size_t>(max_points_per_centroid) * k;
  auto sampled_points = parlay::delayed_tabulate(m, [&](size_t i) {
    size_t id = parlay::hash32(static_cast<uint32_t>(i)) % n;
    return data[id];
  });
  if (use_tq8) {
    if (have_cache) {
      parlay::sequence<uint32_t> sampled_vec_idx(m);
      parlay::parallel_for(0, m, [&](size_t i) {
        size_t id = parlay::hash32(static_cast<uint32_t>(i)) % n;
        sampled_vec_idx[i] = (*tq8_vec_root_indices)[id];
      });
      ::mvsic::lloyds::TQ8LloydsBackend<metric> B(sampled_points, dims, *tq8_cache,
                                                  sampled_vec_idx);
      return ::mvsic::lloyds::lloyds_kmeans(B, k, seed_algo, /*niters=*/10, verbose);
    }
    ::mvsic::lloyds::TQ8LloydsBackend<metric> B(sampled_points, dims);
    return ::mvsic::lloyds::lloyds_kmeans(B, k, seed_algo, /*niters=*/10, verbose);
  }
  ::mvsic::lloyds::FloatLloydsBackend<metric> B(sampled_points, dims);
  return ::mvsic::lloyds::lloyds_kmeans(B, k, seed_algo, /*niters=*/10, verbose);
}

// Runs weighted kmeans on a subsample of size max_points_per_centroid*k.
// `weights` provides a per-point weight for each row in `data`.
// When max_points_per_centroid * k < n, both data and weights are subsampled
// using the same random indices.  The optional TQ8 cache args mirror the
// non-weighted version: when supplied, the backend skips per-row
// rotate+quantize and gathers from `*tq8_cache` via `*tq8_vec_root_indices`.
template<bool metric>
::mvsic::lloyds::CenterSet kmeans_weighted_subsample(
    const parlay::sequence<parlay::sequence<float>>& data,
    const parlay::sequence<float>& weights, uint32_t k, uint32_t max_points_per_centroid,
    bool verbose = false, bool use_tq8 = false,
    const ::mvsic::lloyds::TQ8VectorCache<metric>* tq8_cache = nullptr,
    const parlay::sequence<uint32_t>* tq8_vec_root_indices = nullptr) {
  size_t n = data.size();
  uint32_t dims = static_cast<uint32_t>(data[0].size());
  if (weights.size() != n) {
    std::cerr << "[kmeans_weighted_subsample] Error: weights.size() != data.size()." << std::endl;
    abort();
  }

  const char* seed_algo = metric ? "PrefixDoubling" : "UniformlyRandom";
  const bool have_cache = use_tq8 && tq8_cache != nullptr && tq8_cache->model != nullptr &&
                          tq8_vec_root_indices != nullptr &&
                          tq8_vec_root_indices->size() == n;

  if (static_cast<size_t>(max_points_per_centroid) * k >= n) {
    if (use_tq8) {
      if (have_cache) {
        ::mvsic::lloyds::TQ8LloydsBackend<metric> B(data, dims, *tq8_cache,
                                                    *tq8_vec_root_indices);
        return ::mvsic::lloyds::lloyds_kmeans(B, k, seed_algo, /*niters=*/10, verbose, &weights);
      }
      ::mvsic::lloyds::TQ8LloydsBackend<metric> B(data, dims);
      return ::mvsic::lloyds::lloyds_kmeans(B, k, seed_algo, /*niters=*/10, verbose, &weights);
    }
    ::mvsic::lloyds::FloatLloydsBackend<metric> B(data, dims);
    return ::mvsic::lloyds::lloyds_kmeans(B, k, seed_algo, /*niters=*/10, verbose, &weights);
  }
  const size_t m = static_cast<size_t>(max_points_per_centroid) * k;
  auto sampled_points = parlay::delayed_tabulate(m, [&](size_t i) {
    size_t id = parlay::hash32(static_cast<uint32_t>(i)) % n;
    return data[id];
  });
  parlay::sequence<float> sampled_weights(m);
  parlay::parallel_for(0, m, [&](size_t i) {
    size_t id = parlay::hash32(static_cast<uint32_t>(i)) % n;
    sampled_weights[i] = weights[id];
  });
  if (use_tq8) {
    if (have_cache) {
      parlay::sequence<uint32_t> sampled_vec_idx(m);
      parlay::parallel_for(0, m, [&](size_t i) {
        size_t id = parlay::hash32(static_cast<uint32_t>(i)) % n;
        sampled_vec_idx[i] = (*tq8_vec_root_indices)[id];
      });
      ::mvsic::lloyds::TQ8LloydsBackend<metric> B(sampled_points, dims, *tq8_cache,
                                                  sampled_vec_idx);
      return ::mvsic::lloyds::lloyds_kmeans(B, k, seed_algo, /*niters=*/10, verbose,
                                            &sampled_weights);
    }
    ::mvsic::lloyds::TQ8LloydsBackend<metric> B(sampled_points, dims);
    return ::mvsic::lloyds::lloyds_kmeans(B, k, seed_algo, /*niters=*/10, verbose,
                                          &sampled_weights);
  }
  ::mvsic::lloyds::FloatLloydsBackend<metric> B(sampled_points, dims);
  return ::mvsic::lloyds::lloyds_kmeans(B, k, seed_algo, /*niters=*/10, verbose,
                                        &sampled_weights);
}

// Runs kmeans on a subsample of size max_points_per_centroid*k
// Returns centers, cluster_ids, and indices of "valid" clusters
// Validity: Not empty and Not containing more than maxsize many identical clusters
template<bool metric>
auto kmeans_subsample_assign(const parlay::sequence<parlay::sequence<float>>& data, size_t k,
                             size_t max_points_per_centroid, size_t maxsize, bool verbose = false) {
  using PointTy =
      std::conditional_t<metric, parlayANN::Euclidian_Point<float>, parlayANN::Mips_Point<float>>;
  using Range = parlayANN::PointRange<PointTy>;
  size_t n = data.size();
  size_t dims = data[0].size();

  const char* seed_algo;
  if constexpr (metric) {
    seed_algo = "PrefixDoubling";
  } else {
    seed_algo = "UniformlyRandom";
  }

  Range data_range = Range(data, dims);
  Range centers;
  if (max_points_per_centroid * k >= n) {
    centers = kmeans<float, PointTy>(data_range, k, seed_algo, "Pairwise", 10, verbose);
  } else {
    auto sampled_points = parlay::delayed_tabulate(max_points_per_centroid * k, [&](size_t i) {
      size_t id = parlay::hash32(static_cast<uint32_t>(i)) % n;
      return data[id];
    });
    Range sampled_data_range = Range(sampled_points, dims);
    centers = kmeans<float, PointTy>(sampled_data_range, k, seed_algo, "Pairwise", 10, verbose);
  }
  parlay::sequence<uint32_t> cluster_ids_ =
      compute_cluster_ids_pairwise_blocked<PointTy>(data_range, centers);
  auto cluster_ids = parlay::sequence<size_t>::from_function(
      n, [&](size_t i) { return static_cast<size_t>(cluster_ids_[i]); });
  // Remove clusters with all almost-duplicates
  auto id_pt =
      parlay::delayed_tabulate(n, [&](size_t i) { return std::make_pair(cluster_ids[i], i); });
  auto grouped = parlay::group_by_index(id_pt, k);
  parlay::sequence<bool> active(k, true);
  parlay::parallel_for(0, k, [&](size_t i) {
    if (grouped[i].size() == 0) {
      active[i] = false;
    } else {
      auto is_duplicate = parlay::sequence<int>::from_function(grouped[i].size(), [&](size_t j) {
        if (data_range[grouped[i][j]] == data_range[grouped[i][0]]) {
          return 1;
        }
        return 0;
      });
      size_t num_duplicates = parlay::reduce(is_duplicate);
      if (num_duplicates >= grouped[i].size() - 2 && grouped[i].size() > maxsize) {
        active[i] = false;
      }
    }
  });
  auto active_indices = parlay::pack_index(active);
  return std::make_tuple(centers, cluster_ids, active_indices);
}

// Runs kmeans on a subsample of size max_points_per_centroid*k
// Returns centers and cluster_ids
template<bool metric>
auto kmeans_subsample_assign_only(const parlay::sequence<parlay::sequence<float>>& data, size_t k,
                                  size_t max_points_per_centroid, bool verbose = false,
                                  size_t lloyds_iterations = 10) {
  using PointTy =
      std::conditional_t<metric, parlayANN::Euclidian_Point<float>, parlayANN::Mips_Point<float>>;
  using Range = parlayANN::PointRange<PointTy>;
  size_t n = data.size();
  size_t dims = data[0].size();

  const char* seed_algo;
  if constexpr (metric) {
    seed_algo = "PrefixDoubling";
  } else {
    seed_algo = "UniformlyRandom";
  }

  Range data_range = Range(data, dims);
  Range centers;
  if (max_points_per_centroid * k >= n) {
    centers =
        kmeans<float, PointTy>(data_range, k, seed_algo, "Pairwise", lloyds_iterations, verbose);
  } else {
    auto sampled_points = parlay::delayed_tabulate(max_points_per_centroid * k, [&](size_t i) {
      size_t id = parlay::hash32(static_cast<uint32_t>(i)) % n;
      return data[id];
    });
    Range sampled_data_range = Range(sampled_points, dims);
    centers = kmeans<float, PointTy>(sampled_data_range, k, seed_algo, "Pairwise",
                                     lloyds_iterations, verbose);
  }
  parlay::sequence<uint32_t> cluster_ids_ =
      compute_cluster_ids_pairwise_blocked<PointTy>(data_range, centers);
  auto cluster_ids = parlay::sequence<size_t>::from_function(
      n, [&](size_t i) { return static_cast<size_t>(cluster_ids_[i]); });
  return std::make_tuple(centers, cluster_ids);
}

// Runs kmeans on a subsample and returns the top N assignments for each point.
template<bool metric>
auto kmeans_subsample_top_n_assign(const parlay::sequence<parlay::sequence<float>>& data, size_t k,
                                   size_t N, size_t max_points_per_centroid, bool verbose = false) {
  using PointTy =
      std::conditional_t<metric, parlayANN::Euclidian_Point<float>, parlayANN::Mips_Point<float>>;
  using Range = parlayANN::PointRange<PointTy>;
  size_t n = data.size();
  size_t dims = data[0].size();

  const char* seed_algo;
  if constexpr (metric) {
    seed_algo = "PrefixDoubling";
  } else {
    seed_algo = "UniformlyRandom";
  }

  Range data_range = Range(data, dims);
  Range centers;
  if (max_points_per_centroid * k >= n) {
    centers = kmeans<float, PointTy>(data_range, k, seed_algo, "Pairwise", 10, verbose);
  } else {
    auto sampled_points = parlay::delayed_tabulate(max_points_per_centroid * k, [&](size_t i) {
      size_t id = parlay::hash32(static_cast<uint32_t>(i)) % n;
      return data[id];
    });
    Range sampled_data_range = Range(sampled_points, dims);
    centers = kmeans<float, PointTy>(sampled_data_range, k, seed_algo, "Pairwise", 10, verbose);
  }

  // For each point, find the top N closest centers
  auto top_n_assignments = parlay::tabulate(n, [&](size_t i) {
    auto dists = parlay::tabulate(
        k, [&](size_t j) { return std::make_pair(j, data_range[i].distance(centers[j])); });
    parlay::sort_inplace(dists, [](const auto& a, const auto& b) { return a.second < b.second; });
    return parlay::tabulate(std::min(N, k), [&](size_t rank) { return dists[rank].first; });
  });

  parlay::sequence<parlay::sequence<float>> final_centers(k);
  parlay::parallel_for(0, k, [&](size_t i) {
    parlay::sequence<float> center(dims);
    for (size_t j = 0; j < dims; j++) {
      center[j] = centers[i][j];
    }
    final_centers[i] = std::move(center);
  });

  return std::make_tuple(final_centers, top_n_assignments);
}

}  // namespace mvsic
