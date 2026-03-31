#pragma once

#include "point_cloud_set.h"

namespace mvsic {

// =========================================================================
// PointCloudSetSlice Implementations
// =========================================================================

template<typename ChPoint>
size_t PointCloudSetSlice<ChPoint>::distances(const ChPoint& query,
                                              std::pair<uint32_t, float>* results) const {
  auto cmps = parlay::sequence<size_t>::uninitialized(n);
  parlay::parallel_for(0, n, [&](uint32_t i) {
    std::tie(results[i].second, cmps[i]) = query.distance_w_cmps(parent[start_idx + i]);
    results[i].first = get_id(i);
  });
  return parlay::reduce(cmps);
}

template<typename ChPoint>
std::pair<parlay::sequence<std::pair<uint32_t, float>>, size_t>
PointCloudSetSlice<ChPoint>::distances(const ChPoint& query) const {
  auto results = parlay::sequence<std::pair<uint32_t, float>>::uninitialized(n);
  auto cmps = distances(query, results.begin());
  return std::make_pair(results, cmps);
}

// =========================================================================
// PointCloudSet Constructors & Initialization
// =========================================================================

// Private Zero-Copy Constructor for Shuffle
template<typename ChPoint>
PointCloudSet<ChPoint>::PointCloudSet(uint32_t n, uint32_t dims, std::shared_ptr<float[]> values,
                                      parlay::sequence<size_t> offsets,
                                      parlay::sequence<uint32_t> ids) noexcept :
    n(n),
    dims(dims),
    aligned_dims(dims),
    values(std::move(values)),
    offsets(std::move(offsets)),
    ids(std::move(ids)) {}

template<typename ChPoint>
PointCloudSet<ChPoint>::PointCloudSet(const char* filename, bool is_mmap) {
  if (is_mmap) {
    read_mmap_file(filename);
  } else {
    std::cout << "filename = " << filename << std::endl;
    std::ifstream file(filename, std::ios::binary | std::ios::in);
    if (!file.is_open()) {
      std::cerr << "Error opening file!" << std::endl;
      exit(-1);
    }
    size_t tmp_n, tmp_dims;
    file.read(reinterpret_cast<char*>(&tmp_dims), sizeof(tmp_dims));
    file.read(reinterpret_cast<char*>(&tmp_n), sizeof(tmp_n));
    n = static_cast<uint32_t>(tmp_n);
    dims = static_cast<uint32_t>(tmp_dims);

    size_t num_vectors;
    file.read(reinterpret_cast<char*>(&num_vectors), sizeof(num_vectors));
    std::cout << "Detected " << n << " point clouds with embedding dimension " << dims << std::endl;
    aligned_dims = dims;
    if (aligned_dims != dims) {
      std::cerr << "Expected dims to be a multiple of cacheline size." << std::endl;
      exit(-1);
    }

    size_t coordinate_size = num_vectors * dims * sizeof(float);
    values = std::shared_ptr<float[]>(static_cast<float*>(parlay::p_malloc(coordinate_size)),
                                      parlay::p_free);
    file.read(reinterpret_cast<char*>(values.get()), coordinate_size);
    for (size_t i = 0; i < num_vectors * dims; ++i) {
      if (!std::isfinite(values.get()[i])) {
        std::cerr << "Error: Non-finite value found in input data at index " << i << std::endl;
        exit(1);
      }
    }

    size_t num_offsets;
    file.read(reinterpret_cast<char*>(&num_offsets), sizeof(num_offsets));
    offsets.resize(num_offsets);
    file.read(reinterpret_cast<char*>(offsets.begin()), num_offsets * sizeof(size_t));
    file.close();
  }
}

template<typename ChPoint>
void PointCloudSet<ChPoint>::read_mmap_file(const char* filename) {
  auto [fileptr, length] = mmap_file(filename);
  char* p = fileptr;
  size_t tmp_n, tmp_dims;
  std::memcpy(&tmp_dims, p, sizeof(size_t));
  p += sizeof(size_t);
  std::memcpy(&tmp_n, p, sizeof(size_t));
  p += sizeof(size_t);
  n = static_cast<uint32_t>(tmp_n);
  dims = static_cast<uint32_t>(tmp_dims);
  aligned_dims = dims;

  size_t num_vectors;
  std::memcpy(&num_vectors, p, sizeof(size_t));
  p += sizeof(size_t);

  size_t coordinate_size = num_vectors * dims * sizeof(float);
  values = std::shared_ptr<float[]>(reinterpret_cast<float*>(p), [](float*) {});
  p += coordinate_size;

  size_t num_offsets;
  std::memcpy(&num_offsets, p, sizeof(size_t));
  p += sizeof(size_t);
  offsets.resize(num_offsets);
  std::memcpy(offsets.begin(), p, num_offsets * sizeof(size_t));
}

template<typename ChPoint>
PointCloudSet<ChPoint>::PointCloudSet(uint32_t n, uint32_t dims, const float* values_,
                                      const size_t* offsets_, const uint32_t* ids_) :
    n(n), dims(dims), aligned_dims(dims) {
  offsets = parlay::tabulate(n + 1, [&](size_t i) { return offsets_[i]; });
  if (ids_ != nullptr) ids = parlay::tabulate(n, [&](size_t i) { return ids_[i]; });

  values = std::shared_ptr<float[]>(
      static_cast<float*>(parlay::p_malloc(offsets[n] * sizeof(float))), parlay::p_free);
  std::memcpy(values.get(), values_, offsets[n] * sizeof(float));
}

template<typename ChPoint>
template<template<typename, typename...> class seq, typename... x>
PointCloudSet<ChPoint>::PointCloudSet(const seq<ChPoint, x...>& point_clouds, uint32_t d) :
    n(point_clouds.size()), dims(d), aligned_dims(dims) {
  offsets = parlay::sequence<size_t>::from_function(
      n + 1, [&](size_t i) { return (i == 0) ? 0 : (point_clouds[i - 1].size() * dims); });
  parlay::scan_inclusive_inplace(offsets);

  size_t total_coords = offsets[n];
  values = std::shared_ptr<float[]>(
      static_cast<float*>(parlay::p_malloc(total_coords * sizeof(float))), parlay::p_free);

  parlay::parallel_for(0, n, [&](size_t i) {
    size_t offset = offsets[i];
    parlay::parallel_for(0, point_clouds[i].size(), [&](size_t j) {
      for (size_t t = 0; t < dims; ++t) {
        values[offset + j * dims + t] = point_clouds[i][j][t];
      }
    });
  });
  ids = parlay::sequence<uint32_t>::from_function(
      n, [&](size_t i) { return point_clouds[i].get_id(); });
}

template<typename ChPoint>
template<template<typename> class seqA, template<typename> class seqB,
         template<typename> class seqC>
PointCloudSet<ChPoint>::PointCloudSet(const seqA<seqB<seqC<float>>>& point_clouds, uint32_t d,
                                      parlay::sequence<uint32_t> ids) :
    n(point_clouds.size()), dims(d), aligned_dims(dims), ids(std::move(ids)) {
  offsets = parlay::sequence<size_t>::from_function(
      n + 1, [&](size_t i) { return (i == 0) ? 0 : (point_clouds[i - 1].size() * dims); });
  parlay::scan_inclusive_inplace(offsets);

  size_t total_coords = offsets[n];
  values = std::shared_ptr<float[]>(
      static_cast<float*>(parlay::p_malloc(total_coords * sizeof(float))), parlay::p_free);

  parlay::parallel_for(0, n, [&](size_t i) {
    size_t offset = offsets[i];
    parlay::parallel_for(0, point_clouds[i].size(), [&](size_t j) {
      for (size_t t = 0; t < dims; ++t) {
        values[offset + j * dims + t] = point_clouds[i][j][t];
      }
    });
  });
}

template<typename ChPoint>
PointCloudSet<ChPoint>::PointCloudSet(uint32_t n, uint32_t k, uint32_t d) :
    n(n), dims(d), aligned_dims(d) {
  size_t total_coords = n * k * dims;
  values = std::shared_ptr<float[]>(
      static_cast<float*>(parlay::p_malloc(total_coords * sizeof(float))), parlay::p_free);
  offsets = parlay::sequence<size_t>::from_function(n + 1, [&](size_t i) { return i * k * dims; });
}

// =========================================================================
// Slicing and Shuffling
// =========================================================================

template<typename ChPoint>
PointCloudSet<ChPoint> PointCloudSet<ChPoint>::shuffle(
    const parlay::sequence<uint32_t>& new_order) const {
  size_t new_n = new_order.size();

  // 1. Calculate new point cloud sizes based on the specified order
  auto pc_sizes =
      parlay::delayed_tabulate(new_n, [&](size_t i) { return static_cast<size_t>(get_size(new_order[i])) * dims; });

  // 2. Scan to compute the exact offsets for the new CSR layout
  auto scan_res = parlay::scan(pc_sizes);
  auto new_offsets = std::move(scan_res.first);
  size_t total_elements = scan_res.second;
  new_offsets.push_back(total_elements);

  // 3. Reorder the IDs
  auto new_ids = parlay::sequence<uint32_t>::uninitialized(new_n);
  parlay::parallel_for(0, new_n, [&](size_t i) { new_ids[i] = get_id(new_order[i]); });

  // 4. Allocate and parallel-copy the embeddings into the new contiguous blocks
  auto new_values = std::shared_ptr<float[]>(
      static_cast<float*>(parlay::p_malloc(total_elements * sizeof(float))), parlay::p_free);

  parlay::parallel_for(0, new_n, [&](size_t i) {
    size_t original_idx = new_order[i];
    size_t old_offset = offsets[original_idx];
    size_t new_offset = new_offsets[i];
    size_t num_elements = pc_sizes[i];

    if (num_elements > 0) {
      std::memcpy(new_values.get() + new_offset, values.get() + old_offset,
                  num_elements * sizeof(float));
    }
  });

  // 5. Construct and return utilizing the zero-copy constructor
  return PointCloudSet<ChPoint>(new_n, dims, std::move(new_values), std::move(new_offsets),
                                std::move(new_ids));
}

// =========================================================================
// Distance Calculations
// =========================================================================

template<typename ChPoint>
size_t PointCloudSet<ChPoint>::distances(const ChPoint& query,
                                         std::pair<uint32_t, float>* results) const {
  auto cmps = parlay::sequence<size_t>::uninitialized(n);
  parlay::parallel_for(0, n, [&](uint32_t i) {
    std::tie(results[i].second, cmps[i]) = query.distance_w_cmps((*this)[i]);
    results[i].first = get_id(i);
  });
  return parlay::reduce(cmps);
}

template<typename ChPoint>
std::pair<parlay::sequence<std::pair<uint32_t, float>>, size_t> PointCloudSet<ChPoint>::distances(
    const ChPoint& query) const {
  auto results = parlay::sequence<std::pair<uint32_t, float>>::uninitialized(n);
  auto cmps = distances(query, results.begin());
  return std::make_pair(results, cmps);
}

template<typename ChPoint>
size_t PointCloudSet<ChPoint>::distances_blocked(const ChPoint& query,
                                                 std::pair<uint32_t, float>* results) const {
  auto cmps = query.size() * dims + offsets[n];
  auto dists = OneToMany<ChPoint, PointCloudSet<ChPoint>>::AllDistances(query, *this);
  parlay::parallel_for(0, n, [&](uint32_t i) { results[i] = std::make_pair(get_id(i), dists[i]); });
  return cmps;
}

template<typename ChPoint>
std::pair<parlay::sequence<std::pair<uint32_t, float>>, size_t>
PointCloudSet<ChPoint>::distances_blocked(const ChPoint& query) const {
  auto results = parlay::sequence<std::pair<uint32_t, float>>::uninitialized(n);
  auto cmps = distances_blocked(query, results.begin());
  return std::make_pair(results, cmps);
}

template<typename ChPoint>
template<typename Seq>
size_t PointCloudSet<ChPoint>::distances_subset(const ChPoint& query, const Seq& indices,
                                                std::pair<uint32_t, float>* results) const {
  auto cmps = parlay::sequence<size_t>::uninitialized(indices.size());
  parlay::parallel_for(0, indices.size(), [&](uint32_t i) {
    std::tie(results[i].second, cmps[i]) = query.distance_w_cmps((*this)[indices[i]]);
    results[i].first = get_id(indices[i]);
  });
  return parlay::reduce(cmps);
}

template<typename ChPoint>
template<typename Seq>
std::pair<parlay::sequence<std::pair<uint32_t, float>>, size_t>
PointCloudSet<ChPoint>::distances_subset(const ChPoint& query, const Seq& indices) const {
  auto results = parlay::sequence<std::pair<uint32_t, float>>::uninitialized(indices.size());
  auto cmps = distances_subset(query, indices, results.begin());
  return std::make_pair(results, cmps);
}

template<typename ChPoint>
parlay::sequence<std::pair<uint32_t, float>> PointCloudSet<ChPoint>::distances(
    const PointCloudSet<ChPoint>& Queries, size_t k) const {
  k = std::min<size_t>(k, n);
  size_t num_queries = Queries.size();
  auto results = parlay::sequence<std::pair<uint32_t, float>>::uninitialized(num_queries * k);

  if (num_queries >= 64) {
    mvsic::ManyToMany<PointCloudSet<ChPoint>>::TopKIntoUninitialized(Queries, *this, k,
                                                                     results.data());
  } else {
    parlay::parallel_for(0, num_queries, [&](size_t i) {
      ChPoint q_i = Queries[i];
      auto all_dists = parlay::sequence<std::pair<uint32_t, float>>::uninitialized(n);
      this->distances(q_i, all_dists.data());
      if (k < n) {
        mvsic::sort_inplace_kv(all_dists);
      }
      parlay::parallel_for(0, k, [&](size_t j) { results[i * k + j] = all_dists[j]; });
    });
  }
  return results;
}

// =========================================================================
// Utilities
// =========================================================================

template<typename ChPoint>
template<typename Seq>
parlay::sequence<parlay::sequence<float>> PointCloudSet<ChPoint>::filter_flattened(
    const Seq& cluster_ids) const {
  size_t num = cluster_ids.size();
  auto sizes = parlay::delayed_seq<size_t>(num, [&](size_t i) { return get_size(cluster_ids[i]); });

  size_t total_size;
  parlay::sequence<size_t> csizes;
  std::tie(csizes, total_size) = parlay::scan(sizes);

  auto result = parlay::sequence<parlay::sequence<float>>(
      total_size, parlay::sequence<float>::uninitialized(dims));

  parlay::parallel_for(0, num, [&](size_t i) {
    auto ind = cluster_ids[i];
    size_t offset = csizes[i];
    auto coords = data(ind);
    parlay::parallel_for(0, get_size(ind), [&](size_t j) {
      std::memcpy(result[offset + j].begin(), coords + j * dims, dims * sizeof(float));
    });
  });
  return result;
}

template<typename ChPoint>
template<typename Seq>
void PointCloudSet<ChPoint>::set_point_cloud(size_t i, Seq& point_cloud) {
  assert(point_cloud.size() <= get_size(i));
  auto values_i = values.get() + offsets[i];
  parlay::parallel_for(0, point_cloud.size(), [&](size_t j) {
    for (size_t t = 0; t < dims; ++t) {
      values_i[j * dims + t] = point_cloud[j][t];
    }
  });
}

}  // namespace mvsic