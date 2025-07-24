#pragma once

#include <fstream>
#include "parlay/primitives.h"
#include "mmap.h"

namespace mvivf {

/* =================================Set of Point Clouds Type================================ */
template<typename ChPoint>
struct PointCloudSet {
 private:
  size_t n = 0;
  size_t dims = 0;
  size_t aligned_dims = 0;
  // float *values = nullptr;
  std::shared_ptr<float[]> values;
  parlay::sequence<size_t> offsets;
  parlay::sequence<size_t> ids;

 public:
  PointCloudSet() noexcept {}
  // Load point clouds from file
  PointCloudSet(const char *filename, bool is_mmap = false);
  // Helper to read mmap file
  void read_mmap_file(const char *filename);
  // Input in array format (values_ has offsets_[n] elts, offsets_ has n+1 elts,
  // ids_ has n elts)
  PointCloudSet(size_t n, size_t dims, const float *values_, const size_t *offsets_,
                const size_t *ids_);
  // Sequence of ChPoint type objects given
  template<template<typename, typename...> class seq, typename... x>
  PointCloudSet(const seq<ChPoint, x...> &data, size_t d);
  // Arbitrary random-access range given, along with sequence of ids
  template<template<typename> class seqA, template<typename> class seqB,
           template<typename> class seqC>
  PointCloudSet(const seqA<seqB<seqC<float>>> &data, size_t d, parlay::sequence<size_t> ids);

  // Returns number of point clouds
  inline size_t size() const noexcept { return n; }
  // Returns total number of individual embeddings
  inline size_t total_size() const noexcept { return offsets[n] / dims; }
  // Returns embedding dimension
  inline size_t get_dims() const noexcept { return dims; }
  // Returns number of embeddings of pointcloud i
  inline size_t get_size(size_t i) const noexcept { return (offsets[i + 1] - offsets[i]) / dims; }
  // Returns id of pointcloud i
  inline uint32_t get_id(size_t i) const noexcept { return (ids.size() > 0) ? ids[i] : i; }
  // Returns pointer to embeddings of pointcloud i
  inline float *get_coords(size_t i) const noexcept { return values.get() + offsets[i]; }
  // Returns pointer to embeddings of all point clouds
  inline float *data() const noexcept { return values.get(); }
  // Returns ChPoint type object on the embeddings of point cloud i
  inline ChPoint operator[](size_t i) const {
    return ChPoint(get_size(i), dims, get_coords(i), get_id(i));
  }
  // Returns non-owning sequence of offsets
  inline auto get_offsets() const noexcept {
    return parlay::make_slice(offsets.begin(), offsets.end());
  }
  // Returns non-owning sequence of ChPoint type objects of the point
  // clouds whose indices are given in sequence cluster_ids
  template<typename seq>
  inline auto filter(const seq &cluster_ids) const {
    return parlay::delayed_tabulate(cluster_ids.size(),
                                    [&](size_t i) { return (*this)[cluster_ids[i]]; });
  }
  // Returns sequence of individual embeddings (as sequences) of the point
  // point clouds whose indices are given in sequence cluster_ids
  template<typename seq>
  auto filter_flattened(const seq &cluster_ids) const;
  // Return list of distances from a query to all point clouds
  // TODO: make this blocked, and thread_local
  inline std::pair<parlay::sequence<float>, size_t> distances(const ChPoint &query) {
    auto dists = parlay::sequence<float>::from_function(
        n, [&](size_t i) { return query.distance((*this)[i]); });
    return std::make_pair(dists, query.size() + this->total_size());
  }
};

/* =======================================Implementation======================================= */

// Load point clouds from file
template<typename ChPoint>
PointCloudSet<ChPoint>::PointCloudSet(const char *filename, bool is_mmap) {
  if (is_mmap) {
    read_mmap_file(filename);
  } else {
    std::cout << "filename = " << filename << std::endl;
    std::ifstream file(filename, std::ios::binary | std::ios::in);
    if (!file.is_open()) {
      std::cerr << "Error opening file!" << std::endl;
      exit(-1);
    }
    // Step 1: Read num points and dimension [size_t, size_t]
    file.read(reinterpret_cast<char *>(&dims), sizeof(dims));
    file.read(reinterpret_cast<char *>(&n), sizeof(n));
    size_t num_vectors;
    file.read(reinterpret_cast<char *>(&num_vectors), sizeof(num_vectors));
    std::cout << "Detected " << n << " point clouds with embedding dimension " << dims << std::endl;
    aligned_dims = dims;
    std::cout << "Aligned dims = " << aligned_dims << std::endl;
    if (aligned_dims != dims) {
      std::cerr << "Expected dims to be a multiple of cacheline size." << std::endl;
      exit(-1);
    }
    // Step 2: Read values
    size_t coordinate_size = num_vectors * dims * sizeof(float);
    values = std::shared_ptr<float[]>(static_cast<float *>(parlay::p_malloc(coordinate_size)),
                                      parlay::p_free);
    // values = std::shared_ptr<float[]>(static_cast<float *>(std::aligned_alloc(64,
    // coordinate_size)), std::free);
    file.read(reinterpret_cast<char *>(values.get()), coordinate_size);
    // Step 3: Read offsets
    size_t num_offsets;
    file.read(reinterpret_cast<char *>(&num_offsets), sizeof(num_offsets));
    offsets.resize(num_offsets);
    file.read(reinterpret_cast<char *>(offsets.begin()), num_offsets * sizeof(size_t));
    // Step 4: Set ids
    ids = parlay::sequence<size_t>::from_function(n,
                                                  [&](size_t i) { return static_cast<size_t>(i); });
    file.close();
  }
}

// Helper to read mmap file
template<typename ChPoint>
void PointCloudSet<ChPoint>::read_mmap_file(const char *filename) {
  auto [fileptr, length] = mmap_file(filename);
  char *p = fileptr;
  std::memcpy(&dims, p, sizeof(size_t));
  p += sizeof(size_t);
  std::memcpy(&n, p, sizeof(size_t));
  p += sizeof(size_t);

  size_t num_vectors;
  std::memcpy(&num_vectors, p, sizeof(size_t));
  p += sizeof(size_t);

  aligned_dims = dims;

  size_t coordinate_size = num_vectors * dims * sizeof(float);
  values = std::shared_ptr<float[]>(reinterpret_cast<float *>(p), [](float *) {});
  p += coordinate_size;

  size_t num_offsets;
  std::memcpy(&num_offsets, p, sizeof(size_t));
  p += sizeof(size_t);
  offsets = parlay::sequence<size_t>(num_offsets);
  std::memcpy(offsets.begin(), p, num_offsets * sizeof(size_t));
  p += num_offsets * sizeof(size_t);

  ids = parlay::sequence<size_t>::from_function(n, [&](size_t i) { return i; });
}

// Input in array format (values_ has offsets_[n] elts, offsets_ has n+1 elts,
// ids_ has n elts)
template<typename ChPoint>
PointCloudSet<ChPoint>::PointCloudSet(size_t n, size_t dims, const float *values_,
                                      const size_t *offsets_, const size_t *ids_) :
    n(n), dims(dims), aligned_dims(dims) {
  offsets = parlay::tabulate(n + 1, [&](size_t i) { return offsets_[i]; });
  if (ids_ != nullptr) {
    ids = parlay::tabulate(n, [&](size_t i) { return ids_[i]; });
  }
  values = std::shared_ptr<float[]>(
      static_cast<float *>(parlay::p_malloc(offsets[n] * sizeof(float))), parlay::p_free);
  // values = std::shared_ptr<float[]>(
  //     static_cast<float *>(std::aligned_alloc(64, offsets[n] * sizeof(float))), std::free);
  std::memcpy(values.get(), values_, offsets[n] * sizeof(float));
}

// Sequence of ChPoint type objects given
template<typename ChPoint>
template<template<typename, typename...> class seq, typename... x>
PointCloudSet<ChPoint>::PointCloudSet(const seq<ChPoint, x...> &data, size_t d) :
    n(data.size()), dims(d), aligned_dims(dims) {
  offsets = parlay::sequence<size_t>::from_function(
      n + 1, [&](size_t i) { return (i == 0) ? 0 : (data[i - 1].size() * dims); });
  parlay::scan_inclusive_inplace(offsets);
  size_t total_coords = offsets[n];
  values = std::shared_ptr<float[]>(
      static_cast<float *>(parlay::p_malloc(total_coords * sizeof(float))), parlay::p_free);
  // values = std::shared_ptr<float[]>(
  //     static_cast<float *>(std::aligned_alloc(64, total_coords * sizeof(float))), std::free);
  parlay::parallel_for(0, n, [&](size_t i) {
    size_t offset = offsets[i];
    parlay::parallel_for(0, data[i].size(), [&](size_t j) {
      for (size_t t = 0; t < dims; ++t) {
        values[offset + j * dims + t] = data[i][j][t];
      }
    });
  });
  ids = parlay::sequence<size_t>::from_function(n, [&](size_t i) { return data[i].get_id(); });
}

// Arbitrary random-access range given, along with sequence of ids
template<typename ChPoint>
template<template<typename> class seqA, template<typename> class seqB,
         template<typename> class seqC>
PointCloudSet<ChPoint>::PointCloudSet(const seqA<seqB<seqC<float>>> &data, size_t d,
                                      parlay::sequence<size_t> ids) :
    n(data.size()), dims(d), aligned_dims(dims), ids(std::move(ids)) {
  offsets = parlay::sequence<size_t>::from_function(
      n + 1, [&](size_t i) { return (i == 0) ? 0 : (data[i - 1].size() * dims); });
  parlay::scan_inclusive_inplace(offsets);
  size_t total_coords = offsets[n];
  values = std::shared_ptr<float[]>(
      static_cast<float *>(parlay::p_malloc(total_coords * sizeof(float))), parlay::p_free);
  // values = std::shared_ptr<float[]>(
  //     static_cast<float *>(std::aligned_alloc(64, total_coords * sizeof(float))), std::free);
  parlay::parallel_for(0, n, [&](size_t i) {
    size_t offset = offsets[i];
    parlay::parallel_for(0, data[i].size(), [&](size_t j) {
      for (size_t t = 0; t < dims; ++t) {
        values[offset + j * dims + t] = data[i][j][t];
      }
    });
  });
}

template<typename ChPoint>
template<typename seq>
auto PointCloudSet<ChPoint>::filter_flattened(const seq &cluster_ids) const {
  size_t num = cluster_ids.size();
  auto sizes = parlay::delayed_tabulate(num, [&](size_t i) { return get_size(cluster_ids[i]); });
  auto [csizes, total_size] = parlay::scan(sizes);
  auto result = parlay::sequence<parlay::sequence<float>>(
      total_size, parlay::sequence<float>::uninitialized(dims));
  parlay::parallel_for(0, num, [&](size_t i) {
    auto ind = cluster_ids[i];
    auto offset = csizes[i];
    auto coords = get_coords(ind);
    parlay::parallel_for(0, get_size(ind), [&](size_t j) {
      std::memcpy(result[offset + j].begin(), coords + j * dims, dims * sizeof(float));
    });
  });
  return result;
}

}  // namespace mvivf