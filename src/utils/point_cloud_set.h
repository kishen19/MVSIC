#pragma once

#include <fstream>
#include "parlay/primitives.h"

namespace mvivf {

/* Set of Point Clouds Type */
template<typename ChPoint>
struct PointCloudSet {
 private:
  size_t n = 0;
  size_t dims = 0;
  size_t aligned_dims = 0;
  float *values = nullptr;
  parlay::sequence<size_t> offsets;
  parlay::sequence<size_t> ids;

 public:
  PointCloudSet() noexcept {}
  // Load point clouds from file
  PointCloudSet(const char *filename);
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
  // Move Constructor
  PointCloudSet(PointCloudSet &&other) noexcept;
  // Copy Constructor
  PointCloudSet(const PointCloudSet &other);
  // Destructor
  ~PointCloudSet() noexcept;
  // Move Assignment Operator
  PointCloudSet &operator=(PointCloudSet &&other) noexcept;
  // Copy Assignment Operator
  PointCloudSet &operator=(const PointCloudSet &other);

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
  inline float *get_coords(size_t i) const noexcept { return values + offsets[i]; }
  // Returns ChPoint type object on the embeddings of point cloud i
  inline ChPoint operator[](size_t i) const {
    return ChPoint(get_size(i), dims, get_coords(i), get_id(i));
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
  inline float *data() const noexcept { return values; }
  inline auto get_offsets() const noexcept {
    return parlay::make_slice(offsets.begin(), offsets.end());
  }
};

/* -----------------------------------------Implementation-----------------------------------------*/

// Load point clouds from file
template<typename ChPoint>
PointCloudSet<ChPoint>::PointCloudSet(const char *filename) {
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
  values = static_cast<float *>(parlay::p_malloc(coordinate_size));
  file.read(reinterpret_cast<char *>(values), coordinate_size);
  // Step 3: Read offsets
  size_t num_offsets;
  file.read(reinterpret_cast<char *>(&num_offsets), sizeof(num_offsets));
  offsets.resize(num_offsets);
  file.read(reinterpret_cast<char *>(offsets.begin()), num_offsets * sizeof(size_t));
  // Step 4: Set ids
  ids =
      parlay::sequence<size_t>::from_function(n, [&](size_t i) { return static_cast<size_t>(i); });
  file.close();
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
  values = static_cast<float *>(parlay::p_malloc(offsets[n] * sizeof(float)));
  std::memcpy(values, values_, offsets[n] * sizeof(float));
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
  values = static_cast<float *>(parlay::p_malloc(total_coords * sizeof(float)));
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
  values = static_cast<float *>(parlay::p_malloc(total_coords * sizeof(float)));
  parlay::parallel_for(0, n, [&](size_t i) {
    size_t offset = offsets[i];
    parlay::parallel_for(0, data[i].size(), [&](size_t j) {
      for (size_t t = 0; t < dims; ++t) {
        values[offset + j * dims + t] = data[i][j][t];
      }
    });
  });
}

// Move Constructor
template<typename ChPoint>
PointCloudSet<ChPoint>::PointCloudSet(PointCloudSet &&other) noexcept :
    n(other.n),
    dims(other.dims),
    aligned_dims(other.aligned_dims),
    values(other.values),
    offsets(std::move(other.offsets)),
    ids(std::move(other.ids)) {
  other.n = 0;
  other.dims = 0;
  other.aligned_dims = 0;
  other.values = nullptr;
}

// Copy Constructor
template<typename ChPoint>
PointCloudSet<ChPoint>::PointCloudSet(const PointCloudSet &other) :
    n(other.n), dims(other.dims), aligned_dims(other.aligned_dims) {
  offsets = other.offsets;
  ids = other.ids;
  if (other.values) {
    size_t total_coords = offsets[n];
    size_t coordinate_size = total_coords * sizeof(float);
    values = static_cast<float *>(parlay::p_malloc(coordinate_size));
    std::memcpy(values, other.values, coordinate_size);
  }
}

template<typename ChPoint>
PointCloudSet<ChPoint>::~PointCloudSet() noexcept {
  if (values != nullptr) {
    parlay::p_free(values);
    values = nullptr;
  }
}

// Move Assignment Operator
template<typename ChPoint>
PointCloudSet<ChPoint> &PointCloudSet<ChPoint>::operator=(PointCloudSet &&other) noexcept {
  if (this != &other) {
    if (values != nullptr) {
      parlay::p_free(values);
    }
    n = other.n;
    dims = other.dims;
    aligned_dims = other.aligned_dims;
    values = other.values;
    offsets = std::move(other.offsets);
    ids = std::move(other.ids);

    other.n = 0;
    other.dims = 0;
    other.aligned_dims = 0;
    other.values = nullptr;
  }
  return *this;
}

// Copy Assignment Operator
template<typename ChPoint>
PointCloudSet<ChPoint> &PointCloudSet<ChPoint>::operator=(const PointCloudSet &other) {
  if (this != &other) {
    n = other.n;
    dims = other.dims;
    aligned_dims = other.aligned_dims;
    offsets = other.offsets;
    ids = other.ids;
    if (values != nullptr) {
      parlay::p_free(values);
      values = nullptr;
    }
    if (other.values) {
      size_t total_coords = offsets[n];
      size_t coordinate_size = total_coords * sizeof(float);
      values = static_cast<float *>(parlay::p_malloc(coordinate_size));
      std::memcpy(values, other.values, coordinate_size);
    }
  }
  return *this;
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