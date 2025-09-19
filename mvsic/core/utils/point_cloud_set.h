#pragma once

#include <fstream>
#include "parlay/primitives.h"
#include "mmap.h"

namespace mvsic {

/* =================================Set of Point Clouds Type================================ */
template<typename ChPoint>
struct PointCloudSet {
 private:
  uint32_t n = 0;
  uint32_t dims = 0;
  uint32_t aligned_dims = 0;
  std::shared_ptr<float[]> values;
  parlay::sequence<size_t> offsets;
  parlay::sequence<uint32_t> ids = {};

 public:
  PointCloudSet() noexcept {}
  // Load point clouds from file
  PointCloudSet(const char *filename, bool is_mmap = false);
  // Input in array format (values_ has offsets_[n] elts, offsets_ has n+1 elts,
  // ids_ has n elts)
  PointCloudSet(uint32_t n, uint32_t dims, const float *values_, const size_t *offsets_,
                const uint32_t *ids_);
  // Sequence of ChPoint type objects given
  template<template<typename, typename...> class seq, typename... x>
  PointCloudSet(const seq<ChPoint, x...> &point_clouds, uint32_t d);
  // Arbitrary random-access range given, along with sequence of ids
  template<template<typename> class seqA, template<typename> class seqB,
           template<typename> class seqC>
  PointCloudSet(const seqA<seqB<seqC<float>>> &point_clouds, uint32_t d,
                parlay::sequence<uint32_t> ids);
  // Uninitialized Point Cloud Set of fixed sizes
  PointCloudSet(uint32_t n, uint32_t k, uint32_t d);

  // Helper to read mmap file
  void read_mmap_file(const char *filename);
  // Returns number of point clouds
  inline uint32_t size() const noexcept { return n; }
  // Returns total number of individual embeddings
  inline size_t total_size() const noexcept { return offsets[n] / dims; }
  // Returns embedding dimension
  inline uint32_t get_dims() const noexcept { return dims; }
  // Returns number of embeddings of pointcloud i
  inline uint32_t get_size(size_t i) const noexcept { return (offsets[i + 1] - offsets[i]) / dims; }
  // Returns id of pointcloud i
  inline uint32_t get_id(size_t i) const noexcept { return (ids.size() > 0) ? ids[i] : i; }
  // Returns pointer to embeddings of all point clouds
  inline float *data() const noexcept { return values.get(); }
  // Returns pointer to embeddings of pointcloud i
  inline float *data(size_t i) const noexcept { return values.get() + offsets[i]; }
  // Returns non-owning sequence of offsets
  inline auto get_offsets() const noexcept {
    return parlay::make_slice(offsets.begin(), offsets.end());
  }
  // Returns ChPoint type object on the embeddings of point cloud i
  inline ChPoint operator[](size_t i) const {
    return ChPoint(get_size(i), dims, data(i), get_id(i));
  }

  // Return list of distances from a query to all point clouds
  // TODO: make this blocked, and thread_local
  inline std::pair<parlay::sequence<std::pair<uint32_t, float>>, size_t> distances(
      const ChPoint &query) {
    auto cmps = parlay::sequence<size_t>::uninitialized(n);
    auto dists = parlay::sequence<std::pair<uint32_t, float>>::from_function(n, [&](uint32_t i) {
      float dist;
      size_t cmp;
      std::tie(dist, cmp) = query.distance_w_cmps((*this)[i]);
      cmps[i] = cmp;
      return std::make_pair(i, dist);
    });
    return std::make_pair(dists, parlay::reduce(cmps));
  }

  // Returns non-owning sequence of ChPoint type objects of the point
  // clouds whose indices are given in sequence cluster_ids
  template<typename Seq>
  inline auto filter(const Seq &cluster_ids) const {
    return parlay::delayed_tabulate(cluster_ids.size(),
                                    [&](size_t i) { return (*this)[cluster_ids[i]]; });
  }
  // Returns sequence of individual embeddings (as sequences) of the point
  // point clouds whose indices are given in sequence cluster_ids
  template<typename Seq>
  parlay::sequence<parlay::sequence<float>> filter_flattened(const Seq &cluster_ids) const;

  // Mutating Functions
  template<typename Seq>
  void set_point_cloud(size_t i, Seq &point_cloud) {
    assert(point_cloud.size() <= get_size(i));
    auto values_i = values.get() + offsets[i];
    parlay::parallel_for(0, point_cloud.size(), [&](size_t j) {
      for (size_t t = 0; t < dims; ++t) {
        values_i[j * dims + t] = point_cloud[j][t];
      }
    });
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

    size_t tmp_n, tmp_dims;
    file.read(reinterpret_cast<char *>(&tmp_dims), sizeof(tmp_dims));
    file.read(reinterpret_cast<char *>(&tmp_n), sizeof(tmp_n));
    n = static_cast<uint32_t>(tmp_n);
    dims = static_cast<uint32_t>(tmp_dims);
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
    file.close();
  }
}

// Helper to read mmap file
template<typename ChPoint>
void PointCloudSet<ChPoint>::read_mmap_file(const char *filename) {
  auto [fileptr, length] = mmap_file(filename);
  char *p = fileptr;
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
  values = std::shared_ptr<float[]>(reinterpret_cast<float *>(p), [](float *) {});
  p += coordinate_size;

  size_t num_offsets;
  std::memcpy(&num_offsets, p, sizeof(size_t));
  p += sizeof(size_t);
  offsets.resize(num_offsets);
  std::memcpy(offsets.begin(), p, num_offsets * sizeof(size_t));
  p += num_offsets * sizeof(size_t);
}

// Input in array format (values_ has offsets_[n] elts, offsets_ has n+1 elts,
// ids_ has n elts)
template<typename ChPoint>
PointCloudSet<ChPoint>::PointCloudSet(uint32_t n, uint32_t dims, const float *values_,
                                      const size_t *offsets_, const uint32_t *ids_) :
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
PointCloudSet<ChPoint>::PointCloudSet(const seq<ChPoint, x...> &point_clouds, uint32_t d) :
    n(point_clouds.size()), dims(d), aligned_dims(dims) {
  offsets = parlay::sequence<size_t>::from_function(
      n + 1, [&](size_t i) { return (i == 0) ? 0 : (point_clouds[i - 1].size() * dims); });
  parlay::scan_inclusive_inplace(offsets);
  size_t total_coords = offsets[n];
  values = std::shared_ptr<float[]>(
      static_cast<float *>(parlay::p_malloc(total_coords * sizeof(float))), parlay::p_free);
  // values = std::shared_ptr<float[]>(
  //     static_cast<float *>(std::aligned_alloc(64, total_coords * sizeof(float))), std::free);
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

// Arbitrary random-access range given, along with sequence of ids
template<typename ChPoint>
template<template<typename> class seqA, template<typename> class seqB,
         template<typename> class seqC>
PointCloudSet<ChPoint>::PointCloudSet(const seqA<seqB<seqC<float>>> &point_clouds, uint32_t d,
                                      parlay::sequence<uint32_t> ids) :
    n(point_clouds.size()), dims(d), aligned_dims(dims), ids(std::move(ids)) {
  offsets = parlay::sequence<size_t>::from_function(
      n + 1, [&](size_t i) { return (i == 0) ? 0 : (point_clouds[i - 1].size() * dims); });
  parlay::scan_inclusive_inplace(offsets);
  size_t total_coords = offsets[n];
  values = std::shared_ptr<float[]>(
      static_cast<float *>(parlay::p_malloc(total_coords * sizeof(float))), parlay::p_free);
  // values = std::shared_ptr<float[]>(
  //     static_cast<float *>(std::aligned_alloc(64, total_coords * sizeof(float))), std::free);
  parlay::parallel_for(0, n, [&](size_t i) {
    size_t offset = offsets[i];
    parlay::parallel_for(0, point_clouds[i].size(), [&](size_t j) {
      for (size_t t = 0; t < dims; ++t) {
        values[offset + j * dims + t] = point_clouds[i][j][t];
      }
    });
  });
}

// Uninitialized Point Cloud Set of fixed sizes
template<typename ChPoint>
PointCloudSet<ChPoint>::PointCloudSet(uint32_t n, uint32_t k, uint32_t d) :
    n(n), dims(d), aligned_dims(d) {
  size_t total_coords = n * k * dims;
  values = std::shared_ptr<float[]>(
      static_cast<float *>(parlay::p_malloc(total_coords * sizeof(float))), parlay::p_free);
  offsets = parlay::sequence<size_t>::from_function(n + 1, [&](size_t i) { return i * k * dims; });
}

template<typename ChPoint>
template<typename Seq>
parlay::sequence<parlay::sequence<float>> PointCloudSet<ChPoint>::filter_flattened(
    const Seq &cluster_ids) const {
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

}  // namespace mvsic