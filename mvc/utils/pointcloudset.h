#pragma once

#include <fstream>
#include "parlay/primitives.h"

template <typename ChPoint>
struct PointCloudSet {
  size_t n = 0;
  size_t dims = 0;
  size_t aligned_dims = 0;
  float* values = nullptr;
  parlay::sequence<size_t> offsets;
  parlay::sequence<size_t> ids;

  PointCloudSet() {}
  // Load point clouds from file
  PointCloudSet(const char* filename);
  PointCloudSet(size_t n, size_t dims, const float* values_, const size_t* offsets_, 
                const size_t* ids_);
  // Sequence of ChPoint type objects given
  template <typename seq>
  PointCloudSet(const seq& data, size_t d);
  // Arbitrary random-access range given, along with sequence of ids
  template <typename seq>
  PointCloudSet(const seq& data, size_t d, parlay::sequence<size_t> ids);
  // Copy constructor
  PointCloudSet& operator=(const PointCloudSet& other);
  // Move constructor
  PointCloudSet(const PointCloudSet& other);
  // Destructor
  ~PointCloudSet();

  // Returns number of point clouds
  inline size_t size() const { return n; }

  // Returns total number of individual embeddings
  inline size_t total_size() const { return offsets[n]/dims; }

  // Returns embedding dimension
  inline size_t get_dims() const { return dims; }

  // Returns number of embeddings of pointcloud i
  inline size_t get_size(size_t i) const{
    return (offsets[i + 1] - offsets[i]) / dims;}

  // Returns id of pointcloud i
  inline uint32_t get_id(uint32_t i) const{
    return (ids.size()>0)? ids[i]:i;}
    
  // Returns pointer to embeddings of pointcloud i
  inline float* get_coords(long i) const{ return values + offsets[i]; }
    
  // Returns ChPoint type object on the embeddings of point cloud i
  ChPoint operator[](long i) const{
    return ChPoint(ids[i], get_coords(i), get_size(i), dims);
  }

  // Returns non-owning sequence of ChPoint type objects of the point 
  // clouds whose indices are given in sequence cluster_ids
  template <typename seq>
  inline auto filter(const seq& cluster_ids) const{
    return parlay::delayed_tabulate(cluster_ids.size(), [&](size_t i) {
      return (*this)[cluster_ids[i]]; });
  }

  // Returns sequence of individual embeddings (as sequences) of the point 
  // point clouds whose indices are given in sequence cluster_ids
  template <typename seq>
  inline auto filter_flattened(const seq& cluster_ids) const;
  
};

template <typename ChPoint>
template <typename seq>
inline auto PointCloudSet<ChPoint>::filter_flattened(const seq& cluster_ids) const {
  size_t num = cluster_ids.size();
  auto sizes = parlay::delayed_seq<size_t>(num, [&](size_t i) {
    return get_size(cluster_ids[i]); });
  auto [csizes, total_size] = parlay::scan(sizes);
  auto starts = parlay::sequence<float*>::uninitialized(total_size);
  parlay::parallel_for(0, num, [&](size_t i) {
    auto ind = cluster_ids[i];
    auto offset = csizes[i];
    auto coords = get_coords(ind);
    parlay::parallel_for(0, sizes[i], [&](size_t j) {
      starts[offset + j] = coords + j * dims;});
  });
  return parlay::tabulate(total_size, [&](size_t i) {
    return parlay::tabulate(dims, [&](size_t j) {
      return *(starts[i] + j); }); });
}

template<typename ChPoint>
PointCloudSet<ChPoint>::PointCloudSet(size_t n, size_t dims, const float* values_, 
    const size_t* offsets_, const size_t* ids_) 
    : n(n), dims(dims), aligned_dims(dims) {
  values = static_cast<float*>(parlay::p_malloc(offsets[n] * sizeof(float)));
  std::memcpy(values, values_, offsets[n] * sizeof(float));
  offsets = parlay::tabulate(n+1, [&](size_t i) { return offsets_[i]; });
  if (ids_ != nullptr) {
    ids = parlay::tabulate(n, [&](size_t i) { return ids_[i]; });
  }
}

template <typename ChPoint>
PointCloudSet<ChPoint>::PointCloudSet(const char* filename) {
  std::cout << "filename = " << filename << std::endl;
  std::ifstream file(filename, std::ios::binary | std::ios::in);
  if (!file.is_open()) {
    std::cerr << "Error opening file!" << std::endl;
    exit(-1);
  }
  // Step 1: Read num points and dimension [uint32_t, uint32_t]
  file.read(reinterpret_cast<char*>(&dims), sizeof(dims));
  file.read(reinterpret_cast<char*>(&n), sizeof(n));
  size_t num_vectors;
  file.read(reinterpret_cast<char*>(&num_vectors), sizeof(num_vectors));
  std::cout << "Detected " << n
    << " points with embedding dimension " << dims << std::endl;
  aligned_dims = dims;
  std::cout << "Aligned dims = " << aligned_dims << std::endl;
  if (aligned_dims != dims) {
    std::cerr << "Expected dims to be a multiple of cacheline size."
      << std::endl;
    exit(-1);
  }
  // Step 2: Read values
  size_t coordinate_size = num_vectors * dims * sizeof(float);
  values = static_cast<float*>(parlay::p_malloc(coordinate_size));
  file.read(reinterpret_cast<char*>(values), coordinate_size);
  // Step 3: Read offsets
  size_t num_offsets;
  file.read(reinterpret_cast<char*>(&num_offsets), sizeof(num_offsets));
  offsets.resize(num_offsets);
  file.read(reinterpret_cast<char*>(offsets.begin()), num_offsets * sizeof(size_t));
  // Step 4: Set ids
  ids = parlay::sequence<size_t>::from_function(n, [&](size_t i) {
    return static_cast<size_t>(i); });
  file.close();
}

template <typename ChPoint>
template <typename seq>
PointCloudSet<ChPoint>::PointCloudSet(const seq& data, size_t d)
  : n(data.size()), dims(d), aligned_dims(dims){
  offsets = parlay::sequence<size_t>::from_function(n + 1, [&](size_t i) {
    return (i == 0) ? 0 : (data[i - 1].size() * dims);});
  parlay::scan_inclusive_inplace(offsets);
  size_t total_coords = offsets[n];
  values = static_cast<float*>(parlay::p_malloc(total_coords * sizeof(float)));
  parlay::parallel_for(0, n, [&](size_t i) {
    size_t offset = offsets[i];
    parlay::parallel_for(0, data[i].size(), [&](size_t j) {
      for (size_t t = 0; t < dims; ++t) {
        values[offset + j * dims + t] = data[i][j][t];
      }
    });
  });
  ids = parlay::sequence<size_t>::from_function(n, [&](size_t i) {
    return data[i].get_id();});
}

template <typename ChPoint>
template <typename seq>
PointCloudSet<ChPoint>::PointCloudSet(const seq& data, size_t d, 
                              parlay::sequence<size_t> ids_)
  : n(data.size()), dims(d), aligned_dims(dims){
  offsets = parlay::sequence<size_t>::from_function(n + 1, [&](size_t i) {
    return (i == 0) ? 0 : (data[i - 1].size() * dims); });
  parlay::scan_inclusive_inplace(offsets);
  size_t total_coords = offsets[n];
  values = static_cast<float*>(parlay::p_malloc(total_coords * sizeof(float)));
  parlay::parallel_for(0, n, [&](size_t i) {
    size_t offset = offsets[i];
    parlay::parallel_for(0, data[i].size(), [&](size_t j) {
      for (size_t t = 0; t < dims; ++t) {
        values[offset + j * dims + t] = data[i][j][t];
      }
    });
  });
  ids = ids_;
}

template <typename ChPoint>
PointCloudSet<ChPoint>& PointCloudSet<ChPoint>::operator=(const PointCloudSet<ChPoint>& other) {
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
      values = static_cast<float*>(parlay::p_malloc(coordinate_size));
      std::memcpy(values, other.values, coordinate_size);
    }
  }
  return *this;
}

template <typename ChPoint>
PointCloudSet<ChPoint>::PointCloudSet(const PointCloudSet& other) {
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
    values = static_cast<float*>(parlay::p_malloc(coordinate_size));
    std::memcpy(values, other.values, coordinate_size);
  }
}

template <typename ChPoint>
PointCloudSet<ChPoint>::~PointCloudSet() {
  if (values != nullptr) {
    parlay::p_free(values);
    values = nullptr;
  }
}