#pragma once

#include <fstream>
#include <memory>
#include <cassert>
#include <cmath>
#include <cstring>
#include "parlay/primitives.h"
#include "mvsic/core/utils/mmap.h"
#include "mvsic/core/utils/util.h"
#include "mvsic/core/distance_measures/one_to_many.h"
#include "mvsic/core/distance_measures/many_to_many.h"

namespace mvsic {

// Forward declaration
template<typename ChPoint>
struct PointCloudSet;

/* =================================Slice of Point Clouds================================ */
// A lightweight, non-owning view of a contiguous range of point clouds within a PointCloudSet.
// Designed to mimic the PointCloudSet API for zero-copy distance computations.
template<typename ChPoint>
struct PointCloudSetSlice {
 private:
  const PointCloudSet<ChPoint>& parent;
  size_t start_idx;
  size_t end_idx;
  size_t n;

 public:
  PointCloudSetSlice(const PointCloudSet<ChPoint>& p, size_t i, size_t j) noexcept :
      parent(p), start_idx(i), end_idx(j), n(j - i) {}

  inline size_t size() const noexcept { return n; }
  inline uint32_t get_dims() const noexcept { return parent.get_dims(); }
  inline uint32_t get_size(size_t i) const noexcept { return parent.get_size(start_idx + i); }
  inline uint32_t get_id(size_t i) const noexcept { return parent.get_id(start_idx + i); }

  // Returns total number of individual embeddings in this slice
  inline size_t total_size() const noexcept {
    return (parent.get_offsets()[end_idx] - parent.get_offsets()[start_idx]) / get_dims();
  }

  // Raw data access (exploiting the underlying contiguous CSR layout)
  inline float* data() const noexcept { return parent.data(start_idx); }
  inline float* data(size_t i) const noexcept { return parent.data(start_idx + i); }

  inline ChPoint operator[](size_t i) const { return parent[start_idx + i]; }

  // Distance calculations
  size_t distances(const ChPoint& query, std::pair<uint32_t, float>* results) const;
  std::pair<parlay::sequence<std::pair<uint32_t, float>>, size_t> distances(
      const ChPoint& query) const;
};

/* =================================Set of Point Clouds================================ */
template<typename ChPoint>
struct PointCloudSet {
 private:
  uint32_t n = 0;
  uint32_t dims = 0;
  uint32_t aligned_dims = 0;
  std::shared_ptr<float[]> values;
  parlay::sequence<size_t> offsets;
  parlay::sequence<uint32_t> ids = {};

  // Private zero-copy constructor used by shuffle()
  PointCloudSet(uint32_t n, uint32_t dims, std::shared_ptr<float[]> values,
                parlay::sequence<size_t> offsets, parlay::sequence<uint32_t> ids) noexcept;

 public:
  PointCloudSet() noexcept {}

  // Load point clouds from file
  PointCloudSet(const char* filename, bool is_mmap = false);

  // Input in array format (values_ has offsets_[n] elts, offsets_ has n+1 elts, ids_ has n elts)
  PointCloudSet(uint32_t n, uint32_t dims, const float* values_, const size_t* offsets_,
                const uint32_t* ids_);

  // Sequence of ChPoint type objects given
  template<template<typename, typename...> class seq, typename... x>
  PointCloudSet(const seq<ChPoint, x...>& point_clouds, uint32_t d);

  // Arbitrary random-access range given, along with sequence of ids
  template<template<typename> class seqA, template<typename> class seqB,
           template<typename> class seqC>
  PointCloudSet(const seqA<seqB<seqC<float>>>& point_clouds, uint32_t d,
                parlay::sequence<uint32_t> ids);

  // Uninitialized Point Cloud Set of fixed sizes
  PointCloudSet(uint32_t n, uint32_t k, uint32_t d);

  // Core Accessors
  inline size_t size() const noexcept { return n; }
  inline size_t total_size() const noexcept { return offsets.empty() ? 0 : offsets[n] / dims; }
  inline float average_size() const noexcept {
    return n == 0 ? 0 : static_cast<float>(offsets[n]) / (n * dims);
  }
  inline uint32_t get_dims() const noexcept { return dims; }
  inline uint32_t get_size(size_t i) const noexcept { return (offsets[i + 1] - offsets[i]) / dims; }
  inline uint32_t get_id(size_t i) const noexcept { return (ids.size() > 0) ? ids[i] : i; }

  inline float* data() const noexcept { return values.get(); }
  inline float* data(size_t i) const noexcept { return values.get() + offsets[i]; }

  inline auto get_offsets() const noexcept {
    return parlay::make_slice(offsets.begin(), offsets.end());
  }
  inline auto get_ids() const noexcept { return parlay::make_slice(ids.begin(), ids.end()); }

  inline ChPoint operator[](size_t i) const {
    return ChPoint(get_size(i), dims, data(i), get_id(i));
  }
  static constexpr bool is_metric() noexcept { return ChPoint::is_metric(); }

  // ---------------------------------------------------------
  // Slicing and Shuffling
  // ---------------------------------------------------------
  // Returns a lightweight, non-owning contiguous slice [i, j)
  inline PointCloudSetSlice<ChPoint> slice(size_t i, size_t j) const {
    assert(i <= j && j <= n);
    return PointCloudSetSlice<ChPoint>(*this, i, j);
  }

  // Returns a new PointCloudSet with data physically reordered to match the new_order indices
  PointCloudSet<ChPoint> shuffle(const parlay::sequence<uint32_t>& new_order) const;

  // ---------------------------------------------------------
  // Distance Methods
  // ---------------------------------------------------------
  size_t distances(const ChPoint& query, std::pair<uint32_t, float>* results) const;
  std::pair<parlay::sequence<std::pair<uint32_t, float>>, size_t> distances(
      const ChPoint& query) const;

  size_t distances_blocked(const ChPoint& query, std::pair<uint32_t, float>* results) const;
  std::pair<parlay::sequence<std::pair<uint32_t, float>>, size_t> distances_blocked(
      const ChPoint& query) const;

  template<typename Seq>
  size_t distances_subset(const ChPoint& query, const Seq& indices,
                          std::pair<uint32_t, float>* results) const;

  template<typename Seq>
  std::pair<parlay::sequence<std::pair<uint32_t, float>>, size_t> distances_subset(
      const ChPoint& query, const Seq& indices) const;

  parlay::sequence<std::pair<uint32_t, float>> distances(const PointCloudSet<ChPoint>& Queries,
                                                         size_t k) const;

  // ---------------------------------------------------------
  // Utilities
  // ---------------------------------------------------------
  void read_mmap_file(const char* filename);

  template<typename Seq>
  inline auto filter(const Seq& cluster_ids) const {
    return parlay::delayed_tabulate(cluster_ids.size(),
                                    [&](size_t i) { return (*this)[cluster_ids[i]]; });
  }

  template<typename Seq>
  parlay::sequence<parlay::sequence<float>> filter_flattened(const Seq& cluster_ids) const;

  template<typename Seq>
  void set_point_cloud(size_t i, Seq& point_cloud);

  inline void prefetch(size_t i) const noexcept {
    if (i >= n) return;
    const char* curr = reinterpret_cast<const char*>(values.get() + offsets[i]);
    const char* end = reinterpret_cast<const char*>(values.get() + offsets[i + 1]);
    const size_t max_prefetch_bytes = 2048;
    const char* limit = std::min(end, curr + max_prefetch_bytes);
    while (curr < limit) {
      __builtin_prefetch(curr, 0, 3);
      curr += 64;
    }
    __builtin_prefetch(&offsets[i + 1], 0, 3);
  }
};

/* =======================================Flattened View======================================= */
template<typename PCSet>
struct FlattenedPCRange {
  const float* raw_data;
  size_t _size;
  uint32_t _dim;

  explicit FlattenedPCRange(const PCSet& s) :
      raw_data(s.data()), _size(s.total_size()), _dim(s.get_dims()) {}

  size_t size() const { return _size; }
  uint32_t get_dims() const { return _dim; }
  const uint8_t* location(size_t i) const {
    return reinterpret_cast<const uint8_t*>(raw_data + i * static_cast<size_t>(_dim));
  }
  const float* data() const { return raw_data; }
};

}  // namespace mvsic

// Include the implementations
#include "point_cloud_set-inl.h"