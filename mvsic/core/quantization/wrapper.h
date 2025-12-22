#pragma once

#include <vector>
#include <limits>
#include <algorithm>
#include <cmath>
#include <cstring>

#include "parlay/sequence.h"
#include "parlay/parallel.h"

namespace mvsic {

// ---------------------------------------------------------
// Helper: Flatten PointCloudSet into a PointRange-like view
// ---------------------------------------------------------
template<typename PCSet>
struct FlattenedPCRange {
  const float* raw_data;
  size_t _size;
  uint32_t _dim;

  FlattenedPCRange(const PCSet& s) :
      raw_data(s.data()), _size(s.total_size()), _dim(s.get_dims()) {}

  size_t size() const { return _size; }
  uint32_t get_dims() const { return _dim; }

  const uint8_t* location(size_t i) const {
    return reinterpret_cast<const uint8_t*>(raw_data + i * _dim);
  }
};

// ---------------------------------------------------------
// Quantized Point Cloud: Aux Type
// ---------------------------------------------------------
template<typename VectorQuantizer, bool Metric>
class Quantized_Point_Cloud {
 public:
  const VectorQuantizer* quantizer;
  size_t start_idx;
  size_t end_idx;

  Quantized_Point_Cloud(const VectorQuantizer* q, size_t start, size_t end) :
      quantizer(q), start_idx(start), end_idx(end) {}

  size_t size() const { return end_idx - start_idx; }

  // Access the j-th quantized vector handle
  auto operator[](size_t j) const { return (*quantizer)[start_idx + j]; }
};

// ---------------------------------------------------------
// Quantized Query Point Cloud
// ---------------------------------------------------------
template<typename QuantizedQueryVec, bool Metric>
class Quantized_Query_Point_Cloud {
 public:
  using distanceType = float;
  std::vector<QuantizedQueryVec> vec_queries;

  // Optimized Chamfer Distance for Quantized Handles
  // This takes a Quantized_Point_Cloud and uses its operator[]
  template<typename CloudHandle>
  float distance(const CloudHandle& cloud) const {
    size_t num_q = vec_queries.size();
    if (num_q == 0) return 0.0f;

    size_t cloud_size = cloud.size();
    float total_chamfer = 0.0f;
    for (const auto& q_vec : vec_queries) {
      float min_dist = std::numeric_limits<float>::max();
      for (size_t i = 0; i < cloud_size; ++i) {
        if (i + 1 < cloud_size) {
          cloud[i + 1].prefetch();
        }
        float d = q_vec.distance(cloud[i]);
        if (d < min_dist) {
          min_dist = d;
        }
      }
      total_chamfer += min_dist;
    }
    return total_chamfer / static_cast<float>(num_q);
  }

  template<typename CloudHandle>
  std::pair<float, size_t> distance_w_cmps(const CloudHandle& cloud) const {
    return {this->distance(cloud), vec_queries.size()};
  }

  static constexpr bool is_metric() { return Metric; }
};

// ---------------------------------------------------------
// Quantized Point Cloud Set (The Main Container)
// ---------------------------------------------------------
template<typename VectorQuantizer, bool Metric>
class Quantized_Point_Cloud_Set {
 public:
  VectorQuantizer vec_quantizer;
  parlay::sequence<size_t> offsets;
  uint32_t n_clouds;

  Quantized_Point_Cloud_Set() {}

  template<typename PCSet, typename... Args>
  Quantized_Point_Cloud_Set(const PCSet& pcs, Args&&... args) :
      vec_quantizer(FlattenedPCRange<PCSet>(pcs), std::forward<Args>(args)...) {
    n_clouds = pcs.size();
    auto pcs_offsets = pcs.get_offsets();
    offsets = parlay::sequence<size_t>(pcs_offsets.begin(), pcs_offsets.end());

    uint32_t dimension = pcs.get_dims();
    if (dimension > 0) {
      parlay::parallel_for(0, offsets.size(), [&](size_t i) { offsets[i] /= dimension; });
    }
  }

  // Returns a aux handle (Quantized_Point_Cloud)
  auto operator[](size_t i) const {
    return Quantized_Point_Cloud<VectorQuantizer, Metric>(&vec_quantizer, offsets[i],
                                                          offsets[i + 1]);
  }

  // LUTs Construction
  template<typename ChPoint>
  auto quantize_query(const ChPoint& query_cloud) const {
    using QVecType = decltype(vec_quantizer.quantize_query(query_cloud[0]));
    Quantized_Query_Point_Cloud<QVecType, Metric> qqc;

    size_t n_q = query_cloud.size();
    if (n_q >= 8) {
      vec_quantizer.quantize_query_batch(query_cloud, qqc.vec_queries);
    } else {
      qqc.vec_queries.reserve(n_q);
      for (size_t i = 0; i < n_q; ++i) {
        qqc.vec_queries.push_back(vec_quantizer.quantize_query(query_cloud[i].data()));
      }
    }
    return qqc;
  }

  template<typename QuantizedQueryTy, typename Seq>
  size_t distances(const QuantizedQueryTy& q_query, const Seq indices, size_t n,
                   std::pair<uint32_t, float>* results) const {
    parlay::parallel_for(0, n, [&](size_t i) {
      uint32_t cloud_id = indices[i];
      // target_cloud is a Quantized_Point_Cloud
      auto target_cloud = (*this)[cloud_id];
      results[i] = {cloud_id, q_query.distance(target_cloud)};
    });
    return q_query.vec_queries.size();
  }

  inline size_t get_dist_cmps() const { return vec_quantizer.num_blocks; }

  void save(std::ofstream& out) const {
    out.write((char*)&n_clouds, sizeof(n_clouds));
    size_t off_size = offsets.size();
    out.write((char*)&off_size, sizeof(off_size));
    out.write((char*)offsets.data(), off_size * sizeof(size_t));
    vec_quantizer.save(out);
  }

  void load(std::ifstream& in) {
    in.read((char*)&n_clouds, sizeof(n_clouds));
    size_t off_size;
    in.read((char*)&off_size, sizeof(off_size));
    offsets.resize(off_size);
    in.read((char*)offsets.data(), off_size * sizeof(size_t));
    vec_quantizer.load(in);
  }
};

}  // namespace mvsic