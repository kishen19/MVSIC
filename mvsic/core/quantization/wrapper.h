#pragma once

#include <vector>
#include <limits>
#include <algorithm>
#include <cmath>

#include "parlay/sequence.h"
#include "parlay/parallel.h"

namespace mvsic {

// ---------------------------------------------------------
// Helper: Flatten PointCloudSet into a PointRange-like view
// ---------------------------------------------------------
// This allows the Vector Quantizers (which expect a flat range of vectors)
// to consume the PointCloudSet without modification.
template<typename PCSet>
struct FlattenedPCRange {
  const PCSet& pcs;
  size_t _size;
  size_t _dim;

  FlattenedPCRange(const PCSet& s) : pcs(s), _size(s.total_size()), _dim(s.get_dims()) {}

  // API expected by Vector Quantizers (PQ, RaBitQ, ScaNN)
  size_t size() const { return _size; }
  size_t get_dims() const { return _dim; }

  // Access the i-th vector in the entire dataset
  const float* location(size_t i) const { return pcs.data() + i * _dim; }
};

// ---------------------------------------------------------
// Quantized Point Cloud (Handle)
// ---------------------------------------------------------
// A lightweight view into the quantized dataset representing one cloud.
template<typename VectorQuantizer>
class Quantized_Point_Cloud {
 public:
  const VectorQuantizer* quantizer;
  size_t start_idx;
  size_t end_idx;

  Quantized_Point_Cloud(const VectorQuantizer* q, size_t start, size_t end) :
      quantizer(q), start_idx(start), end_idx(end) {}

  size_t size() const { return end_idx - start_idx; }

  // Access the j-th quantized vector in this cloud
  // Returns the lightweight handle (Quantized_Point, RaBitQ_Point_Wrapper, etc.)
  auto operator[](size_t j) const { return (*quantizer)[start_idx + j]; }
};

// ---------------------------------------------------------
// Quantized Query Point Cloud
// ---------------------------------------------------------
// Holds the pre-computed query structures (LUTs or Rotated Vectors)
// for every vector in the query point cloud.
template<typename QuantizedQueryVec>
class Quantized_Query_Point_Cloud {
 public:
  std::vector<QuantizedQueryVec> vec_queries;

  // -------------------------------------------------------
  // Normalized Chamfer Distance
  // D(Q, P) = (1/|Q|) * Sum_{q in Q} [ Min_{p in P} dist(q, p) ]
  // -------------------------------------------------------
  template<typename CloudHandle>
  float distance(const CloudHandle& cloud) const {
    float total_chamfer = 0.0f;
    size_t num_q = vec_queries.size();
    if (num_q == 0) return 0.0f;

    // Iterate over every vector in the query cloud
    for (const auto& q_vec : vec_queries) {
      float min_dist = std::numeric_limits<float>::max();

      // Linear scan over the target quantized cloud to find NN for this query vector
      size_t cloud_size = cloud.size();
      for (size_t i = 0; i < cloud_size; ++i) {
        // q_vec is the Quantized_Query wrapper (e.g. holds LUT)
        // cloud[i] is the Quantized_Point wrapper (e.g. holds code pointer)
        float d = q_vec.distance(cloud[i]);
        if (d < min_dist) {
          min_dist = d;
        }
      }
      total_chamfer += min_dist;
    }
    return total_chamfer / num_q;
  }
};

// ---------------------------------------------------------
// Quantized Point Cloud Set (The Main Container)
// ---------------------------------------------------------
// Wraps ANY vector quantizer (PQ, RaBitQ, ScaNN) to handle Point Clouds.
template<typename VectorQuantizer>
class Quantized_Point_Cloud_Set {
 public:
  VectorQuantizer vec_quantizer;
  parlay::sequence<size_t> offsets;  // Boundaries of point clouds
  uint32_t n_clouds;

  // Constructor:
  // 1. Creates a Flattened View of the PointCloudSet
  // 2. Forwards this view + any extra args (like bits, M, K) to the VectorQuantizer
  template<typename PCSet, typename... Args>
  Quantized_Point_Cloud_Set(const PCSet& pcs, Args&&... args) :
      vec_quantizer(FlattenedPCRange<PCSet>(pcs), std::forward<Args>(args)...) {
    n_clouds = pcs.size();

    // Copy offsets to reconstruct cloud boundaries later
    auto pcs_offsets = pcs.get_offsets();
    offsets = parlay::sequence<size_t>(pcs_offsets.begin(), pcs_offsets.end());
  }

  // Access a specific Point Cloud by index
  Quantized_Point_Cloud<VectorQuantizer> operator[](size_t i) const {
    return Quantized_Point_Cloud<VectorQuantizer>(&vec_quantizer, offsets[i], offsets[i + 1]);
  }

  // Quantize a Query Point Cloud
  // Returns a Quantized_Query_Point_Cloud holding vector-queries
  template<typename PointCloudTy>
  auto quantize_query(const PointCloudTy& query_cloud) const {
    // 1. Deduce the type of the single-vector query object returned by the quantizer
    // (e.g., RaBitQ_Query_Wrapper or Quantized_Query)
    using QVecType = decltype(vec_quantizer.quantize_query(query_cloud[0]));

    Quantized_Query_Point_Cloud<QVecType> qqc;
    qqc.vec_queries.reserve(query_cloud.size());

    // 2. Quantize every vector in the query cloud
    for (size_t i = 0; i < query_cloud.size(); ++i) {
      qqc.vec_queries.push_back(vec_quantizer.quantize_query(query_cloud[i]));
    }
    return qqc;
  }
};

}  // namespace mvsic