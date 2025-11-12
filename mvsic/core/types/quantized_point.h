#pragma once

#include "parlay/primitives.h"
#include "mvsic/core/types/quantized_point_range.h"
#include "algorithms/utils/euclidian_point.h"
#include "algorithms/utils/mips_point.h"

namespace mvsic {

template<typename Point>
class QuantizedPoint {
 public:
  using distanceType = typename Point::distanceType;
  QuantizedPoint(const Point& raw_query, const QuantizedPointRange<Point>& qpr) {
    const auto& pq_result = qpr.get_pq_result();
    const auto& config = qpr.get_config();

    auto projector_or = pq_result.model->GetProjection(config.projection());
    if (!projector_or.ok()) {
      throw std::runtime_error("Failed to get projector from model.");
    }
    auto projector = projector_or.value();

    auto quantization_distance_or =
        research_scann::GetDistanceMeasure(config.quantization_distance());
    if (!quantization_distance_or.ok()) {
      throw std::runtime_error("Failed to get quantization distance measure.");
    }
    auto quantization_distance = quantization_distance_or.value();

    scann_pq::ScannPQQueryer queryer(projector, quantization_distance, pq_result.model);
    auto status_or_table = scann_pq::create_lookup_table(raw_query, queryer);

    if (!status_or_table.ok()) {
      throw std::runtime_error("Failed to create lookup table.");
    }
    lookup_table_ = std::move(status_or_table).value();

    num_blocks_ = pq_result.model->num_blocks();
    num_clusters_per_block_ = pq_result.model->num_clusters_per_block();
    dist_cmps_ = (num_clusters_per_block_ + 1) * raw_query.get_dims();
  }

  size_t get_dist_cmps() const { return dist_cmps_; }

  float distance(const QuantizedVectorRef& db_vector) const {
    const uint8_t* vec_start = db_vector.codes;
    float dist = 0.0f;
    const float* lookup_table_ptr = lookup_table_.float_lookup_table.data();
    for (size_t block_idx = 0; block_idx < num_blocks_; ++block_idx) {
      const uint8_t code = vec_start[block_idx];
      dist += lookup_table_ptr[block_idx * num_clusters_per_block_ + code];
    }
    return dist;
  }

 private:
  scann_pq::ScannLookupTable lookup_table_;
  size_t num_blocks_;
  size_t num_clusters_per_block_;
  size_t dist_cmps_;
};

// Implementations for QuantizedVectorRef declared in quantized_point_range.h

inline QuantizedVectorRef::QuantizedVectorRef(const uint8_t* c, size_t nb, bool is_metric) :
    codes(c), num_blocks(nb), metric(is_metric) {}

template<typename Point>
inline float QuantizedVectorRef::distance(const QuantizedPoint<Point>& query) const {
  return query.distance(*this);
}

}  // namespace mvsic
