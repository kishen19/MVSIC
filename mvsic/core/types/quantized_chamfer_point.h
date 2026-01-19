#pragma once

#include "parlay/primitives.h"
#include "mvsic/core/types/chamfer_l2_point.h"
#include "mvsic/core/types/chamfer_ip_point.h"
#include "mvsic/core/types/quantized_point_cloud_set.h"

namespace mvsic {

template<bool metric>
class QuantizedChamferPoint {
 public:
  using ChPoint = std::conditional_t<metric, ChamferL2_Point, ChamferIP_Point>;

  QuantizedChamferPoint(const ChPoint& raw_query, const QuantizedPointCloudSet<ChPoint>& qpcs) {
    const auto& pq_result = qpcs.get_pq_result();
    const auto& config = qpcs.get_config();

    auto projector_or = pq_result.model->GetProjection(config.projection());
    if (!projector_or.ok()) {
      throw std::runtime_error("Failed to get projector from model.");
    }
    auto projector = projector_or.value();

    lookup_tables_ = scann_pq::create_lookup_tables_batched_eigen(raw_query, projector,
                                                                  pq_result.model->centers());

    num_blocks_ = pq_result.model->num_blocks();
    num_clusters_per_block_ = pq_result.model->num_clusters_per_block();
    query_size_ = raw_query.size();
    dist_cmps_ = (num_clusters_per_block_ + query_size_) * raw_query.get_dims();
  }

  size_t get_dist_cmps() const { return dist_cmps_; }

  float distance(const QuantizedDBPoint& db_point) const {
    auto dists_query = parlay::sequence<float>::uninitialized(query_size_);

    parlay::parallel_for(0, query_size_, [&](size_t i) {
      const auto& lut = lookup_tables_[i];
      float min_dist_for_query_point = std::numeric_limits<float>::max();

      for (size_t k = 0; k < db_point.num_vectors; ++k) {
        const uint8_t* vec_start = db_point.codes + k * num_blocks_;
        float dist = 0.0f;
        const float* lookup_table_ptr = lut.data();
        for (size_t block_idx = 0; block_idx < num_blocks_; ++block_idx) {
          const uint8_t code = vec_start[block_idx];
          dist += lookup_table_ptr[block_idx * num_clusters_per_block_ + code];
        }
        min_dist_for_query_point = std::min(min_dist_for_query_point, dist);
      }
      dists_query[i] = min_dist_for_query_point;
    });

    return parlay::reduce(dists_query) / query_size_;
  }

 private:
  parlay::sequence<parlay::sequence<float>> lookup_tables_;
  size_t num_blocks_;
  size_t num_clusters_per_block_;
  size_t query_size_;
  size_t dist_cmps_;
};

using QuantizedChamferL2Point = QuantizedChamferPoint<true>;
using QuantizedChamferIPPoint = QuantizedChamferPoint<false>;

}  // namespace mvsic
