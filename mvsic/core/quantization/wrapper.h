#pragma once

#include <vector>
#include <limits>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <immintrin.h>

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
  parlay::sequence<QuantizedQueryVec> vec_queries;

  // Optimized Chamfer Distance for Quantized Handles
  template<typename QuantizedPointCloud>
  float distance(const QuantizedPointCloud& cloud) const {
    size_t num_q = vec_queries.size();
    if (num_q == 0) return 0.0f;

    size_t cloud_size = cloud.size();
    float total_chamfer = 0.0f;
    // Note: Better to keep this sequential
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
    // TODO: fix the cmps value
    return {this->distance(cloud), vec_queries.size()};
  }

  static constexpr bool is_metric() { return Metric; }
};

// ---------------------------------------------------------
// Quantized Point Cloud Set (The Main Container)
// ---------------------------------------------------------
// VectorQuantizer: The associated Quantized PointRange class
//                  e.g.: pq::Quantized_Point_Range, rabitq::Quantized_Point_Range
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
    // Batching seems to help
    if (n_q >= 8) {
      vec_quantizer.quantize_query_batch(query_cloud, qqc.vec_queries);
    } else {
      qqc.vec_queries = parlay::tabulate(
          n_q, [&](size_t i) { return vec_quantizer.quantize_query(query_cloud[i].data()); });
    }
    return qqc;
  }

  template<typename QuantizedQueryTy, typename Seq>
  size_t distances(const QuantizedQueryTy& q_query, const Seq& indices, size_t n,
                   std::pair<uint32_t, float>* results) const {

    if constexpr (VectorQuantizer::is_fastscan) {

      parlay::parallel_for(0, n, [&](size_t i) {
        const uint32_t cloud_id = indices[i];

        // NEW (no padding): use true vector offsets (vector indices in the *flattened* DB)
        const size_t start = vec_quantizer.cloud_vec_offsets[cloud_id];
        const size_t end = vec_quantizer.cloud_vec_offsets[cloud_id + 1];
        const size_t cloud_size = (end > start) ? (end - start) : 0;

        float total_chamfer = 0.0f;

        const size_t strip_stride = static_cast<size_t>(vec_quantizer.num_blocks) * 32;

        // Precompute strip / lane boundaries
        const size_t strip0 = start / 64;
        const int lane0 = static_cast<int>(start % 64);

        const size_t strip1 = end / 64;
        const int lane1 = static_cast<int>(end % 64);  // if 0 => ends on strip boundary

        // Process each query vector sequentially
        for (const auto& q_vec : q_query.vec_queries) {

          if (cloud_size == 0) {
            total_chamfer += std::numeric_limits<float>::max();
            continue;
          }

          float min_d = std::numeric_limits<float>::max();

          // Helper to get pointer to strip s
          auto strip_ptr = [&](size_t s) -> const uint8_t* {
            return &vec_quantizer.packed_codes[s * strip_stride];
          };

          // Case A: all vectors live in the same strip
          if (strip0 == strip1) {
            // range is [lane0, lane1) within strip0; note lane1 can be 0 only when end%64==0,
            // but if strip0==strip1 and end%64==0, that implies start and end are both in same
            // strip, and lane1==0 means "up to end of strip".
            const int hi = (lane1 == 0) ? 64 : lane1;
            min_d = vec_quantizer.scan_64_chunk_min_masked(q_vec, strip_ptr(strip0), lane0, hi);
            total_chamfer += min_d;
            continue;
          }

          // Case B: multi-strip range

          // 1) First partial strip: lanes [lane0, 64)
          {
            const float d0 =
                vec_quantizer.scan_64_chunk_min_masked(q_vec, strip_ptr(strip0), lane0, 64);
            if (d0 < min_d) min_d = d0;
          }

          // 2) Middle full strips: (strip0+1) .. (strip_last_full)
          // Determine last strip that is fully included.
          // If lane1==0, end is exactly at boundary and strip1 is the first strip AFTER the range,
          // so last full strip is strip1-1.
          // If lane1!=0, strip1 is the last (partial) strip in range, so last full strip is
          // strip1-1.
          const size_t first_full = strip0 + 1;
          const size_t last_full = (lane1 == 0) ? (strip1 - 1) : (strip1 - 1);

          if (first_full <= last_full) {
            // We want min over (possibly many) full strips.
            // Use 4 accumulators for better ILP; reduce to scalar min at end.
            __m512i min0 = _mm512_set1_epi16(0xFFFF);
            __m512i min1 = _mm512_set1_epi16(0xFFFF);
            __m512i min2 = _mm512_set1_epi16(0xFFFF);
            __m512i min3 = _mm512_set1_epi16(0xFFFF);

            size_t s = first_full;

            // Process 4 strips per iteration
            for (; s + 3 <= last_full; s += 4) {
              const uint8_t* p0 = strip_ptr(s);
              min0 = vec_quantizer.scan_64_running_min(q_vec, p0, min0);
              min1 = vec_quantizer.scan_64_running_min(q_vec, p0 + strip_stride, min1);
              min2 = vec_quantizer.scan_64_running_min(q_vec, p0 + 2 * strip_stride, min2);
              min3 = vec_quantizer.scan_64_running_min(q_vec, p0 + 3 * strip_stride, min3);
            }

            __m512i combined =
                _mm512_min_epu16(_mm512_min_epu16(min0, min1), _mm512_min_epu16(min2, min3));

            // Remaining strips one-by-one
            for (; s <= last_full; ++s) {
              combined = vec_quantizer.scan_64_running_min(q_vec, strip_ptr(s), combined);
            }

            const float d_full = vec_quantizer.reduce_running_min(q_vec, combined);
            if (d_full < min_d) min_d = d_full;
          }

          // 3) Last partial strip (only if lane1 != 0)
          if (lane1 != 0) {
            const float d1 =
                vec_quantizer.scan_64_chunk_min_masked(q_vec, strip_ptr(strip1), 0, lane1);
            if (d1 < min_d) min_d = d1;
          }

          total_chamfer += min_d;
        }

        // Store (ID, avg chamfer)
        results[i] = {cloud_id, total_chamfer / static_cast<float>(q_query.vec_queries.size())};
      });

    } else {
      // Standard path for PQ, RaBitQ, or ScaNN
      parlay::parallel_for(0, n, [&](size_t i) {
        uint32_t cloud_id = indices[i];
        results[i] = {cloud_id, q_query.distance((*this)[cloud_id])};
      });
    }

    return q_query.vec_queries.size();
  }

  template<typename QuantizedQueryTy>
  size_t distances_all(const QuantizedQueryTy& q_query, std::pair<uint32_t, float>* results) const {
    const size_t num_q = q_query.vec_queries.size();

    if constexpr (VectorQuantizer::is_fastscan) {
      parlay::parallel_for(0, n_clouds, [&](size_t cid) {
        const uint32_t cloud_id = uint32_t(cid);

        // wrapper offsets are VECTOR indices into the flattened DB
        const size_t start = offsets[cloud_id];
        const size_t end = offsets[cloud_id + 1];
        const size_t cloud_size = (end > start) ? (end - start) : 0;

        if (num_q == 0) {
          results[cid] = {cloud_id, 0.0f};
          return;
        }
        if (cloud_size == 0) {
          results[cid] = {cloud_id, std::numeric_limits<float>::max()};
          return;
        }

        const size_t strip_stride = static_cast<size_t>(vec_quantizer.num_blocks) * 32;

        // aligned-fast path (what our synthetic benchmark uses)
        const bool aligned = ((start & 63) == 0) && ((end & 63) == 0);

        float total = 0.0f;

        if (aligned) {
          const size_t strip0 = start / 64;
          const size_t n_strips = cloud_size / 64;

          auto strip_ptr = [&](size_t s) -> const uint8_t* {
            return &vec_quantizer.packed_codes[(strip0 + s) * strip_stride];
          };

          for (const auto& qv : q_query.vec_queries) {
            __m512i min0 = _mm512_set1_epi16(0xFFFF);
            __m512i min1 = _mm512_set1_epi16(0xFFFF);
            __m512i min2 = _mm512_set1_epi16(0xFFFF);
            __m512i min3 = _mm512_set1_epi16(0xFFFF);

            size_t s = 0;
            for (; s + 3 < n_strips; s += 4) {
              const uint8_t* p0 = strip_ptr(s);
              min0 = vec_quantizer.scan_64_running_min(qv, p0, min0);
              min1 = vec_quantizer.scan_64_running_min(qv, p0 + strip_stride, min1);
              min2 = vec_quantizer.scan_64_running_min(qv, p0 + 2 * strip_stride, min2);
              min3 = vec_quantizer.scan_64_running_min(qv, p0 + 3 * strip_stride, min3);
            }

            __m512i combined =
                _mm512_min_epu16(_mm512_min_epu16(min0, min1), _mm512_min_epu16(min2, min3));

            for (; s < n_strips; ++s) {
              combined = vec_quantizer.scan_64_running_min(qv, strip_ptr(s), combined);
            }

            total += vec_quantizer.reduce_running_min(qv, combined);
          }

          results[cid] = {cloud_id, total / float(num_q)};
          return;
        }

        // general no-padding path (works for misalignment, slower)
        for (const auto& qv : q_query.vec_queries) {
          float md = std::numeric_limits<float>::max();
          for (size_t j = 0; j < cloud_size; ++j) {
            float d = qv.distance(vec_quantizer[start + j]);
            if (d < md) md = d;
          }
          total += md;
        }
        results[cid] = {cloud_id, total / float(num_q)};
      });

      return num_q;
    } else {
      // PQ / other quantizers: use generic wrapper distance() on a cloud handle
      parlay::parallel_for(0, n_clouds, [&](size_t cid) {
        results[cid] = {uint32_t(cid), q_query.distance((*this)[cid])};
      });
      return num_q;
    }
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