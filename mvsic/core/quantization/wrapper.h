// mvsic/core/quantization/wrapper.h
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <immintrin.h>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include "parlay/parallel.h"
#include "parlay/sequence.h"
#include "mvsic/core/quantization/turboquant_4bit.h"
#include "mvsic/core/quantization/turboquant_pq_4bit.h"
#include "mvsic/core/quantization/turboquant_byte.h"

namespace mvsic {

// Trait: query supports fast distances_slice over an EncRange (used for TQ-PQ).
template<typename QueryVec, typename EncRange, typename = void>
struct has_distances_slice_t : std::false_type {};
template<typename QueryVec, typename EncRange>
struct has_distances_slice_t<
    QueryVec, EncRange,
    std::void_t<decltype(std::declval<const QueryVec&>().distances_slice(
        std::declval<const EncRange&>(), size_t{}, size_t{}, static_cast<float*>(nullptr)))>> :
    std::true_type {};

// Trait: EncRange advertises a TQ-PQ fast strip layout.
template<typename EncRange, typename = void>
struct has_tqpq_fast_t : std::false_type {};
template<typename EncRange>
struct has_tqpq_fast_t<EncRange, std::void_t<decltype(EncRange::is_tqpq_fast)>> :
    std::bool_constant<EncRange::is_tqpq_fast> {};

// ---------------------------------------------------------
// Helper: Flatten PointCloudSet into a PointRange-like view
// ---------------------------------------------------------
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

// ---------------------------------------------------------
// Quantized Point Cloud: Aux Type (handle over encoded vectors)
// ---------------------------------------------------------
template<typename EncRange, bool Metric>
class Quantized_Point_Cloud {
 public:
  const EncRange* db = nullptr;
  size_t start_idx = 0;
  size_t end_idx = 0;

  Quantized_Point_Cloud() = default;
  Quantized_Point_Cloud(const EncRange* d, size_t start, size_t end) :
      db(d), start_idx(start), end_idx(end) {}

  size_t size() const { return end_idx - start_idx; }

  auto operator[](size_t j) const { return (*db)[start_idx + j]; }
};

// ---------------------------------------------------------
// Quantized Query Point Cloud
//  - stores per-vector quantized queries (LUTs, etc.)
//  - generic Chamfer over a Quantized_Point_Cloud handle
// ---------------------------------------------------------
template<typename QuantizedQueryVec, bool Metric>
class Quantized_Query_Point_Cloud {
 public:
  using distanceType = float;

  parlay::sequence<QuantizedQueryVec> vec_queries;

 private:
  // SFINAE: detect if QuantizedQueryVec has batch distances support.
  template<typename T, typename = void>
  struct has_batch_distances_t : std::false_type {};
  template<typename T>
  struct has_batch_distances_t<T, std::void_t<decltype(T::has_batch_distances)>> :
      std::bool_constant<T::has_batch_distances> {};

#ifdef __AVX512F__
  // Batch Chamfer: VNNI GEMM scoring (or byte TQ AVX2 GEMM).
  template<typename QuantizedPointCloud>
  float distance_batch(const QuantizedPointCloud& cloud) const {
    const size_t num_q = vec_queries.size();
    if (num_q == 0) return 0.0f;

    const size_t cloud_size = cloud.size();
    if (cloud_size == 0) return std::numeric_limits<float>::max();

    const auto* db = cloud.db;
    const size_t start = cloud.start_idx;

    const size_t strip_idx = start / 64;
    const size_t lane_offset = start % 64;
    const size_t strip_stride = db->stride;
    const size_t n_strips = (lane_offset + cloud_size + 63) / 64;

    const uint8_t* strip_data = db->packed_codes.data() + strip_idx * strip_stride;
    const float* norms = db->norm_scaling_factors.data() + start;
    const float* sqn = db->unquantized_squared_norms.data() + start;

    std::vector<const QuantizedQueryVec*> qptrs(num_q);
    for (size_t i = 0; i < num_q; ++i)
      qptrs[i] = &vec_queries[i];

    // TQ-PQ path: use query's fast slice distances, then reduce to Chamfer.
    using EncRangeT = std::remove_reference_t<decltype(*db)>;
    if constexpr (has_distances_slice_t<QuantizedQueryVec, EncRangeT>::value) {
      static thread_local std::vector<float> tmp;
      tmp.resize(cloud_size);
      float total = 0.0f;
      for (const auto& qv : vec_queries) {
        qv.distances_slice(*db, start, cloud_size, tmp.data());
        float min_d = *std::min_element(tmp.begin(), tmp.begin() + cloud_size);
        total += min_d;
      }
      return total / static_cast<float>(num_q);
    } else if constexpr (std::is_same_v<QuantizedQueryVec,
                                        turboquant_byte::Quantized_Query<Metric>>) {
      const float total = turboquant_byte::chamfer_byte_tq_gemm_512<Metric>(
          qptrs.data(), num_q, strip_data, norms, sqn, strip_stride, n_strips,
          db->num_bytes_per_datapoint, cloud_size);
      return total / static_cast<float>(num_q);
    } else {
      const float total = turboquant_4bit::chamfer_vnni_gemm<Metric>(
          qptrs.data(), num_q, strip_data, norms, sqn, strip_stride, n_strips,
          db->num_bytes_per_datapoint, cloud_size, lane_offset);
      return total / static_cast<float>(num_q);
    }
  }
#endif  // __AVX512F__

#if !defined(__AVX512F__) && defined(__AVX2__)
  // Batch Chamfer: AVX2 GEMM scoring.
  // Dispatches to turboquant_byte or turboquant_4bit GEMM.
  template<typename QuantizedPointCloud>
  float distance_batch(const QuantizedPointCloud& cloud) const {
    const size_t num_q = vec_queries.size();
    if (num_q == 0) return 0.0f;

    const size_t cloud_size = cloud.size();
    if (cloud_size == 0) return std::numeric_limits<float>::max();

    const auto* db = cloud.db;
    const size_t start = cloud.start_idx;

    const size_t strip_idx = start / 64;
    const size_t lane_offset = start % 64;
    const size_t strip_stride = db->stride;
    const size_t n_strips = (lane_offset + cloud_size + 63) / 64;

    const uint8_t* strip_data = db->packed_codes.data() + strip_idx * strip_stride;
    const float* norms = db->norm_scaling_factors.data() + start;
    const float* sqn = db->unquantized_squared_norms.data() + start;

    std::vector<const QuantizedQueryVec*> qptrs(num_q);
    for (size_t i = 0; i < num_q; ++i)
      qptrs[i] = &vec_queries[i];

    using EncRangeT = std::remove_reference_t<decltype(*db)>;
    if constexpr (has_distances_slice_t<QuantizedQueryVec, EncRangeT>::value) {
      static thread_local std::vector<float> tmp;
      tmp.resize(cloud_size);
      float total = 0.0f;
      for (const auto& qv : vec_queries) {
        qv.distances_slice(*db, start, cloud_size, tmp.data());
        float min_d = *std::min_element(tmp.begin(), tmp.begin() + cloud_size);
        total += min_d;
      }
      return total / static_cast<float>(num_q);
    } else if constexpr (std::is_same_v<QuantizedQueryVec,
                                        turboquant_byte::Quantized_Query<Metric>>) {
      const float total = turboquant_byte::chamfer_byte_tq_gemm<Metric>(
          qptrs.data(), num_q, strip_data, norms, sqn, strip_stride, n_strips,
          db->num_bytes_per_datapoint, cloud_size);
      return total / static_cast<float>(num_q);
    } else {
      const float total = turboquant_4bit::chamfer_avx2_gemm<Metric>(
          qptrs.data(), num_q, strip_data, norms, sqn, strip_stride, n_strips,
          db->num_bytes_per_datapoint, cloud_size, lane_offset);
      return total / static_cast<float>(num_q);
    }
  }
#endif  // !__AVX512F__ && __AVX2__

 public:
  // Per-point Chamfer: original path for quantizers without batch support.
  // Also used as scalar reference for verification.
  template<typename QuantizedPointCloud>
  float distance_perpoint(const QuantizedPointCloud& cloud) const {
    const size_t num_q = vec_queries.size();
    if (num_q == 0) return 0.0f;

    const size_t cloud_size = cloud.size();
    if (cloud_size == 0) return std::numeric_limits<float>::max();

    float total_chamfer = 0.0f;

    for (const auto& q_vec : vec_queries) {
      float min_dist = std::numeric_limits<float>::max();
      for (size_t i = 0; i < cloud_size; ++i) {
        if (i + 1 < cloud_size) cloud[i + 1].prefetch();
        float d = q_vec.distance(cloud[i]);
        if (d < min_dist) min_dist = d;
      }
      total_chamfer += min_dist;
    }

    return total_chamfer / static_cast<float>(num_q);
  }

  template<typename QuantizedPointCloud>
  float distance(const QuantizedPointCloud& cloud) const {
    if constexpr (has_batch_distances_t<QuantizedQueryVec>::value) {
      return distance_batch(cloud);
    } else {
      return distance_perpoint(cloud);
    }
  }

  // TODO: fix cmps
  template<typename CloudHandle>
  std::pair<float, size_t> distance_w_cmps(const CloudHandle& cloud) const {
    return {this->distance(cloud), vec_queries.size()};
  }

  static constexpr bool is_metric() {
    return Metric;
  }
};

// ---------------------------------------------------------
// Quantized Point Cloud Set (encoded DB + offsets + distances)
//   EncRange: encoded vector storage type
//            e.g. pq::Quantized_Point_Range<...>, fastscan::Quantized_Point_Range<...>, etc.
// ---------------------------------------------------------
template<typename EncRange, bool Metric>
class Quantized_Point_Cloud_Set {
 public:
  EncRange vec_db;
  parlay::sequence<size_t> offsets;  // vector-index offsets into flattened DB (size n_clouds+1)
  parlay::sequence<uint32_t> sizes_unpadded;  // size = n_clouds (only meaningful for fastscan)

  uint32_t n_clouds = 0;

  Quantized_Point_Cloud_Set() = default;

  Quantized_Point_Cloud_Set(EncRange&& enc, parlay::sequence<size_t>&& offs) :
      vec_db(std::move(enc)), offsets(std::move(offs)) {
    n_clouds = static_cast<uint32_t>(offsets.size() ? offsets.size() - 1 : 0);
  }

  Quantized_Point_Cloud_Set(EncRange&& enc, parlay::sequence<size_t>&& offs,
                            parlay::sequence<uint32_t>&& sizes) :
      vec_db(std::move(enc)), offsets(std::move(offs)), sizes_unpadded(std::move(sizes)) {
    n_clouds = static_cast<uint32_t>(offsets.size() ? offsets.size() - 1 : 0);
  }

  auto operator[](size_t i) const {
    const size_t start = offsets[i];
    size_t end = offsets[i + 1];

    if constexpr (EncRange::is_fastscan) {
      if (sizes_unpadded.size() == static_cast<size_t>(n_clouds)) {
        end = start + static_cast<size_t>(sizes_unpadded[i]);  // unpadded end
      }
    }
    return Quantized_Point_Cloud<EncRange, Metric>(&vec_db, start, end);
  }

  template<typename QuantizedQueryTy, typename Seq>
  size_t distances(const QuantizedQueryTy& q_query, const Seq& indices, size_t n,
                   std::pair<uint32_t, float>* results) const {

    if constexpr (EncRange::is_fastscan) {
#if defined(__AVX512F__) || defined(__AVX2__)
      using RunningMinV = typename EncRange::RunningMinVType;
      parlay::parallel_for(0, n, [&](size_t i) {
        const uint32_t cloud_id = indices[i];
        const size_t start = offsets[cloud_id];
        size_t cloud_size = 0;
        if (sizes_unpadded.size() == static_cast<size_t>(n_clouds)) {
          cloud_size = static_cast<size_t>(sizes_unpadded[cloud_id]);
        } else {
          const size_t end_padded = offsets[cloud_id + 1];
          cloud_size = (end_padded > start) ? (end_padded - start) : 0;
        }
        const size_t num_q = q_query.vec_queries.size();
        if (num_q == 0) {
          results[i] = {cloud_id, 0.0f};
          return;
        }
        if (cloud_size == 0) {
          results[i] = {cloud_id, std::numeric_limits<float>::max()};
          return;
        }
        const size_t true_end = start + cloud_size;
        const size_t strip_stride = static_cast<size_t>(vec_db.num_blocks) * 32;
        const size_t strip0 = start / 64;
        const int lane0 = static_cast<int>(start % 64);
        const size_t strip1 = true_end / 64;
        const int lane1 = static_cast<int>(true_end % 64);
        auto strip_ptr = [&](size_t s) -> const uint8_t* {
          return &vec_db.packed_codes[s * strip_stride];
        };
        float total_chamfer = 0.0f;
        const bool fully_aligned_full = (lane0 == 0) && (lane1 == 0) && (strip1 > strip0);

        for (const auto& q_vec : q_query.vec_queries) {
          float min_d = std::numeric_limits<float>::max();
          if (strip0 == strip1) {
            const int hi = (lane1 == 0) ? 64 : lane1;
            min_d = vec_db.scan_64_chunk_min_masked(q_vec, strip_ptr(strip0), lane0, hi);
            total_chamfer += min_d;
            continue;
          }
          if (fully_aligned_full) {
            const size_t first_full = strip0;
            const size_t last_full = strip1 - 1;
            RunningMinV min0 = RunningMinV::max();
            RunningMinV min1 = RunningMinV::max();
            RunningMinV min2 = RunningMinV::max();
            RunningMinV min3 = RunningMinV::max();
            size_t s = first_full;
            for (; s + 3 <= last_full; s += 4) {
              const uint8_t* p0 = strip_ptr(s);
              vec_db.scan_64_running_min(q_vec, p0, min0);
              vec_db.scan_64_running_min(q_vec, p0 + strip_stride, min1);
              vec_db.scan_64_running_min(q_vec, p0 + 2 * strip_stride, min2);
              vec_db.scan_64_running_min(q_vec, p0 + 3 * strip_stride, min3);
            }
            RunningMinV combined;
#ifdef __AVX512F__
            combined.v = _mm512_min_epu16(_mm512_min_epu16(min0.v, min1.v),
                                          _mm512_min_epu16(min2.v, min3.v));
#else
            combined.lo = _mm256_min_epu16(_mm256_min_epu16(min0.lo, min1.lo), _mm256_min_epu16(min2.lo, min3.lo));
            combined.hi = _mm256_min_epu16(_mm256_min_epu16(min0.hi, min1.hi), _mm256_min_epu16(min2.hi, min3.hi));
#endif
            for (; s <= last_full; ++s) {
              vec_db.scan_64_running_min(q_vec, strip_ptr(s), combined);
            }
            min_d = vec_db.reduce_running_min(q_vec, combined);
            total_chamfer += min_d;
            continue;
          }
          {
            const float d0 = vec_db.scan_64_chunk_min_masked(q_vec, strip_ptr(strip0), lane0, 64);
            if (d0 < min_d) min_d = d0;
          }
          const size_t first_full = strip0 + 1;
          const size_t last_full = strip1 - 1;
          if (first_full <= last_full) {
            RunningMinV min0 = RunningMinV::max();
            RunningMinV min1 = RunningMinV::max();
            RunningMinV min2 = RunningMinV::max();
            RunningMinV min3 = RunningMinV::max();
            size_t s = first_full;
            for (; s + 3 <= last_full; s += 4) {
              const uint8_t* p0 = strip_ptr(s);
              vec_db.scan_64_running_min(q_vec, p0, min0);
              vec_db.scan_64_running_min(q_vec, p0 + strip_stride, min1);
              vec_db.scan_64_running_min(q_vec, p0 + 2 * strip_stride, min2);
              vec_db.scan_64_running_min(q_vec, p0 + 3 * strip_stride, min3);
            }
            RunningMinV combined;
#ifdef __AVX512F__
            combined.v = _mm512_min_epu16(_mm512_min_epu16(min0.v, min1.v),
                                          _mm512_min_epu16(min2.v, min3.v));
#else
            combined.lo = _mm256_min_epu16(_mm256_min_epu16(min0.lo, min1.lo), _mm256_min_epu16(min2.lo, min3.lo));
            combined.hi = _mm256_min_epu16(_mm256_min_epu16(min0.hi, min1.hi), _mm256_min_epu16(min2.hi, min3.hi));
#endif
            for (; s <= last_full; ++s) {
              vec_db.scan_64_running_min(q_vec, strip_ptr(s), combined);
            }
            const float d_full = vec_db.reduce_running_min(q_vec, combined);
            if (d_full < min_d) min_d = d_full;
          }
          if (lane1 != 0) {
            const float d1 = vec_db.scan_64_chunk_min_masked(q_vec, strip_ptr(strip1), 0, lane1);
            if (d1 < min_d) min_d = d1;
          }
          total_chamfer += min_d;
        }
        results[i] = {cloud_id, total_chamfer / static_cast<float>(num_q)};
      });
      return q_query.vec_queries.size();
#else
      parlay::parallel_for(0, n, [&](size_t i) {
        uint32_t cloud_id = indices[i];
        results[i] = {cloud_id, q_query.distance((*this)[cloud_id])};
      });
      return q_query.vec_queries.size();
#endif
    } else if constexpr (has_tqpq_fast_t<EncRange>::value) {
#if defined(__AVX512F__)
      const size_t num_q = q_query.vec_queries.size();

      // Precompute per-query scalars once: beta (with 2.0 baked in for L2) and squared norms.
      std::vector<float> beta_all(num_q);
      std::vector<float> sqn_all(num_q);
      for (size_t qi = 0; qi < num_q; ++qi) {
        const auto& qv = q_query.vec_queries[qi];
        float base_beta =
            (qv.norm_scaling_factor * qv.lut_int8_scale) / turboquant_pq_4bit::kPQ_Int8Scale_D1_K16;
        beta_all[qi] = Metric ? (2.0f * base_beta) : base_beta;
        sqn_all[qi] = qv.unquantized_squared_norm;
      }

      parlay::parallel_for(0, n, [&](size_t i) {
        const uint32_t cloud_id = indices[i];
        const size_t start = offsets[cloud_id];
        const size_t end = offsets[cloud_id + 1];
        const size_t cloud_size = (end > start) ? (end - start) : 0;

        if (num_q == 0) {
          results[i] = {cloud_id, 0.0f};
          return;
        }
        if (cloud_size == 0) {
          results[i] = {cloud_id, std::numeric_limits<float>::max()};
          return;
        }

        const size_t true_end = start + cloud_size;
        const size_t strip_stride = vec_db.stride;
        const size_t strip0 = start / 64;
        const size_t strip1 = true_end / 64;
        const int lane1 = static_cast<int>(true_end % 64);

        const uint8_t* base_codes = vec_db.packed_codes.data();
        const float* base_norms = vec_db.norm_scaling_factors.data();
        const float* base_sqn = vec_db.unquantized_squared_norms.data();

        auto strip_ptr = [&](size_t s) -> const uint8_t* { return &base_codes[s * strip_stride]; };
        auto norms_ptr = [&](size_t s) -> const float* { return base_norms + s * 64; };
        auto sqn_ptr = [&](size_t s) -> const float* { return base_sqn + s * 64; };

        constexpr size_t kQBatch = 8;
        float total = 0.0f;

        for (size_t q0 = 0; q0 < num_q; q0 += kQBatch) {
          const size_t qb = std::min(kQBatch, num_q - q0);
          const auto* qv_arr = q_query.vec_queries.data() + q0;

          float beta_q[kQBatch];
          float sqn_q[kQBatch];
          for (size_t qi = 0; qi < qb; ++qi) {
            beta_q[qi] = beta_all[q0 + qi];
            sqn_q[qi] = sqn_all[q0 + qi];
          }

          const __m256i low_mask = _mm256_set1_epi8(0x0F);
          const __m512 inf_ps = _mm512_set1_ps(std::numeric_limits<float>::infinity());
          __m512 global_min_v[kQBatch];
          for (size_t qi = 0; qi < qb; ++qi)
            global_min_v[qi] = inf_ps;
          const size_t nb = static_cast<size_t>(vec_db.num_blocks_for_scan());

          auto scan_strip_masked = [&](size_t s, int lo, int hi) {
            const uint8_t* codes_ptr = strip_ptr(s);
            const float* ns = norms_ptr(s);
            const float* sq = sqn_ptr(s);

            __m512i acc_even[kQBatch];
            __m512i acc_odd[kQBatch];
            for (size_t qi = 0; qi < qb; ++qi) {
              acc_even[qi] = _mm512_setzero_si512();
              acc_odd[qi] = _mm512_setzero_si512();
            }

            for (size_t b = 0; b < nb; ++b) {
              const __m256i packed =
                  _mm256_loadu_si256(reinterpret_cast<const __m256i*>(codes_ptr));
              codes_ptr += 32;

              const __m256i codes_even = _mm256_and_si256(packed, low_mask);
              const __m256i codes_odd = _mm256_and_si256(_mm256_srli_epi16(packed, 4), low_mask);

              for (size_t qi = 0; qi < qb; ++qi) {
                const __m128i lut128 =
                    _mm_loadu_si128(reinterpret_cast<const __m128i*>(&qv_arr[qi].lut_int8[b * 16]));
                const __m256i lut256 = _mm256_broadcastsi128_si256(lut128);
                const __m256i scores_even_i8 = _mm256_shuffle_epi8(lut256, codes_even);
                const __m256i scores_odd_i8 = _mm256_shuffle_epi8(lut256, codes_odd);
                acc_even[qi] = _mm512_add_epi16(acc_even[qi], _mm512_cvtepi8_epi16(scores_even_i8));
                acc_odd[qi] = _mm512_add_epi16(acc_odd[qi], _mm512_cvtepi8_epi16(scores_odd_i8));
              }
            }

            // Precompute masks for pair vectors (16 pairs at a time) for both even and odd lanes.
            auto mask_pairs16 = [&](int pair_base, bool odd) -> __mmask16 {
              uint16_t m = 0;
              for (int j = 0; j < 16; ++j) {
                const int pair = pair_base + j;
                const int lane = (pair << 1) + (odd ? 1 : 0);
                if (lane >= lo && lane < hi) m |= static_cast<uint16_t>(1u << j);
              }
              return static_cast<__mmask16>(m);
            };
            const __mmask16 me0 = mask_pairs16(/*pair_base=*/0, /*odd=*/false);
            const __mmask16 mo0 = mask_pairs16(/*pair_base=*/0, /*odd=*/true);
            const __mmask16 me1 = mask_pairs16(/*pair_base=*/16, /*odd=*/false);
            const __mmask16 mo1 = mask_pairs16(/*pair_base=*/16, /*odd=*/true);

            alignas(64) float ns_even[32];
            alignas(64) float ns_odd[32];
            alignas(64) float sq_even[32];
            alignas(64) float sq_odd[32];
            for (int p = 0; p < 32; ++p) {
              ns_even[p] = ns[2 * p + 0];
              ns_odd[p] = ns[2 * p + 1];
              sq_even[p] = sq[2 * p + 0];
              sq_odd[p] = sq[2 * p + 1];
            }

            auto lane16_ps = [](const __m512i& acc, int half) -> __m512 {
              const __m256i v16 =
                  (half == 0) ? _mm512_castsi512_si256(acc) : _mm512_extracti64x4_epi64(acc, 1);
              const __m512i i32 = _mm512_cvtepi16_epi32(v16);
              return _mm512_cvtepi32_ps(i32);
            };

            for (size_t qi = 0; qi < qb; ++qi) {
              __m512 min_v = global_min_v[qi];
              const __m512 beta_ps = _mm512_set1_ps(beta_q[qi]);

              // pairs 0..15
              {
                const __m512 s_e = lane16_ps(acc_even[qi], 0);
                const __m512 s_o = lane16_ps(acc_odd[qi], 0);
                const __m512 ns_e = _mm512_load_ps(ns_even + 0);
                const __m512 ns_o = _mm512_load_ps(ns_odd + 0);
                const __m512 t_e = _mm512_mul_ps(s_e, ns_e);
                const __m512 t_o = _mm512_mul_ps(s_o, ns_o);

                __m512 dist_e, dist_o;
                if constexpr (Metric) {
                  const __m512 base_e =
                      _mm512_add_ps(_mm512_load_ps(sq_even + 0), _mm512_set1_ps(sqn_q[qi]));
                  const __m512 base_o =
                      _mm512_add_ps(_mm512_load_ps(sq_odd + 0), _mm512_set1_ps(sqn_q[qi]));
                  dist_e =
                      _mm512_fmadd_ps(_mm512_sub_ps(_mm512_setzero_ps(), beta_ps), t_e, base_e);
                  dist_o =
                      _mm512_fmadd_ps(_mm512_sub_ps(_mm512_setzero_ps(), beta_ps), t_o, base_o);
                } else {
                  dist_e = _mm512_mul_ps(_mm512_sub_ps(_mm512_setzero_ps(), beta_ps), t_e);
                  dist_o = _mm512_mul_ps(_mm512_sub_ps(_mm512_setzero_ps(), beta_ps), t_o);
                }
                dist_e = _mm512_mask_blend_ps(me0, inf_ps, dist_e);
                dist_o = _mm512_mask_blend_ps(mo0, inf_ps, dist_o);
                min_v = _mm512_min_ps(min_v, dist_e);
                min_v = _mm512_min_ps(min_v, dist_o);
              }

              // pairs 16..31
              {
                const __m512 s_e = lane16_ps(acc_even[qi], 1);
                const __m512 s_o = lane16_ps(acc_odd[qi], 1);
                const __m512 ns_e = _mm512_load_ps(ns_even + 16);
                const __m512 ns_o = _mm512_load_ps(ns_odd + 16);
                const __m512 t_e = _mm512_mul_ps(s_e, ns_e);
                const __m512 t_o = _mm512_mul_ps(s_o, ns_o);

                __m512 dist_e, dist_o;
                if constexpr (Metric) {
                  const __m512 base_e =
                      _mm512_add_ps(_mm512_load_ps(sq_even + 16), _mm512_set1_ps(sqn_q[qi]));
                  const __m512 base_o =
                      _mm512_add_ps(_mm512_load_ps(sq_odd + 16), _mm512_set1_ps(sqn_q[qi]));
                  dist_e =
                      _mm512_fmadd_ps(_mm512_sub_ps(_mm512_setzero_ps(), beta_ps), t_e, base_e);
                  dist_o =
                      _mm512_fmadd_ps(_mm512_sub_ps(_mm512_setzero_ps(), beta_ps), t_o, base_o);
                } else {
                  dist_e = _mm512_mul_ps(_mm512_sub_ps(_mm512_setzero_ps(), beta_ps), t_e);
                  dist_o = _mm512_mul_ps(_mm512_sub_ps(_mm512_setzero_ps(), beta_ps), t_o);
                }
                dist_e = _mm512_mask_blend_ps(me1, inf_ps, dist_e);
                dist_o = _mm512_mask_blend_ps(mo1, inf_ps, dist_o);
                min_v = _mm512_min_ps(min_v, dist_e);
                min_v = _mm512_min_ps(min_v, dist_o);
              }

              global_min_v[qi] = min_v;
            }
          };

          if (strip0 == strip1) {
            const int hi = (lane1 == 0) ? 64 : lane1;
            scan_strip_masked(strip0, 0, hi);
          } else {
            for (size_t s = strip0; s < strip1; ++s)
              scan_strip_masked(s, 0, 64);
            if (lane1 != 0) scan_strip_masked(strip1, 0, lane1);
          }

          for (size_t qi = 0; qi < qb; ++qi) {
            const float m = _mm512_reduce_min_ps(global_min_v[qi]);
            total += m;
          }
        }

        results[i] = {cloud_id, total / static_cast<float>(num_q)};
      });
      return q_query.vec_queries.size();
#else
      parlay::parallel_for(0, n, [&](size_t i) {
        const uint32_t cloud_id = indices[i];
        results[i] = {cloud_id, q_query.distance((*this)[cloud_id])};
      });
      return q_query.vec_queries.size();
#endif
    } else {
      parlay::parallel_for(0, n, [&](size_t i) {
        uint32_t cloud_id = indices[i];
        results[i] = {cloud_id, q_query.distance((*this)[cloud_id])};
      });
      return q_query.vec_queries.size();
    }
  }

  template<typename QuantizedQueryTy>
  size_t distances_all(const QuantizedQueryTy& q_query, std::pair<uint32_t, float>* results) const {

    const size_t num_q = q_query.vec_queries.size();

    if constexpr (EncRange::is_fastscan) {
      // FastScan strip kernel (VNNI/AVX-512/AVX2 via EncRange::RunningMinVType).
#if defined(__AVX512F__) || defined(__AVX2__)
      using RunningMinV = typename EncRange::RunningMinVType;
      parlay::parallel_for(0, n_clouds, [&](size_t cid) {
        const uint32_t cloud_id = static_cast<uint32_t>(cid);
        const size_t start = offsets[cloud_id];
        size_t cloud_size = 0;
        if (sizes_unpadded.size() == static_cast<size_t>(n_clouds)) {
          cloud_size = static_cast<size_t>(sizes_unpadded[cloud_id]);
        } else {
          const size_t end_padded = offsets[cloud_id + 1];
          cloud_size = (end_padded > start) ? (end_padded - start) : 0;
        }
        if (num_q == 0) {
          results[cid] = {cloud_id, 0.0f};
          return;
        }
        if (cloud_size == 0) {
          results[cid] = {cloud_id, std::numeric_limits<float>::max()};
          return;
        }
        const size_t true_end = start + cloud_size;
        const size_t strip_stride = static_cast<size_t>(vec_db.num_blocks) * 32;
        const size_t strip0 = start / 64;
        const int lane0 = static_cast<int>(start % 64);
        const size_t strip1 = true_end / 64;
        const int lane1 = static_cast<int>(true_end % 64);
        auto strip_ptr = [&](size_t s) -> const uint8_t* {
          return &vec_db.packed_codes[s * strip_stride];
        };
        float total = 0.0f;
        const bool fully_aligned_full = (lane0 == 0) && (lane1 == 0) && (strip1 > strip0);

        for (const auto& qv : q_query.vec_queries) {
          float min_d = std::numeric_limits<float>::max();
          if (strip0 == strip1) {
            const int hi = (lane1 == 0) ? 64 : lane1;
            min_d = vec_db.scan_64_chunk_min_masked(qv, strip_ptr(strip0), lane0, hi);
            total += min_d;
            continue;
          }

          if (fully_aligned_full) {
            const size_t first_full = strip0;
            const size_t last_full = strip1 - 1;
            RunningMinV min0 = RunningMinV::max();
            RunningMinV min1 = RunningMinV::max();
            RunningMinV min2 = RunningMinV::max();
            RunningMinV min3 = RunningMinV::max();
            size_t s = first_full;
            for (; s + 3 <= last_full; s += 4) {
              const uint8_t* p0 = strip_ptr(s);
              vec_db.scan_64_running_min(qv, p0, min0);
              vec_db.scan_64_running_min(qv, p0 + strip_stride, min1);
              vec_db.scan_64_running_min(qv, p0 + 2 * strip_stride, min2);
              vec_db.scan_64_running_min(qv, p0 + 3 * strip_stride, min3);
            }
            RunningMinV combined;
#ifdef __AVX512F__
            combined.v = _mm512_min_epu16(_mm512_min_epu16(min0.v, min1.v),
                                          _mm512_min_epu16(min2.v, min3.v));
#else
            combined.lo = _mm256_min_epu16(_mm256_min_epu16(min0.lo, min1.lo), _mm256_min_epu16(min2.lo, min3.lo));
            combined.hi = _mm256_min_epu16(_mm256_min_epu16(min0.hi, min1.hi), _mm256_min_epu16(min2.hi, min3.hi));
#endif
            for (; s <= last_full; ++s) {
              vec_db.scan_64_running_min(qv, strip_ptr(s), combined);
            }
            min_d = vec_db.reduce_running_min(qv, combined);
            total += min_d;
            continue;
          }

          {
            const float d0 = vec_db.scan_64_chunk_min_masked(qv, strip_ptr(strip0), lane0, 64);
            if (d0 < min_d) min_d = d0;
          }
          const size_t first_full = strip0 + 1;
          const size_t last_full = strip1 - 1;
          if (first_full <= last_full) {
            RunningMinV min0 = RunningMinV::max();
            RunningMinV min1 = RunningMinV::max();
            RunningMinV min2 = RunningMinV::max();
            RunningMinV min3 = RunningMinV::max();
            size_t s = first_full;
            for (; s + 3 <= last_full; s += 4) {
              const uint8_t* p0 = strip_ptr(s);
              vec_db.scan_64_running_min(qv, p0, min0);
              vec_db.scan_64_running_min(qv, p0 + strip_stride, min1);
              vec_db.scan_64_running_min(qv, p0 + 2 * strip_stride, min2);
              vec_db.scan_64_running_min(qv, p0 + 3 * strip_stride, min3);
            }
            RunningMinV combined;
#ifdef __AVX512F__
            combined.v = _mm512_min_epu16(_mm512_min_epu16(min0.v, min1.v),
                                          _mm512_min_epu16(min2.v, min3.v));
#else
            combined.lo = _mm256_min_epu16(_mm256_min_epu16(min0.lo, min1.lo), _mm256_min_epu16(min2.lo, min3.lo));
            combined.hi = _mm256_min_epu16(_mm256_min_epu16(min0.hi, min1.hi), _mm256_min_epu16(min2.hi, min3.hi));
#endif
            for (; s <= last_full; ++s) {
              vec_db.scan_64_running_min(qv, strip_ptr(s), combined);
            }
            const float d_full = vec_db.reduce_running_min(qv, combined);
            if (d_full < min_d) min_d = d_full;
          }
          if (lane1 != 0) {
            const float d1 = vec_db.scan_64_chunk_min_masked(qv, strip_ptr(strip1), 0, lane1);
            if (d1 < min_d) min_d = d1;
          }
          total += min_d;
        }
        results[cid] = {cloud_id, total / float(num_q)};
      });
      return num_q;
#else
      parlay::parallel_for(0, n_clouds, [&](size_t cid) {
        results[cid] = {static_cast<uint32_t>(cid), q_query.distance((*this)[cid])};
      });
      return num_q;
#endif
    } else if constexpr (has_tqpq_fast_t<EncRange>::value) {
      // TQ-PQ: strip-interleaved layout (like FastScan), but scoring uses per-query LUTs.
#if defined(__AVX512F__)
      const size_t num_q = q_query.vec_queries.size();
      const size_t nb = static_cast<size_t>(vec_db.num_blocks_for_scan());

      // Precompute per-query scalars once: beta (with 2.0 baked in for L2) and squared norms.
      std::vector<float> beta_all(num_q);
      std::vector<float> sqn_all(num_q);
      for (size_t qi = 0; qi < num_q; ++qi) {
        const auto& qv = q_query.vec_queries[qi];
        float base_beta =
            (qv.norm_scaling_factor * qv.lut_int8_scale) / turboquant_pq_4bit::kPQ_Int8Scale_D1_K16;
        beta_all[qi] = Metric ? (2.0f * base_beta) : base_beta;
        sqn_all[qi] = qv.unquantized_squared_norm;
      }

      parlay::parallel_for(0, n_clouds, [&](size_t cid) {
        const uint32_t cloud_id = static_cast<uint32_t>(cid);
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

        const size_t true_end = start + cloud_size;
        const size_t strip_stride = vec_db.stride;
        const size_t strip0 = start / 64;
        const size_t strip1 = true_end / 64;
        const int lane1 = static_cast<int>(true_end % 64);

        const uint8_t* base_codes = vec_db.packed_codes.data();
        const float* base_norms = vec_db.norm_scaling_factors.data();
        const float* base_sqn = vec_db.unquantized_squared_norms.data();

        auto strip_ptr = [&](size_t s) -> const uint8_t* { return &base_codes[s * strip_stride]; };
        auto norms_ptr = [&](size_t s) -> const float* { return base_norms + s * 64; };
        auto sqn_ptr = [&](size_t s) -> const float* { return base_sqn + s * 64; };

        // Process query vectors in batches to reuse code loads.
        constexpr size_t kQBatch = 8;
        float total = 0.0f;
        const float inv_num_q = 1.0f / static_cast<float>(num_q);

        for (size_t q0 = 0; q0 < num_q; q0 += kQBatch) {
          const size_t qb = std::min(kQBatch, num_q - q0);
          const auto* qv_arr = q_query.vec_queries.data() + q0;

          float beta_q[kQBatch];
          float sqn_q[kQBatch];
          for (size_t qi = 0; qi < qb; ++qi) {
            beta_q[qi] = beta_all[q0 + qi];
            sqn_q[qi] = sqn_all[q0 + qi];
          }

          float min_q[kQBatch];
          for (size_t qi = 0; qi < qb; ++qi)
            min_q[qi] = std::numeric_limits<float>::max();

          const __m256i low_mask = _mm256_set1_epi8(0x0F);

          auto scan_strip_masked = [&](size_t s, int lo, int hi) {
            const uint8_t* codes_ptr = strip_ptr(s);
            const float* ns = norms_ptr(s);
            const float* sq = sqn_ptr(s);

            __m512i acc_even[kQBatch];
            __m512i acc_odd[kQBatch];
            for (size_t qi = 0; qi < qb; ++qi) {
              acc_even[qi] = _mm512_setzero_si512();
              acc_odd[qi] = _mm512_setzero_si512();
            }

            for (size_t b = 0; b < nb; ++b) {
              const __m256i packed =
                  _mm256_loadu_si256(reinterpret_cast<const __m256i*>(codes_ptr));
              codes_ptr += 32;

              const __m256i codes_even = _mm256_and_si256(packed, low_mask);
              const __m256i codes_odd = _mm256_and_si256(_mm256_srli_epi16(packed, 4), low_mask);

              for (size_t qi = 0; qi < qb; ++qi) {
                const __m128i lut128 =
                    _mm_loadu_si128(reinterpret_cast<const __m128i*>(&qv_arr[qi].lut_int8[b * 16]));
                const __m256i lut256 = _mm256_broadcastsi128_si256(lut128);
                const __m256i scores_even_i8 = _mm256_shuffle_epi8(lut256, codes_even);
                const __m256i scores_odd_i8 = _mm256_shuffle_epi8(lut256, codes_odd);
                acc_even[qi] = _mm512_add_epi16(acc_even[qi], _mm512_cvtepi8_epi16(scores_even_i8));
                acc_odd[qi] = _mm512_add_epi16(acc_odd[qi], _mm512_cvtepi8_epi16(scores_odd_i8));
              }
            }

            // Keep everything in registers: convert acc -> float, apply norms, compute L2/IP,
            // reduce min.
            const __m512 inf_ps = _mm512_set1_ps(std::numeric_limits<float>::infinity());

            auto mask_pairs16 = [&](int pair_base, bool odd) -> __mmask16 {
              uint16_t m = 0;
              for (int j = 0; j < 16; ++j) {
                const int pair = pair_base + j;
                const int lane = (pair << 1) + (odd ? 1 : 0);
                if (lane >= lo && lane < hi) m |= static_cast<uint16_t>(1u << j);
              }
              return static_cast<__mmask16>(m);
            };
            const __mmask16 me0 = mask_pairs16(/*pair_base=*/0, /*odd=*/false);
            const __mmask16 mo0 = mask_pairs16(/*pair_base=*/0, /*odd=*/true);
            const __mmask16 me1 = mask_pairs16(/*pair_base=*/16, /*odd=*/false);
            const __mmask16 mo1 = mask_pairs16(/*pair_base=*/16, /*odd=*/true);

            alignas(64) float ns_even[32];
            alignas(64) float ns_odd[32];
            alignas(64) float sq_even[32];
            alignas(64) float sq_odd[32];
            for (int p = 0; p < 32; ++p) {
              ns_even[p] = ns[2 * p + 0];
              ns_odd[p] = ns[2 * p + 1];
              sq_even[p] = sq[2 * p + 0];
              sq_odd[p] = sq[2 * p + 1];
            }

            auto lane16_ps = [](const __m512i& acc, int half) -> __m512 {
              const __m256i v16 =
                  (half == 0) ? _mm512_castsi512_si256(acc) : _mm512_extracti64x4_epi64(acc, 1);
              const __m512i i32 = _mm512_cvtepi16_epi32(v16);
              return _mm512_cvtepi32_ps(i32);
            };

            for (size_t qi = 0; qi < qb; ++qi) {
              __m512 running_min = inf_ps;
              const __m512 beta_ps = _mm512_set1_ps(beta_q[qi]);
              const __m512 neg_beta_ps = _mm512_sub_ps(_mm512_setzero_ps(), beta_ps);
              const __m512 sqn_q_ps = _mm512_set1_ps(sqn_q[qi]);

              // pairs 0..15
              {
                const __m512 s_e = lane16_ps(acc_even[qi], 0);
                const __m512 s_o = lane16_ps(acc_odd[qi], 0);
                const __m512 ns_e = _mm512_load_ps(ns_even + 0);
                const __m512 ns_o = _mm512_load_ps(ns_odd + 0);
                const __m512 t_e = _mm512_mul_ps(s_e, ns_e);
                const __m512 t_o = _mm512_mul_ps(s_o, ns_o);

                __m512 dist_e, dist_o;
                if constexpr (Metric) {
                  const __m512 base_e = _mm512_add_ps(_mm512_load_ps(sq_even + 0), sqn_q_ps);
                  const __m512 base_o = _mm512_add_ps(_mm512_load_ps(sq_odd + 0), sqn_q_ps);
                  dist_e = _mm512_fmadd_ps(neg_beta_ps, t_e, base_e);
                  dist_o = _mm512_fmadd_ps(neg_beta_ps, t_o, base_o);
                } else {
                  dist_e = _mm512_mul_ps(neg_beta_ps, t_e);
                  dist_o = _mm512_mul_ps(neg_beta_ps, t_o);
                }
                dist_e = _mm512_mask_blend_ps(me0, inf_ps, dist_e);
                dist_o = _mm512_mask_blend_ps(mo0, inf_ps, dist_o);
                running_min = _mm512_min_ps(running_min, dist_e);
                running_min = _mm512_min_ps(running_min, dist_o);
              }

              // pairs 16..31
              {
                const __m512 s_e = lane16_ps(acc_even[qi], 1);
                const __m512 s_o = lane16_ps(acc_odd[qi], 1);
                const __m512 ns_e = _mm512_load_ps(ns_even + 16);
                const __m512 ns_o = _mm512_load_ps(ns_odd + 16);
                const __m512 t_e = _mm512_mul_ps(s_e, ns_e);
                const __m512 t_o = _mm512_mul_ps(s_o, ns_o);

                __m512 dist_e, dist_o;
                if constexpr (Metric) {
                  const __m512 base_e = _mm512_add_ps(_mm512_load_ps(sq_even + 16), sqn_q_ps);
                  const __m512 base_o = _mm512_add_ps(_mm512_load_ps(sq_odd + 16), sqn_q_ps);
                  dist_e = _mm512_fmadd_ps(neg_beta_ps, t_e, base_e);
                  dist_o = _mm512_fmadd_ps(neg_beta_ps, t_o, base_o);
                } else {
                  dist_e = _mm512_mul_ps(neg_beta_ps, t_e);
                  dist_o = _mm512_mul_ps(neg_beta_ps, t_o);
                }
                dist_e = _mm512_mask_blend_ps(me1, inf_ps, dist_e);
                dist_o = _mm512_mask_blend_ps(mo1, inf_ps, dist_o);
                running_min = _mm512_min_ps(running_min, dist_e);
                running_min = _mm512_min_ps(running_min, dist_o);
              }

              const float m = _mm512_reduce_min_ps(running_min);
              if (m < min_q[qi]) min_q[qi] = m;
            }
          };

          if (strip0 == strip1) {
            // Cloud fits entirely within a single strip; only mask the end.
            const int hi = (lane1 == 0) ? 64 : lane1;
            scan_strip_masked(strip0, 0, hi);
          } else {
            // Full strips from strip0 to strip1 - 1.
            for (size_t s = strip0; s < strip1; ++s)
              scan_strip_masked(s, 0, 64);
            // Final partial strip if needed.
            if (lane1 != 0) scan_strip_masked(strip1, 0, lane1);
          }

          for (size_t qi = 0; qi < qb; ++qi)
            total += min_q[qi];
        }

        results[cid] = {cloud_id, total * inv_num_q};
      });
      return num_q;
#else
      parlay::parallel_for(0, n_clouds, [&](size_t cid) {
        results[cid] = {static_cast<uint32_t>(cid), q_query.distance((*this)[cid])};
      });
      return num_q;
#endif
    } else {
      parlay::parallel_for(0, n_clouds, [&](size_t cid) {
        results[cid] = {static_cast<uint32_t>(cid), q_query.distance((*this)[cid])};
      });
      return num_q;
    }
  }

  // TODO: fix this
  inline size_t get_dist_cmps() const {
    return 0;
  }

  void save(std::ofstream& out) const {
    out.write(reinterpret_cast<const char*>(&n_clouds), sizeof(n_clouds));

    // offsets
    size_t off_size = offsets.size();
    out.write(reinterpret_cast<const char*>(&off_size), sizeof(off_size));
    if (off_size) {
      out.write(reinterpret_cast<const char*>(offsets.data()), off_size * sizeof(size_t));
    }

    // sizes_unpadded (needed for fastscan correctness; empty for non-fastscan is fine)
    size_t sz_size = sizes_unpadded.size();
    out.write(reinterpret_cast<const char*>(&sz_size), sizeof(sz_size));
    if (sz_size) {
      out.write(reinterpret_cast<const char*>(sizes_unpadded.data()), sz_size * sizeof(uint32_t));
    }

    vec_db.save(out);
  }

  void load(std::ifstream& in) {
    in.read(reinterpret_cast<char*>(&n_clouds), sizeof(n_clouds));

    // offsets
    size_t off_size = 0;
    in.read(reinterpret_cast<char*>(&off_size), sizeof(off_size));
    offsets.resize(off_size);
    if (off_size) {
      in.read(reinterpret_cast<char*>(offsets.data()), off_size * sizeof(size_t));
    }

    // sizes_unpadded
    size_t sz_size = 0;
    in.read(reinterpret_cast<char*>(&sz_size), sizeof(sz_size));
    sizes_unpadded.resize(sz_size);
    if (sz_size) {
      in.read(reinterpret_cast<char*>(sizes_unpadded.data()), sz_size * sizeof(uint32_t));
    }

    vec_db.load(in);
  }
};

// ---------------------------------------------------------
// Model wrapper: owns the vector-quantizer model and provides
//  - train
//  - encode(pcs) -> Quantized_Point_Cloud_Set
//  - quantize_query(query_cloud) -> Quantized_Query_Point_Cloud
// ---------------------------------------------------------
template<typename VecModel, bool Metric>
class MultiVecQuantizer {
 public:
  VecModel vec_model;

  MultiVecQuantizer() = default;

  // Train on a PointCloudSet by flattening
  template<typename PCSet, typename... Args>
  void train(const PCSet& pcs, Args&&... args) {
    vec_model.train(FlattenedPCRange<PCSet>(pcs), std::forward<Args>(args)...);
  }

  // Encode a PointCloudSet into a cloud-set container
  template<typename PCSet>
  auto encode(const PCSet& pcs) {
    FlattenedPCRange<PCSet> flat(pcs);

    // offsets are in FLOAT indices into flattened float buffer
    const auto pcs_offsets_float = pcs.get_offsets();

    if constexpr (VecModel::is_fastscan) {
      auto enc = vec_model.encode(flat, pcs_offsets_float);
      using EncRange = decltype(enc);
      // FastScan range already stores vector-index offsets (possibly padded per cloud)
      parlay::sequence<size_t> offs = enc.cloud_vec_offsets;
      parlay::sequence<uint32_t> sizes(offs.size() ? offs.size() - 1 : 0);
      const size_t dim = static_cast<size_t>(pcs.get_dims());
      parlay::parallel_for(0, sizes.size(), [&](size_t c) {
        const size_t start_f = static_cast<size_t>(pcs_offsets_float[c]);
        const size_t end_f = static_cast<size_t>(pcs_offsets_float[c + 1]);
        sizes[c] = static_cast<uint32_t>((end_f - start_f) / dim);  // UNPADDED vectors
      });
      return Quantized_Point_Cloud_Set<EncRange, Metric>(std::move(enc), std::move(offs),
                                                         std::move(sizes));
    } else {
      auto enc = vec_model.encode(flat);
      using EncRange = decltype(enc);
      parlay::sequence<size_t> offs(pcs_offsets_float.begin(), pcs_offsets_float.end());
      const size_t dim = static_cast<size_t>(pcs.get_dims());
      if (dim > 0) {
        parlay::parallel_for(0, offs.size(), [&](size_t i) { offs[i] /= dim; });
      }
      return Quantized_Point_Cloud_Set<EncRange, Metric>(std::move(enc), std::move(offs));
    }
  }

  // Quantize an input query cloud (per-vector LUTs etc.)
  template<typename ChPoint>
  auto quantize_query(const ChPoint& query_cloud) const {
    using QVecType = decltype(vec_model.quantize_query(query_cloud[0].data()));
    Quantized_Query_Point_Cloud<QVecType, Metric> qqc;

    const size_t n_q = query_cloud.size();
    if (n_q == 0) return qqc;

    if (n_q >= 8) {
      vec_model.quantize_query_batch(query_cloud, qqc.vec_queries);
      return qqc;
    }
    qqc.vec_queries = parlay::tabulate(
        n_q, [&](size_t i) { return vec_model.quantize_query(query_cloud[i].data()); });
    return qqc;
  }

  void save(std::ofstream& out) const { vec_model.save(out); }
  void load(std::ifstream& in) { vec_model.load(in); }
};

}  // namespace mvsic
