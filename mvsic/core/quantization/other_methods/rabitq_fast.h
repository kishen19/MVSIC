#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

#include <immintrin.h>

#include "parlay/parallel.h"
#include "parlay/primitives.h"

#include "rabitqlib/index/estimator.hpp"
#include "rabitqlib/index/query.hpp"
#include "rabitqlib/quantization/data_layout.hpp"
#include "rabitqlib/quantization/rabitq.hpp"
#include "rabitqlib/utils/rotator.hpp"
#include "rabitqlib/utils/space.hpp"

namespace mvsic {
namespace rabitq_fast {

template<bool Metric>
class Quantized_Query;

// ---------------------------------------------------------
// Quantized Point: Aux Data Type
// ---------------------------------------------------------
template<bool Metric>
class Quantized_Point {
 public:
  const char* bin_data = nullptr;
  const char* ex_data = nullptr;

  Quantized_Point() = default;
  Quantized_Point(const char* bin, const char* ex) : bin_data(bin), ex_data(ex) {}

  inline float distance(const Quantized_Query<Metric>& qq) const;
  bool same_as(const Quantized_Point<Metric>&) const { return false; }
  bool same_as(const Quantized_Query<Metric>&) const { return false; }

  void prefetch() const {
    __builtin_prefetch(bin_data, 0, 3);
    if (ex_data) __builtin_prefetch(ex_data, 0, 3);
  }
  bool is_metric() const { return Metric; }
};

// ---------------------------------------------------------
// FastScan Multi-Tier SIMD Kernels
//
// Quality note (RaBitQ-fast vs RaBitQ): This path uses a linear formula
//   distance = g_add - (weighted_popcount) * g_error
// with fixed weights (8,4,2,1). The original RaBitQ uses rabitqlib's
// split_single_estdist(), which may use a different (e.g. LUT-based or
// non-linear) mapping. If RaBitQ-fast quality is much worse than RaBitQ,
// consider aligning this formula with the library's estimator (e.g. use
// the same per-byte or per-dimension scaling as in split_single_estdist).
// ---------------------------------------------------------

#if defined(__AVX512BITALG__) && defined(__AVX512VNNI__) && defined(__AVX512VL__)
// TIER 1: AVX-512 Native Popcount + VNNI Weighted Accumulation (32 distances per block)
inline void fastscan_compute_block_simd(const char* db_block, const char* q0, const char* q1,
                                        const char* q2, const char* q3, size_t num_bytes,
                                        float* out_distances, float g_add, float g_error) {
  __m256i total_acc[4] = {_mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256(),
                          _mm256_setzero_si256()};

  __m256i weight = _mm256_set1_epi32(0x08040201);  // 8, 4, 2, 1 per byte for VNNI

  size_t i = 0;
  while (i < num_bytes) {
    size_t chunk_end = std::min(num_bytes, i + 30);
    __m256i pop8_0 = _mm256_setzero_si256(), pop8_1 = _mm256_setzero_si256();
    __m256i pop8_2 = _mm256_setzero_si256(), pop8_3 = _mm256_setzero_si256();

    for (; i < chunk_end; ++i) {
      __m256i db = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(db_block + i * 32));
      pop8_0 = _mm256_add_epi8(pop8_0,
                               _mm256_popcnt_epi8(_mm256_and_si256(db, _mm256_set1_epi8(q0[i]))));
      pop8_1 = _mm256_add_epi8(pop8_1,
                               _mm256_popcnt_epi8(_mm256_and_si256(db, _mm256_set1_epi8(q1[i]))));
      pop8_2 = _mm256_add_epi8(pop8_2,
                               _mm256_popcnt_epi8(_mm256_and_si256(db, _mm256_set1_epi8(q2[i]))));
      pop8_3 = _mm256_add_epi8(pop8_3,
                               _mm256_popcnt_epi8(_mm256_and_si256(db, _mm256_set1_epi8(q3[i]))));
    }

    __m256i lo01 = _mm256_unpacklo_epi8(pop8_0, pop8_1);
    __m256i hi01 = _mm256_unpackhi_epi8(pop8_0, pop8_1);
    __m256i lo23 = _mm256_unpacklo_epi8(pop8_2, pop8_3);
    __m256i hi23 = _mm256_unpackhi_epi8(pop8_2, pop8_3);
    __m256i v0 = _mm256_unpacklo_epi16(lo01, lo23);
    __m256i v1 = _mm256_unpackhi_epi16(lo01, lo23);
    __m256i v2 = _mm256_unpacklo_epi16(hi01, hi23);
    __m256i v3 = _mm256_unpackhi_epi16(hi01, hi23);

    total_acc[0] = _mm256_dpbusd_epi32(total_acc[0], v0, weight);
    total_acc[1] = _mm256_dpbusd_epi32(total_acc[1], v1, weight);
    total_acc[2] = _mm256_dpbusd_epi32(total_acc[2], v2, weight);
    total_acc[3] = _mm256_dpbusd_epi32(total_acc[3], v3, weight);
  }

  alignas(32) uint32_t counts[32];
  _mm256_storeu_si256((__m256i*)&counts[0], total_acc[0]);
  _mm256_storeu_si256((__m256i*)&counts[8], total_acc[1]);
  _mm256_storeu_si256((__m256i*)&counts[16], total_acc[2]);
  _mm256_storeu_si256((__m256i*)&counts[24], total_acc[3]);

  for (int v = 0; v < 32; ++v)
    out_distances[v] = g_add - counts[v] * g_error;
}

#elif defined(__AVX2__)
// TIER 2: AVX2 Emulated Popcount + Vertical Expansion (32 distances per block)
inline void fastscan_compute_block_simd(const char* db_block, const char* q0, const char* q1,
                                        const char* q2, const char* q3, size_t num_bytes,
                                        float* out_distances, float g_add, float g_error) {
  __m256i acc_q0[4] = {_mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256(),
                       _mm256_setzero_si256()};
  __m256i acc_q1[4] = {_mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256(),
                       _mm256_setzero_si256()};
  __m256i acc_q2[4] = {_mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256(),
                       _mm256_setzero_si256()};
  __m256i acc_q3[4] = {_mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256(),
                       _mm256_setzero_si256()};

  __m256i lookup = _mm256_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1,
                                    2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
  __m256i low_mask = _mm256_set1_epi8(0x0F);
  __m256i zero = _mm256_setzero_si256();

  size_t i = 0;
  while (i < num_bytes) {
    size_t chunk_end = std::min(num_bytes, i + 30);
    __m256i pop8_0 = zero, pop8_1 = zero, pop8_2 = zero, pop8_3 = zero;

    for (; i < chunk_end; ++i) {
      __m256i db = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(db_block + i * 32));
      auto popcnt_accum = [&](const char* q, __m256i& pop8) {
        __m256i m = _mm256_and_si256(db, _mm256_set1_epi8(q[i]));
        __m256i lo = _mm256_and_si256(m, low_mask);
        __m256i hi = _mm256_and_si256(_mm256_srli_epi16(m, 4), low_mask);
        pop8 = _mm256_add_epi8(pop8,
                              _mm256_add_epi8(_mm256_shuffle_epi8(lookup, lo),
                                              _mm256_shuffle_epi8(lookup, hi)));
      };
      popcnt_accum(q0, pop8_0);
      popcnt_accum(q1, pop8_1);
      popcnt_accum(q2, pop8_2);
      popcnt_accum(q3, pop8_3);
    }

    auto unpack_and_add = [&](__m256i pop8, __m256i* acc32) {
      __m256i p_lo16 = _mm256_unpacklo_epi8(pop8, zero);
      __m256i p_hi16 = _mm256_unpackhi_epi8(pop8, zero);
      __m256i lo_lo = _mm256_unpacklo_epi16(p_lo16, zero);
      __m256i lo_hi = _mm256_unpackhi_epi16(p_lo16, zero);
      __m256i hi_lo = _mm256_unpacklo_epi16(p_hi16, zero);
      __m256i hi_hi = _mm256_unpackhi_epi16(p_hi16, zero);
      acc32[0] = _mm256_add_epi32(acc32[0], lo_lo);
      acc32[1] = _mm256_add_epi32(acc32[1], lo_hi);
      acc32[2] = _mm256_add_epi32(acc32[2], hi_lo);
      acc32[3] = _mm256_add_epi32(acc32[3], hi_hi);
    };
    unpack_and_add(pop8_0, acc_q0);
    unpack_and_add(pop8_1, acc_q1);
    unpack_and_add(pop8_2, acc_q2);
    unpack_and_add(pop8_3, acc_q3);
  }

  alignas(32) uint32_t c0[32], c1[32], c2[32], c3[32];
  for (int r = 0; r < 4; ++r) {
    _mm256_storeu_si256((__m256i*)&c0[r * 8], acc_q0[r]);
    _mm256_storeu_si256((__m256i*)&c1[r * 8], acc_q1[r]);
    _mm256_storeu_si256((__m256i*)&c2[r * 8], acc_q2[r]);
    _mm256_storeu_si256((__m256i*)&c3[r * 8], acc_q3[r]);
  }
  for (int v = 0; v < 32; ++v)
    out_distances[v] = g_add - (c0[v] + 2 * c1[v] + 4 * c2[v] + 8 * c3[v]) * g_error;
}
#endif

// ---------------------------------------------------------
// Encoded Point Range
// ---------------------------------------------------------
template<typename PointRange, bool Metric>
class Quantized_Point_Range {
 public:
  static constexpr bool is_fastscan = false;
  static constexpr bool is_rabitq_fast = true;

  size_t n = 0;
  size_t dim = 0;
  size_t padded_dim = 0;
  size_t total_bits = 0;
  size_t ex_bits = 0;
  size_t bin_stride = 0;
  size_t ex_stride = 0;

  parlay::sequence<char> bin_data;
  parlay::sequence<char> ex_data;

  static constexpr size_t BLOCK_SIZE = 32;
  size_t num_blocks = 0;
  parlay::sequence<char> interleaved_bin_data;

  Quantized_Point_Range() = default;

  Quantized_Point<Metric> operator[](size_t i) const {
    const char* bin = &bin_data[i * bin_stride];
    const char* ex = (ex_bits > 0) ? &ex_data[i * ex_stride] : nullptr;
    return Quantized_Point<Metric>(bin, ex);
  }

  inline uint32_t size() const noexcept { return static_cast<uint32_t>(n); }
  inline uint32_t get_dims() const noexcept { return static_cast<uint32_t>(dim); }

  void build_interleaved_layout() {
    num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    interleaved_bin_data.resize(num_blocks * BLOCK_SIZE * bin_stride, 0);

    parlay::parallel_for(0, num_blocks, [&](size_t b) {
      for (size_t v = 0; v < BLOCK_SIZE; ++v) {
        size_t vec_idx = b * BLOCK_SIZE + v;
        if (vec_idx >= n) continue;
        for (size_t d = 0; d < bin_stride; ++d) {
          size_t target_idx = b * (BLOCK_SIZE * bin_stride) + (d * BLOCK_SIZE) + v;
          size_t src_idx = vec_idx * bin_stride + d;
          interleaved_bin_data[target_idx] = bin_data[src_idx];
        }
      }
    });
  }

  void save(std::ostream& out) const {
    out.write(reinterpret_cast<const char*>(&n), sizeof(n));
    out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    out.write(reinterpret_cast<const char*>(&padded_dim), sizeof(padded_dim));
    out.write(reinterpret_cast<const char*>(&total_bits), sizeof(total_bits));
    out.write(reinterpret_cast<const char*>(&ex_bits), sizeof(ex_bits));
    out.write(reinterpret_cast<const char*>(&bin_stride), sizeof(bin_stride));
    out.write(reinterpret_cast<const char*>(&ex_stride), sizeof(ex_stride));

    size_t bin_sz = bin_data.size();
    out.write(reinterpret_cast<const char*>(&bin_sz), sizeof(bin_sz));
    if (bin_sz) out.write(bin_data.data(), bin_sz);

    size_t ex_sz = ex_data.size();
    out.write(reinterpret_cast<const char*>(&ex_sz), sizeof(ex_sz));
    if (ex_sz) out.write(ex_data.data(), ex_sz);
  }

  void load(std::istream& in) {
    in.read(reinterpret_cast<char*>(&n), sizeof(n));
    in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    in.read(reinterpret_cast<char*>(&padded_dim), sizeof(padded_dim));
    in.read(reinterpret_cast<char*>(&total_bits), sizeof(total_bits));
    in.read(reinterpret_cast<char*>(&ex_bits), sizeof(ex_bits));
    in.read(reinterpret_cast<char*>(&bin_stride), sizeof(bin_stride));
    in.read(reinterpret_cast<char*>(&ex_stride), sizeof(ex_stride));

    size_t bin_sz = 0;
    in.read(reinterpret_cast<char*>(&bin_sz), sizeof(bin_sz));
    bin_data.resize(bin_sz);
    if (bin_sz) in.read(bin_data.data(), bin_sz);

    size_t ex_sz = 0;
    in.read(reinterpret_cast<char*>(&ex_sz), sizeof(ex_sz));
    ex_data.resize(ex_sz);
    if (ex_sz) in.read(ex_data.data(), ex_sz);

    build_interleaved_layout();
  }
};

// ---------------------------------------------------------
// Quantized Query
// ---------------------------------------------------------
template<bool Metric>
class Quantized_Query {
 public:
  using distanceType = float;

  std::vector<float> q_rot_storage;
  rabitqlib::SplitSingleQuery<float> q_obj;

  float (*ip_func)(const float*, const uint8_t*, size_t) = nullptr;

  float g_add = 0.0f;
  float g_error = 0.0f;

  size_t padded_dim = 0;
  size_t ex_bits = 0;

  std::vector<char> q_plane_0, q_plane_1, q_plane_2, q_plane_3;

  Quantized_Query(std::vector<float>&& q_rot, const float* c_rot, size_t padded_dim_,
                  size_t ex_bits_, const rabitqlib::quant::RabitqConfig& cfg) :
      q_rot_storage(std::move(q_rot)),
      q_obj(q_rot_storage.data(), padded_dim_, ex_bits_, cfg,
            Metric ? rabitqlib::METRIC_L2 : rabitqlib::METRIC_IP),
      padded_dim(padded_dim_),
      ex_bits(ex_bits_) {
    if (ex_bits > 0) {
      ip_func = rabitqlib::select_excode_ipfunc(ex_bits);
    }

    if constexpr (Metric) {
      float dist_sq = rabitqlib::euclidean_sqr(q_rot_storage.data(), c_rot, padded_dim);
      float norm = std::sqrt(dist_sq);
      q_obj.set_g_add(norm);
      g_add = norm * norm;
      g_error = norm;
    } else {
      float ip = rabitqlib::dot_product(q_rot_storage.data(), c_rot, padded_dim);
      q_obj.set_g_add(ip);
      g_add = ip;
      g_error = 0.0f;
    }

    decompose_into_planes();
  }

  void decompose_into_planes() {
    size_t byte_dim = padded_dim / 8;
    q_plane_0.resize(byte_dim, 0);
    q_plane_1.resize(byte_dim, 0);
    q_plane_2.resize(byte_dim, 0);
    q_plane_3.resize(byte_dim, 0);

    for (size_t i = 0; i < padded_dim; ++i) {
      uint8_t q_val =
          static_cast<uint8_t>(std::min(15.0f, std::max(0.0f, (q_rot_storage[i] * 15.0f))));
      size_t byte_idx = i / 8;
      size_t bit_idx = i % 8;
      if (q_val & 1) q_plane_0[byte_idx] |= (1 << bit_idx);
      if (q_val & 2) q_plane_1[byte_idx] |= (1 << bit_idx);
      if (q_val & 4) q_plane_2[byte_idx] |= (1 << bit_idx);
      if (q_val & 8) q_plane_3[byte_idx] |= (1 << bit_idx);
    }
  }

  inline float distance(const Quantized_Point<Metric>& p) const {
    return distance_raw(p.bin_data, p.ex_data);
  }

  inline float distance_raw(const char* bin, const char* ex) const {
    float est_dist = 0.0f, low_dist = 0.0f, ip_x0 = 0.0f;

    if (ex_bits > 0) {
      rabitqlib::split_single_fulldist(bin, ex, ip_func, q_obj, padded_dim, ex_bits, est_dist,
                                       low_dist, ip_x0, g_add, g_error);
    } else {
      rabitqlib::split_single_estdist(bin, q_obj, padded_dim, ip_x0, est_dist, low_dist, g_add,
                                      g_error);
    }
    return est_dist;
  }

  template<typename EncRange>
  void distances_all(const EncRange& db, float* out) const {
    const size_t n = static_cast<size_t>(db.size());
    if (n == 0) return;

#if defined(__AVX2__) || (defined(__AVX512BITALG__) && defined(__AVX512VNNI__) && defined(__AVX512VL__))
    if (db.interleaved_bin_data.size() > 0) {
      parlay::parallel_for(0, db.num_blocks, [&](size_t b) {
        const char* block_ptr =
            db.interleaved_bin_data.data() + (b * EncRange::BLOCK_SIZE * db.bin_stride);
        float block_out[32];
        fastscan_compute_block_simd(block_ptr, q_plane_0.data(), q_plane_1.data(), q_plane_2.data(),
                                    q_plane_3.data(), db.bin_stride, block_out, g_add, g_error);
        const size_t base = b * EncRange::BLOCK_SIZE;
        const size_t items = std::min(EncRange::BLOCK_SIZE, n - base);
        for (size_t v = 0; v < items; ++v)
          out[base + v] = block_out[v];
      });
      return;
    }
#endif

    const char* bin_base = db.bin_data.data();
    const char* ex_base =
        (db.ex_bits > 0 && db.ex_stride > 0 && db.ex_data.size() > 0) ? db.ex_data.data() : nullptr;
    parlay::parallel_for(0, n, [&](size_t i) {
      out[i] = distance_raw(bin_base + i * db.bin_stride,
                            ex_base ? (ex_base + i * db.ex_stride) : nullptr);
    });
  }

  // Slice [start, start+n) of db: write distances to out[0..n-1]. Used by wrapper for per-cloud fast path.
  template<typename EncRange>
  void distances_slice(const EncRange& db, size_t start, size_t n, float* out) const {
    if (n == 0) return;
    constexpr size_t kBlockSize = EncRange::BLOCK_SIZE;
    const size_t bin_stride = db.bin_stride;
    const size_t ex_stride = db.ex_stride;
    const char* bin_base = db.bin_data.data();
    const char* ex_base =
        (db.ex_bits > 0 && db.ex_stride > 0 && db.ex_data.size() > 0) ? db.ex_data.data() : nullptr;

#if defined(__AVX2__) || (defined(__AVX512BITALG__) && defined(__AVX512VNNI__) && defined(__AVX512VL__))
    if (db.interleaved_bin_data.size() > 0) {
      const size_t first_block = start / kBlockSize;
      const size_t last_block = (start + n - 1) / kBlockSize;
      for (size_t b = first_block; b <= last_block; ++b) {
        float block_out[32];
        const char* block_ptr =
            db.interleaved_bin_data.data() + (b * kBlockSize * bin_stride);
        fastscan_compute_block_simd(block_ptr, q_plane_0.data(), q_plane_1.data(), q_plane_2.data(),
                                    q_plane_3.data(), bin_stride, block_out, g_add, g_error);
        const size_t base = b * kBlockSize;
        for (size_t v = 0; v < kBlockSize; ++v) {
          const size_t global = base + v;
          if (global >= start && global < start + n)
            out[global - start] = block_out[v];
        }
      }
      return;
    }
#endif

    for (size_t i = 0; i < n; ++i) {
      out[i] = distance_raw(bin_base + (start + i) * bin_stride,
                           ex_base ? (ex_base + (start + i) * ex_stride) : nullptr);
    }
  }
};

template<bool Metric>
inline float Quantized_Point<Metric>::distance(const Quantized_Query<Metric>& qq) const {
  return qq.distance_raw(bin_data, ex_data);
}

// ---------------------------------------------------------
// RabitQ Model
// ---------------------------------------------------------
template<bool Metric>
class Model {
 public:
  static constexpr bool is_fastscan = false;
  size_t dim = 0, padded_dim = 0, total_bits = 0, ex_bits = 0;
  size_t bin_stride = 0, ex_stride = 0;

  rabitqlib::RotatorType rotator_type = rabitqlib::RotatorType::FhtKacRotator;
  std::unique_ptr<rabitqlib::Rotator<float>> rotator;
  std::vector<float> centroid_rot;
  rabitqlib::quant::RabitqConfig config;

  Model() = default;

  template<typename PointRange>
  Model(const PointRange& train_data, size_t bits = 8) {
    train(train_data, bits);
  }

  template<typename PointRange>
  void train(const PointRange& data, size_t bits = 8) {
    total_bits = bits;
    ex_bits = (total_bits > 0) ? (total_bits - 1) : 0;
    dim = data.get_dims();

    rotator.reset(rabitqlib::choose_rotator<float>(dim, rotator_type));
    padded_dim = rotator->size();
    bin_stride = rabitqlib::BinDataMap<float>::data_bytes(padded_dim);
    ex_stride = (ex_bits > 0) ? rabitqlib::ExDataMap<float>::data_bytes(padded_dim, ex_bits) : 0;

    if (total_bits > 1)
      config = rabitqlib::quant::faster_config(padded_dim, total_bits);
    else
      config = rabitqlib::quant::RabitqConfig{};

    const size_t n = data.size();
    const size_t sample_size = std::min(n, size_t(50000));
    std::vector<float> mean(dim, 0.0f);
    for (size_t i = 0; i < sample_size; ++i) {
      const float* p = reinterpret_cast<const float*>(data.location(i));
      for (size_t j = 0; j < dim; ++j)
        mean[j] += p[j];
    }
    const float inv = (sample_size > 0) ? (1.0f / float(sample_size)) : 0.0f;
    for (float& x : mean)
      x *= inv;

    centroid_rot.resize(padded_dim);
    rotator->rotate(mean.data(), centroid_rot.data());
  }

  template<typename PointRange>
  Quantized_Point_Range<PointRange, Metric> encode(const PointRange& data) const {
    Quantized_Point_Range<PointRange, Metric> enc;
    enc.n = data.size();
    enc.dim = data.get_dims();
    enc.padded_dim = padded_dim;
    enc.total_bits = total_bits;
    enc.ex_bits = ex_bits;
    enc.bin_stride = bin_stride;
    enc.ex_stride = ex_stride;

    enc.bin_data.resize(enc.n * bin_stride);
    if (ex_stride > 0) enc.ex_data.resize(enc.n * ex_stride);

    const rabitqlib::MetricType m_type = Metric ? rabitqlib::METRIC_L2 : rabitqlib::METRIC_IP;

    struct EncodingWorkspace {
      std::vector<float> p_rot;
      void ensure_size(size_t sz) {
        if (p_rot.size() != sz) p_rot.resize(sz);
      }
    };

    parlay::parallel_for(0, enc.n, [&](size_t i) {
      static thread_local EncodingWorkspace ws;
      ws.ensure_size(padded_dim);
      const float* p = reinterpret_cast<const float*>(data.location(i));
      rotator->rotate(p, ws.p_rot.data());

      char* bin_out = enc.bin_data.data() + i * bin_stride;
      char* ex_out = (ex_stride > 0) ? (enc.ex_data.data() + i * ex_stride) : nullptr;
      rabitqlib::quant::quantize_split_single(ws.p_rot.data(), centroid_rot.data(), padded_dim,
                                              ex_bits, bin_out, ex_out, m_type, config);
    });

    enc.build_interleaved_layout();
    return enc;
  }

  template<typename PointTy>
  Quantized_Query<Metric> quantize_query(const PointTy& query) const {
    std::vector<float> tmp(dim);
    for (size_t i = 0; i < dim; ++i)
      tmp[i] = query[i];
    std::vector<float> q_rot(padded_dim);
    rotator->rotate(tmp.data(), q_rot.data());
    return Quantized_Query<Metric>(std::move(q_rot), centroid_rot.data(), padded_dim, ex_bits,
                                   config);
  }

  Quantized_Query<Metric> quantize_query(const float* qptr) const {
    std::vector<float> tmp(dim);
    std::memcpy(tmp.data(), qptr, sizeof(float) * dim);
    std::vector<float> q_rot(padded_dim);
    rotator->rotate(tmp.data(), q_rot.data());
    return Quantized_Query<Metric>(std::move(q_rot), centroid_rot.data(), padded_dim, ex_bits,
                                   config);
  }

  template<typename PointCloudTy>
  void quantize_query_batch(const PointCloudTy& query_cloud,
                            parlay::sequence<Quantized_Query<Metric>>& out_queries) const {
    const uint32_t num_q = query_cloud.size();
    const uint32_t dims = query_cloud.get_dims();
    const float* base = query_cloud.data();
    out_queries.clear();
    out_queries.reserve(num_q);
    std::vector<float> tmp(dim);
    for (uint32_t i = 0; i < num_q; ++i) {
      std::memcpy(tmp.data(), base + static_cast<size_t>(i) * dims, sizeof(float) * dim);
      std::vector<float> q_rot(padded_dim);
      rotator->rotate(tmp.data(), q_rot.data());
      out_queries.emplace_back(std::move(q_rot), centroid_rot.data(), padded_dim, ex_bits, config);
    }
  }

  void save(std::ofstream& out) const {
    out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    out.write(reinterpret_cast<const char*>(&padded_dim), sizeof(padded_dim));
    out.write(reinterpret_cast<const char*>(&total_bits), sizeof(total_bits));
    out.write(reinterpret_cast<const char*>(&ex_bits), sizeof(ex_bits));
    out.write(reinterpret_cast<const char*>(&bin_stride), sizeof(bin_stride));
    out.write(reinterpret_cast<const char*>(&ex_stride), sizeof(ex_stride));
    int rot_type_int = static_cast<int>(rotator_type);
    out.write(reinterpret_cast<const char*>(&rot_type_int), sizeof(rot_type_int));
    rotator->save(out);
    size_t c_size = centroid_rot.size();
    out.write(reinterpret_cast<const char*>(&c_size), sizeof(c_size));
    out.write(reinterpret_cast<const char*>(centroid_rot.data()), c_size * sizeof(float));
  }

  void load(std::ifstream& in) {
    in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    in.read(reinterpret_cast<char*>(&padded_dim), sizeof(padded_dim));
    in.read(reinterpret_cast<char*>(&total_bits), sizeof(total_bits));
    in.read(reinterpret_cast<char*>(&ex_bits), sizeof(ex_bits));
    in.read(reinterpret_cast<char*>(&bin_stride), sizeof(bin_stride));
    in.read(reinterpret_cast<char*>(&ex_stride), sizeof(ex_stride));
    int rot_type_int = 0;
    in.read(reinterpret_cast<char*>(&rot_type_int), sizeof(rot_type_int));
    rotator_type = static_cast<rabitqlib::RotatorType>(rot_type_int);
    rotator.reset(rabitqlib::choose_rotator<float>(dim, rotator_type));
    rotator->load(in);
    config = (total_bits > 1) ? rabitqlib::quant::faster_config(padded_dim, total_bits)
                              : rabitqlib::quant::RabitqConfig{};
    size_t c_size = 0;
    in.read(reinterpret_cast<char*>(&c_size), sizeof(c_size));
    centroid_rot.resize(c_size);
    in.read(reinterpret_cast<char*>(centroid_rot.data()), c_size * sizeof(float));
  }
};

}  // namespace rabitq_fast
}  // namespace mvsic