#pragma once

#include <immintrin.h>
#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include "parlay/parallel.h"
#include "parlay/sequence.h"

#include "mvsic/core/utils/kmeans_util.h"

namespace mvsic {
namespace fastscan {

template<bool Metric>
class Quantized_Point;

// ---------------------------------------------------------
// FastScan Quantized Query (8-bit Fixed Point)
// ---------------------------------------------------------
template<bool Metric>
class Quantized_Query {
 public:
  using distanceType = float;

  alignas(64) std::vector<uint8_t> int_lut;  // [num_blocks * K]
  float min_dist = 0.0f;
  float scale = 1.0f;
  uint32_t num_blocks = 0;
  static constexpr uint32_t K = 16;

  Quantized_Query() = default;
  explicit Quantized_Query(uint32_t m) : num_blocks(m) {
    int_lut.resize(static_cast<size_t>(m) * K);
  }

  inline float decode(uint16_t int_dist) const {
    return (min_dist * static_cast<float>(num_blocks)) + (static_cast<float>(int_dist) * scale);
  }

  inline float distance(const Quantized_Point<Metric>& p) const {
    uint16_t acc = 0;
    for (uint32_t b = 0; b < num_blocks; ++b) {
      uint8_t packed = p.code_ptr[b * 32];
      uint8_t code = (p.lane_idx % 2 == 0) ? (packed & 0x0F) : (packed >> 4);
      acc += static_cast<uint16_t>(int_lut[static_cast<size_t>(b) * K + code]);
    }
    return decode(acc);
  }

  template<typename EncRange>
  void distances_all(const EncRange& db, float* out) const {
    // We only output distances for the "real" vectors (unpadded).
    const size_t N = (db.n_points_raw_unpadded != 0) ? db.n_points_raw_unpadded : db.n_points_raw;
    if (N == 0) return;

    const size_t strip_stride = static_cast<size_t>(db.num_blocks) * 32;  // bytes per strip
    const size_t n_full_strips = N / 64;                                  // full strips only
    const size_t full = n_full_strips * 64;

    // Parallel over full strips
    // Grain: tune if you want; 8 or 16 strips per task is usually a decent start.
    parlay::parallel_for(
        0, n_full_strips,
        [&](size_t s) {
          const uint8_t* codes_ptr = db.packed_codes.data() + s * strip_stride;
          db.scan_64_chunk(*this, codes_ptr, out + s * 64);
        },
        /*granularity=*/16);

    // Tail (at most 63 vectors)
    if (full < N) {
      alignas(64) float tmp[64];
      const size_t tail_strip = n_full_strips;  // the next strip
      const uint8_t* codes_ptr = db.packed_codes.data() + tail_strip * strip_stride;
      db.scan_64_chunk(*this, codes_ptr, tmp);

      const size_t tail = N - full;
      std::memcpy(out + full, tmp, tail * sizeof(float));
    }
  }
};

// ---------------------------------------------------------
// FastScan Point Handle
// ---------------------------------------------------------
template<bool Metric>
class Quantized_Point {
 public:
  const uint8_t* code_ptr = nullptr;  // points at lane_pair byte for block0
  uint32_t lane_idx = 0;              // 0..63 within strip

  Quantized_Point() = default;
  Quantized_Point(const uint8_t* ptr, uint32_t lane) : code_ptr(ptr), lane_idx(lane) {}

  inline float distance(const Quantized_Query<Metric>& qq) const { return qq.distance(*this); }
  void prefetch() const { __builtin_prefetch(code_ptr, 0, 3); }

  bool same_as(const Quantized_Point<Metric>&) const { return false; }
  bool same_as(const Quantized_Query<Metric>&) const { return false; }
  bool is_metric() const { return Metric; }
};

// ---------------------------------------------------------
// Encoded FastScan Point Range (NO codebooks; only packed codes + offsets + kernels)
// ---------------------------------------------------------
template<typename PointRange, bool Metric>
class Quantized_Point_Range {
 public:
  static constexpr bool is_fastscan = true;
  static constexpr uint32_t K = 16;

  uint32_t num_blocks = 0;

  size_t n_points_raw = 0;           // padded total vectors
  size_t n_points_raw_unpadded = 0;  // original flattened vectors (no padding)
  size_t dim = 0;
  size_t dim_per_block = 0;

  // Storage: strip-major for 64 vectors.
  // For each strip s and block b: 32 bytes (each byte packs 2x4-bit codes).
  parlay::sequence<uint8_t> packed_codes;

  // Offsets in VECTOR indices into the padded flattened DB: size = n_clouds+1
  parlay::sequence<size_t> cloud_vec_offsets;

  Quantized_Point_Range() = default;

  Quantized_Point<Metric> operator[](size_t i) const {
    const size_t strip_idx = i / 64;
    const size_t lane_idx = i % 64;
    const size_t strip_stride = static_cast<size_t>(num_blocks) * 32;
    const uint8_t* ptr = &packed_codes[strip_idx * strip_stride + (lane_idx / 2)];
    return Quantized_Point<Metric>(ptr, static_cast<uint32_t>(lane_idx));
  }

  // ---------------------------------------------------------
  // SIMD kernels (unchanged)
  // ---------------------------------------------------------
  void scan_64_chunk(const Quantized_Query<Metric>& q, const uint8_t* codes_ptr,
                     float* results) const {
    __m512i acc_even = _mm512_setzero_si512();
    __m512i acc_odd = _mm512_setzero_si512();
    const __m256i low_mask = _mm256_set1_epi8(0x0F);

    for (uint32_t b = 0; b < num_blocks; ++b) {
      const __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(codes_ptr));
      codes_ptr += 32;

      const __m256i codes_even = _mm256_and_si256(packed, low_mask);
      const __m256i codes_odd = _mm256_and_si256(_mm256_srli_epi16(packed, 4), low_mask);

      const __m128i lut128 =
          _mm_loadu_si128(reinterpret_cast<const __m128i*>(&q.int_lut[static_cast<size_t>(b) * K]));
      const __m256i lut256 = _mm256_broadcastsi128_si256(lut128);

      const __m256i scores_even_u8 = _mm256_shuffle_epi8(lut256, codes_even);
      const __m256i scores_odd_u8 = _mm256_shuffle_epi8(lut256, codes_odd);

      acc_even = _mm512_add_epi16(acc_even, _mm512_cvtepu8_epi16(scores_even_u8));
      acc_odd = _mm512_add_epi16(acc_odd, _mm512_cvtepu8_epi16(scores_odd_u8));
    }

    alignas(64) uint16_t raw_even[32];
    alignas(64) uint16_t raw_odd[32];
    _mm512_store_si512(reinterpret_cast<__m512i*>(raw_even), acc_even);
    _mm512_store_si512(reinterpret_cast<__m512i*>(raw_odd), acc_odd);

    for (int i = 0; i < 32; ++i) {
      results[i * 2] = q.decode(raw_even[i]);
      results[i * 2 + 1] = q.decode(raw_odd[i]);
    }
  }

  inline uint16_t hmin_512_epu16(__m512i v) const {
    v = _mm512_min_epu16(v, _mm512_shuffle_i32x4(v, v, _MM_SHUFFLE(1, 0, 3, 2)));
    v = _mm512_min_epu16(v, _mm512_shuffle_i32x4(v, v, _MM_SHUFFLE(0, 0, 1, 1)));

    __m128i v128 = _mm512_castsi512_si128(v);
    v128 = _mm_min_epu16(v128, _mm_srli_si128(v128, 8));
    v128 = _mm_min_epu16(v128, _mm_srli_si128(v128, 4));
    v128 = _mm_min_epu16(v128, _mm_srli_si128(v128, 2));
    return static_cast<uint16_t>(_mm_extract_epi16(v128, 0));
  }

  float scan_64_chunk_min(const Quantized_Query<Metric>& q, const uint8_t* codes_ptr) const {
    __m512i acc_even = _mm512_setzero_si512();
    __m512i acc_odd = _mm512_setzero_si512();
    const __m256i low_mask = _mm256_set1_epi8(0x0F);

    for (uint32_t b = 0; b < num_blocks; ++b) {
      const __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(codes_ptr));
      codes_ptr += 32;

      const __m256i codes_even = _mm256_and_si256(packed, low_mask);
      const __m256i codes_odd = _mm256_and_si256(_mm256_srli_epi16(packed, 4), low_mask);

      const __m128i lut128 =
          _mm_loadu_si128(reinterpret_cast<const __m128i*>(&q.int_lut[static_cast<size_t>(b) * K]));
      const __m256i lut256 = _mm256_broadcastsi128_si256(lut128);

      const __m256i scores_even_u8 = _mm256_shuffle_epi8(lut256, codes_even);
      const __m256i scores_odd_u8 = _mm256_shuffle_epi8(lut256, codes_odd);

      acc_even = _mm512_add_epi16(acc_even, _mm512_cvtepu8_epi16(scores_even_u8));
      acc_odd = _mm512_add_epi16(acc_odd, _mm512_cvtepu8_epi16(scores_odd_u8));
    }

    const __m512i vmin = _mm512_min_epu16(acc_even, acc_odd);
    const uint16_t best = hmin_512_epu16(vmin);
    return q.decode(best);
  }

  inline __m512i scan_64_running_min(const Quantized_Query<Metric>& q, const uint8_t* codes_ptr,
                                     __m512i current_min_v) const {
    __m512i acc_even = _mm512_setzero_si512();
    __m512i acc_odd = _mm512_setzero_si512();
    const __m256i low_mask = _mm256_set1_epi8(0x0F);

    for (uint32_t b = 0; b < num_blocks; ++b) {
      const __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(codes_ptr));
      codes_ptr += 32;

      const __m256i codes_even = _mm256_and_si256(packed, low_mask);
      const __m256i codes_odd = _mm256_and_si256(_mm256_srli_epi16(packed, 4), low_mask);

      const __m128i lut128 =
          _mm_loadu_si128(reinterpret_cast<const __m128i*>(&q.int_lut[static_cast<size_t>(b) * K]));
      const __m256i lut256 = _mm256_broadcastsi128_si256(lut128);

      const __m256i scores_even_u8 = _mm256_shuffle_epi8(lut256, codes_even);
      const __m256i scores_odd_u8 = _mm256_shuffle_epi8(lut256, codes_odd);

      acc_even = _mm512_add_epi16(acc_even, _mm512_cvtepu8_epi16(scores_even_u8));
      acc_odd = _mm512_add_epi16(acc_odd, _mm512_cvtepu8_epi16(scores_odd_u8));
    }

    current_min_v = _mm512_min_epu16(current_min_v, acc_even);
    current_min_v = _mm512_min_epu16(current_min_v, acc_odd);
    return current_min_v;
  }

  inline float reduce_running_min(const Quantized_Query<Metric>& q, __m512i running_min_v) const {
    const uint16_t best = hmin_512_epu16(running_min_v);
    return q.decode(best);
  }

  inline void scan_64_dual_query(const Quantized_Query<Metric>& q1,
                                 const Quantized_Query<Metric>& q2, const uint8_t* codes_ptr,
                                 __m512i& min_v1, __m512i& min_v2) const {
    __m512i acc1_even = _mm512_setzero_si512();
    __m512i acc1_odd = _mm512_setzero_si512();
    __m512i acc2_even = _mm512_setzero_si512();
    __m512i acc2_odd = _mm512_setzero_si512();
    const __m256i low_mask = _mm256_set1_epi8(0x0F);

    for (uint32_t b = 0; b < num_blocks; ++b) {
      const __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(codes_ptr));
      codes_ptr += 32;

      const __m256i codes_even = _mm256_and_si256(packed, low_mask);
      const __m256i codes_odd = _mm256_and_si256(_mm256_srli_epi16(packed, 4), low_mask);

      const __m128i lut1_128 = _mm_loadu_si128(
          reinterpret_cast<const __m128i*>(&q1.int_lut[static_cast<size_t>(b) * K]));
      const __m128i lut2_128 = _mm_loadu_si128(
          reinterpret_cast<const __m128i*>(&q2.int_lut[static_cast<size_t>(b) * K]));
      const __m256i lut1_256 = _mm256_broadcastsi128_si256(lut1_128);
      const __m256i lut2_256 = _mm256_broadcastsi128_si256(lut2_128);

      const __m256i s1_even_u8 = _mm256_shuffle_epi8(lut1_256, codes_even);
      const __m256i s1_odd_u8 = _mm256_shuffle_epi8(lut1_256, codes_odd);
      const __m256i s2_even_u8 = _mm256_shuffle_epi8(lut2_256, codes_even);
      const __m256i s2_odd_u8 = _mm256_shuffle_epi8(lut2_256, codes_odd);

      acc1_even = _mm512_add_epi16(acc1_even, _mm512_cvtepu8_epi16(s1_even_u8));
      acc1_odd = _mm512_add_epi16(acc1_odd, _mm512_cvtepu8_epi16(s1_odd_u8));
      acc2_even = _mm512_add_epi16(acc2_even, _mm512_cvtepu8_epi16(s2_even_u8));
      acc2_odd = _mm512_add_epi16(acc2_odd, _mm512_cvtepu8_epi16(s2_odd_u8));
    }

    min_v1 = _mm512_min_epu16(min_v1, _mm512_min_epu16(acc1_even, acc1_odd));
    min_v2 = _mm512_min_epu16(min_v2, _mm512_min_epu16(acc2_even, acc2_odd));
  }

  // ---------------------------------------------------------
  // Masked min for partial strips
  // ---------------------------------------------------------
  static inline __mmask32 mask_even_lanes(int lo, int hi) {
    __mmask32 m = 0;
    for (int i = 0; i < 32; ++i) {
      const int lane = 2 * i;
      if (lo <= lane && lane < hi) m |= (__mmask32(1) << i);
    }
    return m;
  }
  static inline __mmask32 mask_odd_lanes(int lo, int hi) {
    __mmask32 m = 0;
    for (int i = 0; i < 32; ++i) {
      const int lane = 2 * i + 1;
      if (lo <= lane && lane < hi) m |= (__mmask32(1) << i);
    }
    return m;
  }

  float scan_64_chunk_min_masked(const Quantized_Query<Metric>& q, const uint8_t* codes_ptr, int lo,
                                 int hi) const {
    lo = std::max(lo, 0);
    hi = std::min(hi, 64);
    if (hi <= lo) return std::numeric_limits<float>::infinity();

    __m512i acc_even = _mm512_setzero_si512();
    __m512i acc_odd = _mm512_setzero_si512();
    const __m256i low_mask = _mm256_set1_epi8(0x0F);

    for (uint32_t b = 0; b < num_blocks; ++b) {
      const __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(codes_ptr));
      codes_ptr += 32;

      const __m256i codes_even = _mm256_and_si256(packed, low_mask);
      const __m256i codes_odd = _mm256_and_si256(_mm256_srli_epi16(packed, 4), low_mask);

      const __m128i lut128 =
          _mm_loadu_si128(reinterpret_cast<const __m128i*>(&q.int_lut[static_cast<size_t>(b) * K]));
      const __m256i lut256 = _mm256_broadcastsi128_si256(lut128);

      const __m256i scores_even_u8 = _mm256_shuffle_epi8(lut256, codes_even);
      const __m256i scores_odd_u8 = _mm256_shuffle_epi8(lut256, codes_odd);

      acc_even = _mm512_add_epi16(acc_even, _mm512_cvtepu8_epi16(scores_even_u8));
      acc_odd = _mm512_add_epi16(acc_odd, _mm512_cvtepu8_epi16(scores_odd_u8));
    }

    const __m512i INF = _mm512_set1_epi16(0xFFFF);
    const __mmask32 me = mask_even_lanes(lo, hi);
    const __mmask32 mo = mask_odd_lanes(lo, hi);

    acc_even = _mm512_mask_mov_epi16(INF, me, acc_even);
    acc_odd = _mm512_mask_mov_epi16(INF, mo, acc_odd);

    const __m512i vmin = _mm512_min_epu16(acc_even, acc_odd);
    const uint16_t best = hmin_512_epu16(vmin);
    return q.decode(best);
  }

  // ---------------------------------------------------------
  // Save/load encoded range ONLY
  // ---------------------------------------------------------
  void save(std::ostream& out) const {
    out.write(reinterpret_cast<const char*>(&num_blocks), sizeof(num_blocks));
    out.write(reinterpret_cast<const char*>(&n_points_raw), sizeof(n_points_raw));
    out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    out.write(reinterpret_cast<const char*>(&dim_per_block), sizeof(dim_per_block));
    out.write(reinterpret_cast<const char*>(&n_points_raw_unpadded), sizeof(n_points_raw_unpadded));

    size_t n_off = cloud_vec_offsets.size();
    out.write(reinterpret_cast<const char*>(&n_off), sizeof(n_off));
    if (n_off)
      out.write(reinterpret_cast<const char*>(cloud_vec_offsets.data()), n_off * sizeof(size_t));

    size_t sz = packed_codes.size();
    out.write(reinterpret_cast<const char*>(&sz), sizeof(sz));
    if (sz) out.write(reinterpret_cast<const char*>(packed_codes.data()), sz * sizeof(uint8_t));
  }

  void load(std::istream& in) {
    in.read(reinterpret_cast<char*>(&num_blocks), sizeof(num_blocks));
    in.read(reinterpret_cast<char*>(&n_points_raw), sizeof(n_points_raw));
    in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    in.read(reinterpret_cast<char*>(&dim_per_block), sizeof(dim_per_block));
    in.read(reinterpret_cast<char*>(&n_points_raw_unpadded), sizeof(n_points_raw_unpadded));

    size_t n_off = 0;
    in.read(reinterpret_cast<char*>(&n_off), sizeof(n_off));
    cloud_vec_offsets.resize(n_off);
    if (n_off) in.read(reinterpret_cast<char*>(cloud_vec_offsets.data()), n_off * sizeof(size_t));

    size_t sz = 0;
    in.read(reinterpret_cast<char*>(&sz), sizeof(sz));
    packed_codes.resize(sz);
    if (sz) in.read(reinterpret_cast<char*>(packed_codes.data()), sz * sizeof(uint8_t));
  }

  inline uint32_t size() const noexcept { return static_cast<uint32_t>(n_points_raw); }
  inline uint32_t get_dims() const noexcept { return static_cast<uint32_t>(dim); }
};

// ---------------------------------------------------------
// FastScan model: owns codebooks, trains, encodes, builds LUTs
// ---------------------------------------------------------
template<bool Metric>
class Model {
 public:
  static constexpr bool is_fastscan = true;
  static constexpr uint32_t K = 16;

  uint32_t num_blocks = 0;
  size_t dim = 0;
  size_t dim_per_block = 0;

  std::vector<Eigen::MatrixXf> codebooks;       // [b] (K x dim_per_block)
  std::vector<Eigen::VectorXf> codebook_norms;  // [b] (K)

  Model() = default;

  template<typename PointRange>
  Model(const PointRange& train_data, uint32_t block_size = 32) {
    train(train_data, block_size);
  }

  template<typename PointRange>
  void train(const PointRange& data, uint32_t block_size = 32) {
    dim = data.get_dims();
    dim_per_block = block_size;

    if (dim_per_block == 0 || (dim % dim_per_block) != 0) {
      std::cerr << "Error: FastScan dim=" << dim << " not divisible by block_size=" << dim_per_block
                << "\n";
      abort();
    }
    num_blocks = static_cast<uint32_t>(dim / dim_per_block);

    codebooks.resize(num_blocks);
    codebook_norms.resize(num_blocks);

    const size_t n_points_raw_unpadded = data.size();
    const size_t sample_size = std::min(static_cast<size_t>(K * 50), n_points_raw_unpadded);

    parlay::parallel_for(0, num_blocks, [&](size_t b) {
      const size_t offset = b * dim_per_block;
      parlay::sequence<parlay::sequence<float>> sub(sample_size);

      std::mt19937 rng(static_cast<unsigned>(b + 1));
      std::uniform_int_distribution<size_t> distu(0, n_points_raw_unpadded - 1);

      for (size_t i = 0; i < sample_size; ++i) {
        const float* raw = reinterpret_cast<const float*>(data.location(distu(rng)));
        sub[i] = parlay::sequence<float>(raw + offset, raw + offset + dim_per_block);
      }

      auto [centers, _] = mvsic::kmeans_subsample_assign_only<Metric>(sub, K, sample_size, false);

      codebooks[b] =
          Eigen::MatrixXf(static_cast<Eigen::Index>(K), static_cast<Eigen::Index>(dim_per_block));
      for (size_t c = 0; c < centers.size(); ++c) {
        for (size_t d = 0; d < dim_per_block; ++d) {
          codebooks[b](static_cast<Eigen::Index>(c), static_cast<Eigen::Index>(d)) = centers[c][d];
        }
      }
      codebook_norms[b] = codebooks[b].rowwise().squaredNorm();
    });
  }

  // Encode with explicit padding-by-cloud
  template<typename PointRange, typename SeqOffsetsFloat>
  Quantized_Point_Range<PointRange, Metric> encode(
      const PointRange& data, const SeqOffsetsFloat& cloud_offsets_float) const {
    Quantized_Point_Range<PointRange, Metric> enc;
    enc.num_blocks = num_blocks;
    enc.dim = data.get_dims();
    enc.dim_per_block = dim_per_block;

    enc.n_points_raw_unpadded = data.size();

    const size_t n_clouds = (cloud_offsets_float.size() > 0) ? (cloud_offsets_float.size() - 1) : 0;

    // Build padded offsets in VECTOR indices.
    enc.cloud_vec_offsets.resize(n_clouds + 1);
    size_t cur = 0;
    enc.cloud_vec_offsets[0] = 0;

    for (size_t c = 0; c < n_clouds; ++c) {
      const size_t start_f = static_cast<size_t>(cloud_offsets_float[c]);
      const size_t end_f = static_cast<size_t>(cloud_offsets_float[c + 1]);
      const size_t sz_vecs = (end_f - start_f) / enc.dim;
      const size_t padded = ((sz_vecs + 63) / 64) * 64;
      enc.cloud_vec_offsets[c] = cur;
      cur += padded;
    }
    enc.cloud_vec_offsets[n_clouds] = cur;
    enc.n_points_raw = cur;

    const size_t strip_stride = static_cast<size_t>(enc.num_blocks) * 32;
    const size_t num_strips = (enc.n_points_raw + 63) / 64;  // exact since padded
    enc.packed_codes.resize(num_strips * strip_stride);
    std::fill(enc.packed_codes.begin(), enc.packed_codes.end(), uint8_t{0});

    // Encode each cloud independently (cloud spans whole strips)
    parlay::parallel_for(0, n_clouds, [&](size_t c) {
      const size_t start_f = static_cast<size_t>(cloud_offsets_float[c]);
      const size_t end_f = static_cast<size_t>(cloud_offsets_float[c + 1]);

      const size_t start_vec_src = start_f / enc.dim;      // unpadded vector index
      const size_t sz_vecs = (end_f - start_f) / enc.dim;  // original vectors in cloud

      const size_t start_vec_dst = enc.cloud_vec_offsets[c];
      const size_t padded_sz = enc.cloud_vec_offsets[c + 1] - enc.cloud_vec_offsets[c];
      const size_t n_strips_c = padded_sz / 64;

      const size_t strip0_dst = start_vec_dst / 64;  // aligned

      for (size_t s = 0; s < n_strips_c; ++s) {
        uint8_t* strip_base = enc.packed_codes.data() + (strip0_dst + s) * strip_stride;

        for (size_t lane_pair = 0; lane_pair < 32; ++lane_pair) {
          const size_t v_even_in_cloud = s * 64 + lane_pair * 2;
          const size_t v_odd_in_cloud = v_even_in_cloud + 1;

          const bool has_even = (v_even_in_cloud < sz_vecs);
          const bool has_odd = (v_odd_in_cloud < sz_vecs);

          const size_t v_even_src = start_vec_src + v_even_in_cloud;
          const size_t v_odd_src = start_vec_src + v_odd_in_cloud;

          for (uint32_t b = 0; b < enc.num_blocks; ++b) {
            const uint8_t c_e = has_even ? find_best(data.location(v_even_src), b) : 0;
            const uint8_t c_o = has_odd ? find_best(data.location(v_odd_src), b) : 0;

            strip_base[static_cast<size_t>(b) * 32 + lane_pair] =
                (c_e & 0x0F) | ((c_o & 0x0F) << 4);
          }
        }
      }
    });

    return enc;
  }

  // Encode assuming the entire dataset is a single point cloud.
  // We still pad to a multiple of 64 vectors (since FastScan scans in 64-wide strips),
  // but no per-cloud structure is needed by the caller.
  template<typename PointRange>
  Quantized_Point_Range<PointRange, Metric> encode(const PointRange& data) const {
    std::vector<size_t> cloud_offsets_float;
    cloud_offsets_float.reserve(2);

    const size_t N = data.size();
    const size_t D = static_cast<size_t>(data.get_dims());
    cloud_offsets_float.push_back(0);
    cloud_offsets_float.push_back(N * D);

    return encode(data, cloud_offsets_float);
  }

  // Query LUT build
  template<typename PointTy>
  Quantized_Query<Metric> quantize_query(const PointTy& query) const {
    Quantized_Query<Metric> qq(num_blocks);
    std::vector<float> float_lut(static_cast<size_t>(num_blocks) * K);

    float g_min = std::numeric_limits<float>::max();
    float g_max = std::numeric_limits<float>::lowest();

    for (uint32_t b = 0; b < num_blocks; ++b) {
      Eigen::VectorXf q_sub(static_cast<Eigen::Index>(dim_per_block));
      const size_t offset = static_cast<size_t>(b) * dim_per_block;
      for (size_t j = 0; j < dim_per_block; ++j) {
        q_sub[static_cast<Eigen::Index>(j)] = query[offset + j];
      }

      Eigen::VectorXf dots = codebooks[b] * q_sub;
      float q_sq = Metric ? q_sub.squaredNorm() : 0.0f;

      for (uint32_t i = 0; i < K; ++i) {
        float val = Metric ? (codebook_norms[b][i] - 2.0f * dots[i] + q_sq) : -dots[i];
        float_lut[static_cast<size_t>(b) * K + i] = val;
        g_min = std::min(g_min, val);
        g_max = std::max(g_max, val);
      }
    }
    return finish_quantize_query(qq, float_lut, g_min, g_max);
  }

  Quantized_Query<Metric> quantize_query(const float* qptr) const {
    Quantized_Query<Metric> qq(num_blocks);
    std::vector<float> float_lut(static_cast<size_t>(num_blocks) * K);

    float g_min = std::numeric_limits<float>::max();
    float g_max = std::numeric_limits<float>::lowest();

    for (uint32_t b = 0; b < num_blocks; ++b) {
      Eigen::Map<const Eigen::VectorXf> q_sub(qptr + static_cast<size_t>(b) * dim_per_block,
                                              static_cast<Eigen::Index>(dim_per_block));
      Eigen::VectorXf dots = codebooks[b] * q_sub;
      float q_sq = Metric ? q_sub.squaredNorm() : 0.0f;

      for (uint32_t i = 0; i < K; ++i) {
        float val = Metric ? (codebook_norms[b][i] - 2.0f * dots[i] + q_sq) : -dots[i];
        float_lut[static_cast<size_t>(b) * K + i] = val;
        g_min = std::min(g_min, val);
        g_max = std::max(g_max, val);
      }
    }
    return finish_quantize_query(qq, float_lut, g_min, g_max);
  }

  template<typename PointCloudTy>
  void quantize_query_batch(const PointCloudTy& query_cloud,
                            parlay::sequence<Quantized_Query<Metric>>& out_luts) const {
    const uint32_t num_q = query_cloud.size();
    const uint32_t dims = query_cloud.get_dims();
    const float* base = query_cloud.data();

    out_luts.clear();
    out_luts.reserve(num_q);
    for (uint32_t i = 0; i < num_q; ++i)
      out_luts.emplace_back(num_blocks);

    using RowMajorMat = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    RowMajorMat Q_sub(static_cast<Eigen::Index>(num_q), static_cast<Eigen::Index>(dim_per_block));
    Eigen::MatrixXf dot_products(static_cast<Eigen::Index>(K), static_cast<Eigen::Index>(num_q));

    std::vector<std::vector<float>> batch_float_luts(
        num_q, std::vector<float>(static_cast<size_t>(num_blocks) * K));
    std::vector<float> q_mins(num_q, std::numeric_limits<float>::max());
    std::vector<float> q_maxs(num_q, std::numeric_limits<float>::lowest());

    for (uint32_t b = 0; b < num_blocks; ++b) {
      const uint32_t offset = b * static_cast<uint32_t>(dim_per_block);
      for (uint32_t i = 0; i < num_q; ++i) {
        std::memcpy(&Q_sub(i, 0), base + static_cast<size_t>(i) * dims + offset,
                    sizeof(float) * dim_per_block);
      }

      Eigen::VectorXf q_sq;
      if constexpr (Metric) q_sq = Q_sub.rowwise().squaredNorm();

      dot_products.noalias() = codebooks[b] * Q_sub.transpose();

      for (uint32_t i = 0; i < num_q; ++i) {
        for (uint32_t k_idx = 0; k_idx < K; ++k_idx) {
          float val = Metric ? (codebook_norms[b][k_idx] - 2.0f * dot_products(k_idx, i) + q_sq[i])
                             : -dot_products(k_idx, i);
          batch_float_luts[i][static_cast<size_t>(b) * K + k_idx] = val;
          q_mins[i] = std::min(q_mins[i], val);
          q_maxs[i] = std::max(q_maxs[i], val);
        }
      }
    }

    for (uint32_t i = 0; i < num_q; ++i) {
      finish_quantize_query(out_luts[i], batch_float_luts[i], q_mins[i], q_maxs[i]);
    }
  }

  // Save/load MODEL only
  void save(std::ostream& out) const {
    out.write(reinterpret_cast<const char*>(&num_blocks), sizeof(num_blocks));
    out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    out.write(reinterpret_cast<const char*>(&dim_per_block), sizeof(dim_per_block));

    for (uint32_t b = 0; b < num_blocks; ++b) {
      size_t rs = static_cast<size_t>(codebooks[b].rows());
      size_t cs = static_cast<size_t>(codebooks[b].cols());
      out.write(reinterpret_cast<const char*>(&rs), sizeof(rs));
      out.write(reinterpret_cast<const char*>(&cs), sizeof(cs));
      out.write(reinterpret_cast<const char*>(codebooks[b].data()), rs * cs * sizeof(float));
    }
  }

  void load(std::istream& in) {
    in.read(reinterpret_cast<char*>(&num_blocks), sizeof(num_blocks));
    in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    in.read(reinterpret_cast<char*>(&dim_per_block), sizeof(dim_per_block));

    codebooks.resize(num_blocks);
    codebook_norms.resize(num_blocks);

    for (uint32_t b = 0; b < num_blocks; ++b) {
      size_t rs = 0, cs = 0;
      in.read(reinterpret_cast<char*>(&rs), sizeof(rs));
      in.read(reinterpret_cast<char*>(&cs), sizeof(cs));
      codebooks[b] = Eigen::MatrixXf(static_cast<Eigen::Index>(rs), static_cast<Eigen::Index>(cs));
      in.read(reinterpret_cast<char*>(codebooks[b].data()), rs * cs * sizeof(float));
      codebook_norms[b] = codebooks[b].rowwise().squaredNorm();
    }
  }

 private:
  Quantized_Query<Metric>& finish_quantize_query(Quantized_Query<Metric>& qq,
                                                 const std::vector<float>& float_lut, float g_min,
                                                 float g_max) const {
    qq.min_dist = g_min;
    const float range = (g_max - g_min);
    qq.scale = std::max(1e-6f, range / 255.0f);

    for (size_t i = 0; i < float_lut.size(); ++i) {
      qq.int_lut[i] = static_cast<uint8_t>((float_lut[i] - g_min) / qq.scale);
    }
    return qq;
  }

  template<typename VecLocPtr>
  uint8_t find_best(const VecLocPtr vec_ptr, uint32_t b) const {
    const float* raw = reinterpret_cast<const float*>(vec_ptr);
    const float* sub_ptr = raw + static_cast<size_t>(b) * dim_per_block;

    Eigen::Map<const Eigen::VectorXf> q_sub(sub_ptr, static_cast<Eigen::Index>(dim_per_block));
    Eigen::VectorXf dots = codebooks[b] * q_sub;

    float min_val = std::numeric_limits<float>::max();
    uint8_t best = 0;
    for (uint32_t i = 0; i < K; ++i) {
      float val = Metric ? (codebook_norms[b][i] - 2.0f * dots[i]) : -dots[i];
      if (val < min_val) {
        min_val = val;
        best = static_cast<uint8_t>(i);
      }
    }
    return best;
  }
};

}  // namespace fastscan
}  // namespace mvsic
