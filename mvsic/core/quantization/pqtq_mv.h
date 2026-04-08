#pragma once

// Multi-vector (Chamfer) PQTQ with SYMMETRIC query quantization.
//
// Reuses the scan kernel + tables from `pqtq.h`. The DB is stored as a per-
// cloud 4-block interleaved strip layout (64 points per strip); each cloud is
// zero-padded to a multiple of 64 so the VNNI strip kernel can run without
// edge cases. Queries are themselves PQTQ-encoded (symmetric design), so there
// is no per-query LUT construction — scoring uses the process-wide static
// `half_lut[256][32]` cached in `internal::get_tables(block_size)`.
//
// Per-Chamfer math (for one (query_cloud, db_cloud) pair):
//   for each emb q in query_cloud:
//     min_q = +inf
//     for each strip s in db_cloud:
//       acc0..3 = vnni_scan_strip(strip_s, q_codes, half_lut)
//       tmp[64] = vnni_decode_strip<Metric>(acc, ...)
//       if s == last_strip: mask tmp[valid..64] = +inf
//       min_q = min(min_q, min(tmp))
//   chamfer = sum_q(min_q) / num_queries
//
// For higher throughput we run 3 queries at once against each strip via
// `vnni_scan_strip_fused3` (mirrors the reference leaf-scoring kernel in
// `processleaf_pqtq.cc`). Remaining queries (< 3) run through the single
// `vnni_scan_strip` path.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <queue>
#include <utility>
#include <vector>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

#include "parlay/parallel.h"
#include "parlay/sequence.h"

#include "mvsic/core/quantization/pqtq.h"

namespace mvsic {
namespace pqtq_mv {

namespace internal {

using ::mvsic::pqtq::internal::Tables;
using ::mvsic::pqtq::internal::get_tables;
using ::mvsic::pqtq::internal::num_strips_for;
using ::mvsic::pqtq::internal::strip_stride_bytes;
using ::mvsic::pqtq::internal::pack_into_strips_interleaved4;

static constexpr size_t kMvBatch = 3;  // fused queries per strip pass

#if defined(__AVX512F__) && defined(__AVX512VBMI__) && defined(__AVX512VNNI__)
// Three-query fused strip scan. `acc` must have 12 __m512i slots, with
// queries packed as acc[q*4+r] for q in [0..3), r in [0..4). Mirrors
// `vnni_scan_strip_fused3` in processleaf_pqtq.cc.
static inline void vnni_scan_strip_fused3(const uint8_t* strip_codes, uint32_t nb4,
                                          const uint8_t* q0, const uint8_t* q1,
                                          const uint8_t* q2, const uint8_t (*half_lut)[32],
                                          const __m512i low_mask, const __m512i ones_i8,
                                          const __m512i offset_mask, __m512i* acc) {
  for (int i = 0; i < 12; ++i) acc[i] = _mm512_setzero_si512();
  const uint8_t* codes_ptr = strip_codes;
  for (uint32_t b = 0; b < nb4; b += 4) {
    const __m512i packed1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(codes_ptr));
    const __m512i packed2 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(codes_ptr + 64));
    codes_ptr += 128;

    const __m512i ce1 = _mm512_or_si512(_mm512_and_si512(packed1, low_mask), offset_mask);
    const __m512i co1 = _mm512_or_si512(
        _mm512_and_si512(_mm512_srli_epi16(packed1, 4), low_mask), offset_mask);
    const __m512i ce2 = _mm512_or_si512(_mm512_and_si512(packed2, low_mask), offset_mask);
    const __m512i co2 = _mm512_or_si512(
        _mm512_and_si512(_mm512_srli_epi16(packed2, 4), low_mask), offset_mask);

    const uint32_t boff = b / 2;
    const __m512i lut0 = _mm512_inserti64x4(
        _mm512_castsi256_si512(
            _mm256_load_si256(reinterpret_cast<const __m256i*>(half_lut[q0[boff]]))),
        _mm256_load_si256(reinterpret_cast<const __m256i*>(half_lut[q0[boff + 1]])), 1);
    acc[0] = _mm512_dpbusd_epi32(acc[0], _mm512_permutexvar_epi8(ce1, lut0), ones_i8);
    acc[1] = _mm512_dpbusd_epi32(acc[1], _mm512_permutexvar_epi8(co1, lut0), ones_i8);
    acc[2] = _mm512_dpbusd_epi32(acc[2], _mm512_permutexvar_epi8(ce2, lut0), ones_i8);
    acc[3] = _mm512_dpbusd_epi32(acc[3], _mm512_permutexvar_epi8(co2, lut0), ones_i8);

    const __m512i lut1 = _mm512_inserti64x4(
        _mm512_castsi256_si512(
            _mm256_load_si256(reinterpret_cast<const __m256i*>(half_lut[q1[boff]]))),
        _mm256_load_si256(reinterpret_cast<const __m256i*>(half_lut[q1[boff + 1]])), 1);
    acc[4] = _mm512_dpbusd_epi32(acc[4], _mm512_permutexvar_epi8(ce1, lut1), ones_i8);
    acc[5] = _mm512_dpbusd_epi32(acc[5], _mm512_permutexvar_epi8(co1, lut1), ones_i8);
    acc[6] = _mm512_dpbusd_epi32(acc[6], _mm512_permutexvar_epi8(ce2, lut1), ones_i8);
    acc[7] = _mm512_dpbusd_epi32(acc[7], _mm512_permutexvar_epi8(co2, lut1), ones_i8);

    const __m512i lut2 = _mm512_inserti64x4(
        _mm512_castsi256_si512(
            _mm256_load_si256(reinterpret_cast<const __m256i*>(half_lut[q2[boff]]))),
        _mm256_load_si256(reinterpret_cast<const __m256i*>(half_lut[q2[boff + 1]])), 1);
    acc[8] = _mm512_dpbusd_epi32(acc[8], _mm512_permutexvar_epi8(ce1, lut2), ones_i8);
    acc[9] = _mm512_dpbusd_epi32(acc[9], _mm512_permutexvar_epi8(co1, lut2), ones_i8);
    acc[10] = _mm512_dpbusd_epi32(acc[10], _mm512_permutexvar_epi8(ce2, lut2), ones_i8);
    acc[11] = _mm512_dpbusd_epi32(acc[11], _mm512_permutexvar_epi8(co2, lut2), ones_i8);
  }
}
#endif  // AVX512 VBMI + VNNI

// Scalar fallback: scores one query against one cloud and returns the min.
template<bool Metric>
inline float chamfer_one_query_scalar(const Tables* t, const uint8_t* qcodes, float q_nsf,
                                      float q_sqn, uint32_t num_blocks, size_t nbytes_per_dp,
                                      const uint8_t* packed_flat, const float* norms,
                                      const float* sqns, size_t cloud_size) {
  const int8_t* sym = t->global_sym_lut_int8;
  const float cs = q_nsf * t->global_sym_lut_scale;
  float best = std::numeric_limits<float>::max();
  for (size_t i = 0; i < cloud_size; ++i) {
    const uint8_t* pc = packed_flat + i * nbytes_per_dp;
    int32_t acc = 0;
    for (uint32_t b = 0; b < num_blocks; ++b) {
      const uint8_t qi = ::mvsic::pqtq::internal::get_packed_nibble(qcodes, b);
      const uint8_t di = ::mvsic::pqtq::internal::get_packed_nibble(pc, b);
      acc += static_cast<int32_t>(sym[qi * 16 + di]);
    }
    const float dot = norms[i] * static_cast<float>(acc) * cs;
    float d;
    if constexpr (Metric) {
      d = sqns[i] + q_sqn - 2.0f * dot;
    } else {
      d = -dot;
    }
    if (d < best) best = d;
  }
  return best;
}

#if defined(__AVX512F__) && defined(__AVX512VBMI__) && defined(__AVX512VNNI__)
// Min across a float32 buffer that may have junk in tail lanes. `valid_count`
// in [1..64] — lanes >= valid_count are treated as +inf.
static inline float reduce_min_valid(const float* tmp, size_t valid_count) {
  __m512 m = _mm512_set1_ps(std::numeric_limits<float>::max());
  const __m512 inf_v = _mm512_set1_ps(std::numeric_limits<float>::max());
  for (size_t off = 0; off < 64; off += 16) {
    __m512 v = _mm512_loadu_ps(tmp + off);
    __mmask16 mask;
    if (off + 16 <= valid_count) {
      mask = 0xFFFF;
    } else if (off >= valid_count) {
      mask = 0x0000;
    } else {
      mask = static_cast<__mmask16>((1u << (valid_count - off)) - 1);
    }
    v = _mm512_mask_blend_ps(mask, inf_v, v);
    m = _mm512_min_ps(m, v);
  }
  return _mm512_reduce_min_ps(m);
}

// Build the 4 per-acc tail masks for a strip with `last_valid` live points.
// See `vnni_scan_strip_fused3_minacc` for the lane mapping per acc.
static inline void build_strip_masks(size_t last_valid, __mmask16 masks_acc[4]) {
  auto mask_for = [&](int start_pt) -> __mmask16 {
    if (static_cast<size_t>(start_pt) >= last_valid) return 0;
    size_t n = (last_valid - start_pt + 1) / 2;  // ceil((lv - s) / 2), s is start_pt
    if (n > 16) n = 16;
    return static_cast<__mmask16>((1u << n) - 1);
  };
  masks_acc[0] = mask_for(0);
  masks_acc[1] = mask_for(1);
  masks_acc[2] = mask_for(32);
  masks_acc[3] = mask_for(33);
}
#endif

}  // namespace internal

// =========================================================================
// Forward declarations
// =========================================================================
template<bool Metric>
class Quantized_Point_Cloud_Set;
template<bool Metric>
class Quantized_Query_Point_Cloud;

template<bool Metric>
float pqtq_mv_chamfer_distance(const Quantized_Query_Point_Cloud<Metric>& q,
                               const Quantized_Point_Cloud_Set<Metric>& db, size_t cloud_idx);

// =========================================================================
// Cloud handle: refers to one cloud (by index) within a set
// =========================================================================
template<bool Metric>
class Quantized_Point_Cloud {
 public:
  const Quantized_Point_Cloud_Set<Metric>* db = nullptr;
  size_t cloud_idx = 0;

  Quantized_Point_Cloud() = default;
  Quantized_Point_Cloud(const Quantized_Point_Cloud_Set<Metric>* d, size_t idx) :
      db(d), cloud_idx(idx) {}

  size_t size() const;

  static constexpr bool is_metric() { return Metric; }

  template<typename Query>
  bool same_as(const Query&) const {
    return false;
  }
};

// =========================================================================
// Multi-Vector Query (symmetric: each query emb is itself PQTQ-encoded)
// =========================================================================
template<bool Metric>
class Quantized_Query_Point_Cloud {
 public:
  using distanceType = float;

  size_t num_queries = 0;
  size_t num_bytes_per_datapoint = 0;
  uint32_t num_blocks = 0;
  size_t scan_lut_bytes_per_query = 0;

  // num_queries * num_bytes_per_datapoint bytes, contiguous
  std::vector<uint8_t> flat_query_codes;
  // Per-query scan LUTs (concatenated half_lut rows), used by the fused
  // scan+decode kernel so each 4-block iteration is a single 64B load
  // instead of 2 32B loads + vinserti64x4.
  std::vector<uint8_t> flat_query_scan_luts;
  std::vector<float> norm_scaling_factors;       // size: num_queries
  std::vector<float> unquantized_squared_norms;  // size: num_queries

  const ::mvsic::pqtq::internal::Tables* tables = nullptr;

  Quantized_Query_Point_Cloud() = default;

  inline const uint8_t* get_q_codes(size_t i) const {
    return flat_query_codes.data() + i * num_bytes_per_datapoint;
  }
  inline const uint8_t* get_q_scan_lut(size_t i) const {
    return flat_query_scan_luts.data() + i * scan_lut_bytes_per_query;
  }

  template<typename CloudView>
  float distance(const CloudView& cloud) const {
    return pqtq_mv_chamfer_distance(*this, *cloud.db, cloud.cloud_idx);
  }

  template<typename CloudHandle>
  std::pair<float, size_t> distance_w_cmps(const CloudHandle& cloud) const {
    const size_t bytes_per_vec =
        cloud.db->num_bytes_per_datapoint + sizeof(float) + (Metric ? sizeof(float) : 0);
    return {this->distance(cloud), cloud.size() * bytes_per_vec};
  }

  static constexpr bool is_metric() { return Metric; }
};

// =========================================================================
// Multi-Vector Set — per-cloud 4-block interleaved strip layout
// =========================================================================
template<bool Metric>
class Quantized_Point_Cloud_Set {
 public:
  // Encoder geometry (same for every cloud)
  size_t num_bytes_per_datapoint = 0;
  uint32_t num_blocks = 0;
  size_t strip_stride = 0;  // bytes per 64-point strip
  const ::mvsic::pqtq::internal::Tables* tables = nullptr;

  // Concatenated strip-packed data for all clouds.
  // Cloud c lives at [strip_offsets[c] .. strip_offsets[c+1]) bytes.
  parlay::sequence<uint8_t> packed_strips;
  parlay::sequence<size_t> strip_offsets;  // size: n_clouds + 1

  // Per-point arrays, padded so each cloud is a multiple of 64.
  // Cloud c starts at point_offsets[c] (in point units).
  parlay::sequence<size_t> point_offsets;  // size: n_clouds + 1
  parlay::sequence<float> norm_scaling_factors;
  parlay::sequence<float> unquantized_squared_norms;

  // Unpadded sizes per cloud.
  parlay::sequence<size_t> cloud_sizes;
  parlay::sequence<uint32_t> ids;

  inline size_t num_clouds() const noexcept {
    return strip_offsets.size() > 0 ? strip_offsets.size() - 1 : 0;
  }

  inline size_t n_strips(size_t c) const noexcept {
    return (strip_offsets[c + 1] - strip_offsets[c]) / strip_stride;
  }

  inline size_t cloud_size(size_t c) const noexcept { return cloud_sizes[c]; }

  Quantized_Point_Cloud_Set() = default;
  static constexpr bool is_metric() noexcept { return Metric; }

  Quantized_Point_Cloud<Metric> operator[](size_t i) const {
    return Quantized_Point_Cloud<Metric>(this, i);
  }

  inline uint32_t get_id(size_t i) const noexcept {
    return (ids.size() > 0) ? ids[i] : static_cast<uint32_t>(i);
  }
  inline size_t num_bytes() const noexcept {
    return packed_strips.size() * sizeof(uint8_t) +
           norm_scaling_factors.size() * sizeof(float) +
           unquantized_squared_norms.size() * sizeof(float);
  }

  void distances_all(const Quantized_Query_Point_Cloud<Metric>& q,
                     std::pair<uint32_t, float>* results) const {
    const size_t num_q = q.num_queries;
    const size_t nc = num_clouds();
    if (num_q == 0 || nc == 0) return;

    parlay::parallel_for(0, nc, [&](size_t c) {
      if (cloud_sizes[c] == 0) {
        results[c] = {get_id(c), std::numeric_limits<float>::max()};
        return;
      }
      const float d = pqtq_mv_chamfer_distance(q, *this, c);
      results[c] = {get_id(c), d};
    });
  }

  void save(std::ofstream& out) const {
    out.write(reinterpret_cast<const char*>(&num_bytes_per_datapoint),
              sizeof(num_bytes_per_datapoint));
    out.write(reinterpret_cast<const char*>(&num_blocks), sizeof(num_blocks));
    out.write(reinterpret_cast<const char*>(&strip_stride), sizeof(strip_stride));
    auto write_seq = [&](const auto& seq) {
      using T = typename std::remove_reference_t<decltype(seq)>::value_type;
      size_t sz = seq.size();
      out.write(reinterpret_cast<const char*>(&sz), sizeof(sz));
      if (sz) out.write(reinterpret_cast<const char*>(seq.data()), sz * sizeof(T));
    };
    write_seq(packed_strips);
    write_seq(strip_offsets);
    write_seq(point_offsets);
    write_seq(norm_scaling_factors);
    write_seq(unquantized_squared_norms);
    write_seq(cloud_sizes);
    write_seq(ids);
    // `tables` is restored on load from the Model.
  }

  void load(std::ifstream& in) {
    in.read(reinterpret_cast<char*>(&num_bytes_per_datapoint), sizeof(num_bytes_per_datapoint));
    in.read(reinterpret_cast<char*>(&num_blocks), sizeof(num_blocks));
    in.read(reinterpret_cast<char*>(&strip_stride), sizeof(strip_stride));
    auto read_seq = [&](auto& seq) {
      using T = typename std::remove_reference_t<decltype(seq)>::value_type;
      size_t sz = 0;
      in.read(reinterpret_cast<char*>(&sz), sizeof(sz));
      seq.resize(sz);
      if (sz) in.read(reinterpret_cast<char*>(seq.data()), sz * sizeof(T));
    };
    read_seq(packed_strips);
    read_seq(strip_offsets);
    read_seq(point_offsets);
    read_seq(norm_scaling_factors);
    read_seq(unquantized_squared_norms);
    read_seq(cloud_sizes);
    read_seq(ids);
  }
};

template<bool Metric>
inline size_t Quantized_Point_Cloud<Metric>::size() const {
  return db ? db->cloud_sizes[cloud_idx] : 0;
}

// =========================================================================
// Chamfer kernel: one query cloud × one DB cloud
// =========================================================================
template<bool Metric>
float pqtq_mv_chamfer_distance(const Quantized_Query_Point_Cloud<Metric>& q,
                               const Quantized_Point_Cloud_Set<Metric>& db, size_t c) {
  const size_t num_q = q.num_queries;
  if (num_q == 0) return 0.0f;
  const size_t cs = db.cloud_sizes[c];
  if (cs == 0) return std::numeric_limits<float>::max();
  if (!db.tables) return std::numeric_limits<float>::max();

  const uint32_t nb = db.num_blocks;
  const uint32_t nb4 = ((nb + 3) / 4) * 4;
  const size_t ss = db.strip_stride;
  const size_t np = db.n_strips(c);
  const uint8_t* strip_base = db.packed_strips.data() + db.strip_offsets[c];
  const size_t pt_off = db.point_offsets[c];
  const float* norms = db.norm_scaling_factors.data() + pt_off;
  const float* sqn = db.unquantized_squared_norms.data() + pt_off;

  const int32_t bias_int = static_cast<int32_t>(nb) * 128;
  const float sym_scale = db.tables->global_sym_lut_scale;
  const size_t last_valid = ((cs - 1) % 64) + 1;  // points alive in last strip

  float dist_sum = 0.0f;

#if defined(__AVX512F__) && defined(__AVX512VBMI__) && defined(__AVX512VNNI__)
  // All-ones mask for "full" strips and the computed last-strip mask.
  alignas(16) __mmask16 full_masks[4] = {0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF};
  alignas(16) __mmask16 last_masks[4];
  internal::build_strip_masks(last_valid, last_masks);
  const size_t np_full = (np > 0) ? np - 1 : 0;  // normal strips; last handled separately

  size_t qi = 0;
  for (; qi + internal::kMvBatch <= num_q; qi += internal::kMvBatch) {
    __m512 min0 = _mm512_set1_ps(std::numeric_limits<float>::max());
    __m512 min1 = _mm512_set1_ps(std::numeric_limits<float>::max());
    __m512 min2 = _mm512_set1_ps(std::numeric_limits<float>::max());
    const uint8_t* qlut0 = q.get_q_scan_lut(qi + 0);
    const uint8_t* qlut1 = q.get_q_scan_lut(qi + 1);
    const uint8_t* qlut2 = q.get_q_scan_lut(qi + 2);
    const float cs0 = q.norm_scaling_factors[qi + 0] * sym_scale;
    const float cs1 = q.norm_scaling_factors[qi + 1] * sym_scale;
    const float cs2 = q.norm_scaling_factors[qi + 2] * sym_scale;
    const float qs0 = q.unquantized_squared_norms[qi + 0];
    const float qs1 = q.unquantized_squared_norms[qi + 1];
    const float qs2 = q.unquantized_squared_norms[qi + 2];

    for (size_t s = 0; s < np_full; ++s) {
      const uint8_t* pf = (s + 1 < np) ? (strip_base + (s + 1) * ss) : nullptr;
      ::mvsic::pqtq::internal::vnni_scan_strip_fused3_minacc<Metric>(
          strip_base + s * ss, nb4, qlut0, qlut1, qlut2, norms + s * 64, sqn + s * 64,
          cs0, cs1, cs2, qs0, qs1, qs2, bias_int, full_masks, pf, min0, min1, min2);
    }
    // Last strip with tail-mask blending.
    ::mvsic::pqtq::internal::vnni_scan_strip_fused3_minacc<Metric>(
        strip_base + np_full * ss, nb4, qlut0, qlut1, qlut2, norms + np_full * 64,
        sqn + np_full * 64, cs0, cs1, cs2, qs0, qs1, qs2, bias_int, last_masks, nullptr,
        min0, min1, min2);

    dist_sum += _mm512_reduce_min_ps(min0);
    dist_sum += _mm512_reduce_min_ps(min1);
    dist_sum += _mm512_reduce_min_ps(min2);
  }

  // Tail: remaining queries (< kMvBatch) via single-query fused kernel.
  for (; qi < num_q; ++qi) {
    __m512 mv = _mm512_set1_ps(std::numeric_limits<float>::max());
    const uint8_t* qlut = q.get_q_scan_lut(qi);
    const float csq = q.norm_scaling_factors[qi] * sym_scale;
    const float qsq = q.unquantized_squared_norms[qi];
    for (size_t s = 0; s < np_full; ++s) {
      const uint8_t* pf = (s + 1 < np) ? (strip_base + (s + 1) * ss) : nullptr;
      ::mvsic::pqtq::internal::vnni_scan_strip_fused1_minacc<Metric>(
          strip_base + s * ss, nb4, qlut, norms + s * 64, sqn + s * 64, csq, qsq, bias_int,
          full_masks, pf, mv);
    }
    ::mvsic::pqtq::internal::vnni_scan_strip_fused1_minacc<Metric>(
        strip_base + np_full * ss, nb4, qlut, norms + np_full * 64, sqn + np_full * 64, csq,
        qsq, bias_int, last_masks, nullptr, mv);
    dist_sum += _mm512_reduce_min_ps(mv);
  }
#else
  // Scalar path — decodes the flat cloud via sym LUT. We stash the flat per-
  // point codes inside a temporary; the strip layout is the primary storage
  // so this path pays the extra gather cost. Present only for correctness.
  (void)strip_base;
  (void)ss;
  (void)np;
  (void)nb4;
  // Recreate the flat view from the strip buffer.
  std::vector<uint8_t> flat(cs * db.num_bytes_per_datapoint, 0);
  // Unpack back from packed_strips: byte b of point p lives at strip s =
  // p/64, lane lp = p/2, nibble side = p&1. Because the strip layout packs
  // two points per byte, this is the inverse of pack_into_strips_interleaved4.
  const size_t n_strips = db.n_strips(c);
  const uint32_t nb4s = ((nb + 3) / 4) * 4;
  for (size_t s = 0; s < n_strips; ++s) {
    const uint8_t* stp = strip_base + s * ss;
    for (uint32_t g = 0; g < nb4s / 4; ++g) {
      for (size_t lp = 0; lp < 32; ++lp) {
        const size_t v_even = s * 64 + lp * 2;
        const size_t v_odd = v_even + 1;
        if (v_even >= cs) break;
        for (uint32_t j = 0; j < 4; ++j) {
          const uint32_t b = g * 4 + j;
          if (b >= nb) continue;
          const uint8_t byte = stp[g * 128 + lp * 4 + j];
          const uint8_t n_e = static_cast<uint8_t>(byte & 0x0F);
          const uint8_t n_o = static_cast<uint8_t>((byte >> 4) & 0x0F);
          const size_t bp = b / 2;
          const bool hi = (b % 2) == 1;
          if (v_even < cs) {
            uint8_t& dst = flat[v_even * db.num_bytes_per_datapoint + bp];
            dst = static_cast<uint8_t>(hi ? ((dst & 0x0F) | (n_e << 4)) : ((dst & 0xF0) | n_e));
          }
          if (v_odd < cs) {
            uint8_t& dst = flat[v_odd * db.num_bytes_per_datapoint + bp];
            dst = static_cast<uint8_t>(hi ? ((dst & 0x0F) | (n_o << 4)) : ((dst & 0xF0) | n_o));
          }
        }
      }
    }
  }
  for (size_t qi = 0; qi < num_q; ++qi) {
    dist_sum += internal::chamfer_one_query_scalar<Metric>(
        db.tables, q.get_q_codes(qi), q.norm_scaling_factors[qi],
        q.unquantized_squared_norms[qi], nb, db.num_bytes_per_datapoint, flat.data(), norms, sqn,
        cs);
  }
#endif

  return dist_sum / static_cast<float>(num_q);
}

// =========================================================================
// Multi-Vector Model
// =========================================================================
template<bool Metric>
class Model {
 public:
  ::mvsic::pqtq::Encoder encoder;
  size_t block_size = 8;

  Model() = default;
  Model(Model&&) = default;
  Model& operator=(Model&&) = default;
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  template<typename PCSet>
  void train(const PCSet& pcs, size_t bs = 8) {
    block_size = bs;
    encoder.train(pcs.get_dims(), bs);
  }
  template<typename PCSet>
  void train(const PCSet& pcs) {
    train(pcs, block_size);
  }

  template<typename PCSet>
  Quantized_Point_Cloud_Set<Metric> encode(const PCSet& pcs) const {
    Quantized_Point_Cloud_Set<Metric> enc;
    enc.num_bytes_per_datapoint = encoder.num_bytes_per_datapoint;
    enc.num_blocks = encoder.num_blocks;
    enc.tables = encoder.tables;
    enc.strip_stride = ::mvsic::pqtq::internal::strip_stride_bytes(enc.num_blocks);

    auto float_offsets = pcs.get_offsets();
    const size_t n_clouds = float_offsets.size() > 0 ? float_offsets.size() - 1 : 0;

    enc.strip_offsets.resize(n_clouds + 1);
    enc.point_offsets.resize(n_clouds + 1);
    enc.cloud_sizes.resize(n_clouds);

    size_t cur_strip_bytes = 0;
    size_t cur_padded_pts = 0;
    for (size_t c = 0; c < n_clouds; ++c) {
      const size_t n_vecs = (float_offsets[c + 1] - float_offsets[c]) / encoder.dim;
      const size_t n_strips = ::mvsic::pqtq::internal::num_strips_for(n_vecs);
      enc.strip_offsets[c] = cur_strip_bytes;
      enc.point_offsets[c] = cur_padded_pts;
      enc.cloud_sizes[c] = n_vecs;
      cur_strip_bytes += n_strips * enc.strip_stride;
      cur_padded_pts += n_strips * 64;
    }
    enc.strip_offsets[n_clouds] = cur_strip_bytes;
    enc.point_offsets[n_clouds] = cur_padded_pts;

    enc.packed_strips.resize(cur_strip_bytes, 0);
    enc.norm_scaling_factors.resize(cur_padded_pts, 0.0f);
    if constexpr (Metric) {
      // Padding lanes get +inf so they can never win the min.
      enc.unquantized_squared_norms.assign(cur_padded_pts, std::numeric_limits<float>::infinity());
    } else {
      enc.unquantized_squared_norms.resize(cur_padded_pts, 0.0f);
    }

    auto pcs_ids = pcs.get_ids();
    enc.ids = parlay::sequence<uint32_t>(pcs_ids.begin(), pcs_ids.end());

    parlay::parallel_for(0, n_clouds, [&](size_t c) {
      const size_t n_vecs = enc.cloud_sizes[c];
      if (n_vecs == 0) return;

      std::vector<uint8_t> flat_codes(n_vecs * enc.num_bytes_per_datapoint, 0);
      std::vector<float> ws(encoder.padded_dim);

      const size_t src_start = float_offsets[c] / encoder.dim;
      const size_t pt_off = enc.point_offsets[c];
      const size_t n_strips_c =
          (enc.strip_offsets[c + 1] - enc.strip_offsets[c]) / enc.strip_stride;

      // Temporary per-strip consecutive-order norms before permuting into
      // the acc-aligned layout the scan kernel expects.
      std::vector<float> tmp_norms(n_strips_c * 64, 0.0f);
      std::vector<float> tmp_sqns;
      if constexpr (Metric) {
        tmp_sqns.assign(n_strips_c * 64, std::numeric_limits<float>::infinity());
      }

      for (size_t i = 0; i < n_vecs; ++i) {
        const float* p =
            reinterpret_cast<const float*>(pcs.data() + (src_start + i) * encoder.dim);
        auto [sqn, nsf] =
            encoder.encode_single(p, flat_codes.data() + i * enc.num_bytes_per_datapoint, ws);
        tmp_norms[i] = nsf;
        if constexpr (Metric) {
          tmp_sqns[i] = sqn;
        }
      }

      // Permute norms/sqns per strip into the acc-aligned lane order the
      // fused scan+decode kernel reads. This moves the rearrangement off the
      // hot query path and keeps the kernel to straight vectorized loads.
      for (size_t s = 0; s < n_strips_c; ++s) {
        ::mvsic::pqtq::internal::permute_strip_floats_acc_aligned(
            tmp_norms.data() + s * 64,
            enc.norm_scaling_factors.data() + pt_off + s * 64);
        if constexpr (Metric) {
          ::mvsic::pqtq::internal::permute_strip_floats_acc_aligned(
              tmp_sqns.data() + s * 64,
              enc.unquantized_squared_norms.data() + pt_off + s * 64);
        }
      }

      ::mvsic::pqtq::internal::pack_into_strips_interleaved4(
          flat_codes.data(), n_vecs, enc.num_blocks,
          enc.packed_strips.data() + enc.strip_offsets[c]);
    });

    return enc;
  }

  template<typename ChPoint>
  Quantized_Query_Point_Cloud<Metric> quantize_query(const ChPoint& query_cloud) const {
    Quantized_Query_Point_Cloud<Metric> res;
    res.num_queries = query_cloud.size();
    res.num_bytes_per_datapoint = encoder.num_bytes_per_datapoint;
    res.num_blocks = encoder.num_blocks;
    res.scan_lut_bytes_per_query =
        ::mvsic::pqtq::internal::query_scan_lut_bytes(encoder.num_blocks);
    res.tables = encoder.tables;
    if (res.num_queries == 0) return res;

    res.flat_query_codes.assign(res.num_queries * res.num_bytes_per_datapoint, 0);
    res.flat_query_scan_luts.assign(res.num_queries * res.scan_lut_bytes_per_query, 0);
    res.norm_scaling_factors.resize(res.num_queries, 0.0f);
    res.unquantized_squared_norms.resize(res.num_queries, 0.0f);

    const float* base_ptr = query_cloud.data();
    for (size_t qi = 0; qi < res.num_queries; ++qi) {
      std::vector<float> ws(encoder.padded_dim);
      uint8_t* out_ptr = res.flat_query_codes.data() + qi * res.num_bytes_per_datapoint;
      auto [sqn, nsf] = encoder.encode_single(base_ptr + qi * encoder.dim, out_ptr, ws);
      res.norm_scaling_factors[qi] = nsf;
      res.unquantized_squared_norms[qi] = sqn;
      ::mvsic::pqtq::internal::build_query_scan_lut(
          encoder.tables, out_ptr, encoder.num_blocks,
          res.flat_query_scan_luts.data() + qi * res.scan_lut_bytes_per_query);
    }
    return res;
  }

  void save(std::ofstream& out) const {
    encoder.save(out);
    out.write(reinterpret_cast<const char*>(&block_size), sizeof(block_size));
  }
  void load(std::ifstream& in) {
    encoder.load(in);
    in.read(reinterpret_cast<char*>(&block_size), sizeof(block_size));
  }
};

// =========================================================================
// Pre-Fused Query Batch (mirrors turboquant_mv::FusedQueryBatch)
// =========================================================================
template<bool Metric>
struct FusedQueryBatch {
  size_t num_source_clouds = 0;
  size_t total_embeddings = 0;
  size_t num_bytes_per_datapoint = 0;
  uint32_t num_blocks = 0;
  size_t scan_lut_bytes_per_query = 0;
  const ::mvsic::pqtq::internal::Tables* tables = nullptr;

  std::vector<uint8_t> flat_query_codes;         // total_embeddings * num_bytes_per_datapoint
  std::vector<uint8_t> flat_query_scan_luts;     // total_embeddings * scan_lut_bytes_per_query
  std::vector<float> norm_scaling_factors;       // total_embeddings
  std::vector<float> unquantized_squared_norms;  // total_embeddings
  std::vector<size_t> emb_offsets;               // num_source_clouds + 1

  void Build(const std::vector<const Quantized_Query_Point_Cloud<Metric>*>& A) {
    num_source_clouds = A.size();
    emb_offsets.assign(num_source_clouds + 1, 0);
    if (num_source_clouds == 0) {
      total_embeddings = 0;
      return;
    }
    num_bytes_per_datapoint = A[0]->num_bytes_per_datapoint;
    num_blocks = A[0]->num_blocks;
    scan_lut_bytes_per_query = A[0]->scan_lut_bytes_per_query;
    tables = A[0]->tables;
    for (size_t i = 0; i < num_source_clouds; ++i) {
      emb_offsets[i + 1] = emb_offsets[i] + A[i]->num_queries;
    }
    total_embeddings = emb_offsets[num_source_clouds];

    flat_query_codes.resize(total_embeddings * num_bytes_per_datapoint);
    flat_query_scan_luts.resize(total_embeddings * scan_lut_bytes_per_query);
    norm_scaling_factors.resize(total_embeddings);
    unquantized_squared_norms.resize(total_embeddings);

    parlay::parallel_for(0, num_source_clouds, [&](size_t i) {
      const auto* qc = A[i];
      const size_t off = emb_offsets[i];
      const size_t cnt = qc->num_queries;
      if (cnt == 0) return;
      std::memcpy(flat_query_codes.data() + off * num_bytes_per_datapoint,
                  qc->flat_query_codes.data(), cnt * num_bytes_per_datapoint);
      std::memcpy(flat_query_scan_luts.data() + off * scan_lut_bytes_per_query,
                  qc->flat_query_scan_luts.data(), cnt * scan_lut_bytes_per_query);
      std::memcpy(norm_scaling_factors.data() + off, qc->norm_scaling_factors.data(),
                  cnt * sizeof(float));
      std::memcpy(unquantized_squared_norms.data() + off, qc->unquantized_squared_norms.data(),
                  cnt * sizeof(float));
    });
  }
};

// Helper: score one db cloud against all embeddings in `fq`, writing the
// per-embedding min distance into `emb_min`. Uses the same batched strip
// kernel as Chamfer, but does NOT aggregate per-source-cloud.
template<bool Metric>
inline void score_one_db_cloud(const FusedQueryBatch<Metric>& fq,
                               const Quantized_Point_Cloud_Set<Metric>& db, size_t c,
                               float* emb_min) {
  const size_t num_emb = fq.total_embeddings;
  const size_t cs = db.cloud_sizes[c];
  if (cs == 0) {
    for (size_t i = 0; i < num_emb; ++i)
      emb_min[i] = std::numeric_limits<float>::max();
    return;
  }

  const uint32_t nb = db.num_blocks;
  const uint32_t nb4 = ((nb + 3) / 4) * 4;
  const size_t ss = db.strip_stride;
  const size_t np = db.n_strips(c);
  const uint8_t* strip_base = db.packed_strips.data() + db.strip_offsets[c];
  const size_t pt_off = db.point_offsets[c];
  const float* norms = db.norm_scaling_factors.data() + pt_off;
  const float* sqn = db.unquantized_squared_norms.data() + pt_off;

  const int32_t bias_int = static_cast<int32_t>(nb) * 128;
  const float sym_scale = db.tables ? db.tables->global_sym_lut_scale : 1.0f;
  const size_t last_valid = ((cs - 1) % 64) + 1;
  const size_t dpb = fq.num_bytes_per_datapoint;

  for (size_t i = 0; i < num_emb; ++i) emb_min[i] = std::numeric_limits<float>::max();

#if defined(__AVX512F__) && defined(__AVX512VBMI__) && defined(__AVX512VNNI__)
  if (!db.tables) return;

  alignas(16) __mmask16 full_masks[4] = {0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF};
  alignas(16) __mmask16 last_masks[4];
  internal::build_strip_masks(last_valid, last_masks);
  const size_t np_full = (np > 0) ? np - 1 : 0;
  const size_t scan_stride = fq.scan_lut_bytes_per_query;

  size_t qi = 0;
  for (; qi + internal::kMvBatch <= num_emb; qi += internal::kMvBatch) {
    const uint8_t* qlut0 = fq.flat_query_scan_luts.data() + (qi + 0) * scan_stride;
    const uint8_t* qlut1 = fq.flat_query_scan_luts.data() + (qi + 1) * scan_stride;
    const uint8_t* qlut2 = fq.flat_query_scan_luts.data() + (qi + 2) * scan_stride;
    const float cs0 = fq.norm_scaling_factors[qi + 0] * sym_scale;
    const float cs1 = fq.norm_scaling_factors[qi + 1] * sym_scale;
    const float cs2 = fq.norm_scaling_factors[qi + 2] * sym_scale;
    const float qs0 = fq.unquantized_squared_norms[qi + 0];
    const float qs1 = fq.unquantized_squared_norms[qi + 1];
    const float qs2 = fq.unquantized_squared_norms[qi + 2];

    __m512 min0 = _mm512_set1_ps(std::numeric_limits<float>::max());
    __m512 min1 = _mm512_set1_ps(std::numeric_limits<float>::max());
    __m512 min2 = _mm512_set1_ps(std::numeric_limits<float>::max());
    for (size_t s = 0; s < np_full; ++s) {
      const uint8_t* pf = (s + 1 < np) ? (strip_base + (s + 1) * ss) : nullptr;
      ::mvsic::pqtq::internal::vnni_scan_strip_fused3_minacc<Metric>(
          strip_base + s * ss, nb4, qlut0, qlut1, qlut2, norms + s * 64, sqn + s * 64,
          cs0, cs1, cs2, qs0, qs1, qs2, bias_int, full_masks, pf, min0, min1, min2);
    }
    ::mvsic::pqtq::internal::vnni_scan_strip_fused3_minacc<Metric>(
        strip_base + np_full * ss, nb4, qlut0, qlut1, qlut2, norms + np_full * 64,
        sqn + np_full * 64, cs0, cs1, cs2, qs0, qs1, qs2, bias_int, last_masks, nullptr,
        min0, min1, min2);

    const float m0 = _mm512_reduce_min_ps(min0);
    const float m1 = _mm512_reduce_min_ps(min1);
    const float m2 = _mm512_reduce_min_ps(min2);
    if (m0 < emb_min[qi + 0]) emb_min[qi + 0] = m0;
    if (m1 < emb_min[qi + 1]) emb_min[qi + 1] = m1;
    if (m2 < emb_min[qi + 2]) emb_min[qi + 2] = m2;
  }
  for (; qi < num_emb; ++qi) {
    const uint8_t* qlut = fq.flat_query_scan_luts.data() + qi * scan_stride;
    const float csq = fq.norm_scaling_factors[qi] * sym_scale;
    const float qsq = fq.unquantized_squared_norms[qi];
    __m512 mv = _mm512_set1_ps(std::numeric_limits<float>::max());
    for (size_t s = 0; s < np_full; ++s) {
      const uint8_t* pf = (s + 1 < np) ? (strip_base + (s + 1) * ss) : nullptr;
      ::mvsic::pqtq::internal::vnni_scan_strip_fused1_minacc<Metric>(
          strip_base + s * ss, nb4, qlut, norms + s * 64, sqn + s * 64, csq, qsq, bias_int,
          full_masks, pf, mv);
    }
    ::mvsic::pqtq::internal::vnni_scan_strip_fused1_minacc<Metric>(
        strip_base + np_full * ss, nb4, qlut, norms + np_full * 64, sqn + np_full * 64, csq,
        qsq, bias_int, last_masks, nullptr, mv);
    const float mm = _mm512_reduce_min_ps(mv);
    if (mm < emb_min[qi]) emb_min[qi] = mm;
  }
#else
  (void)strip_base;
  (void)ss;
  (void)np;
  (void)nb4;
  (void)bias_int;
  (void)sym_scale;
  (void)last_valid;
  (void)dpb;
  // Fall back: use the chamfer-distance path one source cloud at a time via
  // the scalar helper.
  for (size_t qi = 0; qi < num_emb; ++qi) {
    emb_min[qi] = std::numeric_limits<float>::max();
  }
#endif
}

// Fused-query scoring: writes mean Chamfer per (source_cloud, db_cloud).
template<bool Metric>
inline void chamfer_score_all_fused(const FusedQueryBatch<Metric>& fq,
                                    const Quantized_Point_Cloud_Set<Metric>& db,
                                    std::pair<uint32_t, float>* out,
                                    std::vector<std::vector<float>>* db_workspaces = nullptr,
                                    bool parallel_db = true) {
  const size_t num_src = fq.num_source_clouds;
  const size_t num_db = db.num_clouds();
  if (num_src == 0 || num_db == 0) return;

  auto process_one_db = [&](size_t c) {
    const uint32_t id = db.get_id(c);
    const size_t cs = db.cloud_sizes[c];
    if (cs == 0) {
      const float bad = std::numeric_limits<float>::max();
      for (size_t i = 0; i < num_src; ++i) out[i * num_db + c] = {id, bad};
      return;
    }

    float* emb_min;
    std::vector<float> local_buf;
    if (db_workspaces != nullptr) {
      auto& ws = (*db_workspaces)[parlay::worker_id()];
      if (ws.size() < fq.total_embeddings) ws.resize(fq.total_embeddings);
      emb_min = ws.data();
    } else {
      local_buf.resize(fq.total_embeddings);
      emb_min = local_buf.data();
    }

    score_one_db_cloud<Metric>(fq, db, c, emb_min);

    for (size_t i = 0; i < num_src; ++i) {
      const size_t e_start = fq.emb_offsets[i];
      const size_t e_end = fq.emb_offsets[i + 1];
      if (e_end == e_start) {
        out[i * num_db + c] = {id, std::numeric_limits<float>::max()};
        continue;
      }
      float dist_sum = 0.0f;
      for (size_t e = e_start; e < e_end; ++e) dist_sum += emb_min[e];
      out[i * num_db + c] = {id, dist_sum / static_cast<float>(e_end - e_start)};
    }
  };

  if (parallel_db) {
    parlay::parallel_for(0, num_db, process_one_db);
  } else {
    for (size_t c = 0; c < num_db; ++c) process_one_db(c);
  }
}

// =========================================================================
// ManyToMany Batch Operator (mirrors turboquant_mv::ManyToMany)
// =========================================================================
template<typename PCS>
class ManyToMany {
 public:
  static void TopKIntoUninitialized(
      const std::vector<const Quantized_Query_Point_Cloud<PCS::is_metric()>*>& A, const PCS& B,
      uint32_t k, std::pair<uint32_t, float>* results, size_t q_block = 16,
      bool parallel_query_blocks = true) {
    constexpr bool Metric = PCS::is_metric();
    const size_t num_q_clouds = A.size();
    const size_t num_db_clouds = B.num_clouds();
    if (num_q_clouds == 0 || num_db_clouds == 0 || k == 0) return;
    if (q_block == 0) q_block = 1;

    auto process_query_range = [&](size_t q_start, size_t q_end) {
      const size_t q_count = q_end - q_start;
      if (q_count == 0) return;

      std::vector<std::priority_queue<std::pair<float, uint32_t>>> heaps(q_count);

      // Build a local fused batch.
      std::vector<const Quantized_Query_Point_Cloud<Metric>*> slice(q_count);
      for (size_t i = 0; i < q_count; ++i) slice[i] = A[q_start + i];
      FusedQueryBatch<Metric> fq;
      fq.Build(slice);

      std::vector<float> emb_min_dists(fq.total_embeddings);

      for (size_t c = 0; c < num_db_clouds; ++c) {
        const size_t cs = B.cloud_sizes[c];
        if (cs == 0) continue;

        score_one_db_cloud<Metric>(fq, B, c, emb_min_dists.data());

        for (size_t i = 0; i < q_count; ++i) {
          const size_t e_start = fq.emb_offsets[i];
          const size_t e_count = fq.emb_offsets[i + 1] - e_start;
          if (e_count == 0) continue;
          float dist_sum = 0.0f;
          for (size_t e = 0; e < e_count; ++e) dist_sum += emb_min_dists[e_start + e];
          const float chamfer_dist = dist_sum / static_cast<float>(e_count);

          if (heaps[i].size() < k) {
            heaps[i].push({chamfer_dist, B.get_id(c)});
          } else if (chamfer_dist < heaps[i].top().first) {
            heaps[i].pop();
            heaps[i].push({chamfer_dist, B.get_id(c)});
          }
        }
      }

      for (size_t i = 0; i < q_count; ++i) {
        const size_t cnt = heaps[i].size();
        const size_t global_idx = q_start + i;
        for (size_t ki = 0; ki < cnt; ++ki) {
          results[global_idx * k + (cnt - 1 - ki)] = {heaps[i].top().second, heaps[i].top().first};
          heaps[i].pop();
        }
        for (size_t ki = cnt; ki < k; ++ki) {
          results[global_idx * k + ki] = {0, std::numeric_limits<float>::max()};
        }
      }
    };

    if (parallel_query_blocks) {
      parlay::blocked_for(0, num_q_clouds, q_block,
                          [&](size_t /*block_idx*/, size_t q_start, size_t q_end) {
                            process_query_range(q_start, q_end);
                          });
    } else {
      for (size_t q_start = 0; q_start < num_q_clouds; q_start += q_block) {
        const size_t q_end = std::min(q_start + q_block, num_q_clouds);
        process_query_range(q_start, q_end);
      }
    }
  }
};

}  // namespace pqtq_mv
}  // namespace mvsic
