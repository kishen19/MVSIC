#pragma once

#include <vector>
#include <cstdint>
#include <limits>
#include <algorithm>
#include <cstring>
#include <random>
#include <iostream>
#include <fstream>
#include <Eigen/Core>

#include "parlay/parallel.h"
#include "parlay/sequence.h"

#include "mvsic/core/utils/kmeans_util.h"

namespace mvsic {
namespace pq_mv {

// =========================================================================
// Forward Declarations & Proxy Objects
// =========================================================================
template<bool Metric>
class Quantized_Point_Cloud_Set;

template<bool Metric>
class Quantized_Point_Cloud {
 public:
  const Quantized_Point_Cloud_Set<Metric>* db = nullptr;
  size_t start_idx = 0;
  size_t end_idx = 0;

  Quantized_Point_Cloud() = default;
  Quantized_Point_Cloud(const Quantized_Point_Cloud_Set<Metric>* d, size_t s, size_t e) :
      db(d), start_idx(s), end_idx(e) {}

  size_t size() const { return end_idx - start_idx; }

  static constexpr bool is_metric() { return Metric; }

  // Required by beam_search
  template<typename Query>
  bool same_as(const Query&) const {
    return false;
  }
};

// =========================================================================
// Multi-Vector Query (Struct of Arrays / Contiguous Layout)
// =========================================================================

template<bool Metric>
class Quantized_Query_Point_Cloud {
 public:
  using distanceType = float;
  size_t num_queries = 0;
  uint32_t num_blocks = 0;
  uint32_t K = 0;  // Usually 256 for standard PQ

  // Flat memory arrays for all queries
  std::vector<uint8_t> flat_int_luts;  // Size: num_queries * num_blocks * K
  std::vector<float> min_dists;        // Size: num_queries
  std::vector<float> scales;           // Size: num_queries

  Quantized_Query_Point_Cloud() = default;

  inline const uint8_t* get_lut(size_t qi) const {
    return flat_int_luts.data() + qi * num_blocks * K;
  }

  inline float decode(size_t qi, uint32_t int_dist) const {
    return (min_dists[qi] * static_cast<float>(num_blocks)) +
           (static_cast<float>(int_dist) * scales[qi]);
  }

  template<typename QuantizedPointCloud>
  float distance(const QuantizedPointCloud& cloud) const {
    const size_t start = cloud.start_idx;
    const size_t end = cloud.end_idx;
    const size_t cloud_size = end - start;
    if (cloud_size == 0) {
      return std::numeric_limits<float>::max();
    }

    float total_chamfer = 0.0f;
    const uint32_t m = num_blocks;
    const uint32_t k_size = K;
    const auto* db = cloud.db;
    for (size_t qi = 0; qi < num_queries; ++qi) {
      const uint8_t* q_lut = get_lut(qi);
      uint32_t min_dist_raw = 0xFFFFFFFF;
      for (size_t v = start; v < end; ++v) {
        const uint8_t* code = db->packed_codes.data() + v * m;
        uint32_t acc = 0;
#pragma GCC unroll 16
        for (uint32_t b = 0; b < m; ++b) {
          acc += static_cast<uint32_t>(q_lut[b * k_size + code[b]]);
        }

        if (acc < min_dist_raw) {
          min_dist_raw = acc;
        }
      }
      total_chamfer += decode(qi, min_dist_raw);
    }
    return total_chamfer / static_cast<float>(num_queries);
  }

  template<typename CloudHandle>
  std::pair<float, size_t> distance_w_cmps(const CloudHandle& cloud) const {
    const size_t cloud_size = cloud.end_idx - cloud.start_idx;
    const size_t bytes = cloud_size * static_cast<size_t>(cloud.db->num_blocks) * sizeof(uint8_t);
    return {this->distance(cloud), bytes};
  }

  static constexpr bool is_metric() { return Metric; }
};

// =========================================================================
// Multi-Vector Set
// =========================================================================
template<bool Metric>
class Quantized_Point_Cloud_Set {
 public:
  parlay::sequence<uint8_t> packed_codes;  // Size: total_vecs * num_blocks
  uint32_t num_blocks = 0;
  uint32_t K = 0;

  parlay::sequence<size_t> offsets;  // Flat vector offsets (tightly packed)
  parlay::sequence<uint32_t> ids;

  Quantized_Point_Cloud_Set() = default;

  Quantized_Point_Cloud<Metric> operator[](size_t i) const {
    const size_t start = offsets[i];
    size_t end = offsets[i + 1];
    return Quantized_Point_Cloud<Metric>(this, start, end);
  }

  inline uint32_t get_id(size_t i) const noexcept { return (ids.size() > 0) ? ids[i] : i; }
  inline size_t num_bytes() const noexcept { return packed_codes.size() * sizeof(uint8_t); }

  void distances_all(const Quantized_Query_Point_Cloud<Metric>& q,
                     std::pair<uint32_t, float>* results) const {
    const size_t num_q = q.num_queries;
    const size_t n_clouds = (offsets.size() > 0) ? offsets.size() - 1 : 0;
    if (num_q == 0 || n_clouds == 0) return;
    parlay::parallel_for(0, n_clouds, [&](size_t cid) {
      const uint32_t cloud_id = static_cast<uint32_t>(cid);
      const size_t start = offsets[cloud_id];
      const size_t end = offsets[cloud_id + 1];
      const size_t cloud_size = end - start;
      if (cloud_size == 0) {
        results[cid] = {get_id(cloud_id), std::numeric_limits<float>::max()};
        return;
      }

      float total_chamfer = 0.0f;
      const uint32_t m = num_blocks;
      const uint32_t k_size = K;
      // Loop inversion: query on the outside, cloud vectors on the inside.
      // Since memory is tightly packed, we just scan straight through the array.
      for (size_t qi = 0; qi < num_q; ++qi) {
        const uint8_t* q_lut = q.get_lut(qi);
        uint32_t min_dist_raw = 0xFFFFFFFF;  // Max uint32_t
        for (size_t v = start; v < end; ++v) {
          const uint8_t* code = packed_codes.data() + v * m;
          uint32_t acc = 0;
// Scalar loop: look up the distance from the pre-computed query LUT
#pragma GCC unroll 16
          for (uint32_t b = 0; b < m; ++b) {
            acc += static_cast<uint32_t>(q_lut[b * k_size + code[b]]);
          }
          if (acc < min_dist_raw) {
            min_dist_raw = acc;
          }
        }
        total_chamfer += q.decode(qi, min_dist_raw);
      }
      results[cid] = {get_id(cloud_id), total_chamfer / static_cast<float>(num_q)};
    });
  }

  void save(std::ofstream& out) const {
    out.write(reinterpret_cast<const char*>(&num_blocks), sizeof(num_blocks));
    out.write(reinterpret_cast<const char*>(&K), sizeof(K));
    size_t pc_sz = packed_codes.size();
    out.write(reinterpret_cast<const char*>(&pc_sz), sizeof(pc_sz));
    if (pc_sz)
      out.write(reinterpret_cast<const char*>(packed_codes.data()), pc_sz * sizeof(uint8_t));
    size_t off_size = offsets.size();
    out.write(reinterpret_cast<const char*>(&off_size), sizeof(off_size));
    if (off_size)
      out.write(reinterpret_cast<const char*>(offsets.data()), off_size * sizeof(size_t));
  }

  void load(std::ifstream& in) {
    in.read(reinterpret_cast<char*>(&num_blocks), sizeof(num_blocks));
    in.read(reinterpret_cast<char*>(&K), sizeof(K));
    size_t pc_sz = 0;
    in.read(reinterpret_cast<char*>(&pc_sz), sizeof(pc_sz));
    packed_codes.resize(pc_sz);
    if (pc_sz) in.read(reinterpret_cast<char*>(packed_codes.data()), pc_sz * sizeof(uint8_t));
    size_t off_size = 0;
    in.read(reinterpret_cast<char*>(&off_size), sizeof(off_size));
    offsets.resize(off_size);
    if (off_size) in.read(reinterpret_cast<char*>(offsets.data()), off_size * sizeof(size_t));
  }
};

// =========================================================================
// Multi-Vector Model
// =========================================================================
template<bool Metric>
class Model {
 public:
  // Compile-time identity for save-format dispatch, CSV / log strings, and the
  // query-compression batch width (replaces qc_internal::batch_alignment).
  static constexpr uint32_t kClassId = 1;  // QuantizerTag::kPQ
  static constexpr uint32_t kBatchAlignment = 1;
  static constexpr const char* kName = "pq";
  // Convenience aliases so index classes can write `LeafModel::EncodedSet`.
  // NOTE: intentionally NOT named `Quantized_Point_Cloud_Set` to avoid
  // shadowing the outer-namespace template inside the class body.
  using EncodedSet = ::mvsic::pq_mv::Quantized_Point_Cloud_Set<Metric>;
  using EncodedQuery = ::mvsic::pq_mv::Quantized_Query_Point_Cloud<Metric>;
  // Typed hyper-parameter struct passed to the templated index's constructor;
  // avoids bloating IndexParams with quantizer-specific fields.
  struct Params {
    uint32_t block_size = 8;
    uint32_t num_clusters_per_block = 256;
    uint32_t num_points_per_cluster = 20;
  };

  uint32_t num_blocks = 0;              // m
  uint32_t num_clusters_per_block = 0;  // K
  uint32_t num_points_per_cluster = 0;  // subsample_mult
  size_t dim = 0;
  size_t dim_per_block = 0;
  std::vector<Eigen::MatrixXf> codebooks;
  std::vector<Eigen::VectorXf> codebook_norms;

  Model() = default;

  template<typename PCSet>
  Model(const PCSet& train_data, uint32_t block_size = 8, uint32_t k = 256,
        uint32_t subsample_mult = 20) {
    train(train_data, block_size, k, subsample_mult);
  }

  template<typename PCSet>
  void train(const PCSet& pcs, const Params& p) {
    train(pcs, p.block_size, p.num_clusters_per_block, p.num_points_per_cluster);
  }

  template<typename PCSet>
  void train(const PCSet& pcs, uint32_t block_size = 8, uint32_t k = 256,
             uint32_t subsample_mult = 20) {
    num_clusters_per_block = k;
    num_points_per_cluster = subsample_mult;
    dim = pcs.get_dims();
    dim_per_block = block_size;
    if (num_clusters_per_block > 256) {
      std::cerr << "Error: PQ K=" << num_clusters_per_block
                << " > 256 not supported with uint8 codes.\n";
      abort();
    }
    if (dim_per_block == 0 || (dim % dim_per_block) != 0) {
      std::cerr << "Error: PQ Dimension " << dim << " not divisible by block_size=" << dim_per_block
                << "\n";
      abort();
    }

    num_blocks = static_cast<uint32_t>(dim / dim_per_block);
    codebooks.resize(num_blocks);
    codebook_norms.resize(num_blocks);
    const size_t n_points = pcs.total_size();
    const size_t sample_size =
        std::min(static_cast<size_t>(num_clusters_per_block) * num_points_per_cluster, n_points);
    parlay::parallel_for(0, num_blocks, [&](size_t b) {
      const size_t offset = b * dim_per_block;
      parlay::sequence<parlay::sequence<float>> sub(sample_size);
      std::mt19937 rng(static_cast<unsigned>(b + 1));
      std::uniform_int_distribution<size_t> distu(0, n_points - 1);
      for (size_t i = 0; i < sample_size; ++i) {
        const float* raw = reinterpret_cast<const float*>(pcs.data() + distu(rng) * dim);
        sub[i] = parlay::sequence<float>(raw + offset, raw + offset + dim_per_block);
      }
      // Explicitly true for L2 kmeans, bounding the Inner Product error optimally.
      auto [centers, _] = mvsic::kmeans_subsample_assign_only<true>(sub, num_clusters_per_block,
                                                                    sample_size, false);
      codebooks[b] = Eigen::MatrixXf(static_cast<Eigen::Index>(num_clusters_per_block),
                                     static_cast<Eigen::Index>(dim_per_block));
      for (size_t c = 0; c < centers.size(); ++c) {
        for (size_t d = 0; d < dim_per_block; ++d) {
          codebooks[b](static_cast<Eigen::Index>(c), static_cast<Eigen::Index>(d)) = centers[c][d];
        }
      }
      // pad any unused centers with 0s
      for (size_t c = centers.size(); c < num_clusters_per_block; ++c) {
        codebooks[b].row(static_cast<Eigen::Index>(c)).setZero();
      }
      codebook_norms[b] = codebooks[b].rowwise().squaredNorm();
    });
  }

  template<typename VecLocPtr>
  uint8_t find_best(const VecLocPtr vec_ptr, uint32_t b) const {
    const float* raw = reinterpret_cast<const float*>(vec_ptr);
    const float* sub_ptr = raw + b * dim_per_block;
    Eigen::Map<const Eigen::VectorXf> q_sub(sub_ptr, static_cast<Eigen::Index>(dim_per_block));
    Eigen::VectorXf dots = codebooks[b] * q_sub;
    float best_val = std::numeric_limits<float>::max();
    uint8_t best_code = 0;
    const auto& c_sq_norms = codebook_norms[b];
    for (uint32_t c = 0; c < num_clusters_per_block; ++c) {
      float val;
      if constexpr (Metric) {
        val = c_sq_norms[static_cast<Eigen::Index>(c)] - 2.0f * dots[static_cast<Eigen::Index>(c)];
      } else {
        val = -dots[static_cast<Eigen::Index>(c)];
      }
      if (val < best_val) {
        best_val = val;
        best_code = static_cast<uint8_t>(c);
      }
    }
    return best_code;
  }

  template<typename PCSet>
  Quantized_Point_Cloud_Set<Metric> encode(const PCSet& pcs) const {
    Quantized_Point_Cloud_Set<Metric> res;
    res.num_blocks = num_blocks;
    res.K = num_clusters_per_block;
    auto float_offsets = pcs.get_offsets();
    size_t n_clouds = float_offsets.size() > 0 ? float_offsets.size() - 1 : 0;
    // Tight Packing: Just map float dimensions directly to vector offsets
    res.offsets.resize(n_clouds + 1);
    for (size_t c = 0; c <= n_clouds; ++c) {
      res.offsets[c] = float_offsets[c] / dim;
    }
    size_t total_vecs = res.offsets.back();
    res.packed_codes.resize(total_vecs * static_cast<size_t>(num_blocks));
    auto pcs_ids = pcs.get_ids();
    res.ids = parlay::sequence<uint32_t>(pcs_ids.begin(), pcs_ids.end());
    // Encode all vectors flatly, completely ignoring cloud boundaries!
    parlay::parallel_for(0, total_vecs, [&](size_t v) {
      uint8_t* out_codes = res.packed_codes.data() + v * num_blocks;
      for (uint32_t b = 0; b < num_blocks; ++b) {
        out_codes[b] = find_best(pcs.data() + v * dim, b);
      }
    });
    return res;
  }

  template<typename ChPoint>
  Quantized_Query_Point_Cloud<Metric> quantize_query(const ChPoint& query_cloud) const {
    Quantized_Query_Point_Cloud<Metric> res;
    res.num_queries = query_cloud.size();
    res.num_blocks = num_blocks;
    res.K = num_clusters_per_block;

    if (res.num_queries == 0) return res;

    res.flat_int_luts.resize(res.num_queries * num_blocks * num_clusters_per_block, 0);
    res.min_dists.resize(res.num_queries, 0.0f);
    res.scales.resize(res.num_queries, 0.0f);
    const float* base = query_cloud.data();
    size_t q_dim = query_cloud.get_dims();
    for (size_t qi = 0; qi < res.num_queries; ++qi) {
      const float* qptr = base + qi * q_dim;
      std::vector<float> float_lut(num_blocks * num_clusters_per_block);
      float g_min = std::numeric_limits<float>::max();
      float g_max = std::numeric_limits<float>::lowest();
      for (uint32_t b = 0; b < num_blocks; ++b) {
        Eigen::Map<const Eigen::VectorXf> q_sub(qptr + b * dim_per_block,
                                                static_cast<Eigen::Index>(dim_per_block));
        Eigen::VectorXf dots = codebooks[b] * q_sub;
        float q_sq = Metric ? q_sub.squaredNorm() : 0.0f;
        for (uint32_t c = 0; c < num_clusters_per_block; ++c) {
          float val = Metric ? (codebook_norms[b][c] - 2.0f * dots[c] + q_sq) : -dots[c];
          float_lut[b * num_clusters_per_block + c] = val;
          g_min = std::min(g_min, val);
          g_max = std::max(g_max, val);
        }
      }
      res.min_dists[qi] = g_min;
      const float range = g_max - g_min;
      res.scales[qi] = (range > 1e-9f) ? (range / 255.0f) : 1e-9f;
      uint8_t* out_lut = res.flat_int_luts.data() + qi * num_blocks * num_clusters_per_block;
      for (size_t i = 0; i < float_lut.size(); ++i) {
        float v = (float_lut[i] - g_min) / res.scales[qi];
        out_lut[i] = static_cast<uint8_t>(std::clamp(static_cast<int>(v + 0.5f), 0, 255));
      }
    }
    return res;
  }

  void save(std::ostream& out) const {
    out.write(reinterpret_cast<const char*>(&num_blocks), sizeof(num_blocks));
    out.write(reinterpret_cast<const char*>(&num_clusters_per_block),
              sizeof(num_clusters_per_block));
    out.write(reinterpret_cast<const char*>(&num_points_per_cluster),
              sizeof(num_points_per_cluster));
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
    in.read(reinterpret_cast<char*>(&num_clusters_per_block), sizeof(num_clusters_per_block));
    in.read(reinterpret_cast<char*>(&num_points_per_cluster), sizeof(num_points_per_cluster));
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
};

}  // namespace pq_mv
}  // namespace mvsic