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

#include "parlay/parallel.h"
#include "parlay/sequence.h"

#include "rabitqlib/index/estimator.hpp"
#include "rabitqlib/index/query.hpp"
#include "rabitqlib/quantization/data_layout.hpp"
#include "rabitqlib/quantization/rabitq.hpp"
#include "rabitqlib/utils/rotator.hpp"
#include "rabitqlib/utils/space.hpp"

namespace mvsic {
namespace rabitq_mv {

// =========================================================================
// Forward declarations
// =========================================================================
template<bool Metric>
class Quantized_Point_Cloud_Set;
template<bool Metric>
class Quantized_Query_Point_Cloud;

// One-cloud Chamfer (mean min distance); same logic as distances_all inner loop.
template<bool Metric>
float rabitq_mv_chamfer_distance(const Quantized_Query_Point_Cloud<Metric>& q,
                                 const Quantized_Point_Cloud_Set<Metric>& db, size_t start,
                                 size_t end);

// =========================================================================
// Proxy: one multi-vector object as a range of flat vector indices
// =========================================================================
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

  template<typename Query>
  bool same_as(const Query&) const {
    return false;
  }
};

// =========================================================================
// Multi-Vector Query Object
// =========================================================================
template<bool Metric>
class Quantized_Query_Point_Cloud {
 public:
  using distanceType = float;

  size_t num_queries = 0;
  size_t padded_dim = 0;
  size_t ex_bits = 0;
  float (*ip_func)(const float*, const uint8_t*, size_t) = nullptr;

  // Encapsulated state per query vector to safely hold the opaque rabitqlib object
  struct QueryState {
    std::vector<float> q_rot_storage;
    std::unique_ptr<rabitqlib::SplitSingleQuery<float>> q_obj;
    float g_add = 0.0f;
    float g_error = 0.0f;
  };

  std::vector<QueryState> states;

  Quantized_Query_Point_Cloud() = default;

  // Matches mvsic::rabitq::Quantized_Query::distance_raw (returns est_dist).
  inline float distance_raw(const QueryState& qs, const char* bin, const char* ex) const {
    float est_dist = 0.0f;
    float low_dist = 0.0f;
    float ip_x0 = 0.0f;

    if (ex_bits > 0) {
      rabitqlib::split_single_fulldist(bin, ex, ip_func, *qs.q_obj, padded_dim, ex_bits, est_dist,
                                       low_dist, ip_x0, qs.g_add, qs.g_error);
    } else {
      rabitqlib::split_single_estdist(bin, *qs.q_obj, padded_dim, ip_x0, est_dist, low_dist,
                                      qs.g_add, qs.g_error);
    }
    return est_dist;
  }

  template<typename CloudView>
  float distance(const CloudView& cloud) const {
    return rabitq_mv_chamfer_distance(*this, *cloud.db, cloud.start_idx, cloud.end_idx);
  }

  template<typename CloudHandle>
  std::pair<float, size_t> distance_w_cmps(const CloudHandle& cloud) const {
    const size_t cloud_size = cloud.end_idx - cloud.start_idx;
    const size_t bytes_per_vec = cloud.db->bin_stride + cloud.db->ex_stride;
    return {this->distance(cloud), cloud_size * bytes_per_vec};
  }

  static constexpr bool is_metric() { return Metric; }
};

// =========================================================================
// Multi-Vector Set (Tightly Packed Flat Arrays)
// =========================================================================
template<bool Metric>
class Quantized_Point_Cloud_Set {
 public:
  size_t total_vecs = 0;
  size_t dim = 0;
  size_t padded_dim = 0;
  size_t total_bits = 0;
  size_t ex_bits = 0;

  size_t bin_stride = 0;
  size_t ex_stride = 0;

  parlay::sequence<char> bin_data;
  parlay::sequence<char> ex_data;

  parlay::sequence<size_t> offsets;
  parlay::sequence<uint32_t> ids;

  Quantized_Point_Cloud_Set() = default;

  Quantized_Point_Cloud<Metric> operator[](size_t i) const {
    const size_t start = offsets[i];
    const size_t end = offsets[i + 1];
    return Quantized_Point_Cloud<Metric>(this, start, end);
  }

  inline uint32_t get_id(size_t i) const noexcept { return (ids.size() > 0) ? ids[i] : i; }
  inline size_t num_bytes() const noexcept { return bin_data.size() + ex_data.size(); }

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

      const float d = rabitq_mv_chamfer_distance(q, *this, start, end);
      results[cid] = {get_id(cloud_id), d};
    });
  }

  void save(std::ostream& out) const {
    out.write(reinterpret_cast<const char*>(&total_vecs), sizeof(total_vecs));
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

    size_t off_size = offsets.size();
    out.write(reinterpret_cast<const char*>(&off_size), sizeof(off_size));
    if (off_size)
      out.write(reinterpret_cast<const char*>(offsets.data()), off_size * sizeof(size_t));
  }

  void load(std::istream& in) {
    in.read(reinterpret_cast<char*>(&total_vecs), sizeof(total_vecs));
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

    size_t off_size = 0;
    in.read(reinterpret_cast<char*>(&off_size), sizeof(off_size));
    offsets.resize(off_size);
    if (off_size) in.read(reinterpret_cast<char*>(offsets.data()), off_size * sizeof(size_t));
  }
};

template<bool Metric>
float rabitq_mv_chamfer_distance(const Quantized_Query_Point_Cloud<Metric>& q,
                                 const Quantized_Point_Cloud_Set<Metric>& db, size_t start,
                                 size_t end) {
  const size_t num_q = q.num_queries;
  if (num_q == 0) return 0.0f;
  if (end <= start) return std::numeric_limits<float>::max();

  const char* bin_base = db.bin_data.data();
  const char* ex_base =
      (db.ex_bits > 0 && db.ex_stride > 0 && db.ex_data.size() > 0) ? db.ex_data.data() : nullptr;

  float total_chamfer = 0.0f;
  for (size_t qi = 0; qi < num_q; ++qi) {
    const auto& qs = q.states[qi];
    float min_dist = std::numeric_limits<float>::max();
    for (size_t v = start; v < end; ++v) {
      const char* bin = bin_base + v * db.bin_stride;
      const char* ex = ex_base ? (ex_base + v * db.ex_stride) : nullptr;
      const float d = q.distance_raw(qs, bin, ex);
      if (d < min_dist) min_dist = d;
    }
    total_chamfer += min_dist;
  }
  return total_chamfer / static_cast<float>(num_q);
}

// =========================================================================
// RabitQ Model
// =========================================================================
template<bool Metric>
class Model {
 public:
  static constexpr uint32_t kClassId = 3;  // QuantizerTag::kRaBitQ
  static constexpr uint32_t kBatchAlignment = 1;
  static constexpr const char* kName = "rabitq";
  using EncodedSet = ::mvsic::rabitq_mv::Quantized_Point_Cloud_Set<Metric>;
  using EncodedQuery = ::mvsic::rabitq_mv::Quantized_Query_Point_Cloud<Metric>;
  struct Params {
    uint32_t bits = 4;
  };

  static constexpr bool is_fastscan = false;
  size_t dim = 0;
  size_t padded_dim = 0;
  size_t total_bits = 0;
  size_t ex_bits = 0;

  size_t bin_stride = 0;
  size_t ex_stride = 0;

  rabitqlib::RotatorType rotator_type = rabitqlib::RotatorType::FhtKacRotator;
  std::unique_ptr<rabitqlib::Rotator<float>> rotator;
  std::vector<float> centroid_rot;
  rabitqlib::quant::RabitqConfig config;

  Model() = default;

  template<typename PCSet>
  Model(const PCSet& train_data, size_t bits = 8) {
    train(train_data, bits);
  }

  template<typename PCSet>
  void train(const PCSet& pcs, const Params& p) {
    train(pcs, static_cast<size_t>(p.bits));
  }

  template<typename PCSet>
  void train(const PCSet& pcs, size_t bits = 8) {
    total_bits = bits;
    ex_bits = (total_bits > 0) ? (total_bits - 1) : 0;
    dim = pcs.get_dims();

    rotator.reset(rabitqlib::choose_rotator<float>(dim, rotator_type));
    padded_dim = rotator->size();

    bin_stride = rabitqlib::BinDataMap<float>::data_bytes(padded_dim);
    ex_stride = (ex_bits > 0) ? rabitqlib::ExDataMap<float>::data_bytes(padded_dim, ex_bits) : 0;

    if (total_bits > 1) {
      config = rabitqlib::quant::faster_config(padded_dim, total_bits);
    } else {
      config = rabitqlib::quant::RabitqConfig{};
    }

    const size_t n = pcs.total_size();
    const size_t sample_size = std::min(n, size_t(50000));

    std::vector<float> mean(dim, 0.0f);
    std::mt19937 rng(42);
    std::uniform_int_distribution<size_t> distu(0, n - 1);

    for (size_t i = 0; i < sample_size; ++i) {
      const float* p = reinterpret_cast<const float*>(pcs.data() + distu(rng) * dim);
      for (size_t j = 0; j < dim; ++j) {
        mean[j] += p[j];
      }
    }

    const float inv = (sample_size > 0) ? (1.0f / float(sample_size)) : 0.0f;
    for (float& x : mean)
      x *= inv;

    centroid_rot.resize(padded_dim);
    rotator->rotate(mean.data(), centroid_rot.data());
  }

  template<typename PCSet>
  Quantized_Point_Cloud_Set<Metric> encode(const PCSet& pcs) const {
    if (!rotator) {
      std::cerr << "RabitQ::encode called before train/load_model.\n";
      std::abort();
    }

    Quantized_Point_Cloud_Set<Metric> res;
    res.dim = dim;
    res.padded_dim = padded_dim;
    res.total_bits = total_bits;
    res.ex_bits = ex_bits;
    res.bin_stride = bin_stride;
    res.ex_stride = ex_stride;

    auto float_offsets = pcs.get_offsets();
    size_t n_clouds = float_offsets.size() > 0 ? float_offsets.size() - 1 : 0;

    // Tight Packing: Map exact float dimensions to vector offsets
    res.offsets.resize(n_clouds + 1);
    for (size_t c = 0; c <= n_clouds; ++c) {
      res.offsets[c] = float_offsets[c] / dim;
    }

    size_t total_vecs = res.offsets.back();
    res.total_vecs = total_vecs;

    res.bin_data.resize(total_vecs * bin_stride);
    if (ex_stride > 0) res.ex_data.resize(total_vecs * ex_stride);

    auto pcs_ids = pcs.get_ids();
    res.ids = parlay::sequence<uint32_t>(pcs_ids.begin(), pcs_ids.end());

    const rabitqlib::MetricType m_type = Metric ? rabitqlib::METRIC_L2 : rabitqlib::METRIC_IP;

    struct EncodingWorkspace {
      std::vector<float> p_rot;
      void ensure_size(size_t sz) {
        if (p_rot.size() != sz) p_rot.resize(sz);
      }
    };

    // Encode totally flat, ignoring cloud boundaries
    parlay::parallel_for(0, total_vecs, [&](size_t v) {
      static thread_local EncodingWorkspace ws;
      ws.ensure_size(padded_dim);

      const float* p = pcs.data() + v * dim;
      rotator->rotate(p, ws.p_rot.data());

      char* bin_out = res.bin_data.data() + v * bin_stride;
      char* ex_out = (ex_stride > 0) ? (res.ex_data.data() + v * ex_stride) : nullptr;

      rabitqlib::quant::quantize_split_single(ws.p_rot.data(), centroid_rot.data(), padded_dim,
                                              ex_bits, bin_out, ex_out, m_type, config);
    });

    return res;
  }

  template<typename ChPoint>
  Quantized_Query_Point_Cloud<Metric> quantize_query(const ChPoint& query_cloud) const {
    if (!rotator) {
      std::cerr << "RabitQ::quantize_query called before train.\n";
      std::abort();
    }

    Quantized_Query_Point_Cloud<Metric> res;
    res.num_queries = query_cloud.size();
    res.padded_dim = padded_dim;
    res.ex_bits = ex_bits;

    if (res.ex_bits > 0) {
      res.ip_func = rabitqlib::select_excode_ipfunc(res.ex_bits);
    }

    if (res.num_queries == 0) return res;

    res.states.resize(res.num_queries);

    const float* base = query_cloud.data();
    size_t q_dim = query_cloud.get_dims();
    const rabitqlib::MetricType m_type = Metric ? rabitqlib::METRIC_L2 : rabitqlib::METRIC_IP;

    for (size_t qi = 0; qi < res.num_queries; ++qi) {
      const float* qptr = base + qi * q_dim;
      auto& qs = res.states[qi];

      qs.q_rot_storage.resize(padded_dim);
      rotator->rotate(qptr, qs.q_rot_storage.data());

      qs.q_obj = std::make_unique<rabitqlib::SplitSingleQuery<float>>(
          qs.q_rot_storage.data(), padded_dim, ex_bits, config, m_type);

      if constexpr (Metric) {
        float dist_sq =
            rabitqlib::euclidean_sqr(qs.q_rot_storage.data(), centroid_rot.data(), padded_dim);
        float norm = std::sqrt(dist_sq);
        qs.q_obj->set_g_add(norm);
        qs.g_add = norm * norm;
        qs.g_error = norm;
      } else {
        float ip = rabitqlib::dot_product(qs.q_rot_storage.data(), centroid_rot.data(), padded_dim);
        qs.q_obj->set_g_add(ip);
        qs.g_add = ip;
        qs.g_error = 0.0f;
      }
    }

    return res;
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

    if (!rotator) {
      std::cerr << "RabitQ::save_model: rotator is null.\n";
      std::abort();
    }
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
    const size_t runtime_padded_dim = rotator->size();
    if (runtime_padded_dim != padded_dim) {
      std::cerr << "RaBitQ Error: Saved padded_dim (" << padded_dim << ") != Runtime padded_dim ("
                << runtime_padded_dim << ")\n";
      std::abort();
    }

    rotator->load(in);

    if (total_bits > 1) {
      config = rabitqlib::quant::faster_config(padded_dim, total_bits);
    } else {
      config = rabitqlib::quant::RabitqConfig{};
    }

    size_t c_size = 0;
    in.read(reinterpret_cast<char*>(&c_size), sizeof(c_size));
    centroid_rot.resize(c_size);
    in.read(reinterpret_cast<char*>(centroid_rot.data()), c_size * sizeof(float));
  }
};

}  // namespace rabitq_mv
}  // namespace mvsic