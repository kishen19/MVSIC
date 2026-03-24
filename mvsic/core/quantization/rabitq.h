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
namespace rabitq {

template<bool Metric>
class Quantized_Query;

// ---------------------------------------------------------
// Quantized Point: Aux Data Type (lightweight view)
// ---------------------------------------------------------
template<bool Metric>
class Quantized_Point {
 public:
  const char* bin_data = nullptr;
  const char* ex_data = nullptr;  // nullptr if ex_bits==0

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
// Quantized Query: Holds rotated query + rabitqlib LUT state
// + distances_all(db, out)
// ---------------------------------------------------------
template<bool Metric>
class Quantized_Query {
 public:
  using distanceType = float;

  // Must come before q_obj (it references this storage)
  std::vector<float> q_rot_storage;

  rabitqlib::SplitSingleQuery<float> q_obj;

  float (*ip_func)(const float*, const uint8_t*, size_t) = nullptr;

  float g_add = 0.0f;
  float g_error = 0.0f;

  size_t padded_dim = 0;
  size_t ex_bits = 0;

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
      // Euclidean-specific init
      float dist_sq = rabitqlib::euclidean_sqr(q_rot_storage.data(), c_rot, padded_dim);
      float norm = std::sqrt(dist_sq);
      q_obj.set_g_add(norm);
      g_add = norm * norm;
      g_error = norm;
    } else {
      // IP-specific init
      float ip = rabitqlib::dot_product(q_rot_storage.data(), c_rot, padded_dim);
      q_obj.set_g_add(ip);
      g_add = ip;
      g_error = 0.0f;
    }
  }

  inline float distance(const Quantized_Point<Metric>& p) const {
    return distance_raw(p.bin_data, p.ex_data);
  }

  // Fast path: avoids constructing Quantized_Point in hot loops.
  inline float distance_raw(const char* bin, const char* ex) const {
    float est_dist = 0.0f;
    float low_dist = 0.0f;
    float ip_x0 = 0.0f;

    if (ex_bits > 0) {
      rabitqlib::split_single_fulldist(bin, ex, ip_func, q_obj, padded_dim, ex_bits, est_dist,
                                       low_dist, ip_x0, g_add, g_error);
    } else {
      rabitqlib::split_single_estdist(bin, q_obj, padded_dim, ip_x0, est_dist, low_dist, g_add,
                                      g_error);
    }
    return est_dist;
  }

  // Compute distances to *every* encoded vector in db.
  // Parallelized across points.
  template<typename EncRange>
  void distances_all(const EncRange& db, float* out) const {
    const size_t n = static_cast<size_t>(db.size());
    if (n == 0) return;

    const char* bin_base = db.bin_data.data();
    const char* ex_base =
        (db.ex_bits > 0 && db.ex_stride > 0 && db.ex_data.size() > 0) ? db.ex_data.data() : nullptr;
    const size_t bin_stride = db.bin_stride;
    const size_t ex_stride = db.ex_stride;

    parlay::parallel_for(0, n, [&](size_t i) {
      const char* bin = bin_base + i * bin_stride;
      const char* ex = ex_base ? (ex_base + i * ex_stride) : nullptr;
      out[i] = distance_raw(bin, ex);
    });
  }
};

template<bool Metric>
inline float Quantized_Point<Metric>::distance(const Quantized_Query<Metric>& qq) const {
  return qq.distance_raw(bin_data, ex_data);
}

// ---------------------------------------------------------
// Encoded Point Range: owns only encoded DB payload (no model)
// ---------------------------------------------------------
template<typename PointRange, bool Metric>
class Quantized_Point_Range {
 public:
  static constexpr bool is_fastscan = false;

  size_t n = 0;
  size_t dim = 0;
  size_t padded_dim = 0;
  size_t total_bits = 0;
  size_t ex_bits = 0;

  size_t bin_stride = 0;
  size_t ex_stride = 0;

  parlay::sequence<char> bin_data;
  parlay::sequence<char> ex_data;

  Quantized_Point_Range() = default;

  Quantized_Point<Metric> operator[](size_t i) const {
    const char* bin = &bin_data[i * bin_stride];
    const char* ex = (ex_bits > 0) ? &ex_data[i * ex_stride] : nullptr;
    return Quantized_Point<Metric>(bin, ex);
  }

  inline uint32_t size() const noexcept { return static_cast<uint32_t>(n); }
  inline uint32_t get_dims() const noexcept { return static_cast<uint32_t>(dim); }

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
  }
};

// ---------------------------------------------------------
// RabitQ Model: owns rotator + centroid + config, can encode + quantize queries
// ---------------------------------------------------------
template<bool Metric>
class Model {
 public:
  static constexpr bool is_fastscan = false;
  size_t dim = 0;
  size_t padded_dim = 0;
  size_t total_bits = 0;
  size_t ex_bits = 0;

  size_t bin_stride = 0;
  size_t ex_stride = 0;

  // NOTE: your current code always uses this type; we keep that.
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

    if (total_bits > 1) {
      config = rabitqlib::quant::faster_config(padded_dim, total_bits);
    } else {
      // config unused for total_bits<=1 in your current usage; leave default
      config = rabitqlib::quant::RabitqConfig{};
    }

    // Compute mean (sample) and rotate it -> centroid_rot
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
    if (!rotator) {
      std::cerr << "RabitQ::encode called before train/load_model.\n";
      std::abort();
    }

    Quantized_Point_Range<PointRange, Metric> enc;
    enc.n = data.size();
    enc.dim = data.get_dims();
    enc.padded_dim = padded_dim;
    enc.total_bits = total_bits;
    enc.ex_bits = ex_bits;
    enc.bin_stride = bin_stride;
    enc.ex_stride = ex_stride;

    if (enc.dim != dim) {
      std::cerr << "RabitQ::encode dim mismatch: data dim=" << enc.dim << " model dim=" << dim
                << "\n";
      std::abort();
    }

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

    return enc;
  }

  template<typename PointTy>
  Quantized_Query<Metric> quantize_query(const PointTy& query) const {
    if (!rotator) {
      std::cerr << "RabitQ::quantize_query called before train/load_model.\n";
      std::abort();
    }
    std::vector<float> tmp(dim);
    for (size_t i = 0; i < dim; ++i)
      tmp[i] = query[i];

    std::vector<float> q_rot(padded_dim);
    rotator->rotate(tmp.data(), q_rot.data());

    return Quantized_Query<Metric>(std::move(q_rot), centroid_rot.data(), padded_dim, ex_bits,
                                   config);
  }

  Quantized_Query<Metric> quantize_query(const float* qptr) const {
    if (!rotator) {
      std::cerr << "RabitQ::quantize_query called before train/load_model.\n";
      std::abort();
    }

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
    if (!rotator) {
      std::cerr << "RabitQ::quantize_query_batch called before train/load_model.\n";
      std::abort();
    }

    const uint32_t num_q = query_cloud.size();
    const uint32_t dims = query_cloud.get_dims();
    const float* base = query_cloud.data();

    out_queries.clear();
    out_queries.reserve(num_q);

    // One reusable temp buffer per call (not thread_local).
    std::vector<float> tmp(dim);

    for (uint32_t i = 0; i < num_q; ++i) {
      const float* qi = base + static_cast<size_t>(i) * dims;

      // Copy into tmp to guarantee we never modify the original query storage.
      std::memcpy(tmp.data(), qi, sizeof(float) * dim);

      std::vector<float> q_rot(padded_dim);
      rotator->rotate(tmp.data(), q_rot.data());

      out_queries.emplace_back(std::move(q_rot), centroid_rot.data(), padded_dim, ex_bits, config);
    }
  }

  // Save/load MODEL only
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

}  // namespace rabitq
}  // namespace mvsic
