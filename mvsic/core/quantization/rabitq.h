#pragma once

#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <memory>
#include <fstream>

#include "parlay/parallel.h"
#include "parlay/primitives.h"

#include "rabitqlib/quantization/rabitq.hpp"
#include "rabitqlib/quantization/data_layout.hpp"
#include "rabitqlib/index/query.hpp"
#include "rabitqlib/index/estimator.hpp"
#include "rabitqlib/utils/rotator.hpp"
#include "rabitqlib/utils/space.hpp"

namespace mvsic {
namespace rabitq {

// ---------------------------------------------------------
// Quantized Point: Aux Data Type
// ---------------------------------------------------------
template<bool Metric>
class Quantized_Query;

template<bool Metric>
class Quantized_Point {
 public:
  const char* bin_data;
  const char* ex_data;

  Quantized_Point(const char* bin, const char* ex) : bin_data(bin), ex_data(ex) {}

  inline float distance(const Quantized_Query<Metric>& qq) const;

  // Functions used by beam_search
  bool same_as(const Quantized_Point<Metric>& q) const { return false; }
  bool same_as(const Quantized_Query<Metric>& q) const { return false; }

  // OPTIMIZATION: Prefetching for random-access graph traversal
  void prefetch() const {
    __builtin_prefetch(bin_data, 0, 3);
    if (ex_data) __builtin_prefetch(ex_data, 0, 3);
  }

  bool is_metric() const { return Metric; }
};

// ---------------------------------------------------------
// Quantized Query: Holds the Lookup Table (LUT)
// ---------------------------------------------------------
template<bool Metric>
class Quantized_Query {
 public:
  using distanceType = float;
  // Storage must come first
  std::vector<float> q_rot_storage;

  rabitqlib::SplitSingleQuery<float> q_obj;

  float (*ip_func)(const float*, const uint8_t*, size_t);

  float g_add = 0.0f;
  float g_error = 0.0f;

  size_t padded_dim;
  size_t ex_bits;

  Quantized_Query(std::vector<float>&& q_rot, const float* c_rot, size_t dim, size_t bits,
                  const rabitqlib::quant::RabitqConfig& cfg) :
      q_rot_storage(std::move(q_rot)),
      q_obj(q_rot_storage.data(), dim, bits, cfg,
            Metric ? rabitqlib::METRIC_L2 : rabitqlib::METRIC_IP),
      padded_dim(dim),
      ex_bits(bits) {

    if (ex_bits > 0) {
      ip_func = rabitqlib::select_excode_ipfunc(ex_bits);
    } else {
      ip_func = nullptr;
    }

    if constexpr (Metric) {
      // Euclidean Specific initialization
      float dist_sq = rabitqlib::euclidean_sqr(q_rot_storage.data(), c_rot, dim);
      float norm = std::sqrt(dist_sq);

      q_obj.set_g_add(norm);
      g_add = norm * norm;
      g_error = norm;
    } else {
      // Inner Product Specific initialization
      float ip = rabitqlib::dot_product(q_rot_storage.data(), c_rot, dim);
      q_obj.set_g_add(ip);
      g_add = ip;
      g_error = 0.0f;
    }
  }

  inline float distance(const Quantized_Point<Metric>& p) const {
    float est_dist;
    float low_dist;
    float ip_x0;

    if (ex_bits > 0) {
      // Full Precision (Binary + Residuals)
      rabitqlib::split_single_fulldist(p.bin_data, p.ex_data, ip_func, q_obj, padded_dim, ex_bits,
                                       est_dist, low_dist, ip_x0, g_add, g_error);
    } else {
      // 1-Bit Precision (Binary Only)
      rabitqlib::split_single_estdist(p.bin_data, q_obj, padded_dim, ip_x0, est_dist, low_dist,
                                      g_add, g_error);
    }

    return est_dist;
  }
};

template<bool Metric>
inline float Quantized_Point<Metric>::distance(const Quantized_Query<Metric>& qq) const {
  return qq.distance(*this);
}

// ---------------------------------------------------------
// RaBitQ Point Range
// ---------------------------------------------------------
template<typename PointRange, bool Metric>
class Quantized_Point_Range {
 public:
  size_t n;
  size_t dim;
  size_t padded_dim;
  size_t total_bits;
  size_t ex_bits;

  rabitqlib::Rotator<float>* rotator = nullptr;
  std::vector<float> centroid_rot;
  rabitqlib::quant::RabitqConfig config;

  parlay::sequence<char> bin_data;
  parlay::sequence<char> ex_data;

  size_t bin_stride;
  size_t ex_stride;

  // OPTIMIZATION: Workspace for thread_local storage to avoid repeated heap allocations
  struct EncodingWorkspace {
    std::vector<float> p_rot;
    void ensure_size(size_t sz) {
      if (p_rot.size() != sz) p_rot.resize(sz);
    }
  };

  // Constructor for loading
  Quantized_Point_Range() {}

  Quantized_Point_Range(const PointRange& data, size_t bits = 8) : total_bits(bits) {
    n = data.size();
    dim = data.get_dims();
    ex_bits = (total_bits > 0) ? total_bits - 1 : 0;

    rotator = rabitqlib::choose_rotator<float>(dim, rabitqlib::RotatorType::FhtKacRotator);
    padded_dim = rotator->size();

    bin_stride = rabitqlib::BinDataMap<float>::data_bytes(padded_dim);
    ex_stride = (ex_bits > 0) ? rabitqlib::ExDataMap<float>::data_bytes(padded_dim, ex_bits) : 0;

    std::cout << "RaBitQ Config: Metric=" << (Metric ? "L2" : "IP") << ", PaddedDim=" << padded_dim
              << ", BinStride=" << bin_stride << "B"
              << ", ExStride=" << ex_stride << "B" << std::endl;

    bin_data.resize(n * bin_stride);
    if (ex_stride > 0) ex_data.resize(n * ex_stride);

    // Compute mean for rotation centroid
    std::vector<float> mean(dim, 0.0f);
    size_t sample_size = std::min(n, (size_t)50000);
    for (size_t i = 0; i < sample_size; ++i) {
      const float* p = reinterpret_cast<const float*>(data.location(i));
      for (size_t j = 0; j < dim; ++j)
        mean[j] += p[j];
    }
    for (float& x : mean)
      x /= sample_size;

    centroid_rot.resize(padded_dim);
    rotator->rotate(mean.data(), centroid_rot.data());

    if (total_bits > 1) {
      config = rabitqlib::quant::faster_config(padded_dim, total_bits);
    }

    // Determine metric for library
    rabitqlib::MetricType m_type = Metric ? rabitqlib::METRIC_L2 : rabitqlib::METRIC_IP;

    std::cout << "Encoding " << n << " vectors (thread-local optimized)..." << std::endl;
    parlay::parallel_for(0, n, [&](size_t i) {
      // OPT 1: Reuse buffer to prevent N mallocs/frees
      static thread_local EncodingWorkspace ws;
      ws.ensure_size(padded_dim);

      const float* p = reinterpret_cast<const float*>(data.location(i));
      rotator->rotate(p, ws.p_rot.data());

      rabitqlib::quant::quantize_split_single(
          ws.p_rot.data(), centroid_rot.data(), padded_dim, ex_bits, &bin_data[i * bin_stride],
          (ex_bits > 0) ? &ex_data[i * ex_stride] : nullptr, m_type, config);
    });
  }

  ~Quantized_Point_Range() {
    if (rotator) delete rotator;
  }

  Quantized_Point<Metric> operator[](size_t i) const {
    return Quantized_Point<Metric>(&bin_data[i * bin_stride],
                                   (ex_bits > 0) ? &ex_data[i * ex_stride] : nullptr);
  }

  template<typename PointTy>
  Quantized_Query<Metric> quantize_query(const PointTy& query) const {
    std::vector<float> q_vec(dim);
    for (size_t i = 0; i < dim; ++i)
      q_vec[i] = query[i];

    std::vector<float> q_rot(padded_dim);
    rotator->rotate(q_vec.data(), q_rot.data());

    return Quantized_Query<Metric>(std::move(q_rot), centroid_rot.data(), padded_dim, ex_bits,
                                   config);
  }

  template<typename PointCloudTy>
  void quantize_query_batch(const PointCloudTy& query_cloud,
                            std::vector<Quantized_Query<Metric>>& out_queries) const {
    uint32_t num_q = query_cloud.size();
    out_queries.reserve(num_q);

    for (size_t i = 0; i < num_q; ++i) {
      std::vector<float> q_rot(padded_dim);
      // Assuming your rotator can take raw data from PointCloudTy
      rotator->rotate(query_cloud[i].data(), q_rot.data());
      out_queries.emplace_back(std::move(q_rot), centroid_rot.data(), padded_dim, ex_bits, config);
    }
  }

  void save(std::ofstream& out) const {
    out.write((char*)&n, sizeof(n));
    out.write((char*)&dim, sizeof(dim));
    out.write((char*)&total_bits, sizeof(total_bits));
    out.write((char*)&padded_dim, sizeof(padded_dim));

    // Delegate rotation state saving to the library
    rotator->save(out);

    size_t c_size = centroid_rot.size();
    out.write((char*)&c_size, sizeof(c_size));
    out.write((char*)centroid_rot.data(), c_size * sizeof(float));

    size_t bin_sz = bin_data.size();
    out.write((char*)&bin_sz, sizeof(bin_sz));
    if (bin_sz > 0) out.write(&bin_data[0], bin_sz);

    size_t ex_sz = ex_data.size();
    out.write((char*)&ex_sz, sizeof(ex_sz));
    if (ex_sz > 0) out.write(&ex_data[0], ex_sz);
  }

  void load(std::ifstream& in) {
    in.read((char*)&n, sizeof(n));
    in.read((char*)&dim, sizeof(dim));
    in.read((char*)&total_bits, sizeof(total_bits));

    size_t saved_padded_dim;
    in.read((char*)&saved_padded_dim, sizeof(saved_padded_dim));

    ex_bits = (total_bits > 0) ? total_bits - 1 : 0;

    // 1. Construct Rotator
    rotator = rabitqlib::choose_rotator<float>(dim, rabitqlib::RotatorType::FhtKacRotator);
    padded_dim = rotator->size();

    // 2. Overwrite the random state with the saved state
    rotator->load(in);

    if (padded_dim != saved_padded_dim) {
      std::cerr << "RaBitQ Error: Saved padded_dim (" << saved_padded_dim
                << ") != Runtime padded_dim (" << padded_dim << ")" << std::endl;
      abort();
    }

    bin_stride = rabitqlib::BinDataMap<float>::data_bytes(padded_dim);
    ex_stride = (ex_bits > 0) ? rabitqlib::ExDataMap<float>::data_bytes(padded_dim, ex_bits) : 0;

    if (total_bits > 1) config = rabitqlib::quant::faster_config(padded_dim, total_bits);

    size_t c_size;
    in.read((char*)&c_size, sizeof(c_size));
    centroid_rot.resize(c_size);
    in.read((char*)centroid_rot.data(), c_size * sizeof(float));

    size_t bin_sz;
    in.read((char*)&bin_sz, sizeof(bin_sz));
    bin_data.resize(bin_sz);
    if (bin_sz > 0) in.read(&bin_data[0], bin_sz);

    size_t ex_sz;
    in.read((char*)&ex_sz, sizeof(ex_sz));
    if (ex_sz > 0) {
      ex_data.resize(ex_sz);
      in.read(&ex_data[0], ex_sz);
    }
  }

  // Returns number of points
  inline uint32_t size() const noexcept { return (uint32_t)n; }
  // Returns embedding dimension
  inline uint32_t get_dims() const noexcept { return (uint32_t)dim; }
};

}  // namespace rabitq
}  // namespace mvsic