#pragma once

#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <memory>

#include "parlay/parallel.h"
#include "parlay/primitives.h"

#include "rabitqlib/quantization/rabitq.hpp"
#include "rabitqlib/quantization/data_layout.hpp"
#include "rabitqlib/index/query.hpp"
#include "rabitqlib/index/estimator.hpp"
#include "rabitqlib/utils/rotator.hpp"
#include "rabitqlib/utils/space.hpp"

namespace mvsic {

// ---------------------------------------------------------
// RaBitQ Point Wrapper
// ---------------------------------------------------------
class RaBitQ_Query_Wrapper;

class RaBitQ_Point_Wrapper {
 public:
  const char* bin_data;
  const char* ex_data;

  RaBitQ_Point_Wrapper(const char* bin, const char* ex) : bin_data(bin), ex_data(ex) {}

  inline float distance(const RaBitQ_Query_Wrapper& qq) const;
};

// ---------------------------------------------------------
// RaBitQ Query Wrapper
// ---------------------------------------------------------
class RaBitQ_Query_Wrapper {
 public:
  // Storage must come first
  std::vector<float> q_rot_storage;

  rabitqlib::SplitSingleQuery<float> q_obj;

  float (*ip_func)(const float*, const uint8_t*, size_t);

  float g_add;
  float g_error;

  size_t padded_dim;
  size_t ex_bits;

  RaBitQ_Query_Wrapper(std::vector<float>&& q_rot, const float* c_rot, size_t dim, size_t bits,
                       const rabitqlib::quant::RabitqConfig& cfg) :
      q_rot_storage(std::move(q_rot)),
      q_obj(q_rot_storage.data(), dim, bits, cfg, rabitqlib::METRIC_L2),
      padded_dim(dim),
      ex_bits(bits) {
    if (ex_bits > 0) {
      ip_func = rabitqlib::select_excode_ipfunc(ex_bits);
    } else {
      ip_func = nullptr;
    }

    float dist_sq = rabitqlib::euclidean_sqr(q_rot_storage.data(), c_rot, dim);
    float norm = std::sqrt(dist_sq);

    q_obj.set_g_add(norm);
    g_add = norm * norm;
    g_error = norm;
  }

  inline float distance(const RaBitQ_Point_Wrapper& p) const {
    float est_dist;
    float low_dist;
    float ip_x0;

    if (ex_bits > 0) {
      // Full Precision (Binary + Residuals)
      rabitqlib::split_single_fulldist(p.bin_data, p.ex_data, ip_func, q_obj, padded_dim, ex_bits,
                                       est_dist, low_dist, ip_x0, g_add, g_error);
    } else {
      // 1-Bit Precision (Binary Only)
      // Uses a different kernel that doesn't touch ex_data
      rabitqlib::split_single_estdist(p.bin_data, q_obj, padded_dim, ip_x0, est_dist, low_dist,
                                      g_add, g_error);
    }

    return est_dist;
  }
};

inline float RaBitQ_Point_Wrapper::distance(const RaBitQ_Query_Wrapper& qq) const {
  return qq.distance(*this);
}

// ---------------------------------------------------------
// RaBitQ Point Range
// ---------------------------------------------------------
template<typename PointRange>
class RaBitQ_Point_Range {
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

  RaBitQ_Point_Range(const PointRange& data, size_t bits = 8) : total_bits(bits) {
    n = data.size();
    dim = data.get_dims();
    ex_bits = (total_bits > 0) ? total_bits - 1 : 0;

    rotator = rabitqlib::choose_rotator<float>(dim, rabitqlib::RotatorType::FhtKacRotator);
    padded_dim = rotator->size();

    bin_stride = rabitqlib::BinDataMap<float>::data_bytes(padded_dim);
    ex_stride = (ex_bits > 0) ? rabitqlib::ExDataMap<float>::data_bytes(padded_dim, ex_bits) : 0;

    std::cout << "RaBitQ Config: PaddedDim=" << padded_dim << ", BinStride=" << bin_stride << "B"
              << ", ExStride=" << ex_stride << "B" << std::endl;

    bin_data.resize(n * bin_stride);
    if (ex_stride > 0) ex_data.resize(n * ex_stride);

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

    std::cout << "Encoding " << n << " vectors..." << std::endl;
    parlay::parallel_for(0, n, [&](size_t i) {
      const float* p = reinterpret_cast<const float*>(data.location(i));
      std::vector<float> p_rot(padded_dim);
      rotator->rotate(p, p_rot.data());

      rabitqlib::quant::quantize_split_single(
          p_rot.data(), centroid_rot.data(), padded_dim, ex_bits, &bin_data[i * bin_stride],
          (ex_bits > 0) ? &ex_data[i * ex_stride] : nullptr, rabitqlib::METRIC_L2, config);
    });
  }

  ~RaBitQ_Point_Range() {
    if (rotator) delete rotator;
  }

  RaBitQ_Point_Wrapper operator[](size_t i) const {
    return RaBitQ_Point_Wrapper(&bin_data[i * bin_stride],
                                (ex_bits > 0) ? &ex_data[i * ex_stride] : nullptr);
  }

  template<typename PointTy>
  RaBitQ_Query_Wrapper quantize_query(const PointTy& query) const {
    std::vector<float> q_vec(dim);
    for (size_t i = 0; i < dim; ++i)
      q_vec[i] = query[i];

    std::vector<float> q_rot(padded_dim);
    rotator->rotate(q_vec.data(), q_rot.data());

    return RaBitQ_Query_Wrapper(std::move(q_rot), centroid_rot.data(), padded_dim, ex_bits, config);
  }
};

}  // namespace mvsic