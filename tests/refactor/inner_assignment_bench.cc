// Compare singleton-center panels with packed-center inner assignment. Build
// separate AVX2 and AVX-512 executables; this file does not dispatch at runtime.
// Run only in a coordinated quiet window. The installed baseline stays intact.
#include "mvsic/core/data.h"
#include "mvsic/core/kmeans/lloyds_kmeans.h"
#include "mvsic/core/quantization/turboquant_mv.h"
#include "mvsic/core/quantization/turboquant_8bit_mv.h"

#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
using mvsic::PointCloud;
using mvsic::PointCloudSetView;

PointCloudSetView singletons(const std::vector<float>& values, uint32_t d,
                             std::vector<size_t>& offsets) {
  const size_t n = values.size() / d;
  offsets.resize(n + 1);
  for (size_t i = 0; i <= n; ++i)
    offsets[i] = i * d;
  return {uint32_t(n), d, values.data(), offsets.data()};
}

template<bool Metric, class Query, class Database>
float score(const Query& query, const Database& db, size_t q, size_t center) {
  int32_t dot = 0;
  for (size_t j = 0; j < db.total_tiles * 4; ++j) {
    const size_t offset = (center / 16) * db.panel_bytes + (j / 4) * 64 + (center % 16) * 4 + j % 4;
    dot += (int(db.panel_data[offset]) - 128) * int(query.flat_query_data[q * query.q_stride + j]);
  }
  const float negative =
      -(float(dot) * db.norm_scaling_factors[center] * query.norm_scaling_factors[q]);
  if constexpr (Metric)
    return db.unquantized_squared_norms[center] +
           ((negative + negative) + query.unquantized_squared_norms[q]);
  return negative;
}

template<bool Metric, class Query, class Database>
void packed_assign(const Query& query, const Database& db, std::vector<uint32_t>& out) {
  out.resize(query.num_queries);
  mvsic::turboquant_8bit_mv::SingleCloudArgmin<Metric>::RunFlat(
      query.flat_query_data.data(), query.norm_scaling_factors.data(),
      query.unquantized_squared_norms.data(), query.byte_sums.data(), query.q_stride,
      query.num_queries, db, out.data());
}

template<bool Metric, class Model, template<class> class Batch>
void verify(const std::vector<float>& values, const std::vector<float>& centers, uint32_t d,
            bool exact_ties = false, uint32_t tied_center = 0) {
  const size_t n = values.size() / d, k = centers.size() / d;
  std::vector<size_t> offsets;
  const auto separate = singletons(centers, d, offsets);
  const size_t packed_offsets[] = {0, centers.size()};
  const PointCloudSetView packed{1, d, centers.data(), packed_offsets};
  Model model;
  model.encoder.seed_ = 11;
  model.train(separate);
  const auto db = model.encode(packed);
  const auto query = model.quantize_query(PointCloud{values.data(), uint32_t(n), d});
  std::vector<uint32_t> actual;
  packed_assign<Metric>(query, db, actual);
  for (size_t q = 0; q < n; ++q) {
    uint32_t expected = 0;
    float best = std::numeric_limits<float>::infinity();
    for (size_t j = 0; j < k; ++j) {
      const float distance = score<Metric>(query, db, q, j);
      if (distance < best) {
        best = distance;
        expected = uint32_t(j);
      }
    }
    if (actual[q] >= k || std::abs(score<Metric>(query, db, q, actual[q]) - best) >
                              2e-5f * std::max(1.0f, std::abs(best)))
      throw std::runtime_error("Packed argmin disagrees with decoded byte/norm oracle");
    if (exact_ties && (expected != tied_center || actual[q] != tied_center))
      throw std::runtime_error("Packed argmin did not select the lowest center ID on a tie");
  }
}

template<bool Metric, class Model, template<class> class Batch>
void checks() {
  std::mt19937 random(188);
  std::normal_distribution<float> normal;
  for (uint32_t d : {7, 31, 128}) {
    std::vector<float> values(71 * d);
    for (float& v : values)
      v = normal(random);
    // Includes a zero query and query-block/SIMD tails.
    std::fill_n(values.begin(), d, 0);
    for (size_t k : {1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 65}) {
      std::vector<float> centers(k * d);
      for (float& v : centers)
        v = normal(random) * 3;
      verify<Metric, Model, Batch>(values, centers, d);
    }
  }
  // Equal best scores in different panels: lane order incorrectly chose 16.
  std::vector<float> values(7 * 16, 0), centers(17 * 16, 0);
  for (size_t i = 0; i < 7; ++i)
    values[i * 16] = 1;
  for (size_t i = 0; i < 17; ++i)
    centers[i * 16] = -1;
  centers[16] = centers[16 * 16] = 1;
  verify<Metric, Model, Batch>(values, centers, 16, true, 1);
  // AVX2 reduces two eight-center halves. Exercise ties across halves and
  // across panels with a nonzero winning ID, including one-query tails.
  values.resize(3 * 16);
  centers[16 * 16] = -1;
  centers[8 * 16] = 1;
  verify<Metric, Model, Batch>(values, centers, 16, true, 1);
  values.resize(16);
  centers[16] = -1;
  centers[16 * 16] = 1;
  verify<Metric, Model, Batch>(values, centers, 16, true, 8);
  // All real scores positive under IP; padding must not win. Zero input ties
  // remain real and select center zero even with a partial final panel.
  centers[8 * 16] = centers[16 * 16] = -1;
  verify<Metric, Model, Batch>(values, centers, 16, true, 0);
  std::fill(values.begin(), values.end(), 0);
  verify<Metric, Model, Batch>(values, centers, 16, true, 0);
}

template<bool Metric, class Model, template<class> class Batch>
void verify_weighted_updates() {
  constexpr uint32_t n = 96, k = 17, d = 31;
  std::mt19937 random(317);
  std::normal_distribution<float> normal;
  parlay::sequence<parlay::sequence<float>> data(n);
  std::vector<float> values(size_t(n) * d);
  parlay::sequence<float> weights(n);
  for (size_t i = 0; i < n; ++i) {
    data[i].resize(d);
    weights[i] = 1.0f / float(1 + i % 7);
    for (size_t j = 0; j < d; ++j)
      values[i * d + j] = data[i][j] = normal(random);
  }
  mvsic::lloyds::FloatLloydsBackend<Metric> old_means(data, d, true), new_means(data, d, true);
  mvsic::lloyds::CenterSet centers(k, d);
  std::copy_n(values.data(), size_t(k) * d, centers.data());
  std::vector<size_t> offsets(k + 1);
  for (size_t i = 0; i <= k; ++i)
    offsets[i] = i * d;
  const size_t packed_offsets[] = {0, size_t(k) * d};
  Model model;
  model.encoder.seed_ = 11;
  model.encoder.train(d);
  const auto query = model.quantize_query(PointCloud{values.data(), n, d});
  std::vector<typename Model::EncodedQuery> singles(n);
  std::vector<const typename Model::EncodedQuery*> pointers(n);
  for (size_t i = 0; i < n; ++i) {
    singles[i] = model.quantize_query(PointCloud{values.data() + i * d, 1, d});
    pointers[i] = &singles[i];
  }
  std::vector<std::pair<uint32_t, float>> unpacked(n);
  std::vector<uint32_t> packed;
  for (size_t iteration = 0; iteration < 4; ++iteration) {
    const auto old_db = model.encode(PointCloudSetView{k, d, centers.data(), offsets.data()});
    const auto new_db = model.encode(PointCloudSetView{1, d, centers.data(), packed_offsets});
    Batch<typename Model::EncodedSet>::TopKIntoUninitialized(pointers, old_db, 1, unpacked.data());
    packed_assign<Metric>(query, new_db, packed);
    parlay::sequence<uint32_t> a(n), b(n);
    for (size_t i = 0; i < n; ++i) {
      a[i] = unpacked[i].first;
      b[i] = packed[i];
    }
    const auto before = old_means.compute_means_weighted(a, weights, k);
    auto after = new_means.compute_means_weighted(b, weights, k);
    for (size_t i = 0; i < size_t(k) * d; ++i)
      if (std::abs(before.data()[i] - after.data()[i]) > 1e-6f)
        throw std::runtime_error("Packed assignment changed weighted float updates");
    centers = std::move(after);
  }
}

template<bool Metric, class Model, template<class> class Batch>
void benchmark(const char* codec, uint32_t n, uint32_t k, uint32_t d, uint32_t repeats,
               uint32_t loops) {
  checks<Metric, Model, Batch>();
  verify_weighted_updates<Metric, Model, Batch>();
  std::mt19937 random(91);
  std::normal_distribution<float> normal;
  std::vector<float> values(size_t(n) * d), center_values(size_t(k) * d);
  for (float& value : values)
    value = normal(random);
  for (float& value : center_values)
    value = normal(random);
  std::vector<size_t> offsets;
  const auto centers = singletons(center_values, d, offsets);
  const size_t packed_offsets[] = {0, center_values.size()};
  const PointCloudSetView packed_centers{1, d, center_values.data(), packed_offsets};
  Model model;
  model.encoder.seed_ = 11;
  model.train(centers);
  const auto packed_query = model.quantize_query(PointCloud{values.data(), n, d});
  std::vector<typename Model::EncodedQuery> queries(n);
  std::vector<const typename Model::EncodedQuery*> pointers(n);
  parlay::parallel_for(0, n, [&](size_t i) {
    queries[i] = model.quantize_query(PointCloud{values.data() + i * d, 1, d});
    pointers[i] = &queries[i];
  });
  const auto old_db = model.encode(centers);
  const auto packed_db = model.encode(packed_centers);
  std::vector<std::pair<uint32_t, float>> old_ids(n);
  std::vector<uint32_t> ids(n);
  auto assign = [&](unsigned variant) {
    switch (variant) {
      case 0:
        Batch<typename Model::EncodedSet>::TopKIntoUninitialized(pointers, old_db, 1,
                                                                 old_ids.data());
        break;
      case 1: packed_assign<Metric>(packed_query, packed_db, ids); break;
      case 2: {
        const auto db = model.encode(centers);
        Batch<typename Model::EncodedSet>::TopKIntoUninitialized(pointers, db, 1, old_ids.data());
        break;
      }
      case 3: {
        const auto db = model.encode(packed_centers);
        packed_assign<Metric>(packed_query, db, ids);
        break;
      }
    }
  };
  for (unsigned v = 0; v < 4; ++v)
    assign(v);
  // Random nondegenerate fixture: the original singleton path and the packed
  // path must select equivalent encoded objectives before any timing is used.
  for (size_t i = 0; i < n; ++i) {
    if (std::abs(score<Metric>(packed_query, packed_db, i, old_ids[i].first) -
                 score<Metric>(packed_query, packed_db, i, ids[i])) >
        2e-5f * std::max(1.0f, std::abs(old_ids[i].second)))
      throw std::runtime_error("Unpacked/packed assignment objective mismatch");
  }
  std::vector<std::vector<double>> times(4);
  uint64_t checksum = 0;
  for (size_t rep = 0; rep < repeats; ++rep) {
    for (unsigned position = 0; position < 4; ++position) {
      const unsigned variant = rep % 2 ? 3 - position : position;
      const auto start = Clock::now();
      for (size_t loop = 0; loop < loops; ++loop)
        assign(variant);
      times[variant].push_back(std::chrono::duration<double>(Clock::now() - start).count() / loops);
      for (size_t i = 0; i < n; ++i)
        checksum += variant % 2 ? ids[i] : old_ids[i].first;
    }
  }
#ifdef __AVX512F__
  const char* isa = "avx512";
#elif defined(__AVX2__)
  const char* isa = "avx2 (packed argmin SIMD)";
#else
  const char* isa = "scalar";
#endif
  const char* names[] = {"unpacked_kernel", "packed_kernel", "unpacked_assignment",
                         "packed_assignment"};
  std::cout << std::setprecision(17) << "{\"codec\":\"" << codec << "\",\"metric\":\""
            << (Metric ? "l2" : "ip") << "\",\"isa\":\"" << isa << "\",\"n\":" << n
            << ",\"k\":" << k << ",\"d\":" << d << ",\"loops\":" << loops
            << ",\"checks\":\"passed\",\"checksum\":" << checksum << ",\"seconds\":{";
  for (size_t variant = 0; variant < 4; ++variant) {
    if (variant) std::cout << ',';
    std::cout << '"' << names[variant] << "\":[";
    for (size_t i = 0; i < times[variant].size(); ++i) {
      if (i) std::cout << ',';
      std::cout << times[variant][i];
    }
    std::cout << ']';
  }
  std::cout << "}}\n";
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 3 && argc != 8)
      throw std::invalid_argument(
          "usage: inner_assignment_bench tq4|tq8 ip|l2 [n k d repeats loops]");
    const std::string codec = argv[1], metric = argv[2];
    if ((codec != "tq4" && codec != "tq8") || (metric != "ip" && metric != "l2"))
      throw std::invalid_argument("unsupported codec or inner objective");
    uint32_t params[] = {2800, 140, 128, 7, 10};
    if (argc == 8)
      for (size_t i = 0; i < 5; ++i) {
        size_t used;
        const std::string text = argv[i + 3];
        const auto value = std::stoull(text, &used);
        if (used != text.size() || text.empty() || text[0] == '-' || value == 0 ||
            value > UINT32_MAX)
          throw std::invalid_argument("benchmark sizes must be positive uint32 values");
        params[i] = uint32_t(value);
      }
    if (params[2] > 65535) throw std::invalid_argument("dimension exceeds integer scoring bound");
#define RUN(METRIC, MODEL, BATCH)                                                                  \
  benchmark<METRIC, MODEL<METRIC>, BATCH>(codec.c_str(), params[0], params[1], params[2],          \
                                          params[3], params[4])
    if (codec == "tq4") {
      if (metric == "ip")
        RUN(false, mvsic::turboquant_mv::Model, mvsic::turboquant_mv::ManyToMany);
      else
        RUN(true, mvsic::turboquant_mv::Model, mvsic::turboquant_mv::ManyToMany);
    } else {
      if (metric == "ip")
        RUN(false, mvsic::turboquant_8bit_mv::Model, mvsic::turboquant_8bit_mv::ManyToMany);
      else
        RUN(true, mvsic::turboquant_8bit_mv::Model, mvsic::turboquant_8bit_mv::ManyToMany);
    }
#undef RUN
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
