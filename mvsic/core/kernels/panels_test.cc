#include "mvsic/core/kernels/panels_avx2.h"
#include "mvsic/core/kernels/capabilities.h"
#include "mvsic/core/data.h"
#include "mvsic/algorithms/mvivf/mvivf.h"
#include "mvsic/core/quantization/turboquant_mv.h"
#include "mvsic/core/quantization/turboquant_1bit_mv.h"
#include "mvsic/core/quantization/turboquant_8bit_mv.h"

#include <bit>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
void check(float actual, float expected) {
  if (!std::isfinite(actual) || !std::isfinite(expected) ||
      std::abs(actual - expected) > 2e-5f * std::max(1.0f, std::abs(expected)))
    throw std::runtime_error("AVX2 prepared-panel oracle mismatch");
}

struct MappedClouds {
  const mvsic::PointCloudSetView& source;
  std::vector<uint32_t> rows{3, 0, 4};
  std::vector<size_t> offsets{0};
  std::vector<uint32_t> ids;
  explicit MappedClouds(const mvsic::PointCloudSetView& source) : source(source) {
    for (auto row : rows) {
      offsets.push_back(offsets.back() + source.get_size(row) * size_t(source.d));
      ids.push_back(source.get_id(row));
    }
  }
  uint32_t get_dims() const { return source.d; }
  std::span<const size_t> get_offsets() const { return offsets; }
  std::span<const uint32_t> get_ids() const { return ids; }
  const float* data(size_t i) const { return source.data(rows[i]); }
};

// The Lloyd center view exposes only data(), so also exercise that fallback.
struct ContiguousClouds {
  const mvsic::PointCloudSetView& source;
  uint32_t get_dims() const { return source.d; }
  auto get_offsets() const { return source.get_offsets(); }
  auto get_ids() const { return source.get_ids(); }
  const float* data() const { return source.values; }
};

template<class Model>
void verify_mapped_encoding(const mvsic::PointCloudSetView& points) {
  MappedClouds mapped(points);
  const auto gathered = mvsic::gather_point_clouds(points, mapped.rows);
  Model model;
  model.train(points);
  const auto direct = model.encode(mapped);
  const auto packed = model.encode(gathered);
  const auto contiguous = model.encode(ContiguousClouds{gathered});
  for (const auto* other : {&packed, &contiguous}) {
    if (!std::equal(direct.panel_data.begin(), direct.panel_data.end(),
                    other->panel_data.begin(), other->panel_data.end()))
      throw std::runtime_error("Mapped encoding differs from gathered panels");
    const auto query = model.quantize_query(points[2]);
    const auto unwritten = std::pair<uint32_t, float>{UINT32_MAX, std::numeric_limits<float>::quiet_NaN()};
    std::vector<std::pair<uint32_t, float>> a(mapped.rows.size(), unwritten), b(a.size(), unwritten);
    direct.distances_all(query, a.data());
    other->distances_all(query, b.data());
    for (size_t i = 0; i < a.size(); ++i) {
      if (a[i].first != mapped.ids[i] || a[i].first != b[i].first)
        throw std::runtime_error("Mapped encoding changed sparse document IDs");
      check(a[i].second, b[i].second);
    }
  }
}

void verify_encoding() {
  std::mt19937 random(173);
  std::normal_distribution<float> normal;
  std::vector<float> values;
  std::vector<size_t> offsets{0};
  for (size_t cloud = 0; cloud < 5; ++cloud) {
    for (size_t i = 0; i <= cloud; ++i) {
      const size_t begin = values.size();
      double norm = 0;
      for (size_t j = 0; j < 128; ++j) {
        const float v = normal(random);
        values.push_back(v);
        norm += double(v) * v;
      }
      for (size_t j = 0; j < 128; ++j) values[begin + j] /= std::sqrt(norm);
    }
    offsets.push_back(values.size());
  }
  const auto points = mvsic::own_point_clouds(std::move(values), std::move(offsets), 128,
                                             {1001, 997, 7001, 42, 37000});
  verify_mapped_encoding<mvsic::turboquant_1bit_mv::Model<false>>(points);
  verify_mapped_encoding<mvsic::turboquant_1bit_mv::Model<true>>(points);
  verify_mapped_encoding<mvsic::turboquant_mv::Model<false>>(points);
  verify_mapped_encoding<mvsic::turboquant_mv::Model<true>>(points);
  verify_mapped_encoding<mvsic::turboquant_8bit_mv::Model<false>>(points);
  verify_mapped_encoding<mvsic::turboquant_8bit_mv::Model<true>>(points);
}

// Decode only actual points, independently of the SIMD epilogues and their
// padding metadata. Antipodal vectors make every real IP distance positive,
// so an accidentally included zero padding lane is immediately observable.
template<class Query, class Database>
float scalar_ip(const Query& query, const Database& db, size_t cloud) {
  double total = 0;
  for (size_t q = 0; q < query.num_queries; ++q) {
    double best = std::numeric_limits<double>::infinity();
    for (size_t p = 0; p < db.cloud_sizes[cloud]; ++p) {
      int32_t dot = 0;
      for (size_t j = 0; j < db.total_tiles * 4; ++j) {
        const size_t pos = db.panel_offsets[cloud] + (p / 16) * db.panel_bytes +
                           (j / 4) * 64 + (p % 16) * 4 + j % 4;
        dot += (int(db.panel_data[pos]) - 128) * int(query.flat_query_data[q * query.q_stride + j]);
      }
      const double distance = -double(dot) * db.norm_scaling_factors[db.point_offsets[cloud] + p] *
                              query.norm_scaling_factors[q];
      best = std::min(best, distance);
    }
    total += best;
  }
  return float(total / query.num_queries);
}

template<class Model>
void verify_ip_padding(mvsic::IndexParams::QuantizerType codec) {
  using namespace mvsic;
  std::vector<float> values;
  std::vector<size_t> offsets{0};
  for (size_t n : {1, 7, 15, 16, 17, 33, 65, 1, 7}) {
    const size_t cloud = offsets.size() - 1;
    for (size_t p = 0; p < n; ++p) {
      const size_t begin = values.size();
      values.resize(begin + 128, 0);
      if (cloud != 7 && !(cloud == 8 && p == 0)) values[begin] = 1.0f;
    }
    offsets.push_back(values.size());
  }
  const auto points = own_point_clouds(std::move(values), std::move(offsets), 128);
  values.clear(); offsets = {0};
  for (size_t n : {1, 6, 7, 13}) {
    for (size_t p = 0; p < n; ++p) {
      const size_t begin = values.size();
      values.resize(begin + 128, 0); values[begin] = -1;
    }
    offsets.push_back(values.size());
  }
  const auto queries = own_point_clouds(std::move(values), std::move(offsets), 128);
  Model model;
  model.train(points);
  const auto db = model.encode(points);
  auto params = IndexParams::mvivf();
  params.max_leaf_size = 100;
  params.pq.method = codec;
  IndexMVIVF index(128, params);
  index.build(points);
  auto search = SearchParams::mvivf(points.n, 3, 0, false);
  const auto batch = index.search_all(queries, search);
  for (size_t q = 0; q < queries.n; ++q) {
    const auto encoded_query = model.quantize_query(queries[q]);
    const auto single = index.search(queries[q], search);
    for (const auto* neighbors : {&single, &batch[q]}) {
      if (neighbors->size() != points.n) throw std::runtime_error("IP padding test omitted documents");
      for (const auto& [id, score] : *neighbors) check(score, scalar_ip(encoded_query, db, id));
    }
    check(scalar_ip(encoded_query, db, 7), 0);
    check(scalar_ip(encoded_query, db, 8), 0);
    if (scalar_ip(encoded_query, db, 0) < 0.5f)
      throw std::runtime_error("Antipodal fixture did not produce positive IP distance");
  }
}
void verify_bucket_ties() {
  using Bucket = mvsic::turboquant_1bit_mv::internal::BoundedTopKBucket;
  std::mt19937 random(711);
  for (size_t k : {1, 7, 10, 15, 16, 17, 31, 32, 33, 50, 127, 128, 129, 511, 512}) {
    Bucket bucket;
    bucket.init(k);
    std::vector<std::pair<float, uint32_t>> candidates;
    // Full-score ties, then strictly better arrivals must evict the largest
    // tied ID. High unsigned IDs exercise both halves of the ID domain.
    for (uint32_t i = 0; i < 2 * k + 37; ++i)
      candidates.emplace_back(-float(i % 7), i % 2 ? UINT32_MAX - 3 * i : 3 * i);
    std::shuffle(candidates.begin(), candidates.end(), random);
    for (const auto& candidate : candidates) bucket.try_insert(candidate.first, candidate.second);
    std::sort(candidates.begin(), candidates.end());
    std::vector<std::pair<uint32_t, float>> actual(k);
    for (bool ordered : {true, false}) {
      bucket.finalize(actual.data(), ordered);
      if (!ordered) std::sort(actual.begin(), actual.end(), [](const auto& a, const auto& b) {
        return a.second < b.second || (a.second == b.second && a.first < b.first);
      });
      for (size_t i = 0; i < k; ++i)
        if (actual[i] != std::pair<uint32_t, float>{candidates[i].second, candidates[i].first})
          throw std::runtime_error("Bounded top-k changed distance/ID ordering at a tie");
    }
  }
  // Both drain modes fully write unfilled slots, including empty buckets.
  for (size_t count : {0, 1, 7, 16, 32}) {
    Bucket bucket;
    bucket.init(33);
    for (size_t i = 0; i < count; ++i) bucket.try_insert(-float(i), uint32_t(i));
    std::pair<uint32_t, float> actual[33];
    for (bool ordered : {true, false}) {
      bucket.finalize(actual, ordered);
      for (size_t i = count; i < 33; ++i)
        if (actual[i] != std::pair<uint32_t, float>{UINT32_MAX, std::numeric_limits<float>::max()})
          throw std::runtime_error("Bounded top-k drain did not initialize a sentinel");
    }
  }
  for (bool tied_arrival : {false, true}) {
    Bucket bucket;
    bucket.init(2);
    bucket.try_insert(0, 1);
    bucket.try_insert(0, 2);
    bucket.try_insert(tied_arrival ? 0 : -1, tied_arrival ? 0 : 3);
    std::pair<uint32_t, float> actual[2];
    bucket.finalize_sorted(actual);
    const std::pair<uint32_t, float> expected[] = {{tied_arrival ? 0u : 3u, tied_arrival ? 0.0f : -1.0f}, {1, 0}};
    if (!std::equal(std::begin(actual), std::end(actual), std::begin(expected)))
      throw std::runtime_error("Bounded top-k evicted the wrong equal-score candidate");
  }
}

template<class Model, template<class> class Batch>
void verify_codec_topk_ties() {
  constexpr uint32_t n = 530, d = 128;
  std::vector<float> values(size_t(n) * d, 0);
  std::vector<size_t> offsets(n + 1);
  std::vector<uint32_t> ids(n);
  for (uint32_t i = 0; i < n; ++i) {
    values[size_t(i) * d] = 1;
    offsets[i] = size_t(i) * d;
    ids[i] = UINT32_MAX - 1000 - 7 * i;  // Later equal scores carry smaller IDs.
  }
  offsets[n] = values.size();
  const auto points = mvsic::own_point_clouds(std::move(values), std::move(offsets), d, ids);
  Model model;
  model.train(points);
  const auto database = model.encode(points);
  const auto query = model.quantize_query(points[0]);
  std::vector<const typename Model::EncodedQuery*> queries(9, &query);
  // 513 exercises the TQ1 heap fallback; 7 exercises a partial SIMD bucket.
  for (uint32_t k : {7, 513}) {
    std::vector<std::pair<uint32_t, float>> result(queries.size() * k);
    Batch<typename Model::EncodedSet>::TopKIntoUninitialized(queries, database, k, result.data(), 8, false);
    for (size_t q = 0; q < queries.size(); ++q)
      for (size_t i = 0; i < k; ++i)
        if (result[q * k + i].first != ids[n - 1 - i] ||
            result[q * k + i].second != result[0].second)
          throw std::runtime_error("Codec top-k discarded a smaller-ID tied candidate");
  }
}

template<bool Metric>
void verify(size_t points, size_t dims, size_t nq, std::mt19937& random) {
  const size_t np = (points + 15) / 16, tiles = dims / 4, panel_bytes = tiles * 64;
  std::vector<uint8_t> panels(np * panel_bytes);
  std::vector<int8_t> query(nq * dims);
  std::vector<float> norms(np * 16), sqns(np * 16), qnorms(nq), qsqns(nq);
  std::vector<int32_t> sums(nq, 0);
  for (auto& x : panels) x = uint8_t(random());
  for (auto& x : query) x = int8_t(random());
  for (size_t i = 0; i < norms.size(); ++i) { norms[i] = float(1 + random() % 7) / 100; sqns[i] = float(random() % 13); }
  for (size_t q = 0; q < nq; ++q) {
    qnorms[q] = float(1 + random() % 9) / 100; qsqns[q] = float(random() % 11);
    for (size_t j = 0; j < dims; ++j) sums[q] += query[q * dims + j];
  }
  float expected = 0;
  for (size_t q = 0; q < nq; ++q) {
    float best = std::numeric_limits<float>::max();
    for (size_t p = 0; p < points; ++p) {
      int32_t dot = 0;
      for (size_t j = 0; j < dims; ++j)
        dot += (int(panels[(p / 16) * panel_bytes + (j / 4) * 64 + (p % 16) * 4 + j % 4]) - 128) * int(query[q * dims + j]);
      float score = -float(dot) * norms[p] * qnorms[q];
      if constexpr (Metric) score = sqns[p] + 2 * score + qsqns[q];
      best = std::min(best, score);
    }
    expected += best;
  }
  check(mvsic::cpu::chamfer_panels_avx2<Metric>(query.data(), qnorms.data(), qsqns.data(), sums.data(), dims,
      nq, panels.data(), norms.data(), sqns.data(), tiles, panel_bytes, np, points), expected);

  const size_t bit_bytes = dims / 8, bit_tiles = (bit_bytes + 3) / 4, bit_panel = bit_tiles * 64;
  std::vector<uint8_t> bit_panels(np * bit_panel), bits(nq * bit_bytes);
  for (auto& x : bit_panels) x = uint8_t(random());
  for (auto& x : bits) x = uint8_t(random());
  expected = 0;
  for (size_t q = 0; q < nq; ++q) {
    int best = INT32_MAX;
    for (size_t p = 0; p < points; ++p) {
      int count = 0;
      for (size_t j = 0; j < bit_bytes; ++j) {
        const uint8_t db = bit_panels[(p / 16) * bit_panel + (j / 4) * 64 + (p % 16) * 4 + j % 4];
        count += __builtin_popcount(unsigned(db ^ bits[q * bit_bytes + j]));
      }
      best = std::min(best, count);
    }
    expected += float(best) * (Metric ? 4.0f : 2.0f) / float(dims) - (Metric ? 0.0f : 1.0f);
  }
  check(mvsic::cpu::chamfer_hamming_panels_avx2<Metric>(bits.data(), bit_bytes, nq, bit_bytes,
      bit_panels.data(), bit_panel, bit_tiles, points, dims), expected);
  std::vector<uint8_t> broadcast(nq * bit_tiles * 64, 0);
  for (size_t q = 0; q < nq; ++q)
    for (size_t tile = 0; tile < bit_tiles; ++tile)
      for (size_t lane = 0; lane < 16; ++lane)
        std::memcpy(broadcast.data() + q * bit_tiles * 64 + tile * 64 + lane * 4,
                    bits.data() + q * bit_bytes + tile * 4, 4);
  check(mvsic::cpu::chamfer_hamming_panels_avx2<Metric>(broadcast.data(), bit_tiles * 64, nq, bit_bytes,
      bit_panels.data(), bit_panel, bit_tiles, points, dims, 64), expected);
}
// Decode real points independently. Opposite codes attain the byte-accumulator
// bound, while zero padding would win incorrectly if a tail mask were lost.
template<bool Metric>
void verify_hamming_accumulation(size_t tiles, size_t points, size_t nq, int pattern,
                                 std::mt19937& random) {
  using namespace mvsic::turboquant_1bit_mv;
  const size_t bytes = tiles * 4 - (pattern == 2 ? 1 : 0);
  const size_t dimension = bytes * 8, panel_bytes = tiles * internal::kTileBytes;
  Quantized_Point_Cloud_Set<Metric> db;
  db.num_bytes_per_datapoint = bytes;
  db.padded_dim = dimension;
  db.num_hamming_tiles = tiles;
  db.panel_bytes = panel_bytes;
  db.panel_data.resize(((points + 15) / 16) * panel_bytes, 0);
  db.panel_offsets = {0, db.panel_data.size()};
  db.cloud_sizes = {points};
  for (size_t p = 0; p < points; ++p)
    for (size_t b = 0; b < bytes; ++b)
      db.panel_data[(p / 16) * panel_bytes + (b / 4) * 64 + (p % 16) * 4 + b % 4] =
          pattern == 2   ? uint8_t(random())
          : pattern == 1 ? 255
                         : 0;
  std::vector<uint8_t> bits(nq * bytes, 0);
  if (pattern == 2)
    for (auto& byte : bits)
      byte = uint8_t(random());
  FusedQueryBatch<Metric> queries;
  queries.num_source_clouds = 1;
  queries.total_embeddings = nq;
  queries.num_bytes_per_datapoint = bytes;
  queries.padded_dim = dimension;
  queries.num_hamming_tiles = tiles;
  queries.qbuf_tile_stride = panel_bytes;
  queries.emb_offsets = {0, nq};
  queries.flat_qbuf.resize(nq * panel_bytes);
  for (size_t q = 0; q < nq; ++q)
    for (size_t tile = 0; tile < tiles; ++tile) {
      uint32_t word = 0;
      std::memcpy(&word, bits.data() + q * bytes + tile * 4,
                  std::min<size_t>(4, bytes - tile * 4));
      for (size_t lane = 0; lane < 16; ++lane)
        std::memcpy(queries.flat_qbuf.data() + q * panel_bytes + tile * 64 + lane * 4, &word, 4);
    }
  const float scale = (Metric ? 4.0f : 2.0f) / float(dimension);
  constexpr float addend = Metric ? 0.0f : -1.0f;
  int64_t total = 0;
  float expected_avx2 = 0;
  std::vector<float> expected(nq), actual(nq);
  for (size_t q = 0; q < nq; ++q) {
    int32_t best = INT32_MAX;
    for (size_t p = 0; p < points; ++p) {
      int32_t count = 0;
      for (size_t b = 0; b < bytes; ++b)
        count += __builtin_popcount(
            unsigned(bits[q * bytes + b] ^
                     db.panel_data[(p / 16) * panel_bytes + (b / 4) * 64 + (p % 16) * 4 + b % 4]));
      best = std::min(best, count);
    }
    total += best;
    expected[q] = scale * float(best) + addend;
    // AVX2 preserves its existing per-query float sum; AVX512 converts a
    // combined integer sum below. Test each contract in its original order.
    expected_avx2 += (Metric ? 4.0f : 2.0f) / float(dimension) * float(best) - (Metric ? 0.0f : 1.0f);
  }
  const float packed = mvsic::cpu::chamfer_hamming_panels_avx2<Metric>(
      bits.data(), bytes, nq, bytes, db.panel_data.data(), panel_bytes, tiles, points, dimension);
  const float broadcast = mvsic::cpu::chamfer_hamming_panels_avx2<Metric>(
      queries.flat_qbuf.data(), panel_bytes, nq, bytes, db.panel_data.data(), panel_bytes,
      tiles, points, dimension, 64);
  if (std::bit_cast<uint32_t>(packed) != std::bit_cast<uint32_t>(expected_avx2) ||
      std::bit_cast<uint32_t>(broadcast) != std::bit_cast<uint32_t>(expected_avx2))
    throw std::runtime_error("AVX2 Hamming byte accumulation changed exact scores");
#ifdef __AVX512F__
  score_one_db_cloud(queries, db, 0, actual.data());
  if (actual != expected) throw std::runtime_error("TQ1 per-embedding Hamming oracle mismatch");
  const float expected_sum = scale * float(total) + float(nq) * addend;
  const float sum = internal::chamfer_panels_qbuf<Metric>(queries.flat_qbuf.data(), panel_bytes, nq,
                                                          db.panel_data.data(), panel_bytes, tiles,
                                                          points, dimension);
  float grouped = 0;
  score_one_db_cloud_grouped(queries, db, 0, &grouped);
  if (sum != expected_sum || grouped != expected_sum / float(nq))
    throw std::runtime_error("TQ1 single/grouped Hamming oracle mismatch");
#endif
}

#ifdef __AVX512F__
void verify_hamming_long_cloud() {
  using namespace mvsic::turboquant_1bit_mv;
  constexpr size_t dimension = 65536, tiles = dimension / 32, nq = 32769;
  constexpr size_t bytes = dimension / 8, panel_bytes = tiles * internal::kTileBytes;
  // A zero query stride repeats one antipodal embedding using only 128KiB.
  // The actual scoring loops must sum nq*dimension > INT32_MAX safely.
  FusedQueryBatch<false> query;
  query.num_source_clouds = 1;
  query.total_embeddings = nq;
  query.num_bytes_per_datapoint = bytes;
  query.padded_dim = dimension;
  query.num_hamming_tiles = tiles;
  query.qbuf_tile_stride = 0;
  query.emb_offsets = {0, nq};
  query.flat_qbuf.resize(panel_bytes, 0);
  Quantized_Point_Cloud_Set<false> db;
  db.num_bytes_per_datapoint = bytes;
  db.padded_dim = dimension;
  db.num_hamming_tiles = tiles;
  db.panel_bytes = panel_bytes;
  db.panel_data.resize(panel_bytes, 255);
  db.panel_offsets = {0, panel_bytes};
  db.cloud_sizes = {1};
  const float sum = internal::chamfer_panels_qbuf<false>(
      query.flat_qbuf.data(), 0, nq, db.panel_data.data(), panel_bytes, tiles, 1, dimension);
  float grouped = 0;
  score_one_db_cloud_grouped(query, db, 0, &grouped);
  if (sum != float(nq) || grouped != 1.0f)
    throw std::runtime_error("TQ1 long-cloud Hamming sum overflowed");
}
#endif
}  // namespace
__attribute__((noinline)) static int run_tests() {
  try {
    std::mt19937 random(4242);
    for (size_t points : {1,7,8,15,16,17,41})
      for (size_t dims : {32,64,128,256,1024})
        for (size_t nq : {1,3,8}) {
          verify<false>(points,dims,nq,random);
          verify<true>(points,dims,nq,random);
        }
    for (size_t tiles : {1, 2, 4, 9, 31, 32, 64})
      for (size_t points : {1, 17, 31, 32, 65})
        for (size_t nq : {1, 13})
          for (int pattern : {0, 1, 2}) {
            verify_hamming_accumulation<false>(tiles, points, nq, pattern, random);
            verify_hamming_accumulation<true>(tiles, points, nq, pattern, random);
          }
#ifdef __AVX512F__
    verify_hamming_long_cloud();
#endif
    verify_encoding();
    verify_bucket_ties();
    verify_codec_topk_ties<mvsic::turboquant_1bit_mv::Model<false>, mvsic::turboquant_1bit_mv::ManyToMany>();
    verify_codec_topk_ties<mvsic::turboquant_mv::Model<false>, mvsic::turboquant_mv::ManyToMany>();
    verify_codec_topk_ties<mvsic::turboquant_8bit_mv::Model<false>, mvsic::turboquant_8bit_mv::ManyToMany>();
    verify_ip_padding<mvsic::turboquant_mv::Model<false>>(mvsic::IndexParams::QuantizerType::TurboQuant);
    verify_ip_padding<mvsic::turboquant_8bit_mv::Model<false>>(mvsic::IndexParams::QuantizerType::EightBitTQ);
    std::cout << "Prepared-panel checks passed\n";
    return 0;
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}

// The dedicated AVX-512 binary must check OS vector state before entering any
// helpers compiled for that ISA. The ordinary binary keeps its AVX2 baseline.
#ifdef __AVX512F__
__attribute__((target("arch=x86-64-v3")))
#endif
int main() {
#ifdef __AVX512F__
  if (!mvsic::kernels::cpu_capabilities().usable_avx512()) {
    std::cout << "Skipping AVX-512 panel checks: CPU or OS state unavailable\n";
    return 77;
  }
#endif
  return run_tests();
}
