#include "mvsic/algorithms/mvivf/mvivf_internal.h"

#include <algorithm>
#include <array>
#include <barrier>
#include <bit>
#include <cmath>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>
#include <thread>
#include <unistd.h>

#include "gtest/gtest.h"
#include "parlay/parallel.h"

namespace {
using namespace mvsic;

struct TemporaryFiles {
  std::filesystem::path base;
  explicit TemporaryFiles(const char* name)
      : base(std::filesystem::temp_directory_path() /
             (std::string("mvsic-") + name + "-" + std::to_string(getpid()))) {}
  ~TemporaryFiles() {
    for (const char* extension : {".mvif", ".mvif.bad", ".pcs"}) {
      std::error_code ignored;
      std::filesystem::remove(base.string() + extension, ignored);
    }
  }
};

void check(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
template<class F> void rejects(F operation) {
  bool rejected = false;
  try { operation(); } catch (const std::exception&) { rejected = true; }
  check(rejected, "Invalid input was accepted");
}

PointCloudSetView sample(uint32_t n, uint32_t d, uint32_t seed, bool unit = true) {
  std::mt19937 generator(seed);
  std::normal_distribution<float> normal;
  std::vector<float> values;
  std::vector<size_t> offsets{0};
  std::vector<uint32_t> ids;
  for (uint32_t row = 0; row < n; ++row) {
    for (uint32_t v = 0; v < 1 + row % 5; ++v) {
      const size_t begin = values.size();
      double norm = 0;
      for (uint32_t j = 0; j < d; ++j) { const float x = normal(generator); values.push_back(x); norm += x * x; }
      if (unit) for (uint32_t j = 0; j < d; ++j) values[begin + j] /= std::sqrt(norm);
    }
    offsets.push_back(values.size());
    ids.push_back(1000 + 17 * row);
  }
  return own_point_clouds(std::move(values), std::move(offsets), d, std::move(ids));
}
Neighbors oracle(PointCloud query, PointCloudSetView corpus, size_t k) {
  Neighbors result;
  for (uint32_t row = 0; row < corpus.n; ++row) {
    const auto document = corpus[row];
    double sum = 0;
    for (uint32_t q = 0; q < query.n; ++q) {
      double best = std::numeric_limits<double>::infinity();
      for (uint32_t p = 0; p < document.n; ++p) {
        double value = 0;
        for (uint32_t j = 0; j < query.d; ++j) {
          const double a = query.data(q)[j], b = document.data(p)[j];
          value -= a * b;
        }
        best = std::min(best, value);
      }
      sum += best;
    }
    result.emplace_back(corpus.get_id(row), float(sum / query.n));
  }
  std::sort(result.begin(), result.end(), [](auto a, auto b) {
    return a.second != b.second ? a.second < b.second : a.first < b.first;
  });
  result.resize(std::min(k, result.size()));
  return result;
}
void equal(const Neighbors& a, const Neighbors& b, float tolerance = 2e-5) {
  check(a.size() == b.size(), "Neighbor counts differ");
  for (size_t i = 0; i < a.size(); ++i) {
    check(a[i].first == b[i].first, "Neighbor IDs differ");
    check(std::abs(a[i].second - b[i].second) <= tolerance * std::max(1.0f, std::abs(a[i].second)), "Neighbor scores differ");
  }
}

void verify_retrieval(const std::filesystem::path& path) {
  auto corpus = sample(43, 128, 42);
  const auto queries = sample(11, 128, 53);
  auto params = IndexParams::mvivf(4, 6, false, 0, 2, 100, 20, "Random", 3, true, 2);
  params.quantize_centers = false;
  params.pq.method = IndexParams::QuantizerType::None;
  IndexMVIVF index(corpus.d, params);
  index.build(corpus);
  check(index.get_height() > 1, "Tree did not split");
  auto search = SearchParams::mvivf(10, 1000, 0, false);
  auto batch = index.search_all(queries, search);
  for (size_t i = 0; i < queries.n; ++i) {
    equal(batch[i], oracle(queries[i], corpus, search.k));
    equal(batch[i], index.search(queries[i], search));
  }
  auto many_queries = sample(69, 128, 71);
  auto many = index.search_all(many_queries, search);
  for (size_t i = 0; i < many_queries.n; ++i)
    equal(many[i], oracle(many_queries[i], corpus, search.k));
  index.save(path.string());
  IndexMVIVF loaded(corpus.d, params);
  loaded.load(path.string(), corpus);
  equal(loaded.search(queries[2], search), batch[2]);
  auto changed = sample(corpus.n, corpus.d, 5);
  rejects([&] { loaded.search(queries[0], changed, search); });
  auto bad = search; bad.k = 0;
  rejects([&] { loaded.search(queries[0], bad); });
  auto invalid_values = std::vector<float>(128, 0);
  invalid_values[0] = std::numeric_limits<float>::quiet_NaN();
  rejects([&] { loaded.search(PointCloud{invalid_values.data(), 1, 128, 0}, search); });
  invalid_values[0] = std::numeric_limits<float>::max();
  rejects([&] { loaded.search(PointCloud{invalid_values.data(), 1, 128, 0}, search); });
  invalid_values[0] = std::numeric_limits<float>::quiet_NaN();
  auto invalid_corpus = own_point_clouds(std::move(invalid_values), {0, 128}, 128);
  rejects([&] { loaded.build(invalid_corpus); });
  // The finite accumulation bound also protects shared clustering kernels for
  // long clouds. Reject oversized norms at build/load before any scoring loop.
  std::vector<float> huge_vector(128, 0), zero_query(32 * 128, 0);
  huge_vector[0] = 4e18f;
  auto huge_corpus = own_point_clouds(std::move(huge_vector), {0, 128}, 128);
  rejects([&] {
    loaded.build(huge_corpus);
    loaded.search(PointCloud{zero_query.data(), 32, 128, 0}, search);
  });
  auto duplicate_ids = own_point_clouds(std::vector<float>(256, 0), {0,128,256}, 128, {7,7});
  rejects([&] { loaded.build(duplicate_ids); });
  auto reserved_id = own_point_clouds(std::vector<float>(128, 0), {0,128}, 128, {UINT32_MAX});
  rejects([&] { loaded.build(reserved_id); });
  auto empty_cloud = own_point_clouds(std::vector<float>(128, 0), {0,0,128}, 128);
  rejects([&] { loaded.build(empty_cloud); });
  std::ofstream truncated(path.string() + ".bad", std::ios::binary);
  truncated << "MVIF"; truncated.close();
  rejects([&] { loaded.load(path.string() + ".bad", corpus); });
  equal(loaded.search(queries[2], search), batch[2]);
  std::filesystem::remove(path.string() + ".bad");
  auto stats = loaded.search_with_stats(queries[0], corpus, search);
  check(std::get<2>(stats).size() == 12, "Incorrect timer positions");

  // Every serving representation loads the same skeleton. Full candidate
  // rerank tests ID/ownership/metric semantics independently of approximation.
  BatchNeighbors tq8_reference;
  BatchNeighbors compressed_tq8_reference;
  for (auto codec : {IndexParams::QuantizerType::OneBitTQ, IndexParams::QuantizerType::TurboQuant,
                     IndexParams::QuantizerType::EightBitTQ}) {
    params.pq.method = codec;
    params.quantize_centers = true;
    IndexMVIVF quantized(corpus.d, params);
    quantized.load(path.string(), corpus);
    search.num_rerank = corpus.n;
    auto exact = quantized.search_all(queries, search);
    for (size_t i = 0; i < queries.n; ++i) {
      equal(exact[i], oracle(queries[i], corpus, search.k));
      equal(exact[i], quantized.search(queries[i], search));
    }
    search.tq8_rerank = true;
    auto tq8 = quantized.search_all(queries, search);
    if (tq8_reference.empty()) tq8_reference = tq8;
    else for (size_t i = 0; i < queries.n; ++i) equal(tq8[i], tq8_reference[i]);
    // Force an actual compression to one vector. TQ8 leaves can reuse this
    // query encoding; other leaf codecs independently encode it for rerank.
    search.query_compression = SearchParams::QueryCompression::Wards;
    search.query_alignment = 1;
    search.query_alignment_strict = true;
    search.compress_rerank = true;
    const auto compressed_tq8 = quantized.search_all(queries, search);
    if (compressed_tq8_reference.empty()) compressed_tq8_reference = compressed_tq8;
    else for (size_t i = 0; i < queries.n; ++i) equal(compressed_tq8[i], compressed_tq8_reference[i]);
    search.query_compression = SearchParams::QueryCompression::None;
    search.query_alignment = 0;
    search.query_alignment_strict = false;
    search.compress_rerank = false;
    search.root_m2m = false;
    auto unshared = quantized.search_all(queries, search);
    for (size_t i = 0; i < queries.n; ++i) {
      equal(tq8[i], unshared[i]);
      equal(tq8[i], quantized.search(queries[i], search));
    }
    search.root_m2m = true;
    search.num_rerank = 0;
    search.k = corpus.n;
    auto all = quantized.search_all(queries, search);
    for (size_t i = 0; i < queries.n; ++i) equal(all[i], quantized.search(queries[i], search));
    search.k = 10;
    search.num_rerank = corpus.n;
    search.tq8_rerank = false;
    search.query_compression = SearchParams::QueryCompression::Wards;
    search.query_compression_threshold = 0.5;
    auto compressed = quantized.search_all(queries, search);
    for (size_t i = 0; i < queries.n; ++i) equal(compressed[i], oracle(queries[i], corpus, search.k));
    search.query_compression = SearchParams::QueryCompression::None;
    // Reload after first TQ8 use must replace both its once_flag and corpus.
    quantized.load(path.string(), changed);
    IndexMVIVF fresh(corpus.d, params);
    fresh.load(path.string(), changed);
    search.tq8_rerank = true;
    equal(quantized.search(queries[0], search), fresh.search(queries[0], search));
    search.tq8_rerank = false;
  }

  // A leaf-only root, moved index and released caller owner all remain usable.
  auto tiny = sample(3, 128, 12);
  params.pq.method = IndexParams::QuantizerType::None;
  IndexMVIVF single(128, params);
  single.build(tiny);
  check(single.get_height() == 1, "Leaf-only root has incorrect height");
  auto expected = oracle(queries[0], tiny, 10);
  std::weak_ptr<const void> owner = tiny.owner;
  tiny = {};
  check(!owner.expired(), "Index failed to retain corpus ownership");
  IndexMVIVF moved = std::move(single);
  equal(moved.search(queries[0], search), expected);
  auto scaled = sample(19, 128, 4, false);
  moved.build(scaled);
  search.num_rerank = scaled.n;
  equal(moved.search(queries[0], search), oracle(queries[0], scaled, search.k));
  params.pq.method = IndexParams::QuantizerType::OneBitTQ;
  IndexMVIVF unit_only(128, params);
  rejects([&] { unit_only.build(scaled); });
  auto duplicate = sample(6, 128, 6);
  std::vector<float> same(6 * 128, 0); for (size_t i = 0; i < 6; ++i) same[i * 128] = 1;
  duplicate = own_point_clouds(std::move(same), {0,128,256,384,512,640,768}, 128);
  params.max_leaf_size = 1;
  IndexMVIVF collapsed(128, params);
  collapsed.build(duplicate);
  check(collapsed.get_height() == 1, "Collapsed cluster recursed");
  check(collapsed.search(queries[0], search).size() == 6, "Collapsed root omitted documents");
}

void verify_pcs(const std::filesystem::path& path) {
  auto source = sample(4, 3, 8, false);
  std::ofstream file(path, std::ios::binary);
  auto write = [&](auto value) { file.write(reinterpret_cast<const char*>(&value), sizeof(value)); };
  write(size_t(source.d)); write(size_t(source.n)); write(source.total_size());
  file.write(reinterpret_cast<const char*>(source.values), source.num_bytes());
  write(size_t(source.n) + 1);
  file.write(reinterpret_cast<const char*>(source.offsets), (source.n + 1) * sizeof(size_t));
  file.close();
  auto mapped = load_point_clouds(path.string());
  auto copied = load_point_clouds(path.string(), false);
  check(mapped.d == source.d && mapped.n == source.n, "PCS shape mismatch");
  for (size_t i = 0; i < source.offsets[source.n]; ++i)
    check(mapped.values[i] == source.values[i] && copied.values[i] == source.values[i], "PCS values differ");
  const std::vector<uint32_t> rows{3, 0, 3, 1};
  auto gathered = gather_point_clouds(source, rows);
  auto dense = gather_point_clouds(source, rows, false);
  for (size_t i = 0; i < rows.size(); ++i) {
    check(gathered.get_id(i) == source.get_id(rows[i]) && dense.get_id(i) == i,
          "Gathered IDs differ");
    check(gathered.get_size(i) == source.get_size(rows[i]), "Gathered shape differs");
    for (size_t j = 0; j < size_t(gathered.get_size(i)) * source.d; ++j)
      check(gathered.data(i)[j] == source.data(rows[i])[j] && dense.data(i)[j] == gathered.data(i)[j],
            "Gathered values differ");
  }
  const float retained = gathered.values[0];
  source = {};
  check(gathered.values[0] == retained, "Gathered values lost their owner");
  std::filesystem::resize_file(path, std::filesystem::file_size(path) - 1);
  rejects([&] { load_point_clouds(path.string()); });
}

void verify_concurrent_search(const std::filesystem::path& path) {
  // Each caller owns one scheduler worker. Three application threads exercise
  // concurrent requests without implicitly creating three machine-sized pools.
  parlay::execute_with_scheduler(1, [&] {
    const auto corpus = sample(43, 128, 174);
    const auto queries = sample(11, 128, 62);
    auto params = IndexParams::mvivf(4, 6, false, 0, 2, 100, 20, "Random", 3, true, 2);
    params.quantize_centers = true;
    IndexMVIVF skeleton(128, params);
    skeleton.build(corpus);
    check(skeleton.get_height() > 1, "Concurrent fixture needs an internal root");
    skeleton.save(path.string());
    const auto search = SearchParams::mvivf(10, 1000, corpus.n, true);

    // TQ1 first rerank must train and encode TQ8. With TQ8 leaves the model is
    // already trained, but the shared full-corpus rerank cache is still lazy.
    for (auto codec : {IndexParams::QuantizerType::OneBitTQ, IndexParams::QuantizerType::EightBitTQ}) {
      params.pq.method = codec;
      IndexMVIVF reference(128, params);
      reference.load(path.string(), corpus);
      const auto expected = reference.search_all(queries, search);

      IndexMVIVF shared(128, params);
      shared.load(path.string(), corpus);  // Deliberately no warmup on this index.
      constexpr size_t callers = 3;
      std::barrier<> start(static_cast<std::ptrdiff_t>(callers));
      std::array<std::exception_ptr, callers> failures{};
      std::array<std::jthread, callers> threads;
      size_t launched = 0;
      try {
        for (; launched < callers; ++launched) {
          threads[launched] = std::jthread([&, caller = launched] {
            bool reached_start = false;
            try {
              parlay::execute_with_scheduler(1, [&] {
                reached_start = true;
                start.arrive_and_wait();
                for (unsigned round = 0; round < 3; ++round) {
                  auto request = search;
                  request.root_m2m = round != 1;
                  if (caller == 1) {
                    const auto actual = shared.search_all(queries, request);
                    check(actual.size() == expected.size(), "Concurrent batch size differs");
                    for (size_t q = 0; q < queries.n; ++q) equal(actual[q], expected[q]);
                  } else {
                    for (size_t q = 0; q < queries.n; ++q) {
                      if (caller == 0) equal(shared.search(queries[q], request), expected[q]);
                      else equal(std::get<0>(shared.search_with_stats(queries[q], corpus, request)), expected[q]);
                    }
                  }
                }
              });
            } catch (...) {
              if (!reached_start) start.arrive_and_drop();
              failures[caller] = std::current_exception();
            }
          });
        }
      } catch (...) {
        // Do not strand existing callers at the barrier if thread creation fails.
        for (size_t missing = launched; missing < callers; ++missing) start.arrive_and_drop();
        for (auto& thread : threads) if (thread.joinable()) thread.join();
        throw;
      }
      for (auto& thread : threads) thread.join();
      for (size_t caller = 0; caller < callers; ++caller) {
        if (!failures[caller]) continue;
        try { std::rethrow_exception(failures[caller]); }
        catch (const std::exception& error) {
          throw std::runtime_error("Concurrent caller " + std::to_string(caller) + ": " + error.what());
        }
      }
      // The cache and request-local scratch remain usable after callers exit.
      for (size_t q = 0; q < queries.n; ++q) equal(shared.search(queries[q], search), expected[q]);
    }
  });
}

TEST(MVIVF, NumericValidationPreservesScalarBoundaryDecisions) {
  constexpr double max_norm = double(std::numeric_limits<float>::max()) /
                              (16.0 * std::numeric_limits<uint32_t>::max());
  // This reference retains the previous sequential double sum. Reordering
  // positive terms can cross a rejection threshold even without float math.
  auto scalar_result = [&](const std::vector<float>& values, uint32_t d, bool unit) {
    for (size_t row = 0; row < values.size() / d; ++row) {
      double norm = 0;
      for (size_t j = 0; j < d; ++j) {
        const double value = values[row * d + j];
        norm += value * value;
      }
      if (!std::isfinite(norm)) return std::string("query values must be finite");
      if (norm > max_norm) return std::string("query squared norms exceed the float scoring range");
      if (unit && std::abs(norm - 1) > .002)
        return std::string("OneBitTQ serving requires unit-norm query vectors; use float, TQ4 or TQ8 for general norms");
    }
    return std::string{};
  };
  size_t cases = 0;
  auto verify = [&](const std::vector<float>& values, uint32_t d, bool unit, const char* fixture) {
    SCOPED_TRACE(std::string(fixture) + " d=" + std::to_string(d) + " unit=" + std::to_string(unit));
    ++cases;
    std::string actual;
    try { mvivf_internal::validate_values(values.data(), values.size() / d, d, unit, "query"); }
    catch (const std::invalid_argument& error) { actual = error.what(); }
    EXPECT_EQ(actual, scalar_result(values, d, unit));
  };
  std::mt19937 generator(2161);
  std::normal_distribution<float> normal;
  for (uint32_t d : {1, 3, 4, 7, 8, 15, 31, 32, 127, 128, 129, 257, 513, 65535}) {
    std::vector<float> values(size_t(3) * d);
    for (size_t row = 0; row < 3; ++row) {
      double norm = 0;
      for (size_t j = 0; j < d; ++j) {
        const float value = normal(generator);
        values[row * d + j] = value;
        norm += double(value) * value;
      }
      const double scale = 1 / std::sqrt(norm);
      for (size_t j = 0; j < d; ++j) values[row * d + j] = float(values[row * d + j] * scale);
    }
    for (bool unit : {false, true}) verify(values, d, unit, "normalized");
    for (size_t j : {size_t(0), size_t(d / 2), size_t(d - 1), size_t(3 * d - 1)}) {
      for (float invalid : {std::numeric_limits<float>::quiet_NaN(),
                            std::numeric_limits<float>::infinity(),
                            -std::numeric_limits<float>::infinity(),
                            std::numeric_limits<float>::max(), 4e18f}) {
        const float saved = values[j];
        values[j] = invalid;
        for (bool unit : {false, true}) verify(values, d, unit, "nonfinite/huge");
        values[j] = saved;
      }
    }
    for (float value : {0.0f, -0.0f, std::numeric_limits<float>::denorm_min()}) {
      std::fill(values.begin(), values.end(), value);
      for (bool unit : {false, true}) verify(values, d, unit, "zero/subnormal");
    }
    for (double bound : {1.0 - .002, 1.0 + .002, max_norm}) {
      const float center = float(std::sqrt(bound));
      for (int delta = -8; delta <= 8; ++delta) {
        float value = center;
        for (int step = 0; step < std::abs(delta); ++step)
          value = std::nextafter(value, delta < 0 ? 0.0f : std::numeric_limits<float>::infinity());
        std::fill(values.begin(), values.end(), 0);
        for (size_t row = 0; row < 3; ++row) values[row * d + (row * d / 3) % d] = value;
        for (bool unit : {false, true}) verify(values, d, unit, "boundary-neighbors");
      }
    }
    if (d < 2 || d > 513) continue;
    // Large leading terms followed by tiny squares expose summation-order
    // differences at the unit and finite-scoring bounds. Also reverse them.
    for (double bound : {1.0 - .002, 1.0 + .002, max_norm}) {
      float large = float(std::sqrt(bound));
      while (double(large) * large >= bound) large = std::nextafter(large, 0.0f);
      const float small = float(std::sqrt((bound - double(large) * large) / (d - 1)));
      for (int delta = -32; delta <= 32; ++delta) {
        float adjusted = small;
        for (int step = 0; step < std::abs(delta); ++step)
          adjusted = std::nextafter(adjusted, delta < 0 ? 0.0f : std::numeric_limits<float>::infinity());
        values.assign(d, adjusted);
        values[0] = large;
        for (bool unit : {false, true}) verify(values, d, unit, "mixed-magnitude-boundary");
        std::reverse(values.begin(), values.end());
        for (bool unit : {false, true}) verify(values, d, unit, "reversed-mixed-boundary");
      }
    }
  }
  EXPECT_EQ(cases, 11460);
}

TEST(MVIVF, GroupedLeafTopKPreservesEqualScoreBoundary) {
  constexpr uint32_t n = 50, d = 128, query_count = 37, query_vectors = 7;
  std::vector<float> values(size_t(n) * d, 0), query_values(size_t(query_count) * query_vectors * d, 0);
  std::vector<size_t> offsets(n + 1), query_offsets(query_count + 1);
  std::vector<uint32_t> ids(n);
  for (uint32_t row = 0; row < n; ++row) {
    values[size_t(row) * d] = row < 30 ? -1 : 1;
    offsets[row] = size_t(row) * d;
    ids[row] = 9000 + 17 * row;
  }
  offsets[n] = values.size();
  for (size_t i = 0; i < size_t(query_count) * query_vectors; ++i) query_values[i * d] = 1;
  for (size_t i = 0; i <= query_count; ++i) query_offsets[i] = i * query_vectors * d;
  const auto points = own_point_clouds(std::move(values), std::move(offsets), d, ids);
  const auto queries = own_point_clouds(std::move(query_values), std::move(query_offsets), d);
  parlay::execute_with_scheduler(1, [&] {
    for (auto codec : {IndexParams::QuantizerType::None, IndexParams::QuantizerType::OneBitTQ,
                       IndexParams::QuantizerType::TurboQuant, IndexParams::QuantizerType::EightBitTQ}) {
      auto params = IndexParams::mvivf(0, n);
      params.quantize_centers = false;
      params.pq.method = codec;
      IndexMVIVF index(d, params);
      index.build(points);
      auto request = SearchParams::mvivf(25, 1);
      const auto single = index.search(queries[0], request);
      ASSERT_EQ(single.size(), 25);
      // Twenty better arrivals must leave the five smallest IDs of the tied
      // initial group, despite replacements changing the bucket's slot order.
      for (size_t i = 0; i < single.size(); ++i)
        EXPECT_EQ(single[i].first, ids[i < 20 ? 30 + i : i - 20]);
      request.nprobes = 3;  // Force the grouped path while keeping one leaf.
      const auto batch = index.search_all(queries, request);
      ASSERT_EQ(batch.size(), query_count);
      for (const auto& row : batch) ASSERT_NO_THROW(equal(single, row));
    }
  });
}

TEST(MVIVF, PublicIdTiesSurviveLeafSelectionAndReorderedLoad) {
  constexpr uint32_t n = 8, d = 128, k = 4, query_count = 64;
  const std::vector<uint32_t> ids{UINT32_MAX - 1, 0, 78, 4, 56, 7, 34, 2};
  auto reversed_ids = ids;
  std::reverse(reversed_ids.begin(), reversed_ids.end());
  std::vector<uint32_t> identity_ids(n);
  for (uint32_t i = 0; i < n; ++i) identity_ids[i] = i;
  auto corpus = [&](const std::vector<uint32_t>& labels) {
    std::vector<float> values(size_t(n) * d, 0);
    std::vector<size_t> offsets(n + 1);
    for (size_t i = 0; i < n; ++i) values[i * d] = 1;
    for (size_t i = 0; i <= n; ++i) offsets[i] = i * d;
    return own_point_clouds(std::move(values), std::move(offsets), d, labels);
  };
  std::vector<float> query_values(size_t(query_count) * d, 0);
  std::vector<size_t> query_offsets(query_count + 1);
  for (size_t i = 0; i < query_count; ++i) query_values[i * d] = 1;
  for (size_t i = 0; i <= query_count; ++i) query_offsets[i] = i * d;
  const auto queries = own_point_clouds(std::move(query_values), std::move(query_offsets), d);
  TemporaryFiles files("public-id-ties");
  parlay::execute_with_scheduler(1, [&] {
    for (auto codec : {IndexParams::QuantizerType::None, IndexParams::QuantizerType::OneBitTQ,
                       IndexParams::QuantizerType::TurboQuant, IndexParams::QuantizerType::EightBitTQ}) {
      auto params = IndexParams::mvivf(0, n);
      params.quantize_centers = false;
      params.pq.method = codec;
      // Temporary corpus views release their callers' owners after each call.
      IndexMVIVF original(d, params), reordered(d, params), loaded(d, params);
      IndexMVIVF implicit(d, params), identity(d, params);
      original.build(corpus(ids));
      reordered.build(corpus(reversed_ids));
      original.save(files.base.string() + ".mvif");
      loaded.load(files.base.string() + ".mvif", corpus(reversed_ids));
      implicit.build(corpus({}));
      identity.build(corpus(identity_ids));
      for (const auto [rerank, tq8] : std::array<std::pair<uint32_t, bool>, 5>{
               {{0, false}, {n, false}, {n, true}, {k + 2, false}, {k + 2, true}}}) {
        // Force fused leaf selection while keeping every document in one leaf.
        const auto request = SearchParams::mvivf(k, 3, rerank, tq8);
        const auto control = implicit.search(queries[0], request);
        const auto explicit_identity = identity.search(queries[0], request);
        ASSERT_EQ(control.size(), k);
        ASSERT_EQ(explicit_identity.size(), k);
        for (size_t i = 0; i < k; ++i) {
          EXPECT_EQ(control[i].first, i);
          EXPECT_EQ(explicit_identity[i].first, control[i].first);
          EXPECT_EQ(std::bit_cast<uint32_t>(explicit_identity[i].second),
                    std::bit_cast<uint32_t>(control[i].second));
        }
        for (const auto* index : {&original, &reordered, &loaded, &implicit, &identity}) {
          const auto single = index->search(queries[0], request);
          const auto batch = index->search_all(queries, request);
          const std::array<uint32_t, k> expected = index == &implicit || index == &identity
              ? std::array<uint32_t, k>{0, 1, 2, 3} : std::array<uint32_t, k>{0, 2, 4, 7};
          ASSERT_EQ(single.size(), k);
          ASSERT_EQ(batch.size(), query_count);
          for (size_t i = 0; i < k; ++i) {
            EXPECT_EQ(single[i].first, expected[i]);
            EXPECT_EQ(std::bit_cast<uint32_t>(single[i].second),
                      std::bit_cast<uint32_t>(control[i].second));
          }
          for (const auto& row : batch) {
            ASSERT_EQ(row.size(), k);
            for (size_t i = 0; i < k; ++i) {
              EXPECT_EQ(row[i].first, single[i].first);
              EXPECT_EQ(std::bit_cast<uint32_t>(row[i].second),
                        std::bit_cast<uint32_t>(single[i].second));
            }
          }
        }
      }
    }
  });
}

TEST(MVIVF, RetrievalOwnershipAndPersistence) {
  TemporaryFiles files("retrieval-test");
  ASSERT_NO_THROW(verify_retrieval(files.base.string() + ".mvif"));
}

TEST(MVIVF, BatchValidationReportsFirstInputErrorOnCaller) {
  constexpr uint32_t d = 128, count = 64;
  auto params = IndexParams::mvivf(0, 100);
  params.pq.method = IndexParams::QuantizerType::OneBitTQ;
  params.quantize_centers = false;
  const auto corpus = sample(8, d, 12);
  IndexMVIVF index(d, params);
  index.build(corpus);
  for (unsigned workers : {1, 4})
    parlay::execute_with_scheduler(workers, [&] {
      for (uint32_t first : {0, 17, 63}) {
        std::vector<float> values(size_t(count) * d, 0);
        std::vector<size_t> offsets(count + 1);
        for (size_t row = 0; row < count; ++row) {
          offsets[row] = row * d;
          values[row * d] = 1;
        }
        offsets[count] = values.size();
        // The first error is a norm failure; later NaNs have a different
        // message and can be detected first by a different worker.
        values[size_t(first) * d] = 0;
        for (size_t row = first + 1; row < count; ++row)
          values[row * d] = std::numeric_limits<float>::quiet_NaN();
        const auto queries = own_point_clouds(std::move(values), std::move(offsets), d);
        auto request = SearchParams::mvivf(3, 1);
        std::string expected;
        try { index.search(queries[first], request); }
        catch (const std::invalid_argument& error) { expected = error.what(); }
        ASSERT_FALSE(expected.empty());
        for (uint32_t probes : {1, 3}) {
          request.nprobes = probes;
          std::string actual;
          try { index.search_all(queries, request); }
          catch (const std::invalid_argument& error) { actual = error.what(); }
          EXPECT_EQ(actual, expected);
        }
      }
    });
}

TEST(MVIVF, PointCloudStorageAndMalformedFiles) {
  TemporaryFiles files("pcs-test");
  ASSERT_NO_THROW(verify_pcs(files.base.string() + ".pcs"));
}

TEST(MVIVF, ConcurrentCallersInitializeAndShareTQ8RerankCache) {
  TemporaryFiles files("concurrent-test");
  ASSERT_NO_THROW(verify_concurrent_search(files.base.string() + ".mvif"));
}
}  // namespace
