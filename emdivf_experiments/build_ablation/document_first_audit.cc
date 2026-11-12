#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mvsic/algorithms/emdivf/index.h"
#include "mvsic/core/quantization/turboquant_8bit_mv.h"
#include "parlay/primitives.h"

namespace {
using Clock = std::chrono::steady_clock;
using Set = emdivf::CloudSet;
using TQ8 = mvsic::turboquant_8bit_mv::Model<false>;

double Seconds(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 7)
      throw std::invalid_argument(
          "usage: document_first_audit INDEX TRAIN_PCS DOCS_PCS TQ8_CACHE OUTPUT_CSV M");
    const auto index = emdivf::Load(argv[1]);
    Set train(argv[2], true), docs(argv[3], true);
    const uint32_t m = std::stoul(argv[6]), b = 512;
    if (m > index.centers.size() || m < 256 || index.max_bucket_size < b ||
        train.size() != index.train_size || docs.size() != index.corpus_size ||
        train.get_dims() != docs.get_dims())
      throw std::invalid_argument("index or point-cloud mismatch");
    TQ8 model;
    TQ8::EncodedSet encoded;
    const auto load_start = Clock::now();
    {
      std::ifstream cache(argv[4], std::ios::binary);
      if (!cache) throw std::runtime_error("cannot open TQ8 cache");
      model.load(cache);
      encoded.load(cache);
      if (!cache || encoded.num_clouds() != docs.size())
        throw std::runtime_error("TQ8 cache/document mismatch");
    }
    const double cache_seconds = Seconds(load_start);
    const size_t n = docs.size();
    std::vector<float> scores(static_cast<size_t>(m) * n);
    const auto scan_start = Clock::now();
    constexpr uint32_t batch = 16;
    for (uint32_t begin = 0; begin < m; begin += batch) {
      const uint32_t count = std::min(batch, m - begin);
      std::vector<TQ8::EncodedQuery> queries;
      queries.reserve(count);
      for (uint32_t j = 0; j < count; ++j)
        queries.push_back(model.quantize_query(train[index.centers[begin + j]]));
      std::vector<const TQ8::EncodedQuery*> pointers;
      for (const auto& q : queries) pointers.push_back(&q);
      mvsic::turboquant_8bit_mv::FusedQueryBatch<false> fused;
      fused.Build(pointers);
      std::vector<std::pair<uint32_t, float>> batch_scores(static_cast<size_t>(count) * n);
      std::vector<std::vector<float>> workspaces(parlay::num_workers());
      mvsic::turboquant_8bit_mv::chamfer_score_all_fused(
          fused, encoded, batch_scores.data(), &workspaces, true);
      parlay::parallel_for(0, count, [&](size_t j) {
        for (size_t d = 0; d < n; ++d) {
          const auto [doc, score] = batch_scores[j * n + d];
          if (doc >= n) throw std::runtime_error("invalid scored document ID");
          scores[(begin + j) * n + doc] = score;
        }
      });
      if ((begin + count) % 128 == 0 || begin + count == m)
        std::cerr << "scored centers " << begin + count << '/' << m << '\n';
    }
    const double scan_seconds = Seconds(scan_start);

    std::vector<std::vector<uint16_t>> membership(n);
    std::vector<std::vector<uint16_t>> ranks(n);
    for (uint32_t center = 0; center < m; ++center)
      for (uint32_t j = 0; j < b; ++j)
        membership[index.buckets[center][j].document].push_back(center);
    const auto rank_start = Clock::now();
    parlay::parallel_for(0, n, [&](size_t d) {
      std::vector<std::pair<float, uint16_t>> ranked(m);
      for (uint32_t center = 0; center < m; ++center)
        ranked[center] = {scores[static_cast<size_t>(center) * n + d],
                          static_cast<uint16_t>(center)};
      const uint32_t limit = std::min<uint32_t>(256, m);
      std::partial_sort(ranked.begin(), ranked.begin() + limit, ranked.end());
      std::vector<uint16_t> center_rank(m, std::numeric_limits<uint16_t>::max());
      for (uint32_t j = 0; j < limit; ++j)
        center_rank[ranked[j].second] = static_cast<uint16_t>(j + 1);
      ranks[d].reserve(membership[d].size());
      for (uint16_t center : membership[d])
        ranks[d].push_back(center_rank[center]);
    });
    const double rank_seconds = Seconds(rank_start);
    const std::array<uint32_t, 9> r_values = {1, 2, 4, 8, 16, 32, 64, 128, 256};
    std::array<uint64_t, 9> recovered{};
    std::vector<std::array<uint16_t, 9>> recovered_per_center(m);
    for (size_t d = 0; d < n; ++d) {
      for (size_t j = 0; j < membership[d].size(); ++j) {
        const uint16_t center = membership[d][j], rank = ranks[d][j];
        for (size_t ri = 0; ri < r_values.size(); ++ri)
          if (rank <= r_values[ri]) {
            ++recovered[ri];
            ++recovered_per_center[center][ri];
          }
      }
    }
    std::filesystem::create_directories(std::filesystem::path(argv[5]).parent_path());
    std::ofstream output(argv[5]);
    output << std::setprecision(10)
           << "m,corpus,b,top_r,recovered_exact_postings,exact_posting_recall,"
              "document_first_postings,posting_multiplier_vs_exact,"
              "min_center_bucket_recall,mean_center_bucket_recall,"
              "tq8_cache_load_seconds,tq8_full_scan_seconds,invert_rank_seconds\n";
    for (size_t ri = 0; ri < r_values.size(); ++ri) {
      uint16_t minimum = b;
      for (const auto& row : recovered_per_center)
        minimum = std::min(minimum, row[ri]);
      const uint64_t assigned = static_cast<uint64_t>(n) * r_values[ri];
      output << m << ',' << n << ',' << b << ',' << r_values[ri] << ','
             << recovered[ri] << ',' << static_cast<double>(recovered[ri]) / (m * b)
             << ',' << assigned << ',' << static_cast<double>(assigned) / (m * b)
             << ',' << static_cast<double>(minimum) / b << ','
             << static_cast<double>(recovered[ri]) / (m * b) << ','
             << cache_seconds << ',' << scan_seconds << ',' << rank_seconds << '\n';
    }
    const uint32_t sample_size = std::min<uint32_t>(256, n);
    std::vector<std::array<uint32_t, 9>> sample_exact(sample_size),
                                          sample_tq8(sample_size);
    std::vector<uint32_t> sample_postings(sample_size);
    const auto sample_start = Clock::now();
    parlay::parallel_for(0, sample_size, [&](size_t sample) {
      const size_t doc = sample * n / sample_size;
      std::vector<std::pair<float, uint16_t>> exact(m), approx(m);
      for (uint32_t center = 0; center < m; ++center) {
        exact[center] = {train[index.centers[center]].distance(docs[doc]),
                         static_cast<uint16_t>(center)};
        approx[center] = {scores[static_cast<size_t>(center) * n + doc],
                          static_cast<uint16_t>(center)};
      }
      const uint32_t limit = std::min<uint32_t>(256, m);
      std::partial_sort(exact.begin(), exact.begin() + limit, exact.end());
      std::partial_sort(approx.begin(), approx.begin() + limit, approx.end());
      std::vector<uint16_t> exact_rank(m, std::numeric_limits<uint16_t>::max());
      std::vector<uint16_t> approx_rank(m, std::numeric_limits<uint16_t>::max());
      for (uint32_t j = 0; j < limit; ++j) {
        exact_rank[exact[j].second] = j + 1;
        approx_rank[approx[j].second] = j + 1;
      }
      sample_postings[sample] = static_cast<uint32_t>(membership[doc].size());
      for (uint16_t center : membership[doc]) {
        for (size_t ri = 0; ri < r_values.size(); ++ri) {
          sample_exact[sample][ri] += exact_rank[center] <= r_values[ri];
          sample_tq8[sample][ri] += approx_rank[center] <= r_values[ri];
        }
      }
    });
    const double exact_sample_seconds = Seconds(sample_start);
    std::string sample_path = argv[5];
    if (sample_path.size() >= 4 && sample_path.substr(sample_path.size() - 4) == ".csv")
      sample_path.resize(sample_path.size() - 4);
    sample_path += "_sample.csv";
    std::ofstream sample_out(sample_path);
    sample_out << std::setprecision(10)
               << "rank_source,top_r,sample_docs,sample_exact_memberships,"
                  "retained_exact_memberships,membership_recall,exact_sample_seconds\n";
    uint64_t denominator = 0;
    for (uint32_t count : sample_postings) denominator += count;
    for (size_t ri = 0; ri < r_values.size(); ++ri) {
      uint64_t exact_count = 0, tq8_count = 0;
      for (size_t sample = 0; sample < sample_size; ++sample) {
        exact_count += sample_exact[sample][ri];
        tq8_count += sample_tq8[sample][ri];
      }
      for (const auto& [name, count] :
           {std::pair<const char*, uint64_t>{"exact", exact_count},
            std::pair<const char*, uint64_t>{"tq8", tq8_count}})
        sample_out << name << ',' << r_values[ri] << ',' << sample_size << ','
                   << denominator << ',' << count << ','
                   << static_cast<double>(count) / denominator << ','
                   << exact_sample_seconds << '\n';
    }
    std::cerr << "cache=" << cache_seconds << " scan=" << scan_seconds
              << " inverse ranks=" << rank_seconds
              << " exact sample=" << exact_sample_seconds << " seconds\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "document_first_audit: " << error.what() << '\n';
    return 1;
  }
}
