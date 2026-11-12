#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mvsic/algorithms/emdivf/emd.h"
#include "mvsic/algorithms/emdivf/index.h"
#include "mvsic/algorithms/stats.h"
#include "parlay/primitives.h"

namespace {

using emdivf::CloudSet;

struct Cell {
  std::vector<double> candidates;
  std::vector<double> recall;
};

double Mean(const std::vector<double>& values) {
  return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}

double Percentile(std::vector<double> values, double quantile) {
  std::sort(values.begin(), values.end());
  const double position = quantile * (values.size() - 1);
  const size_t left = static_cast<size_t>(position);
  const size_t right = std::min(left + 1, values.size() - 1);
  return values[left] + (values[right] - values[left]) * (position - left);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 7 && argc != 8) {
      throw std::invalid_argument(
          "usage: probe_curve TRAIN_PCS QUERIES_PCS EXACT_GT EMD_INDEX OUTPUT_CSV K [M]");
    }
    CloudSet train(argv[1]), queries(argv[2]);
    std::string gt_path(argv[3]);
    const auto exact = ReadGT(gt_path, queries.size());
    const auto index = emdivf::Load(argv[4]);
    const std::filesystem::path output(argv[5]);
    const uint32_t k = static_cast<uint32_t>(std::stoul(argv[6]));
    const uint32_t m = argc == 8 ? static_cast<uint32_t>(std::stoul(argv[7]))
                                 : static_cast<uint32_t>(index.centers.size());
    std::vector<uint32_t> bucket_sizes;
    for (uint32_t b : {512u, 1024u, 2048u}) {
      if (b <= index.max_bucket_size) bucket_sizes.push_back(b);
    }
    if (k == 0 || exact.empty() || exact[0].size() < k ||
        m == 0 || m > index.centers.size() || index.train_size != train.size() ||
        index.dimension != queries.get_dims() ||
        bucket_sizes.empty()) {
      throw std::invalid_argument("incompatible index, PCS, or exact GT");
    }
    std::vector<std::vector<Cell>> cells(bucket_sizes.size(), std::vector<Cell>(m));
    std::vector<int32_t> gold(index.corpus_size, -1);
    std::vector<uint32_t> seen(index.corpus_size, 0);
    uint32_t generation = 0;
    for (uint32_t query = 0; query < queries.size(); ++query) {
      const auto q = queries[query];
      for (uint32_t rank = 0; rank < k; ++rank) {
        const uint32_t doc = exact[query][rank].first;
        if (doc >= index.corpus_size) throw std::runtime_error("GT ID outside corpus");
        gold[doc] = rank;
      }
      std::vector<std::pair<double, uint32_t>> route(m);
      parlay::parallel_for(0, m, [&](size_t center) {
        const auto prototype = train[index.centers[center]];
        route[center] = {emdivf::Emd(q.data(), q.size(), prototype.data(),
                                     prototype.size(), index.dimension),
                         static_cast<uint32_t>(center)};
      });
      std::sort(route.begin(), route.end());
      for (size_t bi = 0; bi < bucket_sizes.size(); ++bi) {
        ++generation;
        uint32_t candidate_count = 0, covered = 0;
        for (uint32_t probe = 0; probe < m; ++probe) {
          const auto& bucket = index.buckets[route[probe].second];
          for (uint32_t rank = 0; rank < bucket_sizes[bi]; ++rank) {
            const uint32_t doc = bucket[rank].document;
            if (seen[doc] != generation) {
              seen[doc] = generation;
              ++candidate_count;
              covered += gold[doc] >= 0;
            }
          }
          cells[bi][probe].candidates.push_back(candidate_count);
          cells[bi][probe].recall.push_back(static_cast<double>(covered) / k);
        }
      }
      for (uint32_t rank = 0; rank < k; ++rank) gold[exact[query][rank].first] = -1;
      if ((query + 1) % 100 == 0 || query + 1 == queries.size()) {
        std::cerr << "candidate coverage " << query + 1 << '/' << queries.size() << '\n';
      }
    }
    std::filesystem::create_directories(output.parent_path());
    std::ofstream out(output);
    if (!out) throw std::runtime_error("cannot write probe curve CSV");
    out << std::setprecision(9);
    out << "method,clusters,bucket_b,probes,queries,mean_candidates,mean_recall,"
           "p10_recall,p50_recall,p90_recall\n";
    for (size_t bi = 0; bi < bucket_sizes.size(); ++bi) {
      for (uint32_t probe = 0; probe < m; ++probe) {
        const Cell& cell = cells[bi][probe];
        out << "emdivf," << m << ',' << bucket_sizes[bi] << ',' << probe + 1 << ','
            << queries.size() << ',' << Mean(cell.candidates) << ',' << Mean(cell.recall)
            << ',' << Percentile(cell.recall, 0.1) << ','
            << Percentile(cell.recall, 0.5) << ','
            << Percentile(cell.recall, 0.9) << '\n';
      }
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "probe_curve: " << error.what() << '\n';
    return 1;
  }
}
