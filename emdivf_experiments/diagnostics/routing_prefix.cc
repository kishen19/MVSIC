#include <algorithm>
#include <bit>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "mvsic/emdivf/emd.h"
#include "mvsic/emdivf/index.h"
#include "mvsic/core/stats.h"
#include "parlay/primitives.h"

namespace {
using emdivf::CloudSet;

double Quantile(std::vector<uint32_t> values, double q) {
  std::sort(values.begin(), values.end());
  const double p = q * (values.size() - 1);
  const size_t lo = static_cast<size_t>(p);
  const size_t hi = std::min(lo + 1, values.size() - 1);
  return values[lo] + (values[hi] - values[lo]) * (p - lo);
}

struct Config {
  uint32_t m;
  uint32_t b;
  std::vector<uint32_t> depths;
};
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 7) {
      throw std::invalid_argument(
          "usage: routing_prefix TRAIN_PCS QUERIES_PCS EXACT_GT EMD_INDEX OUTPUT_CSV K");
    }
    CloudSet train(argv[1]), queries(argv[2]);
    std::string gt_path(argv[3]);
    const auto exact = ReadGT(gt_path, queries.size());
    const auto index = emdivf::Load(argv[4]);
    const std::filesystem::path output(argv[5]);
    const uint32_t k = static_cast<uint32_t>(std::stoul(argv[6]));
    std::vector<Config> configs = {{256, 512, {}},   {512, 512, {}},
                                   {1024, 512, {}},  {2048, 512, {}},
                                   {1024, 1024, {}}, {2048, 1024, {}}};
    configs.erase(std::remove_if(configs.begin(), configs.end(), [&](const Config& config) {
      return config.m > index.centers.size() || config.b > index.max_bucket_size;
    }), configs.end());
    if (k == 0 || k > 31 || exact.empty() || exact[0].size() < k ||
        configs.empty() ||
        index.train_size != train.size() || index.dimension != queries.get_dims()) {
      throw std::invalid_argument("incompatible index, PCS, or exact GT");
    }
    std::vector<int32_t> gold(index.corpus_size, -1);
    for (uint32_t qi = 0; qi < queries.size(); ++qi) {
      for (uint32_t rank = 0; rank < k; ++rank) gold[exact[qi][rank].first] = rank;
      const auto query = queries[qi];
      std::vector<std::pair<double, uint32_t>> route(2048);
      parlay::parallel_for(0, route.size(), [&](size_t center) {
        const auto prototype = train[index.centers[center]];
        route[center] = {emdivf::Emd(query.data(), query.size(), prototype.data(),
                                     prototype.size(), index.dimension),
                         static_cast<uint32_t>(center)};
      });
      std::sort(route.begin(), route.end());
      for (auto& config : configs) {
        uint32_t mask = 0, probes = 0;
        for (const auto& [distance, center] : route) {
          (void)distance;
          if (center >= config.m) continue;
          ++probes;
          const auto& bucket = index.buckets[center];
          for (uint32_t rank = 0; rank < config.b; ++rank) {
            const int32_t bit = gold[bucket[rank].document];
            if (bit >= 0) mask |= uint32_t{1} << bit;
          }
          if (std::popcount(mask) == k) {
            config.depths.push_back(probes);
            break;
          }
        }
      }
      for (uint32_t rank = 0; rank < k; ++rank) gold[exact[qi][rank].first] = -1;
      if ((qi + 1) % 100 == 0 || qi + 1 == queries.size())
        std::cerr << "routing depths " << qi + 1 << '/' << queries.size() << '\n';
    }
    std::filesystem::create_directories(output.parent_path());
    std::ofstream out(output);
    out << std::setprecision(9)
        << "method,mean,p90,p99,fully_coverable_fraction\n";
    for (const auto& config : configs) {
      const double mean = std::accumulate(config.depths.begin(), config.depths.end(), 0.0) /
                          config.depths.size();
      out << "\"EMDIVF M=" << config.m << ", B=" << config.b << "\"," << mean << ','
          << Quantile(config.depths, .9) << ',' << Quantile(config.depths, .99) << ','
          << static_cast<double>(config.depths.size()) / queries.size() << '\n';
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "routing_prefix: " << error.what() << '\n';
    return 1;
  }
}
