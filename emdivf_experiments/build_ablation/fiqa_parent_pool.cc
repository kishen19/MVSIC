#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "mvsic/emdivf/emd.h"
#include "mvsic/emdivf/index.h"
#include "mvsic/core/stats.h"
#include "parlay/primitives.h"

namespace {
using Clock = std::chrono::steady_clock;
using emdivf::CloudSet;
using emdivf::Index;
using emdivf::Neighbor;

struct Config {
  uint32_t parents;
  uint32_t pool;
  uint32_t parent_probes = 1;
  std::vector<std::vector<Neighbor>> buckets;
  std::vector<uint32_t> scored_per_center;
  double assignment_seconds = 0;
  double child_scoring_seconds = 0;
  double overlap = 0;
  double minimum_overlap = 1;
  uint64_t scored_documents = 0;
};

double Seconds(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}
bool Better(const Neighbor& a, const Neighbor& b) {
  return a.distance < b.distance ||
         (a.distance == b.distance && a.document < b.document);
}
void Top(std::vector<Neighbor>& a, size_t count) {
  if (a.size() > count) {
    std::nth_element(a.begin(), a.begin() + count, a.end(), Better);
    a.resize(count);
  }
  std::sort(a.begin(), a.end(), Better);
}
double Emd(const CloudSet& train, uint32_t a, uint32_t b) {
  const auto left = train[a], right = train[b];
  return emdivf::Emd(left.data(), left.size(), right.data(), right.size(),
                     train.get_dims());
}

struct EvalRow {
  double candidates = 0;
  double recall = 0;
};

void Evaluate(const std::vector<Config>& configs, const Index& original,
              const CloudSet& train, const CloudSet& queries,
              const std::string& gt_path, const std::string& output,
              uint32_t m, uint32_t b) {
  auto mutable_gt_path = gt_path;
  const auto gt = ReadGT(mutable_gt_path, queries.size());
  if (gt.size() != queries.size() || gt.front().size() < 10)
    throw std::runtime_error("GT/query mismatch");
  const std::array<uint32_t, 10> probes = {1, 2, 4, 8, 12, 16, 24, 32, 64, 128};
  std::vector<std::vector<EvalRow>> sums(configs.size() + 1,
                                         std::vector<EvalRow>(probes.size()));
  std::vector<uint8_t> gold(original.corpus_size);
  std::vector<uint32_t> seen(original.corpus_size);
  uint32_t generation = 0;
  for (size_t qi = 0; qi < queries.size(); ++qi) {
    const auto q = queries[qi];
    std::vector<std::pair<double, uint32_t>> route(m);
    parlay::parallel_for(0, m, [&](size_t i) {
      const auto p = train[original.centers[i]];
      route[i] = {emdivf::Emd(q.data(), q.size(), p.data(), p.size(),
                             train.get_dims()), static_cast<uint32_t>(i)};
    });
    std::sort(route.begin(), route.end());
    for (size_t j = 0; j < 10; ++j) gold[gt[qi][j].first] = 1;
    for (size_t ci = 0; ci <= configs.size(); ++ci) {
      ++generation;
      uint32_t candidate_count = 0, found = 0;
      uint32_t start = 0;
      for (size_t pi = 0; pi < probes.size(); ++pi) {
        const uint32_t end = std::min(probes[pi], m);
        for (uint32_t probe = start; probe < end; ++probe) {
          const uint32_t center = route[probe].second;
          const auto& bucket = ci == 0 ? original.buckets[center]
                                       : configs[ci - 1].buckets[center];
          for (size_t j = 0; j < b; ++j) {
            const uint32_t doc = bucket[j].document;
            if (seen[doc] != generation) {
              seen[doc] = generation;
              ++candidate_count;
              found += gold[doc];
            }
          }
        }
        sums[ci][pi].candidates += candidate_count;
        sums[ci][pi].recall += static_cast<double>(found) / 10;
        start = end;
      }
    }
    for (size_t j = 0; j < 10; ++j) gold[gt[qi][j].first] = 0;
    if ((qi + 1) % 100 == 0 || qi + 1 == queries.size())
      std::cerr << "evaluated " << qi + 1 << '/' << queries.size() << '\n';
  }
  std::ofstream out(output);
  out << std::setprecision(10)
      << "method,m,parents,parent_pool,parent_probes,b,probes,queries,mean_candidates,candidate_recall_10\n";
  for (size_t ci = 0; ci <= configs.size(); ++ci) {
    for (size_t pi = 0; pi < probes.size(); ++pi) {
      const auto& cell = sums[ci][pi];
      out << (ci == 0 ? "original" : "parent_pool") << ',' << m << ','
          << (ci == 0 ? 0 : configs[ci - 1].parents) << ','
          << (ci == 0 ? 0 : configs[ci - 1].pool) << ','
          << (ci == 0 ? 0 : configs[ci - 1].parent_probes) << ',' << b << ','
          << std::min(probes[pi], m) << ',' << queries.size() << ','
          << cell.candidates / queries.size() << ','
          << cell.recall / queries.size() << '\n';
    }
  }
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 8 && argc != 11 && argc != 12) {
      throw std::invalid_argument(
          "usage: fiqa_parent_pool INDEX TRAIN_PCS DOCS_PCS QUERIES_PCS EXACT_GT OUT_PREFIX M [PARENTS POOL PARENT_PROBES [PARENT_INDEX]]");
    }
    const std::string prefix = argv[6];
    const uint32_t m = std::stoul(argv[7]), b = 512;
    auto original = emdivf::Load(argv[1]);
    std::optional<Index> parent_storage;
    const Index* parent_index = &original;
    if (argc == 12) {
      parent_storage.emplace(emdivf::Load(argv[11]));
      parent_index = &*parent_storage;
    }
    CloudSet train(argv[2], true), docs(argv[3], true), queries(argv[4], true);
    if (m > original.centers.size() || m < 128 ||
        original.max_bucket_size < 2048 || docs.size() != original.corpus_size ||
        train.size() != original.train_size || train.get_dims() != docs.get_dims())
      throw std::invalid_argument("index or point-cloud mismatch");
    std::filesystem::create_directories(std::filesystem::path(prefix).parent_path());
    std::array<uint32_t, 3> parent_counts = m == 256
        ? std::array<uint32_t, 3>{16, 32, 64}
        : std::array<uint32_t, 3>{32, 64, 128};
    std::vector<Config> configs;
    for (uint32_t count : parent_counts) {
      for (uint32_t pool : {1024u, 2048u}) {
        Config cfg;
        cfg.parents = count;
        cfg.pool = pool;
        cfg.buckets.resize(m);
        cfg.scored_per_center.resize(m);
        for (uint32_t i = 0; i < count; ++i) {
          cfg.buckets[i].assign(original.buckets[i].begin(),
                                original.buckets[i].begin() + b);
        }
        configs.push_back(std::move(cfg));
      }
    }
    for (uint32_t probes : {2u, 4u}) {
      Config cfg;
      cfg.parents = parent_counts.back();
      cfg.pool = 2048;
      cfg.parent_probes = probes;
      cfg.buckets.resize(m);
      cfg.scored_per_center.resize(m);
      for (uint32_t i = 0; i < cfg.parents; ++i) {
        cfg.buckets[i].assign(original.buckets[i].begin(),
                              original.buckets[i].begin() + b);
      }
      configs.push_back(std::move(cfg));
    }
    if (argc == 11 || argc == 12) {
      const uint32_t requested_parents = std::stoul(argv[8]);
      const uint32_t requested_pool = std::stoul(argv[9]);
      const uint32_t requested_probes = std::stoul(argv[10]);
      if (std::find(parent_counts.begin(), parent_counts.end(), requested_parents) ==
              parent_counts.end() || requested_pool < b ||
          requested_pool > parent_index->max_bucket_size ||
          (requested_probes != 1 && requested_probes != 2 && requested_probes != 4) ||
          requested_probes > requested_parents ||
          parent_index->centers.size() < requested_parents ||
          parent_index->corpus_size != original.corpus_size ||
          parent_index->train_size != original.train_size ||
          !std::equal(original.centers.begin(), original.centers.begin() + requested_parents,
                      parent_index->centers.begin()))
        throw std::invalid_argument("parent index or requested config mismatch");
      Config chosen;
      chosen.parents = requested_parents;
      chosen.pool = requested_pool;
      chosen.parent_probes = requested_probes;
      chosen.buckets.resize(m);
      chosen.scored_per_center.resize(m);
      for (uint32_t i = 0; i < requested_parents; ++i)
        chosen.buckets[i].assign(original.buckets[i].begin(),
                                 original.buckets[i].begin() + b);
      configs.clear();
      configs.push_back(std::move(chosen));
    }
    const auto assignment_start = Clock::now();
    std::vector<std::array<std::array<uint32_t, 4>, 3>> nearest(m);
    parlay::parallel_for(0, m, [&](size_t i) {
      std::vector<std::pair<double, uint32_t>> distances(parent_counts.back());
      for (uint32_t parent = 0; parent < parent_counts.back(); ++parent) {
        const double dist = Emd(train, original.centers[i], original.centers[parent]);
        distances[parent] = {dist, parent};
      }
      for (size_t ci = 0; ci < 3; ++ci) {
        auto ranked = distances;
        ranked.resize(parent_counts[ci]);
        std::partial_sort(ranked.begin(), ranked.begin() + 4, ranked.end());
        for (size_t p = 0; p < 4; ++p) nearest[i][ci][p] = ranked[p].second;
      }
    });
    const double assignment_seconds = Seconds(assignment_start);
    std::cerr << "assigned child centers in " << assignment_seconds << " seconds\n";
    const auto score_start = Clock::now();
    parlay::parallel_for(0, m, [&](size_t i) {
      const auto q = train[original.centers[i]];
      for (auto& cfg : configs) {
        if (i < cfg.parents) continue;
        const size_t group = static_cast<size_t>(std::find(
            parent_counts.begin(), parent_counts.end(), cfg.parents) -
            parent_counts.begin());
        std::vector<Neighbor> row;
        if (cfg.parent_probes == 1) {
          const auto& bucket = parent_index->buckets[nearest[i][group][0]];
          row.reserve(cfg.pool);
          for (size_t j = 0; j < cfg.pool; ++j) {
            const uint32_t doc = bucket[j].document;
            row.push_back({doc, q.distance(docs[doc])});
          }
        } else {
          std::unordered_set<uint32_t> unique;
          for (size_t p = 0; p < cfg.parent_probes; ++p) {
            const auto& bucket = parent_index->buckets[nearest[i][group][p]];
            for (size_t j = 0; j < cfg.pool; ++j)
              unique.insert(bucket[j].document);
          }
          row.reserve(unique.size());
          for (uint32_t doc : unique) row.push_back({doc, q.distance(docs[doc])});
        }
        cfg.scored_per_center[i] = static_cast<uint32_t>(row.size());
        Top(row, b);
        cfg.buckets[i] = std::move(row);
      }
      if ((i + 1) % 32 == 0 || i + 1 == m)
        std::cerr << "built child " << i + 1 << '/' << m << '\n';
    });
    const double child_seconds = Seconds(score_start);
    std::ofstream summary(prefix + "_build.csv");
    summary << std::setprecision(10)
            << "m,parents,parent_pool,parent_probes,b,mean_child_topb_overlap,min_child_topb_overlap,"
               "parent_global_document_scores,child_document_scores,"
               "estimated_total_document_scores,assignment_seconds,"
               "shared_child_rescoring_seconds\n";
    for (auto& cfg : configs) {
      cfg.assignment_seconds = assignment_seconds;
      cfg.child_scoring_seconds = child_seconds;
      double overlap = 0, minimum = 1;
      for (uint32_t i = 0; i < m; ++i) {
        std::unordered_set<uint32_t> original_ids;
        for (size_t j = 0; j < b; ++j)
          original_ids.insert(original.buckets[i][j].document);
        uint32_t common = 0;
        for (const auto& entry : cfg.buckets[i])
          common += original_ids.count(entry.document);
        const double ratio = static_cast<double>(common) / b;
        overlap += ratio;
        minimum = std::min(minimum, ratio);
      }
      const uint64_t child_scores = std::accumulate(
          cfg.scored_per_center.begin(), cfg.scored_per_center.end(), uint64_t(0));
      const uint64_t total = static_cast<uint64_t>(cfg.parents) * docs.size() + child_scores;
      summary << m << ',' << cfg.parents << ',' << cfg.pool << ','
              << cfg.parent_probes << ',' << b << ','
              << overlap / m << ',' << minimum << ','
              << static_cast<uint64_t>(cfg.parents) * docs.size() << ','
              << child_scores << ','
              << total << ',' << assignment_seconds << ',' << child_seconds << '\n';
    }
    Evaluate(configs, original, train, queries, argv[5], prefix + "_curve.csv", m, b);
    std::cerr << "completed parent-pool ablation\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "fiqa_parent_pool: " << error.what() << '\n';
    return 1;
  }
}
