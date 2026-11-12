#include "mvsic/algorithms/emdivf/index.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>

#include "mvsic/algorithms/emdivf/emd.h"
#include "mvsic/core/distance_measures/one_to_many.h"
#include "parlay/primitives.h"

namespace emdivf {
namespace {

using Clock = std::chrono::steady_clock;

double Seconds(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

bool Better(const Neighbor& left, const Neighbor& right) {
  return left.distance < right.distance ||
         (left.distance == right.distance && left.document < right.document);
}

template <typename T>
void Write(std::ostream& out, T value) {
  out.write(reinterpret_cast<const char*>(&value), sizeof(T));
  if (!out) throw std::runtime_error("failed to write EMDIVF index");
}

template <typename T>
T Read(std::istream& in) {
  T value{};
  in.read(reinterpret_cast<char*>(&value), sizeof(T));
  if (!in) throw std::runtime_error("truncated EMDIVF index");
  return value;
}

}  // namespace

Index Build(const CloudSet& train, const CloudSet& corpus, uint32_t m,
            uint32_t max_bucket_size, uint64_t seed, bool random_centers) {
  if (train.get_dims() != corpus.get_dims() || train.size() == 0 || corpus.size() == 0 ||
      m == 0 || m > train.size() || max_bucket_size == 0 || max_bucket_size > corpus.size()) {
    throw std::invalid_argument("invalid training/corpus dimensions or M/B");
  }
  Index index;
  index.dimension = train.get_dims();
  index.corpus_size = static_cast<uint32_t>(corpus.size());
  index.train_size = static_cast<uint32_t>(train.size());
  index.max_bucket_size = max_bucket_size;
  index.seed = seed;
  index.random_centers = random_centers;
  index.centers.reserve(m);
  index.selection_seconds.reserve(m);
  index.cover_radius.reserve(m);
  index.train_nearest_center.assign(train.size(), 0);
  index.train_distance.assign(train.size(), std::numeric_limits<double>::infinity());
  std::vector<bool> chosen(train.size(), false);
  std::mt19937_64 rng(seed);
  std::vector<uint32_t> random_order(train.size());
  std::iota(random_order.begin(), random_order.end(), 0);
  if (random_centers) std::shuffle(random_order.begin(), random_order.end(), rng);
  const uint32_t first = random_centers ? random_order[0] :
      std::uniform_int_distribution<uint32_t>(0, static_cast<uint32_t>(train.size() - 1))(rng);
  const auto selection_start = Clock::now();
  for (uint32_t center_number = 0; center_number < m; ++center_number) {
    uint32_t next = first;
    if (center_number > 0) {
      if (random_centers) {
        next = random_order[center_number];
      } else {
        double farthest = -1;
        for (uint32_t query = 0; query < train.size(); ++query) {
          if (!chosen[query] && index.train_distance[query] > farthest) {
            farthest = index.train_distance[query];
            next = query;
          }
        }
      }
    }
    index.centers.push_back(next);
    chosen[next] = true;
    const auto center = train[next];
    parlay::parallel_for(0, train.size(), [&](size_t query) {
      const auto item = train[query];
      const double distance = Emd(item.data(), item.size(), center.data(), center.size(),
                                  index.dimension);
      if (distance < index.train_distance[query]) {
        index.train_distance[query] = distance;
        index.train_nearest_center[query] = center_number;
      }
    });
    // Force an exact zero for each selected center, avoiding small roundoff
    // from the float32 dot-product used to build EMD ground costs.
    index.train_distance[next] = 0;
    index.train_nearest_center[next] = center_number;
    index.cover_radius.push_back(
        *std::max_element(index.train_distance.begin(), index.train_distance.end()));
    index.selection_seconds.push_back(Seconds(selection_start));
    if ((center_number + 1) % 8 == 0 || center_number + 1 == m) {
      std::cerr << "selected " << center_number + 1 << '/' << m
                << " centers; radius=" << index.cover_radius.back()
                << "; seconds=" << index.selection_seconds.back() << '\n';
    }
  }

  const auto bucket_start = Clock::now();
  index.buckets.reserve(m);
  index.bucket_seconds.reserve(m);
  for (uint32_t center_number = 0; center_number < m; ++center_number) {
    const auto distances =
        mvsic::OneToMany<mvsic::ChamferIP_Point, CloudSet>::AllDistances(
            train[index.centers[center_number]], corpus);
    std::vector<Neighbor> ranked(corpus.size());
    for (uint32_t document = 0; document < corpus.size(); ++document) {
      ranked[document] = {document, distances[document]};
    }
    if (max_bucket_size < ranked.size()) {
      std::nth_element(ranked.begin(), ranked.begin() + max_bucket_size,
                       ranked.end(), Better);
      ranked.resize(max_bucket_size);
    }
    std::sort(ranked.begin(), ranked.end(), Better);
    index.buckets.push_back(std::move(ranked));
    index.bucket_seconds.push_back(Seconds(bucket_start));
    if ((center_number + 1) % 8 == 0 || center_number + 1 == m) {
      std::cerr << "built " << center_number + 1 << '/' << m
                << " buckets; seconds=" << index.bucket_seconds.back() << '\n';
    }
  }
  return index;
}

AssignmentStats AssignTrainingQueries(const CloudSet& train, const Index& index, uint32_t m) {
  if (m == 0 || m > index.centers.size() || train.size() != index.train_size) {
    throw std::invalid_argument("invalid training assignment request");
  }
  AssignmentStats result;
  result.assigned_center.resize(train.size());
  result.distance.resize(train.size());
  result.cluster_size.assign(m, 0);
  result.cluster_radius.assign(m, 0);
  if (m == index.centers.size() && index.train_distance.size() == train.size()) {
    result.assigned_center = index.train_nearest_center;
    result.distance = index.train_distance;
  } else {
    parlay::parallel_for(0, train.size(), [&](size_t query) {
      const auto item = train[query];
      double best = std::numeric_limits<double>::infinity();
      uint32_t closest = 0;
      for (uint32_t center = 0; center < m; ++center) {
        const auto prototype = train[index.centers[center]];
        const double distance = Emd(item.data(), item.size(), prototype.data(),
                                    prototype.size(), index.dimension);
        if (distance < best) {
          best = distance;
          closest = center;
        }
      }
      result.assigned_center[query] = closest;
      result.distance[query] = best;
    });
  }
  for (size_t query = 0; query < train.size(); ++query) {
    const uint32_t center = result.assigned_center[query];
    ++result.cluster_size[center];
    result.cluster_radius[center] = std::max(result.cluster_radius[center], result.distance[query]);
    result.mean_distance += result.distance[query];
    result.radius = std::max(result.radius, result.distance[query]);
  }
  result.mean_distance /= train.size();
  auto sorted = result.distance;
  std::sort(sorted.begin(), sorted.end());
  result.median_distance = (sorted[(sorted.size() - 1) / 2] + sorted[sorted.size() / 2]) / 2;
  return result;
}

void Save(const Index& index, const std::filesystem::path& path) {
  const auto m = index.centers.size();
  if (!index.dimension || !index.corpus_size || !index.max_bucket_size || !m ||
      m > index.train_size || index.max_bucket_size > index.corpus_size ||
      index.selection_seconds.size() != m || index.bucket_seconds.size() != m ||
      index.cover_radius.size() != m || index.buckets.size() != m)
    throw std::invalid_argument("incomplete EMDIVF index");
  for (const auto& bucket : index.buckets)
    if (bucket.size() != index.max_bucket_size)
      throw std::invalid_argument("inconsistent EMDIVF bucket size");
  const auto temporary = std::filesystem::path(path.string() + ".tmp");
  if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
  {
    std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("cannot open EMDIVF index for writing");
    out.write("EMDIVF1\0", 8);
    Write<uint32_t>(out, index.dimension);
    Write<uint32_t>(out, index.corpus_size);
    Write<uint32_t>(out, index.train_size);
    Write<uint32_t>(out, index.max_bucket_size);
    Write<uint64_t>(out, index.seed);
    Write<uint8_t>(out, index.random_centers ? 1 : 0);
    Write<uint32_t>(out, static_cast<uint32_t>(index.centers.size()));
    for (size_t i = 0; i < index.centers.size(); ++i) {
      Write<uint32_t>(out, index.centers[i]);
      Write<double>(out, index.selection_seconds[i]);
      Write<double>(out, index.bucket_seconds[i]);
      Write<double>(out, index.cover_radius[i]);
      for (const auto& entry : index.buckets[i]) {
        Write<uint32_t>(out, entry.document);
        Write<float>(out, entry.distance);
      }
    }
    out.flush();
    if (!out) throw std::runtime_error("failed to flush EMDIVF index");
  }
  std::filesystem::rename(temporary, path);
}

Index Load(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open EMDIVF index: " + path.string());
  char magic[8];
  in.read(magic, 8);
  if (!in || std::string(magic, 8) != std::string("EMDIVF1\0", 8)) {
    throw std::runtime_error("invalid EMDIVF index magic");
  }
  Index index;
  index.dimension = Read<uint32_t>(in);
  index.corpus_size = Read<uint32_t>(in);
  index.train_size = Read<uint32_t>(in);
  index.max_bucket_size = Read<uint32_t>(in);
  index.seed = Read<uint64_t>(in);
  const auto random_flag = Read<uint8_t>(in);
  index.random_centers = random_flag != 0;
  const uint32_t m = Read<uint32_t>(in);
  if (!index.dimension || !index.corpus_size || !index.max_bucket_size ||
      random_flag > 1 || m == 0 || m > index.train_size ||
      index.max_bucket_size > index.corpus_size) {
    throw std::runtime_error("invalid EMDIVF index dimensions");
  }
  const uint64_t expected = 37 + uint64_t(m) * (28 + uint64_t(index.max_bucket_size) * 8);
  if (std::filesystem::file_size(path) != expected)
    throw std::runtime_error("EMDIVF index size does not match header");
  index.centers.resize(m);
  index.selection_seconds.resize(m);
  index.bucket_seconds.resize(m);
  index.cover_radius.resize(m);
  index.buckets.resize(m);
  std::unordered_set<uint32_t> centers;
  for (uint32_t i = 0; i < m; ++i) {
    index.centers[i] = Read<uint32_t>(in);
    index.selection_seconds[i] = Read<double>(in);
    index.bucket_seconds[i] = Read<double>(in);
    index.cover_radius[i] = Read<double>(in);
    if (index.centers[i] >= index.train_size || !centers.insert(index.centers[i]).second)
      throw std::runtime_error("invalid or repeated center ID");
    if (!std::isfinite(index.selection_seconds[i]) || index.selection_seconds[i] < 0 ||
        !std::isfinite(index.bucket_seconds[i]) || index.bucket_seconds[i] < 0 ||
        !std::isfinite(index.cover_radius[i]) || index.cover_radius[i] < 0)
      throw std::runtime_error("invalid center diagnostics");
    index.buckets[i].resize(index.max_bucket_size);
    std::unordered_set<uint32_t> documents;
    float previous = -std::numeric_limits<float>::infinity();
    for (auto& entry : index.buckets[i]) {
      entry.document = Read<uint32_t>(in);
      entry.distance = Read<float>(in);
      if (entry.document >= index.corpus_size || !documents.insert(entry.document).second ||
          !std::isfinite(entry.distance) || entry.distance < previous)
        throw std::runtime_error("invalid, repeated, or unordered bucket entry");
      previous = entry.distance;
    }
  }
  if (in.peek() != std::char_traits<char>::eof()) {
    throw std::runtime_error("unexpected trailing bytes in EMDIVF index");
  }
  return index;
}

}  // namespace emdivf
