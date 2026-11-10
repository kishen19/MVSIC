#include "mvsic/emdivf/emdivf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <unordered_set>
#include <utility>

#include "mvsic/core/query_compression.h"
#include "mvsic/core/quantization/turboquant_1bit_mv.h"
#include "mvsic/core/quantization/turboquant_8bit_mv.h"
#include "mvsic/core/quantization/turboquant_mv.h"
#include "mvsic/core/search_params.h"
#include "mvsic/vamana/beam_search.h"
#include "mvsic/vamana/graph.h"
#include "parlay/primitives.h"
#include "parlay/random.h"

namespace emdivf {
namespace {

using Clock = std::chrono::steady_clock;
using Graph = mvsic::vamana::Graph<uint32_t>;
using TQ1 = mvsic::turboquant_1bit_mv::Model<false>;
using TQ4 = mvsic::turboquant_mv::Model<false>;
using TQ8 = mvsic::turboquant_8bit_mv::Model<false>;
using Scored = std::pair<float, uint32_t>;

double Ms(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void ValidatePrototypes(const CloudSet& train, const Index& index) {
  if (index.centers.empty() || index.dimension != train.get_dims() ||
      index.train_size != train.size())
    throw std::invalid_argument("EMDIVF prototype dimensions do not match training set");
  std::vector<uint8_t> used(train.size());
  for (auto id : index.centers) {
    if (id >= train.size() || used[id] || train.get_size(id) == 0)
      throw std::invalid_argument("invalid, repeated, or empty EMDIVF prototype");
    used[id] = 1;
  }
}

CloudSet PrototypeClouds(const CloudSet& train, const Index& index) {
  auto selected = parlay::tabulate(index.centers.size(), [&](size_t i) {
    const auto cloud = train[index.centers[i]];
    return mvsic::ChamferIP_Point(cloud.size(), cloud.get_dims(), cloud.data(), i);
  });
  return CloudSet(selected, train.get_dims());
}

// Batch-parallel Vamana construction, retaining the existing experiment's
// pruning and insertion policy so its graph settings remain reproducible.
class GraphBuilder {
 public:
  explicit GraphBuilder(const RoutingBuildParams& params) : params_(params) {}

  Graph Build(const CloudSet& points) {
    graph_ = Graph(params_.degree, points.size());
    const auto order = parlay::random_permutation<uint32_t>(points.size());
    const size_t max_batch = std::max<size_t>(1, points.size() / 50);
    size_t begin = 0, increment = 1;
    while (begin < points.size()) {
      const size_t end = std::min(points.size(), begin + increment);
      parlay::sequence<parlay::sequence<uint32_t>> outgoing(end - begin);
      parlay::parallel_for(begin, end, [&](size_t i) {
        const uint32_t point = order[i];
        auto params = mvsic::SearchParams::vamana(0, params_.beam, 0.0);
        auto visited = std::get<0>(mvsic::vamana::beam_search(
            points[point], graph_, points, uint32_t{0}, params)).second;
        std::vector<uint32_t> ids;
        ids.reserve(visited.size() + graph_[point].size());
        for (const auto& candidate : visited) ids.push_back(candidate.first);
        for (auto id : graph_[point]) ids.push_back(id);
        outgoing[i - begin] = Prune(point, std::move(ids), points);
      });
      auto reverse = parlay::flatten(parlay::tabulate(end - begin, [&](size_t i) {
        return parlay::tabulate(outgoing[i].size(), [&](size_t j) {
          return std::make_pair(outgoing[i][j], order[begin + i]);
        });
      }));
      parlay::parallel_for(begin, end, [&](size_t i) {
        graph_[order[i]].update_neighbors(outgoing[i - begin]);
      });
      auto grouped = parlay::group_by_key(reverse);
      parlay::parallel_for(0, grouped.size(), [&](size_t i) {
        auto& [point, incoming] = grouped[i];
        std::vector<uint32_t> ids(incoming.begin(), incoming.end());
        for (auto id : graph_[point]) ids.push_back(id);
        std::sort(ids.begin(), ids.end());
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
        if (ids.size() <= params_.degree) graph_[point].update_neighbors(ids);
        else graph_[point].update_neighbors(Prune(point, std::move(ids), points));
      });
      begin = end;
      increment = std::min(max_batch, increment * 2);
    }
    parlay::parallel_for(0, points.size(), [&](size_t i) {
      graph_[i].sort([&](uint32_t a, uint32_t b) {
        return points[i].distance(points[a]) < points[i].distance(points[b]);
      });
    });
    return graph_;
  }

 private:
  parlay::sequence<uint32_t> Prune(uint32_t point, std::vector<uint32_t> ids,
                                  const CloudSet& points) const {
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    std::vector<Scored> candidates;
    candidates.reserve(ids.size());
    for (auto id : ids)
      if (id != point) candidates.emplace_back(points[point].distance(points[id]), id);
    std::sort(candidates.begin(), candidates.end());
    std::vector<uint32_t> selected;
    for (size_t cursor = 0; cursor < candidates.size() && selected.size() < params_.degree;
         ++cursor) {
      const auto [distance, id] = candidates[cursor];
      if (id == std::numeric_limits<uint32_t>::max()) continue;
      selected.push_back(id);
      for (size_t j = cursor + 1; j < candidates.size(); ++j) {
        if (candidates[j].second == std::numeric_limits<uint32_t>::max()) continue;
        if (params_.alpha * points[id].distance(points[candidates[j].second]) <=
            candidates[j].first)
          candidates[j].second = std::numeric_limits<uint32_t>::max();
      }
    }
    return parlay::to_sequence(selected);
  }

  RoutingBuildParams params_;
  Graph graph_;
};

template <class T>
void Write(std::ostream& out, T value) {
  out.write(reinterpret_cast<const char*>(&value), sizeof(value));
  if (!out) throw std::runtime_error("failed to write EMDIVF routing graph");
}
template <class T>
T Read(std::istream& in) {
  T value{};
  in.read(reinterpret_cast<char*>(&value), sizeof(value));
  if (!in) throw std::runtime_error("truncated EMDIVF routing graph");
  return value;
}

Graph LoadGraph(const std::filesystem::path& path, const Index& index) {
  std::ifstream in(path, std::ios::binary);
  const auto magic = Read<uint32_t>(in), version = Read<uint32_t>(in);
  const auto metric = Read<uint32_t>(in), start = Read<uint32_t>(in);
  if (magic != 0x5056414d || (version != 1 && version != 2) || metric != 1 || start != 0)
    throw std::runtime_error("invalid EMDIVF Chamfer routing graph header");
  if (version == 2) {
    if (Read<uint32_t>(in) != index.dimension || Read<uint32_t>(in) != index.train_size ||
        Read<uint32_t>(in) != index.centers.size())
      throw std::runtime_error("routing graph prototype dimensions do not match index");
    for (auto center : index.centers)
      if (Read<uint32_t>(in) != center)
        throw std::runtime_error("routing graph prototype order does not match index");
  }
  Graph graph(in);
  if (!in || graph.size() != index.centers.size())
    throw std::runtime_error("routing graph size does not match bucket index");
  for (uint32_t i = 0; i < graph.size(); ++i)
    for (auto id : graph[i])
      if (id >= graph.size()) throw std::runtime_error("invalid routing graph neighbor");
  return graph;
}

class PreparedQuery {
 public:
  PreparedQuery(const TQ1::EncodedQuery& query, const TQ1::EncodedSet& db) : query_(query) {
#ifdef __AVX512F__
    stride_ = db.num_hamming_tiles * mvsic::turboquant_1bit_mv::internal::kTileBytes;
    buffer_.resize(query.num_queries * stride_);
    for (size_t i = 0; i < query.num_queries; ++i)
      mvsic::turboquant_1bit_mv::internal::pre_broadcast_query(
          query.get_q_codes(i), query.num_bytes_per_datapoint, db.num_hamming_tiles,
          buffer_.data() + i * stride_);
#endif
  }

  float Distance(const TQ1::EncodedSet& db, uint32_t id) const {
    if (!db.cloud_sizes[id]) return std::numeric_limits<float>::max();
#ifdef __AVX512F__
    return mvsic::turboquant_1bit_mv::internal::chamfer_panels_qbuf<false>(
               buffer_.data(), stride_, query_.num_queries,
               db.panel_data.data() + db.panel_offsets[id], db.panel_bytes,
               db.num_hamming_tiles, db.cloud_sizes[id], db.padded_dim) /
           static_cast<float>(query_.num_queries);
#else
    return query_.distance(db[id]);
#endif
  }

 private:
  const TQ1::EncodedQuery& query_;
  size_t stride_ = 0;
  std::vector<uint8_t> buffer_;
};

std::vector<Scored> ScoreCandidates(const TQ1::EncodedSet& db, const Index& index,
                                  const std::vector<uint32_t>& centers, uint32_t b,
                                  const PreparedQuery& prepared) {
  std::vector<uint64_t> selected((index.corpus_size + size_t{63}) / 64, 0);
  for (auto center : centers) for (uint32_t j = 0; j < b; ++j) {
    const auto doc = index.buckets[center][j].document;
    selected[doc / 64] |= uint64_t{1} << (doc % 64);
  }
  std::vector<uint32_t> ids;
  ids.reserve(std::min<size_t>(size_t(centers.size()) * b, index.corpus_size));
  for (size_t word = 0; word < selected.size(); ++word)
    for (auto bits = selected[word]; bits; bits &= bits - 1)
      ids.push_back(static_cast<uint32_t>(word * 64 + __builtin_ctzll(bits)));
  auto bytes = [&](size_t i) {
    return db.panel_offsets[ids[i] + 1] - db.panel_offsets[ids[i]];
  };
  std::vector<size_t> starts{0};
  size_t current = 0;
  for (size_t i = 0; i < ids.size(); ++i) {
    if (current && current + bytes(i) > 128 * 1024) {
      starts.push_back(i);
      current = 0;
    }
    current += bytes(i);
  }
  starts.push_back(ids.size());
  std::vector<Scored> scores(ids.size());
  parlay::parallel_for(0, starts.size() - 1, [&](size_t chunk) {
    const auto first = starts[chunk], last = starts[chunk + 1];
    std::vector<size_t> prefix(last - first + 1, 0);
    for (size_t i = first; i < last; ++i) prefix[i - first + 1] = prefix[i - first] + bytes(i);
    size_t ahead = first;
    for (size_t i = first; i < last; ++i) {
      ahead = std::max(ahead, i);
      while (ahead < last && prefix[ahead - first] < prefix[i - first] + 8192) ++ahead;
      if (ahead < last) {
        const auto* ptr = db.panel_data.data() + db.panel_offsets[ids[ahead]];
        for (size_t off = 0; off < std::min<size_t>(bytes(ahead), 512); off += 64)
          __builtin_prefetch(ptr + off, 0, 1);
      }
      scores[i] = {prepared.Distance(db, ids[i]), ids[i]};
    }
  }, 1);
  return scores;
}

}  // namespace

void BuildRoutingGraph(const CloudSet& train, const Index& index,
                       const std::filesystem::path& output,
                       const RoutingBuildParams& params) {
  ValidatePrototypes(train, index);
  if (!params.degree || !params.beam || !std::isfinite(params.alpha) || params.alpha < 1)
    throw std::invalid_argument("invalid EMDIVF routing graph parameters");
  auto prototypes = PrototypeClouds(train, index);
  auto graph = GraphBuilder(params).Build(prototypes);
  if (!output.parent_path().empty()) std::filesystem::create_directories(output.parent_path());
  const auto temporary = output.string() + ".tmp";
  std::ofstream out(temporary, std::ios::binary);
  Write<uint32_t>(out, 0x5056414d);
  Write<uint32_t>(out, 2);
  Write<uint32_t>(out, 1);  // Chamfer, matching the original graph metric tag.
  Write<uint32_t>(out, 0);  // Entry point.
  Write(out, index.dimension);
  Write(out, index.train_size);
  Write<uint32_t>(out, index.centers.size());
  for (auto center : index.centers) Write(out, center);
  graph.save(out);
  out.close();
  if (!out) throw std::runtime_error("failed to close EMDIVF routing graph");
  std::filesystem::rename(temporary, output);
}

struct SearchIndex::Impl {
  Index index;
  mutable Graph graph;
  TQ4 routing_model;
  TQ4::EncodedSet routing_db;
  TQ8 prototype_model;
  TQ8::EncodedSet prototype_db;
  TQ1 scoring_model;
  TQ1::EncodedSet scoring_db;
  TQ8 rerank_model;
  TQ8::EncodedSet rerank_db;
};

SearchIndex::SearchIndex(const CloudSet& train, const CloudSet& corpus, Index index,
                         const std::filesystem::path& routing_graph)
    : impl_(std::make_unique<Impl>()) {
  ValidatePrototypes(train, index);
  if (corpus.get_dims() != index.dimension || corpus.size() != index.corpus_size ||
      !index.max_bucket_size || index.buckets.size() != index.centers.size())
    throw std::invalid_argument("EMDIVF corpus or buckets do not match index");
  for (const auto& bucket : index.buckets) {
    if (bucket.size() != index.max_bucket_size)
      throw std::invalid_argument("EMDIVF bucket size mismatch");
    for (auto entry : bucket)
      if (entry.document >= corpus.size() || !std::isfinite(entry.distance))
        throw std::invalid_argument("invalid EMDIVF bucket entry");
  }
  impl_->graph = LoadGraph(routing_graph, index);
  auto prototypes = PrototypeClouds(train, index);
  // Prototype encoding IDs must be contiguous graph IDs, not training row IDs.
  impl_->routing_model.train(prototypes);
  impl_->routing_db = impl_->routing_model.encode(prototypes);
  impl_->prototype_model.train(prototypes);
  impl_->prototype_db = impl_->prototype_model.encode(prototypes);
  impl_->scoring_model.train(corpus);
  impl_->scoring_db = impl_->scoring_model.encode(corpus);
  impl_->rerank_model.train(corpus);
  impl_->rerank_db = impl_->rerank_model.encode(corpus);
  impl_->index = std::move(index);
}

SearchIndex::~SearchIndex() = default;
SearchIndex::SearchIndex(SearchIndex&&) noexcept = default;
SearchIndex& SearchIndex::operator=(SearchIndex&&) noexcept = default;
size_t SearchIndex::NumPrototypes() const noexcept { return impl_->index.centers.size(); }
size_t SearchIndex::NumDocuments() const noexcept { return impl_->index.corpus_size; }

SearchResult SearchIndex::Search(const mvsic::ChamferIP_Point& query,
                                const SearchParams& params) const {
  const auto total_start = Clock::now();
  const auto& index = impl_->index;
  const uint32_t b = params.bucket_size ? params.bucket_size : index.max_bucket_size;
  const uint32_t shortlist = std::min<size_t>(params.shortlist, NumPrototypes());
  if (query.get_dims() != index.dimension || !query.size() || !params.k ||
      params.k > index.corpus_size || !params.nprobes || params.nprobes > shortlist ||
      params.num_rerank < params.k || !b || b > index.max_bucket_size ||
      !std::isfinite(params.compression_threshold))
    throw std::invalid_argument("invalid EMDIVF search query or parameters");
  SearchResult result;
  auto start = Clock::now();
  mvsic::CompressedPointCloud<mvsic::ChamferIP_Point> routing_compressed;
  auto routing_query = query;
  if (params.compress_query) {
    routing_compressed = mvsic::compress_query<mvsic::ChamferIP_Point>(
        query, mvsic::SearchParams::QueryCompression::Wards,
        params.compression_threshold, TQ4::kBatchAlignment, 4, false);
    routing_query = routing_compressed.view();
  }
  auto q4 = impl_->routing_model.quantize_query(routing_query);
  auto route_params = mvsic::SearchParams::vamana(shortlist, shortlist, 1.35, 0);
  auto [coarse, comparisons, routing_bytes] = mvsic::vamana::beam_search(
      q4, impl_->graph, impl_->routing_db, uint32_t{0}, route_params);
  result.prototype_comparisons = comparisons;
  result.routing_ms = Ms(start);
  start = Clock::now();
  auto center_query = impl_->prototype_model.quantize_query(query);
  std::vector<Scored> centers;
  centers.reserve(coarse.first.size());
  for (auto [id, distance] : coarse.first)
    centers.emplace_back(mvsic::turboquant_8bit_mv::turboquant_8bit_mv_chamfer_distance<false>(
                             center_query, impl_->prototype_db, id), id);
  const size_t count = std::min<size_t>(params.nprobes, centers.size());
  std::partial_sort(centers.begin(), centers.begin() + count, centers.end());
  std::vector<uint32_t> selected(count);
  for (size_t i = 0; i < count; ++i) selected[i] = centers[i].second;
  result.prototype_rerank_ms = Ms(start);
  start = Clock::now();
  mvsic::CompressedPointCloud<mvsic::ChamferIP_Point> scoring_compressed;
  auto scoring_query = query;
  if (params.compress_query) {
    scoring_compressed = mvsic::compress_query<mvsic::ChamferIP_Point>(
        query, mvsic::SearchParams::QueryCompression::Wards,
        params.compression_threshold, TQ1::kBatchAlignment, 4, false);
    scoring_query = scoring_compressed.view();
  }
  auto q1 = impl_->scoring_model.quantize_query(scoring_query);
  PreparedQuery prepared(q1, impl_->scoring_db);
  auto scores = ScoreCandidates(impl_->scoring_db, index, selected, b, prepared);
  result.candidates_scored = scores.size();
  const size_t nr = std::min<size_t>(params.num_rerank, scores.size());
  std::partial_sort(scores.begin(), scores.begin() + nr, scores.end());
  scores.resize(nr);
  result.scoring_ms = Ms(start);
  start = Clock::now();
  auto q8 = impl_->rerank_model.quantize_query(query);
  auto candidates = parlay::tabulate(scores.size(), [&](size_t i) {
    return std::make_pair(scores[i].second, scores[i].first);
  });
  auto [top, rerank_bytes] = mvsic::turboquant_8bit_mv::rerank_candidates<false>(
      q8, impl_->rerank_db, candidates, candidates.size(), std::min<size_t>(params.k, nr));
  result.neighbors.reserve(top.size());
  for (auto [id, distance] : top) result.neighbors.push_back({id, distance});
  result.final_rerank_ms = Ms(start);
  result.total_ms = Ms(total_start);
  return result;
}

}  // namespace emdivf
