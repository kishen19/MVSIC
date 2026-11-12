#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

#include "mvsic/algorithms/emdivf/index.h"

namespace emdivf {

struct RoutingBuildParams {
  uint32_t degree = 32;
  uint32_t beam = 200;
  double alpha = 1.2;
};

struct SearchParams {
  uint32_t k = 10;
  uint32_t nprobes = 32;
  uint32_t shortlist = 128;
  uint32_t num_rerank = 500;
  // Zero uses the index's stored bucket size. A smaller value selects a prefix.
  uint32_t bucket_size = 0;
  bool compress_query = true;
  float compression_threshold = 0.5f;
};

struct SearchResult {
  // Negative mean maximum inner product; smaller distances are better.
  std::vector<Neighbor> neighbors;
  size_t candidates_scored = 0;
  size_t prototype_comparisons = 0;
  double routing_ms = 0;
  double prototype_rerank_ms = 0;
  double scoring_ms = 0;
  double final_rerank_ms = 0;
  double total_ms = 0;
};

// Builds a Chamfer Vamana graph over the selected prototypes. The new graph
// format records the prototype IDs so a mismatched bucket index is rejected.
void BuildRoutingGraph(const CloudSet& train, const Index& index,
                       const std::filesystem::path& output,
                       const RoutingBuildParams& params = {});

// The standard one-copy EMDIVF search path: TQ4 graph routing, TQ8 prototype
// reranking, unique TQ1 candidate scoring, and the shared TQ8 document reranker.
// Construction encodes the corpus once. Source point-cloud sets may be released
// after the constructor returns. Search workspaces are query-owned, allowing
// independent queries to run concurrently.
class SearchIndex {
 public:
  SearchIndex(const CloudSet& train, const CloudSet& corpus, Index index,
              const std::filesystem::path& routing_graph);
  ~SearchIndex();
  SearchIndex(SearchIndex&&) noexcept;
  SearchIndex& operator=(SearchIndex&&) noexcept;
  SearchIndex(const SearchIndex&) = delete;
  SearchIndex& operator=(const SearchIndex&) = delete;

  SearchResult Search(const mvsic::ChamferIP_Point& query,
                      const SearchParams& params = {}) const;
  size_t NumPrototypes() const noexcept;
  size_t NumDocuments() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace emdivf
