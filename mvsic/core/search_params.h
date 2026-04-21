#pragma once

#include <cassert>
#include <string>

namespace mvsic {

struct SearchParams {
 public:
  // Common Params
  std::string method;  // Index name
  size_t k;            // Number of nearest neighbors
  size_t num_rerank;   // Number of candidates to rerank: 0 if no reranking

  // mvivf
  size_t nprobes;  // Number of leaf-level nodes probed

  // ANN (muvera, mpool, vamana) Params
  size_t L;
  double cut;

  // muvera, mpool, vamana
  bool norerank = false;

  // Query point cloud compression before search.
  // Carve: greedy ball-carving clustering (MUVERA paper).
  //   IP metric: merge if <q_i, q_j> >= threshold (default 0.7). Centroids = sum.
  //   L2 metric: merge if ||q_i - q_j||^2 <= threshold. Centroids = mean.
  // Wards: agglomerative clustering with Ward linkage criterion.
  //   merge while Ward distance <= threshold. Centroids = weighted mean.
  enum class QueryCompression : uint8_t { None = 0, Carve = 1, Wards = 2 };
  QueryCompression query_compression = QueryCompression::None;
  float query_compression_threshold = 0.7f;
  bool compress_rerank = false;  // If true, use compressed query for reranking too

  // mvivf search params
  static SearchParams mvivf(size_t k, size_t nprobes, size_t num_rerank = 0) {
    SearchParams params;
    params.method = "mvivf";
    params.k = k;
    assert(k > 0);
    params.nprobes = nprobes;
    assert(nprobes > 0);
    params.num_rerank = num_rerank;
    assert(num_rerank == 0 || num_rerank >= k);
    return params;
  }

  // mvivf_spill search params
  static SearchParams mvivf_spill(size_t k, size_t nprobes, size_t num_rerank = 0) {
    SearchParams params;
    params.method = "mvivf_spill";
    params.k = k;
    assert(k > 0);
    params.nprobes = nprobes;
    assert(nprobes > 0);
    params.num_rerank = num_rerank;
    assert(num_rerank == 0 || num_rerank >= k);
    return params;
  }

  // mvivf_flat search params
  static SearchParams mvivf_flat(size_t k, size_t nprobes, size_t num_rerank = 0) {
    SearchParams params;
    params.method = "mvivf_flat";
    params.k = k;
    assert(k > 0);
    params.nprobes = nprobes;
    assert(nprobes > 0);
    params.num_rerank = num_rerank;
    assert(num_rerank == 0 || num_rerank >= k);
    return params;
  }

  // vamana
  static SearchParams vamana(size_t k, size_t L, double cut = 1.35, size_t num_rerank = 0) {
    SearchParams params;
    params.method = "vamana";
    params.k = k;
    params.L = L;
    params.cut = cut;
    params.num_rerank = num_rerank;
    return params;
  }

  // muvera
  static SearchParams muvera(size_t k, size_t L, size_t num_rerank, double cut = 1.35,
                             bool norerank = false) {
    SearchParams params;
    params.method = "muvera";
    params.k = k;
    params.L = L;
    params.cut = cut;
    params.norerank = norerank;
    params.num_rerank = num_rerank;
    assert(num_rerank >= k);
    return params;
  }

  // mean_pooling
  static SearchParams mpool(size_t k, size_t L, size_t num_rerank, double cut = 1.35,
                            bool norerank = false) {
    SearchParams params;
    params.method = "mpool";
    params.k = k;
    params.L = L;
    params.cut = cut;
    params.norerank = norerank;
    params.num_rerank = num_rerank;
    assert(num_rerank >= k);
    return params;
  }

  // svh_ivf search params
  static SearchParams svh_ivf(size_t k, size_t nprobes, size_t num_rerank, bool norerank = false) {
    SearchParams params;
    params.method = "svh_ivf";
    params.k = k;
    assert(k > 0);
    params.nprobes = nprobes;
    assert(nprobes > 0);
    params.norerank = norerank;
    params.num_rerank = num_rerank;
    assert(num_rerank >= k);
    return params;
  }

  // svh_graph search params
  static SearchParams svh_graph(size_t k, size_t L, size_t num_rerank, double cut = 1.35,
                                bool norerank = false) {
    SearchParams params;
    params.method = "svh_graph";
    params.k = k;
    assert(k > 0);
    params.L = L;
    params.cut = cut;
    params.norerank = norerank;
    params.num_rerank = num_rerank;
    assert(num_rerank >= k);
    return params;
  }

  SearchParams() = default;
};

}  // namespace mvsic