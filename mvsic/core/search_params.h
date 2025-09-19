#pragma once

namespace mvsic {

struct SearchParams {
 public:
  // Common Params
  std::string method;  // Index name
  size_t k;            // Number of nearest neighbors
  size_t num_rerank;   // Number of candidates to rerank: 0 if no reranking

  // mvivf
  size_t nprobes;  // Number of leaf-level nodes probed

  // Vamana (muvera, mpool, mvvamana) Params
  size_t L;
  double cut;
  size_t limit;
  size_t degree_limit;

  // mvivf search params
  static SearchParams mvivf(size_t k, size_t nprobes, size_t num_rerank) {
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

  // mvivf_flat search params
  static SearchParams mvivf_flat(size_t k, size_t nprobes, size_t num_rerank) {
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

  // single vector heuristic
  // static SearchParams svh(size_t k, size_t nprobes, size_t cands, size_t beam_length = 0) {
  //   SearchParams params;
  //   params.method = "svh";
  //   params.k = k;
  //   params.nprobes = nprobes;
  //   params.cands = cands;
  //   if (beam_length == 0) {
  //     params.beam_length = nprobes;
  //   } else {
  //     params.beam_length = beam_length;
  //   }
  //   return params;
  // }

  // // vamana
  // static SearchParams mvvamana(size_t k, size_t L, double cut, size_t limit, size_t degree_limit)
  // {
  //   SearchParams params;
  //   params.method = "mvvamana";
  //   params.k = k;
  //   params.L = L;
  //   params.cut = cut;
  //   params.limit = limit;
  //   params.degree_limit = degree_limit;
  //   return params;
  // }

  // // muvera
  // static SearchParams muvera(size_t k, size_t L, double cut, size_t limit, size_t degree_limit,
  //                            bool rerank, size_t num_candidates_to_rerank) {
  //   SearchParams params;
  //   params.method = "muvera";
  //   params.k = k;
  //   params.L = L;
  //   params.cut = cut;
  //   params.limit = limit;
  //   params.degree_limit = degree_limit;
  //   params.rerank = rerank;
  //   params.num_candidates_to_rerank = num_candidates_to_rerank;
  //   return params;
  // }

  // // mean_pooling
  // static SearchParams mpool(size_t k, size_t L, double cut, size_t limit, size_t degree_limit,
  //                           bool rerank, size_t num_candidates_to_rerank) {
  //   SearchParams params;
  //   params.method = "mpool";
  //   params.k = k;
  //   params.L = L;
  //   params.cut = cut;
  //   params.limit = limit;
  //   params.degree_limit = degree_limit;
  //   params.rerank = rerank;
  //   params.num_candidates_to_rerank = num_candidates_to_rerank;
  //   return params;
  // }

  //  private:
  SearchParams() = default;
};

}  // namespace mvsic