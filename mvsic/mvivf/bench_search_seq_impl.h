#pragma once

// Template definition for bench_search_seq's per-cell run_one_seq. Heavy:
// this header is only included by the per-cell instantiation .cc files
// under bench_search_seq_inst/, never by the dispatch TU.
//
// Mirrors bench_search_all_impl.h but drives the per-query
// run_search_sweep_seq path (single-thread search via
// parlay::execute_with_scheduler(1, ...)) and supports random subsampling
// of the query set via num_queries / query_sample_seed.

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "mvsic/core/bench_utils.h"
#include "mvsic/core/stats.h"
#include "mvsic/mvivf/bench_search_seq_inst.h"
#include "parlay/sequence.h"
#include "parlay/delayed_sequence.h"

namespace mvsic {
namespace bench_seq {

inline const char* variant_name(MVIVFVariant v) {
  return v == MVIVFVariant::Flat  ? "MVIVF_Flat"
       : v == MVIVFVariant::Spill ? "MVIVF_Spill"
                                  : "MVIVF";
}

template <typename ChPoint, bool metric, class IndexT>
void run_one_seq(RunOneCtxSeq<ChPoint>& ctx) {
  using PC = PointCloudSet<ChPoint>;

  IndexT index(ctx.points->get_dims(), ctx.ip);
  bench::build_or_load(index, *ctx.points, ctx.io.index_path);

  if (ctx.ds.queries.empty() || ctx.ds.gt.empty()) {
    std::cout << "No queries/GT specified. Done." << std::endl;
    return;
  }
  auto queries = PC(ctx.ds.queries.c_str());
  auto gt = ReadGT(ctx.ds.gt, queries.size());

  // Optional random subsample of the query set. Fixed-seed shuffle so
  // re-runs hit the same subset; override with -query_sample_seed.
  if (ctx.num_queries > 0 && ctx.num_queries < queries.size()) {
    const size_t n_full = queries.size();
    std::vector<uint32_t> idx(n_full);
    std::iota(idx.begin(), idx.end(), 0);
    std::mt19937_64 rng(ctx.query_sample_seed);
    std::shuffle(idx.begin(), idx.end(), rng);
    idx.resize(ctx.num_queries);

    auto sampled = parlay::delayed_tabulate(
        ctx.num_queries, [&](size_t i) { return queries[idx[i]]; });
    auto queries_sub = PC(sampled, queries.get_dims());
    parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>> gt_sub(ctx.num_queries);
    for (size_t i = 0; i < ctx.num_queries; ++i) gt_sub[i] = gt[idx[i]];

    queries = std::move(queries_sub);
    gt = std::move(gt_sub);
    std::cout << "Subsampled " << ctx.num_queries << " of " << n_full
              << " queries (seed=" << ctx.query_sample_seed << ")" << std::endl;
  }

  bench::print_header(std::string(variant_name(ctx.variant)) + " (seq)",
                      ctx.ds.name, ctx.points->size(), queries.size());
  bench::print_compression_stats<ChPoint>(queries, ctx.sp_base, index.quantization_mode);

  const MVIVFVariant variant = ctx.variant;
  const size_t k = ctx.k;
  const size_t num_rerank = ctx.num_rerank;
  const SearchParams sp_base = ctx.sp_base;
  auto make_sp = [variant, k, num_rerank, sp_base](size_t np) {
    SearchParams sp;
    if (variant == MVIVFVariant::Flat)
      sp = SearchParams::mvivf_flat(k, np, num_rerank);
    else if (variant == MVIVFVariant::Spill)
      sp = SearchParams::mvivf_spill(k, np, num_rerank);
    else
      sp = SearchParams::mvivf(k, np, num_rerank);
    sp.query_compression = sp_base.query_compression;
    sp.query_compression_threshold = sp_base.query_compression_threshold;
    sp.compress_rerank = sp_base.compress_rerank;
    sp.tq8_rerank = sp_base.tq8_rerank;
    return sp;
  };

  std::vector<std::string> labels;
  if (variant == MVIVFVariant::Flat) {
    labels = {"n_centers", "probe_cmps", "t_compress", "t_search", "t_leaf_dists",
              "t_leaf_rest", "t_rerank"};
  } else if (variant == MVIVFVariant::Spill) {
    labels = {"search_cmps", "probe_cmps", "t_search_dists", "t_search_beam",
              "t_search_rest", "t_compress", "t_quant", "t_leaf_dists", "t_leaf_dedup",
              "t_leaf_rest", "t_rerank"};
  } else {
    labels = {"search_cmps", "probe_cmps", "t_search_dists", "t_search_beam",
              "t_search_rest", "t_search_top_level", "t_compress", "t_quant",
              "t_leaf_dists", "t_leaf_rest", "t_rerank", "t_greedy"};
  }
  bench::run_search_sweep_seq(index, *ctx.points, queries, gt, "nprobes",
                              ctx.nprobes_list, make_sp, ctx.io.csv_path, labels,
                              ctx.skip_warmup);
}

}  // namespace bench_seq
}  // namespace mvsic
