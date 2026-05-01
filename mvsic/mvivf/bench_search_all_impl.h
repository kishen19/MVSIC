#pragma once

// Template definition for bench_search_all's per-cell run_one. Heavy: this
// header is only included by the per-cell instantiation .cc files under
// bench_search_all_inst/, never by the dispatch TU.

#include <iostream>
#include <string>
#include <utility>

#include "mvsic/core/bench_utils.h"
#include "mvsic/core/stats.h"
#include "mvsic/mvivf/bench_search_all_inst.h"

namespace mvsic {

template <typename ChPoint, bool metric, class IndexT>
void run_one(RunOneCtx<ChPoint>& ctx) {
  using PC = PointCloudSet<ChPoint>;

  IndexT index(ctx.points->get_dims(), ctx.ip);
  bench::build_or_load(index, *ctx.points, ctx.io.index_path, ctx.io.save_path);

  if (ctx.ds.queries.empty() || ctx.ds.gt.empty()) {
    std::cout << "No queries/GT specified. Done." << std::endl;
    return;
  }
  auto queries = PC(ctx.ds.queries.c_str());
  auto gt = ReadGT(ctx.ds.gt, queries.size());

  const char* vname = ctx.variant == MVIVFVariant::Flat  ? "MVIVF_Flat"
                    : ctx.variant == MVIVFVariant::Spill ? "MVIVF_Spill"
                                                         : "MVIVF";
  bench::print_header(std::string(vname) + " (search_all)", ctx.ds.name,
                      ctx.points->size(), queries.size());
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

  bench::run_search_all_sweep(index, *ctx.points, queries, gt, "nprobes",
                              ctx.nprobes_list, make_sp, ctx.io.csv_path);
}

}  // namespace mvsic
