#pragma once

// Template definition for the microbenchmark bench_search_all's per-cell
// run_one. Heavy: this header is only included by the per-cell instantiation
// .cc files under bench_search_all_inst/, never by the dispatch TU.

#include <filesystem>
#include <iostream>
#include <string>
#include <utility>

#include "parlay/primitives.h"

#include "mvsic/core/stats.h"
#include "mvsic/core/types/point_cloud_set.h"
#include "mvsic/mvivf/mvivf.h"
#include "microbenchmark/mvivf/bench_search_all_inst.h"

namespace mvsic {

template <typename ChPoint, bool metric>
void build_skeleton_if_missing(PointCloudSet<ChPoint>& points,
                               const IndexParams& index_params,
                               const std::string& index_path) {
  const std::filesystem::path idx_path(index_path);
  if (std::filesystem::exists(idx_path)) return;

  // save() is only valid on the <metric, false, NoQuantizer> variant; any
  // templated variant can load that skeleton and retrain its center / leaf
  // quantizers on load.
  using SkeletonIndex = IndexMVIVF<metric, false, NoQuantizer<metric>>;
  std::cout << "Index not found at " << index_path
            << ". Building raw skeleton (compress_centers=0, leaf_quantizer=None)..."
            << std::endl;
  SkeletonIndex skeleton(points.get_dims(), index_params);
  parlay::internal::timer tb;
  tb.start();
  skeleton.build(points);
  tb.stop();
  std::cout << "Skeleton built in " << tb.total_time() << " seconds." << std::endl;
  if (!idx_path.parent_path().empty()) {
    std::filesystem::create_directories(idx_path.parent_path());
  }
  std::cout << "Saving skeleton index to " << index_path << std::endl;
  skeleton.save(index_path);
}

template <typename ChPoint, bool metric, class IndexT>
void run_one(MicroSearchAllCtx<ChPoint>& ctx) {
  IndexT index(ctx.points->get_dims(), ctx.index_params);
  std::cout << "Loading index (compress_centers=" << (ctx.compress_centers ? 1 : 0)
            << ", leaf_quantizer=" << ctx.quant_method_name << ") from " << ctx.index_path
            << " ..." << std::endl;
  index.load(ctx.index_path, *ctx.points);

  const bool run_old = (ctx.mode == "both" || ctx.mode == "old");
  const bool run_new = (ctx.mode == "both" || ctx.mode == "new");

  for (std::size_t nprobes : ctx.nprobes_list) {
    std::cout << "\n############################################" << std::endl;
    std::cout << "  nprobes = " << nprobes << std::endl;
    std::cout << "############################################" << std::endl;

    SearchParams search_params = SearchParams::mvivf(ctx.k, nprobes, ctx.num_rerank);
    search_params.query_compression =
        static_cast<SearchParams::QueryCompression>(ctx.query_compression);
    search_params.query_compression_threshold = ctx.query_compression_threshold;
    search_params.compress_rerank = ctx.compress_rerank;
    search_params.tq8_rerank = ctx.tq8_rerank;

    if (search_params.query_compression != SearchParams::QueryCompression::None) {
      const char* mname =
          (search_params.query_compression == SearchParams::QueryCompression::Carve) ? "ball"
                                                                                     : "wards";
      std::cout << "Query compression: " << mname
                << " tau=" << search_params.query_compression_threshold
                << " compress_rerank=" << (search_params.compress_rerank ? 1 : 0) << std::endl;
    }
    if (search_params.tq8_rerank) std::cout << "tq8_rerank=1" << std::endl;

    parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>> pred1, pred2;
    std::size_t cmps1 = 0, cmps2 = 0;
    double time1 = 0, time2 = 0;

    if (run_old) {
      std::cout << "\n========================================" << std::endl;
      std::cout << "Running search_all (Original)..." << std::endl;
      parlay::internal::timer t1;
      t1.start();
      auto [p1, c1] = index.search_all(*ctx.queries, *ctx.points, search_params);
      t1.stop();
      pred1 = std::move(p1);
      cmps1 = c1;
      time1 = t1.total_time();
      std::cout << "search_all time: " << time1 << " seconds. Dist cmps: " << cmps1 << std::endl;
    }

    if (run_new) {
      std::cout << "\n========================================" << std::endl;
      std::cout << "Running search_all_new..." << std::endl;
      parlay::internal::timer t2;
      t2.start();
      auto [p2, c2] = index.search_all_new(*ctx.queries, *ctx.points, search_params);
      t2.stop();
      pred2 = std::move(p2);
      cmps2 = c2;
      time2 = t2.total_time();
      std::cout << "search_all_new time: " << time2 << " seconds. Dist cmps: " << cmps2
                << std::endl;
    }

    if (run_old && run_new) {
      std::cout << "\n========================================" << std::endl;
      std::cout << "Speedup (old / new): " << time1 / time2 << "x" << std::endl;
      std::cout << "========================================" << std::endl;

      if (ctx.has_gt) {
        auto [r1_old, rk_old] = compute_scores(pred1, ctx.gt, ctx.k);
        auto [r1_new, rk_new] = compute_scores(pred2, ctx.gt, ctx.k);
        std::cout << "\nRecall vs Ground Truth:" << std::endl;
        std::cout << "  search_all:     recall@1 = " << r1_old << ", recall@" << ctx.k << " = "
                  << rk_old << std::endl;
        std::cout << "  search_all_new: recall@1 = " << r1_new << ", recall@" << ctx.k << " = "
                  << rk_new << std::endl;
      }

      double agreement = compute_recall(pred2, pred1, ctx.k, ctx.k);
      std::cout << "\nAgreement (new vs old): " << agreement << std::endl;
    } else if (ctx.has_gt) {
      auto& pred = run_old ? pred1 : pred2;
      auto [r1, rk] = compute_scores(pred, ctx.gt, ctx.k);
      std::cout << "\nRecall vs Ground Truth: recall@1 = " << r1 << ", recall@" << ctx.k << " = "
                << rk << std::endl;
    }
  }
}

}  // namespace mvsic
