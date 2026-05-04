#pragma once

// Template definition for bench's per-cell bench_run_one. This header is
// only included by per-cell instantiation .cc files under bench_inst/, never
// by the dispatch TU. It deliberately does NOT include any mvivf*.h or
// quantizer headers — each per-cell .cc adds only the variant + quantizer
// it instantiates, keeping per-TU compile time small.

#include <cstdint>
#include <iostream>
#include <string>

#include "mvsic/core/query_compression.h"
#include "mvsic/core/stats.h"
#include "mvsic/core/types/point_cloud_set.h"
#include "mvsic/mvivf/bench_run_one_decl.h"

namespace mvsic {
namespace bench_main {

inline const char* variant_name(MVIVFVariant v) {
  return v == MVIVFVariant::Flat  ? "MVIVF_Flat"
       : v == MVIVFVariant::Spill ? "MVIVF_Spill"
                                  : "MVIVF";
}

template <typename ChPoint, bool metric, class IndexT>
void bench_run_one(BenchRunOneCtx<ChPoint>& ctx) {
  using PC = PointCloudSet<ChPoint>;

  IndexT index(ctx.points->get_dims(), ctx.index_params);
  if (!ctx.index_file.empty()) {
    std::cout << "Loading index from " << ctx.index_file << std::endl;
    index.load(ctx.index_file, *ctx.points);
    std::cout << "Index loaded" << std::endl;
  } else {
    std::cout << "Building index (" << variant_name(ctx.variant)
              << ", compress_centers=" << (ctx.compress_centers ? 1 : 0)
              << ", leaf=" << ctx.quant_method << ")..." << std::endl;
    parlay::internal::timer it;
    it.start();
    index.build(*ctx.points);
    it.stop();
    std::cout << "Index built in " << it.total_time() << " seconds." << std::endl;
  }
  if (!ctx.out_file.empty()) {
    std::cout << "Saving index to " << ctx.out_file << std::endl;
    index.save(ctx.out_file);
    std::cout << "Index saved." << std::endl;
  }

  if (ctx.q_file.empty()) return;
  auto queries = PC(ctx.q_file.c_str());
  if (ctx.search_params.query_compression != SearchParams::QueryCompression::None) {
    uint32_t ba = qc_internal::batch_alignment(index.quantization_mode);
    double sum_orig = 0.0;
    double sum_comp = 0.0;
    for (size_t j = 0; j < queries.size(); ++j) {
      sum_orig += static_cast<double>(queries[j].size());
      auto c = compress_query<ChPoint>(queries[j], ctx.search_params.query_compression,
                                       ctx.search_params.query_compression_threshold, ba,
                                       ctx.search_params.query_alignment,
                                       ctx.search_params.query_alignment_strict);
      sum_comp += static_cast<double>(c.n);
    }
    const double nq = static_cast<double>(queries.size());
    std::cout << "Avg query points (raw):        " << (sum_orig / nq) << std::endl
              << "Avg query points (compressed): " << (sum_comp / nq) << std::endl
              << "Compression ratio (raw/compr): "
              << (sum_comp > 0 ? sum_orig / sum_comp : 0.0) << std::endl;
  }
  auto gt = ReadGT(ctx.gt_file, queries.size());
  std::cout << "Computing stats..." << std::endl;
  Stats result = compute_stats(index, *ctx.points, queries, gt, ctx.search_params);
  const std::size_t k = ctx.k;
  std::cout << "Number of Queries: " << queries.size() << std::endl
            << "QPS_seq: " << result.QPS_seq << std::endl
            << "QPS_par: " << result.QPS_par << std::endl
            << "Average cmps: " << result.avg_cmps << std::endl
            << "Average recall 1 @ " << k << ": " << result.recall_1_k << std::endl
            << "Average recall " << k << " @ " << k << ": " << result.recall_k_k << std::endl;
}

}  // namespace bench_main
}  // namespace mvsic
