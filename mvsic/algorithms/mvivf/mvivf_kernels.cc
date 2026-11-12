#include "mvsic/algorithms/mvivf/mvivf_internal.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <mutex>
#include <stdexcept>

#include "mvsic/core/distance_measures/one_to_many.h"
#include "mvsic/core/distance_measures/many_to_many.h"
#include "mvsic/core/quantization/turboquant_mv.h"
#include "mvsic/core/quantization/turboquant_1bit_mv.h"
#include "mvsic/core/quantization/turboquant_8bit_mv.h"
#include "mvsic/core/query_compression.h"
#include "mvsic/core/types/chamfer_ip_point.h"
#include "parlay/sequence.h"

namespace mvsic::mvivf_internal {
namespace {

using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

using ChPoint = ChamferIP_Point;

ChPoint typed_point(const PointCloud& p) {
  return {p.n, p.d, const_cast<float*>(p.values), p.id};
}

struct TypedView : PointCloudSetView {
  explicit TypedView(const PointCloudSetView& view) : PointCloudSetView(view) {}
  static constexpr bool is_metric() noexcept { return false; }
  ChPoint operator[](size_t i) const noexcept { return typed_point(PointCloudSetView::operator[](i)); }
};

// Quantized leaves read original clouds directly. Only their small offset
// table is materialized; no temporary float corpus is needed for encoding.
struct SubsetView {
  const PointCloudSetView& points;
  std::span<const uint32_t> rows;
  parlay::sequence<size_t> offsets;
  SubsetView(const PointCloudSetView& points, std::span<const uint32_t> rows)
      : points(points), rows(rows), offsets(parlay::sequence<size_t>::uninitialized(rows.size() + 1)) {
    offsets[0] = 0;
    for (size_t i = 0; i < rows.size(); ++i)
      offsets[i + 1] = offsets[i] + (points.offsets[rows[i] + 1] - points.offsets[rows[i]]);
  }
  uint32_t get_dims() const noexcept { return points.d; }
  std::span<const size_t> get_offsets() const noexcept { return {offsets.data(), offsets.size()}; }
  std::span<const uint32_t> get_ids() const noexcept { return rows; }
  const float* data(size_t i) const noexcept { return points.data(rows[i]); }
};

struct Query final : PreparedQuery {
  CompressedPointCloud<ChPoint> storage;
  turboquant_1bit_mv::Quantized_Query_Point_Cloud<false> onebit;
  turboquant_mv::Quantized_Query_Point_Cloud<false> fourbit;
  turboquant_8bit_mv::Quantized_Query_Point_Cloud<false> eightbit;
};

class IPKernels final : public Kernels {
  using QT = IndexParams::QuantizerType;
  using TQ4 = turboquant_mv::Model<false>;
  using TQ1 = turboquant_1bit_mv::Model<false>;
  using TQ8 = turboquant_8bit_mv::Model<false>;
  IndexParams params_;
  PointCloudSetView points_;
  TQ1 onebit_model_;
  TQ4 fourbit_model_;
  mutable TQ8 eightbit_model_;
  mutable TQ8::EncodedSet rerank_db_;
  mutable std::once_flag rerank_once_;

 public:
  IPKernels(const IndexParams& params, const PointCloudSetView& points)
      : params_(params), points_(points) {
    if (params.quantize_centers || params.pq.method == QT::TurboQuant) fourbit_model_.train(points);
    switch (params.pq.method) {
      case QT::None: break;
      case QT::OneBitTQ: onebit_model_.train(points); break;
      case QT::TurboQuant: break;
      case QT::EightBitTQ: eightbit_model_.train(points); break;
      default: throw std::invalid_argument("Compiled MVIVF supports None, OneBitTQ, TurboQuant, EightBitTQ leaves");
    }
  }

  void prepare_rerank() const override {
    // Float/no-rerank searches pay no full-corpus TQ8 storage or encoding cost.
    // Build/load replace this object, including the flag and bound corpus.
    std::call_once(rerank_once_, [&] {
      if (params_.pq.method != QT::EightBitTQ) eightbit_model_.train(points_);
      auto dense = points_;
      dense.ids = nullptr;
      rerank_db_ = eightbit_model_.encode(dense);
    });
  }

  void prepare_nodes(const std::vector<Node>& nodes, std::vector<EncodedNode>& encoded) const override {
    encoded.resize(nodes.size());
    parlay::parallel_for(0, nodes.size(), [&](size_t i) {
      const Node& node = nodes[i];
      auto& out = encoded[i];
      if (!node.is_leaf()) {
        if (params_.quantize_centers) {
          auto centers = std::make_shared<TQ4::EncodedSet>(fourbit_model_.encode(node.centers));
          out.center_bytes = centers->num_bytes();
          out.centers = std::move(centers);
        } else out.center_bytes = node.centers.num_bytes();
        return;
      }
      if (params_.pq.method == QT::None) {
        auto raw = gather_point_clouds(points_, node.rows, false);
        raw.ids = node.rows.data();
        out.raw_leaf = raw;
        out.leaf_bytes = raw.num_bytes();
        return;
      }
      // Encoded IDs always address rows, independent of public document IDs.
      const SubsetView subset(points_, node.rows);
      switch (params_.pq.method) {
        case QT::OneBitTQ: {
          auto db = std::make_shared<TQ1::EncodedSet>(onebit_model_.encode(subset));
          out.leaf_bytes = db->num_bytes(); out.leaf = std::move(db); break;
        }
        case QT::TurboQuant: {
          auto db = std::make_shared<TQ4::EncodedSet>(fourbit_model_.encode(subset));
          out.leaf_bytes = db->num_bytes(); out.leaf = std::move(db); break;
        }
        case QT::EightBitTQ: {
          auto db = std::make_shared<TQ8::EncodedSet>(eightbit_model_.encode(subset));
          out.leaf_bytes = db->num_bytes(); out.leaf = std::move(db); break;
        }
        default: break;
      }
    }, 1);
  }

  std::unique_ptr<PreparedQuery> prepare_query(const PointCloud& original,
                                             const SearchParams& params) const override {
    auto query = std::make_unique<Query>();
    auto start = Clock::now();
    query->effective = original;
    if (params.query_compression != SearchParams::QueryCompression::None) {
      const uint32_t alignment = params_.pq.method == QT::OneBitTQ ? 4 :
          (params_.pq.method == QT::None ? 1 : 6);
      query->storage = compress_query<ChPoint>(typed_point(original),
          params.query_compression, params.query_compression_threshold, alignment,
          params.query_alignment, params.query_alignment_strict);
      const auto compressed = query->storage.view();
      query->effective = {compressed.data(), compressed.size(), compressed.get_dims(), original.id};
      validate_values(query->effective.values, query->effective.n, query->effective.d, false, "compressed query");
    }
    query->rerank = params.compress_rerank ? query->effective : original;
    query->compression_seconds = seconds(start);
    start = Clock::now();
    // Centers and TQ4 leaves use the same rotation/codec, so encode only once.
    if (params_.quantize_centers || params_.pq.method == QT::TurboQuant)
      query->fourbit = fourbit_model_.quantize_query(query->effective);
    switch (params_.pq.method) {
      case QT::OneBitTQ: query->onebit = onebit_model_.quantize_query(query->effective); break;
      case QT::TurboQuant: break;
      case QT::EightBitTQ: query->eightbit = eightbit_model_.quantize_query(query->effective); break;
      default: break;
    }
    query->quantization_seconds = seconds(start);
    return query;
  }

  void center_distances(const Node& node, const EncodedNode& encoded,
                        const PreparedQuery& prepared, Neighbor* output) const override {
    const auto& query = static_cast<const Query&>(prepared);
    if (params_.quantize_centers) {
      static_cast<const TQ4::EncodedSet*>(encoded.centers.get())->distances_all(query.fourbit, output);
    } else {
      const auto q = typed_point(query.effective);
      const auto distances = OneToMany<ChPoint, TypedView>::AllDistances(q, TypedView(node.centers));
      for (size_t i = 0; i < node.centers.size(); ++i)
        output[i] = {uint32_t(i), distances[i]};
    }
  }

  bool root_distances(const EncodedNode& root,
      std::span<const std::unique_ptr<PreparedQuery>> queries, Neighbor* output) const override {
    if (!params_.quantize_centers || !root.centers) return false;
    std::vector<const TQ4::EncodedQuery*> pointers(queries.size());
    for (size_t i = 0; i < queries.size(); ++i)
      pointers[i] = &static_cast<const Query&>(*queries[i]).fourbit;
    turboquant_mv::FusedQueryBatch<false> fused;
    fused.Build(pointers);
    turboquant_mv::chamfer_score_all_fused(fused,
        *static_cast<const TQ4::EncodedSet*>(root.centers.get()), output);
    return true;
  }

  void leaf_distances(const EncodedNode& leaf, const PreparedQuery& prepared,
                      Neighbor* output) const override {
    const auto& query = static_cast<const Query&>(prepared);
    switch (params_.pq.method) {
      case QT::None: {
        const TypedView raw(leaf.raw_leaf);
        const auto distances = OneToMany<ChPoint, TypedView>::AllDistances(typed_point(query.effective), raw);
        for (size_t i = 0; i < raw.size(); ++i) output[i] = {raw.get_id(i), distances[i]};
        break;
      }
      case QT::OneBitTQ: static_cast<const TQ1::EncodedSet*>(leaf.leaf.get())->distances_all(query.onebit, output); break;
      case QT::TurboQuant: static_cast<const TQ4::EncodedSet*>(leaf.leaf.get())->distances_all(query.fourbit, output); break;
      case QT::EightBitTQ: static_cast<const TQ8::EncodedSet*>(leaf.leaf.get())->distances_all(query.eightbit, output); break;
      default: break;
    }
  }

  template<class Model, class Batch, class Getter>
  void batch(const EncodedNode& leaf, std::span<const PreparedQuery* const> queries,
             size_t k, Neighbor* output, Getter get) const {
    std::vector<const typename Model::EncodedQuery*> pointers(queries.size());
    for (size_t i = 0; i < queries.size(); ++i) pointers[i] = get(static_cast<const Query&>(*queries[i]));
    const auto& encoded = *static_cast<const typename Model::EncodedSet*>(leaf.leaf.get());
    if constexpr (std::is_same_v<Model, TQ1>) {
      // finish() selects/sorts after all leaf results have been collected.
      Batch::TopKIntoUninitialized(pointers, encoded, uint32_t(k), output, 8, true, false);
    } else {
      Batch::TopKIntoUninitialized(pointers, encoded, uint32_t(k), output, 8, true);
    }
  }

  void leaf_topk(const EncodedNode& leaf, std::span<const PreparedQuery* const> queries,
                 size_t k, Neighbor* output) const override {
    if (params_.pq.method == QT::None && queries.size() >= 64) {
      auto offsets = parlay::sequence<size_t>::uninitialized(queries.size() + 1);
      offsets[0] = 0;
      for (size_t i = 0; i < queries.size(); ++i)
        offsets[i + 1] = offsets[i] + size_t(queries[i]->effective.n) * points_.d;
      auto values = parlay::sequence<float>::uninitialized(offsets.back());
      for (size_t i = 0; i < queries.size(); ++i)
        std::copy_n(queries[i]->effective.values, offsets[i + 1] - offsets[i], values.data() + offsets[i]);
      const PointCloudSetView batch{uint32_t(queries.size()), points_.d, values.data(), offsets.data(), nullptr, {}};
      ManyToMany<TypedView>::TopKIntoUninitialized(TypedView(batch), TypedView(leaf.raw_leaf), uint32_t(k), output);
      // The float M2M primitive returns local positions, unlike encoded
      // scorers. Translate once per result to the index's dense corpus rows.
      for (size_t i = 0; i < queries.size() * k; ++i)
        output[i].first = leaf.raw_leaf.get_id(output[i].first);
      return;
    }
    if (params_.pq.method == QT::OneBitTQ) {
      batch<TQ1, turboquant_1bit_mv::ManyToMany<TQ1::EncodedSet>>(leaf, queries, k, output,
          [](const Query& q) { return &q.onebit; });
      return;
    } else if (params_.pq.method == QT::TurboQuant) {
      batch<TQ4, turboquant_mv::ManyToMany<TQ4::EncodedSet>>(leaf, queries, k, output,
          [](const Query& q) { return &q.fourbit; });
      return;
    } else if (params_.pq.method == QT::EightBitTQ) {
      batch<TQ8, turboquant_8bit_mv::ManyToMany<TQ8::EncodedSet>>(leaf, queries, k, output,
          [](const Query& q) { return &q.eightbit; });
      return;
    }
    const size_t n = leaf.raw_leaf.size();
    parlay::parallel_for(0, queries.size(), [&](size_t i) {
      auto distances = parlay::sequence<Neighbor>::uninitialized(n);
      leaf_distances(leaf, *queries[i], distances.data());
      select_neighbors(distances, std::min(k, n));
      std::copy(distances.begin(), distances.end(), output + i * k);
      for (size_t j = distances.size(); j < k; ++j)
        output[i * k + j] = {UINT32_MAX, std::numeric_limits<float>::max()};
    });
  }

  size_t rerank(const PreparedQuery& prepared, std::span<Neighbor> candidates, bool tq8) const override {
    const auto& query = static_cast<const Query&>(prepared);
    size_t bytes = 0;
    if (tq8) {
      TQ8::EncodedQuery encoded_rerank;
      const auto* encoded = &query.eightbit;
      if (params_.pq.method != QT::EightBitTQ || query.rerank.values != query.effective.values ||
          query.rerank.n != query.effective.n) {
        encoded_rerank = eightbit_model_.quantize_query(query.rerank);
        encoded = &encoded_rerank;
      }
      parlay::parallel_for(0, candidates.size(), [&](size_t i) {
        auto& candidate = candidates[i];
        candidate.second = turboquant_8bit_mv::turboquant_8bit_mv_chamfer_distance(*encoded, rerank_db_, candidate.first);
      });
      for (const auto& candidate : candidates)
        bytes += points_.get_size(candidate.first) * (rerank_db_.num_bytes_per_datapoint + sizeof(float));
    } else {
      const auto q = typed_point(query.rerank);
      parlay::parallel_for(0, candidates.size(), [&](size_t i) {
        auto& candidate = candidates[i];
        candidate.second = q.distance(typed_point(points_[candidate.first]));
      });
      for (const auto& candidate : candidates)
        bytes += (points_.offsets[candidate.first + 1] - points_.offsets[candidate.first]) * sizeof(float);
    }
    return bytes;
  }
};

}  // namespace

#ifndef MVSIC_KERNEL_FACTORY
#define MVSIC_KERNEL_FACTORY make_kernels
#endif
__attribute__((visibility("default")))
std::unique_ptr<Kernels> MVSIC_KERNEL_FACTORY(const IndexParams& params,
                                      const PointCloudSetView& points) {
  return std::make_unique<IPKernels>(params, points);
}

}  // namespace mvsic::mvivf_internal
