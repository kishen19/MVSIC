#pragma once

#include <memory>

#include "mvsic/core/index.h"
#include "mvsic/core/index_params.h"

namespace mvsic {

/* Inner-product Multi-Vector IVF tree. Runtime options select prepared kernels.
 * Searches may run concurrently on a built index, including first-time reranking.
 * Build, load, move and destruction require callers to finish active searches.
 * Corpus arrays must stay unchanged; ownerless views require caller-managed lifetime.
 */
class IndexMVIVF final : public Index {
 public:
  // L2 is available to clustering, but is not a retrieval metric.
  IndexMVIVF(uint32_t d, Metric metric, const IndexParams& params = IndexParams::mvivf());
  ~IndexMVIVF() override;
  IndexMVIVF(IndexMVIVF&&) noexcept;
  IndexMVIVF& operator=(IndexMVIVF&&) noexcept;
  IndexMVIVF(const IndexMVIVF&) = delete;
  IndexMVIVF& operator=(const IndexMVIVF&) = delete;

  // Corpus IDs must be unique; UINT32_MAX is reserved for missing neighbors.
  void build(const PointCloudSetView& points) override;
  Neighbors search(const PointCloud& query, const PointCloudSetView& points,
                   const SearchParams& params) const override;
  Neighbors search(const PointCloud& query, const SearchParams& params) const override;
  SearchResultWithStats search_with_stats(const PointCloud& query,
      const PointCloudSetView& points, const SearchParams& params) const override;
  BatchNeighbors search_all(const PointCloudSetView& queries,
      const PointCloudSetView& points, const SearchParams& params) const override;
  BatchNeighbors search_all(const PointCloudSetView& queries,
                           const SearchParams& params) const override;
  void save(const std::string& filename) const override;
  void load(const std::string& filename, const PointCloudSetView& points) override;
  Metric metric() const noexcept override;
  uint32_t get_dims() const noexcept override;
  size_t get_height() const noexcept;
  size_t mean_cluster_size() const noexcept;
  size_t max_cluster_size() const noexcept;
  std::vector<uint32_t> get_flat_clustering() const;
  const IndexParams& params() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace mvsic
