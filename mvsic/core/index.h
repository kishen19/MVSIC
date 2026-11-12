#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "mvsic/core/data.h"
#include "mvsic/core/search_params.h"
#include "mvsic/core/types.h"

namespace mvsic {

using Neighbor = std::pair<uint32_t, float>;
using Neighbors = std::vector<Neighbor>;
using BatchNeighbors = std::vector<Neighbors>;
using SearchResultWithStats = std::tuple<Neighbors, size_t, std::vector<double>>;

/* Common index interface. Implementations and numerical kernels are compiled. */
class Index {
 public:
  virtual ~Index();
  virtual void build(const PointCloudSetView& points) = 0;
  virtual Neighbors search(const PointCloud& query, const PointCloudSetView& points,
                           const SearchParams& params) const = 0;
  virtual Neighbors search(const PointCloud& query, const SearchParams& params) const = 0;
  virtual SearchResultWithStats search_with_stats(const PointCloud& query,
      const PointCloudSetView& points, const SearchParams& params) const = 0;
  virtual BatchNeighbors search_all(const PointCloudSetView& queries,
      const PointCloudSetView& points, const SearchParams& params) const = 0;
  virtual BatchNeighbors search_all(const PointCloudSetView& queries,
                                    const SearchParams& params) const = 0;
  virtual void save(const std::string& filename) const = 0;
  virtual void load(const std::string& filename, const PointCloudSetView& points) = 0;
  virtual Metric metric() const noexcept = 0;
  virtual uint32_t get_dims() const noexcept = 0;
};

}  // namespace mvsic
