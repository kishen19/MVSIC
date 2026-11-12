#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace mvsic {

/* Non-owning point cloud. Its source storage must outlive the query call. */
struct PointCloud {
  const float* values = nullptr;
  uint32_t n = 0;
  uint32_t d = 0;
  uint32_t id = 0;

  uint32_t size() const noexcept { return n; }
  uint32_t get_dims() const noexcept { return d; }
  uint32_t get_id() const noexcept { return id; }
  const float* data() const noexcept { return values; }
  const float* data(size_t i) const noexcept { return values + i * d; }
};

/* Set of point clouds. Offsets count float coordinates, as in the PCS format.
 * The optional owner retains all values, offsets, and IDs used by the view.
 * An index retains this owner when build() or load() binds its input corpus.
 * With no owner, the caller must keep the storage alive until the index is
 * destroyed or rebound. Bound buffers must remain unchanged in either case.
 */
struct PointCloudSetView {
  uint32_t n = 0;
  uint32_t d = 0;
  const float* values = nullptr;
  const size_t* offsets = nullptr;
  const uint32_t* ids = nullptr;
  std::shared_ptr<const void> owner;

  size_t size() const noexcept { return n; }
  uint32_t get_dims() const noexcept { return d; }
  size_t total_size() const noexcept { return n ? offsets[n] / d : 0; }
  float average_size() const noexcept { return n ? float(total_size()) / n : 0; }
  size_t num_bytes() const noexcept { return n ? offsets[n] * sizeof(float) : 0; }
  uint32_t get_size(size_t i) const noexcept { return (offsets[i + 1] - offsets[i]) / d; }
  uint32_t get_id(size_t i) const noexcept { return ids ? ids[i] : uint32_t(i); }
  const float* data() const noexcept { return values; }
  const float* data(size_t i) const noexcept { return values + offsets[i]; }
  std::span<const size_t> get_offsets() const noexcept { return {offsets, size_t(n) + 1}; }
  std::span<const uint32_t> get_ids() const noexcept { return ids ? std::span<const uint32_t>(ids, n) : std::span<const uint32_t>(); }
  PointCloud operator[](size_t i) const noexcept { return {data(i), get_size(i), d, get_id(i)}; }

  void validate(bool require_nonempty = true) const;
};

PointCloudSetView load_point_clouds(const std::string& filename, bool is_mmap = true);
PointCloudSetView own_point_clouds(std::vector<float> values, std::vector<size_t> offsets,
                                 uint32_t d, std::vector<uint32_t> ids = {});
PointCloudSetView gather_point_clouds(const PointCloudSetView& points,
                                    std::span<const uint32_t> rows, bool preserve_ids = true);

}  // namespace mvsic
