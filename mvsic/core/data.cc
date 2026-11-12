#include "mvsic/core/data.h"

#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace mvsic {
namespace {

struct OwnedPoints {
  std::vector<float> values;
  std::vector<size_t> offsets;
  std::vector<uint32_t> ids;
};

struct MappedPoints {
  void* address = MAP_FAILED;
  size_t length = 0;
  std::vector<size_t> offsets;
  ~MappedPoints() { if (address != MAP_FAILED) munmap(address, length); }
};

size_t checked_product(size_t a, size_t b) {
  if (b && a > std::numeric_limits<size_t>::max() / b)
    throw std::runtime_error("PCS size overflow");
  return a * b;
}

}  // namespace

void PointCloudSetView::validate(bool require_nonempty) const {
  if (!d) throw std::invalid_argument("Point cloud dimension must be positive");
  if (require_nonempty && !n) throw std::invalid_argument("Point cloud set is empty");
  if (!offsets || offsets[0] != 0)
    throw std::invalid_argument("Point cloud offsets must start at zero");
  for (size_t i = 0; i < n; ++i) {
    if (offsets[i + 1] < offsets[i] || (offsets[i + 1] - offsets[i]) % d)
      throw std::invalid_argument("Point cloud offsets must be ordered multiples of dimension");
    if ((offsets[i + 1] - offsets[i]) / d > std::numeric_limits<uint32_t>::max())
      throw std::invalid_argument("Point cloud has too many vectors");
    if (require_nonempty && offsets[i + 1] == offsets[i])
      throw std::invalid_argument("Point clouds must contain at least one vector");
  }
  if (offsets[n] && !values) throw std::invalid_argument("Point cloud values are missing");
}

PointCloudSetView own_point_clouds(std::vector<float> values, std::vector<size_t> offsets,
                                 uint32_t d, std::vector<uint32_t> ids) {
  if (offsets.empty() || offsets.size() - 1 > std::numeric_limits<uint32_t>::max())
    throw std::invalid_argument("Invalid point cloud count");
  if (offsets.back() != values.size() || (!ids.empty() && ids.size() + 1 != offsets.size()))
    throw std::invalid_argument("Point cloud storage sizes do not match offsets");
  auto storage = std::make_shared<OwnedPoints>(OwnedPoints{
      std::move(values), std::move(offsets), std::move(ids)});
  PointCloudSetView result{uint32_t(storage->offsets.size() - 1), d, storage->values.data(),
                          storage->offsets.data(), storage->ids.empty() ? nullptr : storage->ids.data(),
                          storage};
  result.validate(false);
  return result;
}

PointCloudSetView gather_point_clouds(const PointCloudSetView& points,
                                    std::span<const uint32_t> rows, bool preserve_ids) {
  if (rows.size() > std::numeric_limits<uint32_t>::max())
    throw std::length_error("Too many gathered point clouds");
  std::vector<size_t> offsets(rows.size() + 1, 0);
  std::vector<uint32_t> ids;
  if (preserve_ids) ids.resize(rows.size());
  for (size_t i = 0; i < rows.size(); ++i) {
    if (rows[i] >= points.size()) throw std::out_of_range("Point cloud row is out of range");
    const size_t count = points.offsets[rows[i] + 1] - points.offsets[rows[i]];
    if (count > std::numeric_limits<size_t>::max() - offsets[i])
      throw std::length_error("Point cloud gather is too large");
    offsets[i + 1] = offsets[i] + count;
    if (preserve_ids) ids[i] = points.get_id(rows[i]);
  }
  // Construct floats directly from each row. resize() would first zero the
  // entire gathered corpus; reserve() alone does not create writable elements.
  std::vector<float> values;
  values.reserve(offsets.back());
  for (size_t i = 0; i < rows.size(); ++i) {
    const float* begin = points.data(rows[i]);
    values.insert(values.end(), begin, begin + (offsets[i + 1] - offsets[i]));
  }
  return own_point_clouds(std::move(values), std::move(offsets), points.d, std::move(ids));
}

PointCloudSetView load_point_clouds(const std::string& filename, bool is_mmap) {
  const int fd = open(filename.c_str(), O_RDONLY);
  if (fd < 0) throw std::runtime_error("Cannot open PCS file: " + filename);
  struct stat st {};
  if (fstat(fd, &st) || st.st_size < 4 * sizeof(size_t)) {
    close(fd);
    throw std::runtime_error("Invalid PCS file: " + filename);
  }
  auto storage = std::make_shared<MappedPoints>();
  storage->length = size_t(st.st_size);
  storage->address = mmap(nullptr, storage->length, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (storage->address == MAP_FAILED) throw std::runtime_error("Cannot map PCS file: " + filename);
  const char* bytes = static_cast<const char*>(storage->address);
  size_t cursor = 0;
  auto read_size = [&]() {
    if (cursor > storage->length - sizeof(size_t)) throw std::runtime_error("Truncated PCS header");
    size_t value;
    std::memcpy(&value, bytes + cursor, sizeof(value));
    cursor += sizeof(value);
    return value;
  };
  const size_t d = read_size(), n = read_size(), vectors = read_size();
  if (!d || d > UINT32_MAX || n > UINT32_MAX) throw std::runtime_error("Invalid PCS dimensions");
  const size_t coordinate_bytes = checked_product(checked_product(vectors, d), sizeof(float));
  if (coordinate_bytes > storage->length - cursor) throw std::runtime_error("Truncated PCS values");
  const float* values = reinterpret_cast<const float*>(bytes + cursor);
  cursor += coordinate_bytes;
  const size_t count = read_size();
  if (count != n + 1 || checked_product(count, sizeof(size_t)) != storage->length - cursor)
    throw std::runtime_error("Invalid PCS offset storage");
  // Offsets can start at an unaligned address after an odd float count.
  storage->offsets.resize(count);
  std::memcpy(storage->offsets.data(), bytes + cursor, count * sizeof(size_t));
  PointCloudSetView result{uint32_t(n), uint32_t(d), values, storage->offsets.data(), nullptr, storage};
  result.validate(false);
  if (result.offsets[n] != vectors * d) throw std::runtime_error("PCS vector count disagrees with offsets");
  if (is_mmap) return result;
  return own_point_clouds(std::vector<float>(values, values + vectors * d),
                          storage->offsets, uint32_t(d));
}

}  // namespace mvsic
