#pragma once

#include "parlay/primitives.h"

template <typename ChPoint, typename Range>
struct PointCloud {
  using T = typename ChPoint::distance_type;
  using range_type = Range;
  using point_type = ChPoint;

  inline size_t size() const { return n; }
  inline unsigned int get_dims() const { return dims; }
  inline ChPoint operator[](long i) const {
    return ChPoint(ids[i], get_coords(i), get_size(i), dims);
  }

  // Return the number of embeddings for point i.
  size_t get_size(size_t i) const {
    size_t num_coords = offsets[i + 1] - offsets[i];
    return num_coords / dims;
  }

  inline uint32_t get_id(uint32_t i) const {
    if (ids.size() > 0) { return ids[i]; }
    else { return i; }
  }

  T* get_coords(long i) const {
    size_t num_coords = offsets[i + 1] - offsets[i];
    return values + offsets[i];
  }

  template <typename Seq>
  inline auto filter(const Seq& cluster_ids) const {
    size_t num = cluster_ids.size();
    return parlay::delayed_tabulate(num, [&](size_t i) {
      return (*this)[cluster_ids[i]]; });
  }

  template <typename Seq>
  inline auto filter_flattened(const Seq& cluster_ids) const {
    size_t num = cluster_ids.size();
    auto sizes = parlay::delayed_seq<size_t>(num, [&](size_t i) {
      return get_size(cluster_ids[i]);
      });
    auto [csizes, total_size] = parlay::scan(sizes);
    auto starts = parlay::sequence<T*>::uninitialized(total_size);
    parlay::parallel_for(0, num, [&](size_t i) {
      auto ind = cluster_ids[i];
      auto offset = csizes[i];
      auto coords = get_coords(ind);
      parlay::parallel_for(0, sizes[i], [&](size_t j) {
        starts[offset + j] = coords + j * dims;
        });
      });
    return parlay::tabulate(total_size, [&](size_t i) {
      return parlay::make_slice(starts[i], starts[i] + dims);
      });
  }

  PointCloud() {}

  PointCloud(const char* filename) { // Doesn't support mmap at the moment.
    std::cout << "filename = " << filename << std::endl;
    if (filename == nullptr) { return; }
    size_t file_size = 0;
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
      std::cerr << "Error opening file!" << std::endl;
      exit(-1);
    }
    file_size = file.tellg();
    std::ifstream reader(filename);
    assert(reader.is_open());
    // [MMAP] auto [fileptr, length] = mmapStringFromFile(filename);
    // Step 1: Read num points and dimension [uint32_t, uint32_t]
    uint32_t num_points;
    unsigned int d;
    reader.read((char*)(&num_points), sizeof(uint32_t));
    n = num_points;
    reader.read((char*)(&d), sizeof(uint32_t));
    dims = d;
    std::cout << "Detected " << num_points
      << " points with embedding dimension " << d << std::endl;
    aligned_dims = dim_round_up(dims, sizeof(T));
    std::cout << "Aligned dims = " << aligned_dims << std::endl;
    if (aligned_dims != dims) {
      std::cerr << "Expected dims to be a multiple of cacheline size."
        << std::endl;
      exit(-1);
    }
    // [MMAP] char* ptr = fileptr + length - (static_cast<int64_t>(n) * sizeof(uint32_t));
    // Step 2: Skip to the last N entries and read perm [uint32_t]
    // Document with id i is stored at perm[i].
    reader.seekg(-static_cast<int64_t>(n) * sizeof(uint32_t), std::ios::end);
    auto perm = parlay::sequence<uint32_t>::uninitialized(n);
    ids = parlay::sequence<uint32_t>::uninitialized(n);
    //[MMAP] std::memcpy(perm.begin(), ptr, n * sizeof(uint32_t));
    reader.read((char*)perm.begin(), n * sizeof(uint32_t));
    parlay::parallel_for(0, n, [&](size_t i) {
      ids[perm[i]] = i;
      });
    // Step 3: Skip to the offsets [size_t]
    reader.seekg(-1LL * (static_cast<int64_t>(n) * sizeof(uint32_t) +
      static_cast<int64_t>(n + 1) * sizeof(size_t)),
      std::ios::end);
    // [MMAP] ptr = fileptr + length - (static_cast<int64_t>(n) * sizeof(uint32_t)) -
          // (static_cast<int64_t>(n + 1) * sizeof(size_t));
    offsets = parlay::sequence<size_t>(n + 1);
    // [MMAP] std::memcpy(offsets.begin(), ptr, (n + 1) * sizeof(size_t));
    reader.read((char*)offsets.begin(), (n + 1) * sizeof(size_t));
    // Step 4: Read the coordinates [T]
    size_t coordinate_size =
      file_size - (2 * sizeof(uint32_t) + ((n) * sizeof(uint32_t)) +
        ((n + 1) * sizeof(size_t)));
    values = static_cast<T*>(std::malloc(coordinate_size)); // TODO: p_malloc
    reader.seekg(static_cast<int64_t>(2) * sizeof(uint32_t), std::ios::beg);
    reader.read((char*)values, coordinate_size);
    // [MMAP] values = reinterpret_cast<T*>(fileptr + 2 * sizeof(uint32_t));
    reader.close();
  }

  // Sequence of Chamfer Points
  template <typename Seq>
  PointCloud(const Seq& data, unsigned int _d)
    : dims(_d), aligned_dims(dim_round_up(dims, sizeof(T))), n(data.size()) {
    offsets = parlay::sequence<size_t>::from_function(n + 1, [&](size_t i) {
      return (i == 0) ? 0 : (data[i - 1].size() * dims);
      });
    parlay::scan_inclusive_inplace(offsets);
    size_t total_coords = offsets[n];
    values = static_cast<T*>(std::malloc(total_coords * sizeof(T))); // TODO: p_malloc
    parlay::parallel_for(0, n, [&](size_t i) {
      size_t offset = offsets[i];
      parlay::parallel_for(0, data[i].size(), [&](size_t j) {
        for (unsigned int t = 0; t < dims; ++t) {
          values[offset + j * dims + t] = data[i][j][t];
        }
        });
      });
    ids = parlay::sequence<uint32_t>::from_function(n, [&](size_t i) {
      return data[i].get_id();
      });
  }

  // Arbitrary random-access range
  template <typename Seq>
  PointCloud(const Seq& data, unsigned int _d, parlay::sequence<uint32_t> _ids)
    : dims(_d), aligned_dims(dim_round_up(dims, sizeof(T))), n(data.size()) {
    offsets = parlay::sequence<size_t>::from_function(n + 1, [&](size_t i) {
      return (i == 0) ? 0 : (data[i - 1].size() * dims);
      });
    parlay::scan_inclusive_inplace(offsets);
    size_t total_coords = offsets[n];
    values = static_cast<T*>(std::malloc(total_coords * sizeof(T))); // TODO: p_malloc
    parlay::parallel_for(0, n, [&](size_t i) {
      size_t offset = offsets[i];
      parlay::parallel_for(0, data[i].size(), [&](size_t j) {
        for (unsigned int t = 0; t < dims; ++t) {
          values[offset + j * dims + t] = data[i][j][t];
        }
        });
      });
    ids = parlay::sequence<uint32_t>::from_function(n, [&](size_t i) {
      return (_ids.size() > 0) ? _ids[i] : (static_cast<uint32_t>(i));
      });
  }

  PointCloud& operator=(const PointCloud& other) {
    if (this != &other) {
      n = other.n;
      dims = other.dims;
      aligned_dims = other.aligned_dims;
      offsets = other.offsets;
      ids = other.ids;
      if (values != nullptr) {
        free(values); // TODO: p_free
        values = nullptr;
      }
      if (other.values) {
        size_t total_coords = offsets[n];
        size_t coordinate_size = total_coords * sizeof(T);
        values = static_cast<T*>(std::malloc(coordinate_size)); // TODO: p_malloc
        std::memcpy(values, other.values, coordinate_size);
      }
    }
    return *this;
  }

  // Copy constructor
  PointCloud(const PointCloud& other) {
    n = other.n;
    dims = other.dims;
    aligned_dims = other.aligned_dims;
    offsets = other.offsets;
    ids = other.ids;
    if (values != nullptr) {
      free(values); // TODO: p_free
      values = nullptr;
    }
    if (other.values) {
      size_t total_coords = offsets[n];
      size_t coordinate_size = total_coords * sizeof(T);
      values = static_cast<T*>(std::malloc(coordinate_size)); // TODO: p_malloc
      std::memcpy(values, other.values, coordinate_size);
    }
  }

  ~PointCloud() {
    if (values != nullptr) {
      free(values); // TODO: p_free
      values = nullptr;
    }
  }

  T* values = nullptr;
  parlay::sequence<size_t> offsets;
  parlay::sequence<uint32_t> ids;
  unsigned int dims = 0;
  unsigned int aligned_dims = 0;
  size_t n = 0;
};
