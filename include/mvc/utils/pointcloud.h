#pragma once

// Reads in a multi-embedding, stored in CSR. The format assumes the num_points
// many offsets are the last num_points many size_t's in the read data.
template <typename ChPoint, typename Range>
struct PointCloud {
  using T = typename ChPoint::distance_type;
  using range_type = Range;
  using point_type = ChPoint;

  PointCloud() {}

  PointCloud(const char* filename) { // Doesn't support mmap at the moment.
    std::cout << "filename = " << filename << std::endl;
    if (filename == nullptr) {
      n = 0;
      dims = 0;
      return;
    }
    size_t file_size = 0;
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
      std::cerr << "Error opening file!" << std::endl;
      exit(-1);
    }
    file_size = file.tellg();
    std::cout << "File size: " << file_size << " bytes" << std::endl;
    std::ifstream reader(filename);
    assert(reader.is_open());
    // [MMAP] auto [fileptr, length] = mmapStringFromFile(filename);
    // Read num points and dimension
    uint32_t num_points;
    uint32_t d;
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
    // Skip to the last N entries and read the last N size_t values
    reader.seekg(-static_cast<int64_t>(n) * sizeof(uint32_t), std::ios::end);
    perm = parlay::sequence<uint32_t>(n);
    //[MMAP] std::memcpy(perm.begin(), ptr, n * sizeof(uint32_t));
    reader.read((char*)perm.begin(), n * sizeof(uint32_t));
    ids = parlay::sequence<uint32_t>::from_function(n, [&](size_t i) {
      return perm[i];
    });
    // Skip to the offsets
    reader.seekg(-1LL * (static_cast<int64_t>(n) * sizeof(uint32_t) +
                         static_cast<int64_t>(n + 1) * sizeof(size_t)),
                 std::ios::end);
    // [MMAP] ptr = fileptr + length - (static_cast<int64_t>(n) * sizeof(uint32_t)) -
          // (static_cast<int64_t>(n + 1) * sizeof(size_t));
    offsets = parlay::sequence<size_t>(n + 1);
    // [MMAP] std::memcpy(offsets.begin(), ptr, (n + 1) * sizeof(size_t));
    reader.read((char*)offsets.begin(), (n + 1) * sizeof(size_t));
    size_t coordinate_size =
        file_size - (2 * sizeof(uint32_t) + ((n) * sizeof(uint32_t)) +
                     ((n + 1) * sizeof(size_t)));
    std::cout << "Coordinate size = " << coordinate_size << std::endl;
    values = static_cast<T*>(malloc(coordinate_size));
    reader.seekg(static_cast<int64_t>(2) * sizeof(uint32_t), std::ios::beg);
    reader.read((char*)values, coordinate_size);
    // [MMAP] values = reinterpret_cast<T*>(fileptr + 2 * sizeof(uint32_t));
    reader.close();
  }

  template <typename Seq>
  PointCloud(const Seq& data, unsigned _d, parlay::sequence<uint32_t> _ids = {})
      : dims(_d),
        aligned_dims(dim_round_up(dims, sizeof(T))),
        n(data.size()) {
    if (_ids.size() > 0){ ids = _ids; }
    offsets = parlay::sequence<size_t>::from_function(n + 1, [&](size_t i) {
      return (i == 0) ? 0 : (data[i-1].size() * dims);
    });
    parlay::scan_inclusive_inplace(offsets);
    size_t total_coords = offsets[n];
    values = static_cast<T*>(malloc(total_coords * sizeof(T)));
    parlay::parallel_for(0, n, [&](size_t i) {
      size_t offset = offsets[i];
      parlay::parallel_for(0, data[i].size(), [&](size_t j) {
        for (unsigned int k = 0; k < dims; ++k) {
          values[offset + j * dims + k] = data[i][j][k];
        }
      });
    });
    perm = parlay::sequence<uint32_t>::from_function(n, [&](size_t i) {
      return static_cast<uint32_t>(i);
    });
  }

  size_t size() const { return n; }
  long dimension() const { return dims; }

  // Return the number of embeddings for point i.
  size_t NumEmb(size_t i) const {
    auto p_i = perm[i];
    size_t num_coords = offsets[p_i + 1] - offsets[p_i];
    return num_coords / dims;
  }

  auto Coords(long i) const {
    auto p_i = perm[i];
    size_t num_coords = offsets[p_i + 1] - offsets[p_i];
    // std::cout << "num_coords = " << num_coords << std::endl;
    return parlay::make_slice(values + offsets[p_i], 
      values + offsets[p_i] + num_coords);
  }

  auto Coords2(long i) const {
    auto p_i = perm[i];
    size_t num_coords = offsets[p_i + 1] - offsets[p_i];
    // std::cout << "num_coords = " << num_coords << std::endl;
    return values + offsets[p_i];
  }

  template <typename Seq>
  Range GetRange(const Seq& cluster_ids) const{
    size_t k = cluster_ids.size();
    auto num_emb = parlay::delayed_seq<size_t>(k, [&](size_t i) {
      return NumEmb(cluster_ids[i]);
    });
    auto [offsets, total_embs] = parlay::scan(num_emb);
    auto data = parlay::sequence<parlay::sequence<T>>(total_embs);
    parlay::parallel_for(0, k, [&](size_t i) {
      auto ind = cluster_ids[i];
      auto offset = offsets[i];
      auto coords = Coords(ind);
      parlay::parallel_for(0, coords.size()/dims, [&](size_t j) {
        data[offset + j] = parlay::sequence<T>::uninitialized(dims);
        for (unsigned int t = 0; t < dims; ++t) {
          data[offset + j][t] = coords[j * dims + t];
        }
      });
    });
    return Range(data, dims);
  }

  template <typename Seq>
  PointCloud GetPointCloud(const Seq& cluster_ids) const{
    size_t k = cluster_ids.size();
    auto points = parlay::delayed_seq<ChPoint>(k, [&](size_t i) {
      return (*this)[cluster_ids[i]];
    });
    auto ids_ = parlay::sequence<uint32_t>::from_function(k, [&](size_t i) {
      return get_id(cluster_ids[i]);
    });
    return PointCloud<ChPoint, Range>(points, dims, ids_);
  }

  ChPoint operator[](long i) const { return ChPoint(i, Coords2(i), NumEmb(i), dims); }

  PointCloud& operator=(const PointCloud& other) {
    if (this != &other) {
      if (values != nullptr) {
        free(values);
        values = nullptr;
      }
      n = other.n;
      dims = other.dims;
      aligned_dims = other.aligned_dims;
      offsets = other.offsets;
      perm = other.perm;
      ids = other.ids;
      size_t total_coords = offsets[n];
      values = static_cast<T*>(malloc(total_coords * sizeof(T)));
      std::memcpy(values, other.values, total_coords * sizeof(T));
    }
    return *this;
  }

  // Copy constructor
  PointCloud(PointCloud& other) {
    n = other.n;
    dims = other.dims;
    aligned_dims = other.aligned_dims;
    offsets = other.offsets;
    perm = other.perm;
    ids = other.ids;
    size_t total_coords = offsets[n];
    values = static_cast<T*>(malloc(total_coords * sizeof(T)));
    std::memcpy(values, other.values, total_coords * sizeof(T));
  }

  ~PointCloud() {
    if (values != nullptr) {
      // std::cout << "Freeing values of PointCloud with " << n 
      //           << " points and " << dims << " dimensions." << std::endl;
      free(values);
      values = nullptr;
    }
  }

  uint32_t get_id(uint32_t i) const {
    if (ids.size() > 0) {
      return ids[i];
    } else {
      return i;
    }
  }

  auto get_ids() const {
    return ids;
  }

  T* values = nullptr;
  parlay::sequence<size_t> offsets;
  parlay::sequence<uint32_t> perm;
  parlay::sequence<uint32_t> ids;
  unsigned int dims = 0;
  unsigned int aligned_dims = 0;
  size_t n = 0;
};
