#pragma once

#include "chamferpoint.h"

// Reads in a multi-embedding, stored in CSR. The format assumes the num_points
// many offsets are the last num_points many size_t's in the read data.
template <typename T, typename Range>
struct PointCloud {
  long dimension() { return dims; }

  PointCloud() : initialized(false) {}

  PointCloud(const char* filename) {
    initialized = true;
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

    auto [fileptr, length] = mmapStringFromFile(filename);

    // Read num points and dimension
    uint32_t num_points;  // Number of points with multi-embeddings.
    uint32_t d;           // Dimensionality per-embedding, e.g., 128.
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

    char* ptr = fileptr + length - (static_cast<int64_t>(n) * sizeof(uint32_t));
    // Skip to the last N entries and read the last N size_t values
    // reader.seekg(-static_cast<int64_t>(n) * sizeof(uint32_t), std::ios::end);
    perm = parlay::sequence<uint32_t>(n);
    std::memcpy(perm.begin(), ptr, n * sizeof(uint32_t));

    // Skip to the offsets
    reader.seekg(-1LL * (static_cast<int64_t>(n) * sizeof(uint32_t) +
                         static_cast<int64_t>(n + 1) * sizeof(size_t)),
                 std::ios::end);
    ptr = fileptr + length - (static_cast<int64_t>(n) * sizeof(uint32_t)) -
          (static_cast<int64_t>(n + 1) * sizeof(size_t));
    offsets = parlay::sequence<size_t>(n + 1);
    std::memcpy(offsets.begin(), ptr, (n + 1) * sizeof(size_t));

    size_t coordinate_size =
        file_size - (2 * sizeof(uint32_t) + ((n) * sizeof(uint32_t)) +
                     ((n + 1) * sizeof(size_t)));
    std::cout << "Coordinate size = " << coordinate_size << std::endl;
    values = reinterpret_cast<T*>(fileptr + 2 * sizeof(uint32_t));
    auto del_seq = parlay::delayed_seq<size_t>(
        coordinate_size, [&](size_t i) { return ((uint8_t*)(values))[i]; });
    std::cout << parlay::reduce(del_seq) << std::endl;
    reader.close();
  }

  template <typename Seq>
  PointCloud(const Seq& data, unsigned _d)
      : initialized(true),
        dims(_d),
        aligned_dims(dim_round_up(dims, sizeof(T))),
        n(data.size()) {
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

  size_t size() { return n; }

  // Return the number of embeddings for point i in the multi-embedding.
  size_t num_embeddings(size_t i) {
    auto p_i = perm[i];
    size_t num_coords = offsets[p_i + 1] - offsets[p_i];
    return num_coords / dims;
  }

  auto Coords(long i) {
    auto p_i = perm[i];
    size_t num_coords = offsets[p_i + 1] - offsets[p_i];
    // std::cout << "num_coords = " << num_coords << std::endl;
    return parlay::make_slice(values + offsets[p_i], 
      values + offsets[p_i] + num_coords);
  }

  auto Coords2(long i) {
    auto p_i = perm[i];
    size_t num_coords = offsets[p_i + 1] - offsets[p_i];
    // std::cout << "num_coords = " << num_coords << std::endl;
    return values + offsets[p_i];
  }

  template <typename Seq>
  Range GetCluster(const Seq& cluster_ids) {
    size_t k = cluster_ids.size();
    auto num_emb = parlay::delayed_seq<size_t>(k, [&](size_t i) {
      return num_embeddings(cluster_ids[i]);
    });
    auto [offsets, total_embs] = parlay::scan(num_emb);
    auto data = parlay::sequence<parlay::sequence<T>>::uninitialized(total_embs);
    parlay::parallel_for(0, k, [&](size_t i) {
      auto ind = cluster_ids[i];
      auto offset = offsets[i];
      auto coords = Coords(ind);
      parlay::parallel_for(0, coords.size(), [&](size_t j) { 
        for (unsigned int t = 0; t < dims; ++t) {
          data[offset + j].push_back(coords[j * dims + t]);
        }
      });
    });
    return Range(data, dims);
  }

  ChamferPoint<T> operator[](long i) { return ChamferPoint<T>(i, Coords2(i), num_embeddings(i), dims); }

  ~PointCloud() {
    std::cout << "Freeing... " << values << " initialized = " << initialized
              << std::endl;
    if (initialized && values != nullptr) {
      //      free(values);
      //      values = nullptr;
      //      std::cout << "Done freeing" << std::endl;
      //      initialized = false;
    }
  }

private:
  T* values = nullptr;
  bool initialized = false;  // false by default
  parlay::sequence<size_t> offsets;
  parlay::sequence<uint32_t> perm;
  unsigned int dims = 0;
  unsigned int aligned_dims = 0;
  size_t n = 0;
};
