#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>
#include <istream>
#include <vector>

#include "algorithms/utils/point_range.h"
#include "algorithms/utils/graph.h" // Added include for parlayANN::Graph

namespace parlayANN {
namespace io {

// --------- internal helpers ---------
inline void write_u32(std::ostream& out, unsigned int v) {
  out.write(reinterpret_cast<const char*>(&v), sizeof(unsigned int));
  if (!out) throw std::runtime_error("write_u32: failed to write 4-byte header");
}

inline unsigned int read_u32(std::istream& in) {
  unsigned int v;
  in.read(reinterpret_cast<char*>(&v), sizeof(unsigned int));
  if (!in) throw std::runtime_error("read_u32: failed to read 4-byte header");
  return v;
}

inline void write_string(std::ostream& out, const std::string& s) {
  write_u32(out, s.length());
  out.write(s.data(), s.length());
  if (!out) throw std::runtime_error("write_string: failed to write string");
}

inline std::string read_string(std::istream& in) {
  unsigned int len = read_u32(in);
  std::string s(len, '\0');
  in.read(s.data(), len);
  if (!in) throw std::runtime_error("read_string: failed to read string");
  return s;
}

// Returns a reasonable block size (in number of points) that keeps scratch <= ~64MB by default.
// You can tweak the cap if needed.
inline std::size_t choose_block_points(std::size_t bytes_per_point,
                                       std::size_t cap_bytes = 64ull * 1024ull * 1024ull) {
  if (bytes_per_point == 0) return 0;
  std::size_t m = cap_bytes / bytes_per_point;
  if (m == 0) m = 1;
  return m;
}

// --------- ostream overload (core implementation) ---------
template<class Point_>
void save_point_range(const PointRange<Point_>& pr, std::ostream& out) {
  using byte = uint8_t;

  const unsigned int n = static_cast<unsigned int>(pr.size());
  const unsigned int dims = static_cast<unsigned int>(pr.params.dims);
  const int num_bytes_i = pr.params.num_bytes();  // bytes per *unaligned* record on disk
  if (num_bytes_i <= 0) throw std::runtime_error("save_point_range: invalid num_bytes()");
  const std::size_t num_bytes = static_cast<std::size_t>(num_bytes_i);

  // Header: [uint32 n][uint32 dims]
  write_u32(out, n);
  write_u32(out, dims);

  if (n == 0) return;  // nothing more to write

  // Stream payload in blocks. Each record is exactly num_bytes (no alignment padding).
  const std::size_t BLOCK_POINTS = choose_block_points(num_bytes);  // ~64MB scratch by default
  std::size_t blk_pts = BLOCK_POINTS ? BLOCK_POINTS : 1;

  std::unique_ptr<byte[]> scratch(new byte[num_bytes * blk_pts]);

  std::size_t idx = 0;
  while (idx < n) {
    const std::size_t floor = idx;
    const std::size_t ceiling = std::min(floor + blk_pts, static_cast<std::size_t>(n));
    const std::size_t m = ceiling - floor;

    // Gather exactly num_bytes per point from in-memory aligned storage.
    for (std::size_t i = 0; i < m; ++i) {
      const byte* src = pr.location(static_cast<long>(floor + i));
      std::memcpy(scratch.get() + i * num_bytes, src, num_bytes);
    }

    out.write(reinterpret_cast<const char*>(scratch.get()),
              static_cast<std::streamsize>(m * num_bytes));
    if (!out) throw std::runtime_error("save_point_range: write failed during payload");

    idx = ceiling;
  }
}

// --------- filename overload (convenience wrapper) ---------
template<class Point_>
void save_point_range(const PointRange<Point_>& pr, const std::string& filename) {
  std::ofstream out(filename, std::ios::binary);
  if (!out) throw std::runtime_error("save_point_range: cannot open file: " + filename);
  save_point_range(pr, static_cast<std::ostream&>(out));
  // out dtor will close; no need to manually check again here
}

// --------- ostream overload (core implementation) ---------
template<typename indexType>
void save_graph(const parlayANN::Graph<indexType>& G, std::ostream& out) {
  const unsigned int n = static_cast<unsigned int>(G.size());
  const unsigned int max_deg = static_cast<unsigned int>(G.max_degree());

  // Header: [uint32 n][uint32 max_deg]
  write_u32(out, n);
  write_u32(out, max_deg);

  if (n == 0) return; // nothing more to write

  // Write degrees of each node
  parlay::sequence<indexType> sizes =
      parlay::tabulate(n, [&](size_t i) { return static_cast<indexType>(G[i].size()); });
  out.write(reinterpret_cast<const char*>(sizes.begin()), sizes.size() * sizeof(indexType));
  if (!out) throw std::runtime_error("save_graph: failed to write sizes");

  // Write edge data in blocks
  size_t BLOCK_SIZE = 1000000; // Same block size as in Graph::save
  size_t index = 0;
  while(index < n){
    size_t floor = index;
    size_t ceiling = index + BLOCK_SIZE <= n ? index + BLOCK_SIZE : n;
    auto edge_data = parlay::tabulate(ceiling - floor, [&] (size_t i){
      return parlay::tabulate(G[i + floor].size(), [&] (size_t j){ return G[i + floor][j];});
    });
    parlay::sequence<indexType> data = parlay::flatten(edge_data);
    out.write(reinterpret_cast<const char*>(data.begin()), data.size() * sizeof(indexType));
    if (!out) throw std::runtime_error("save_graph: write failed during payload");
    index = ceiling;
  }
}

// --------- filename overload (convenience wrapper) ---------
template<typename indexType>
void save_graph(const parlayANN::Graph<indexType>& G, const std::string& filename) {
  std::ofstream out(filename, std::ios::binary);
  if (!out) throw std::runtime_error("save_graph: cannot open file: " + filename);
  save_graph(G, static_cast<std::ostream&>(out));
}

// --------- istream overload (core implementation) ---------
template<typename indexType>
parlayANN::Graph<indexType> load_graph(std::istream& in) {
  unsigned int n_u32 = read_u32(in);
  unsigned int max_deg_u32 = read_u32(in);

  size_t n = static_cast<size_t>(n_u32);
  long max_deg = static_cast<long>(max_deg_u32);

  parlayANN::Graph<indexType> G(max_deg, n);

  if (n == 0) return G; // nothing more to read

  // Read degrees of each node
  parlay::sequence<indexType> sizes(n);
  in.read(reinterpret_cast<char*>(sizes.begin()), sizes.size() * sizeof(indexType));
  if (!in) throw std::runtime_error("load_graph: failed to read sizes");

  // Read edge data in blocks
  size_t BLOCK_SIZE = 1000000; // Same block size as in Graph::save
  size_t index = 0;
  while(index < n){
    size_t floor = index;
    size_t ceiling = index + BLOCK_SIZE <= n ? index + BLOCK_SIZE : n;
    size_t total_edges_in_block = 0;
    for (size_t i = floor; i < ceiling; ++i) {
        total_edges_in_block += sizes[i];
    }

    parlay::sequence<indexType> data(total_edges_in_block);
    in.read(reinterpret_cast<char*>(data.begin()), data.size() * sizeof(indexType));
    if (!in) throw std::runtime_error("load_graph: failed to read edge data");

    size_t current_data_offset = 0;
    for (size_t i = floor; i < ceiling; ++i) {
        parlay::sequence<indexType> node_edges(sizes[i]);
        for (size_t j = 0; j < sizes[i]; ++j) {
            node_edges[j] = data[current_data_offset + j];
        }
        G[i].update_neighbors(node_edges);
        current_data_offset += sizes[i];
    }
    index = ceiling;
  }
  return G;
}

// --------- filename overload (convenience wrapper) ---------
template<typename indexType>
parlayANN::Graph<indexType> load_graph(const std::string& filename) {
  std::ifstream in(filename, std::ios::binary);
  if (!in) throw std::runtime_error("load_graph: cannot open file: " + filename);
  return load_graph<indexType>(static_cast<std::istream&>(in));
}

// Helper to load point_range data into a sequence of vectors
template<class Point_>
std::pair<parlay::sequence<std::vector<float>>, int> read_point_range(std::istream& in) {
  unsigned int n_u32 = read_u32(in);
  unsigned int dims_u32 = read_u32(in);

  size_t n = static_cast<size_t>(n_u32);
  int dims = static_cast<int>(dims_u32);

  if (n == 0) return {parlay::sequence<std::vector<float>>(), dims}; // nothing more to read

  parlay::sequence<std::vector<float>> fdes(n);
  for (size_t i = 0; i < n; ++i) {
    fdes[i].resize(dims);
    in.read(reinterpret_cast<char*>(fdes[i].data()), dims * sizeof(float));
    if (!in) throw std::runtime_error("read_point_range: failed to read fde data");
  }
  return {fdes, dims};
}

}  // namespace io
}  // namespace parlayANN
