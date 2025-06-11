/*
  Disjoint Set-Union Implementation
  - Union by Size
  - Path Compression
*/

#pragma once

#include "parlay/primitives.h"

template<typename indexType>
struct union_find {
  size_t n = 0;
  parlay::sequence<indexType> parents;
  parlay::sequence<indexType> size;

  union_find() noexcept {}
  union_find(size_t n) noexcept : n(n) {
    parents = parlay::sequence<indexType>::from_function(
        n, [](indexType i) noexcept -> indexType { return i; });
    size = parlay::sequence<indexType>(n, 1);
  }
  template<class Seq>
  union_find(size_t n, Seq& weights) noexcept : n(n) {
    parents = parlay::sequence<indexType>::from_function(
        n, [](indexType i) noexcept -> indexType { return i; });
    size = parlay::sequence<indexType>::from_function(
        n, [](indexType i) noexcept -> indexType { return weights[i]; });
  }

  inline indexType find_compress(indexType i) noexcept {
    while (parents[i] != parents[parents[i]]) {
      parents[i] = parents[parents[i]];
      i = parents[i];
    }
    return parents[i];
  }

  // Returns the parent
  inline indexType unite(indexType u_orig, indexType v_orig) {
    indexType u = find_compress(u_orig);
    indexType v = find_compress(v_orig);
    if (u != v) {
      if (size[v] >= size[u]) {
        parents[u] = v;
        size[v] += size[u];
        return v;
      } else {
        parents[v] = u;
        size[u] += size[v];
        return u;
      }
    }
    std::cerr << "Uniting two already merged clusters: " << u_orig << ", " << v_orig << std::endl;
    assert(false);
    exit(-1);
  }

  inline indexType get_size(indexType u) noexcept { return size[find_compress(u)]; }
};