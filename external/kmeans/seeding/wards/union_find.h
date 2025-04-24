/*
	Disjoint Set-Union Implementation
	- Union by Size
	- Path Compression
*/

#pragma once

#include "parlay/primitives.h"
#include "parlay/sequence.h"

template <typename indexType>
struct union_find {
  size_t n;
  parlay::sequence<indexType> parents;
  parlay::sequence<indexType> size;

  union_find() : n(0) {}

  union_find(size_t n) : n(n) {
		parents = parlay::sequence<indexType>::from_function(n, [&](indexType i) { return i; });
		size = parlay::sequence<indexType>::from_function(n, [&](indexType i) { return 1; });
	}

	template <class Seq>
	union_find(size_t n, Seq& weights) : n(n) {
		parents = parlay::sequence<indexType>::from_function(n, [&](indexType i) { return i; });
		size = parlay::sequence<indexType>::from_function(n, [&](indexType i) { return weights[i]; });
	}

  inline indexType find_compress(indexType i) {
		indexType j = i;
		while (parents[j] != j) { j = parents[j]; }
		indexType tmp = parents[i];
		while (tmp != j) {
			parents[i] = j;
			i = tmp;
			tmp = parents[i];
		}
		return j;
	}

  // Returns the parent
	inline indexType unite(indexType u_orig, indexType v_orig) {
		indexType u = find_compress(u_orig);
		indexType v = find_compress(v_orig);
		if (u != v){
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
		std::cerr << "Uniting two already merged clusters: " << u_orig 
				<< ", " << v_orig << std::endl;
		assert(false);
		exit(-1);
	}

  indexType get_size(indexType u) {
    return size[find_compress(u)];
  }
};