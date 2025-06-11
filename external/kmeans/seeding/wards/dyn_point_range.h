// This code is part of the Problem Based Benchmark Suite (PBBS)
// Copyright (c) 2011 Guy Blelloch and the PBBS team
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights (to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once

#include "parlay/internal/file_map.h"
#include "parlay/primitives.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace dyn {

// tp_size must divide 64 evenly--no weird/large types!
long dim_round_up(long dim, long tp_size) {
  long qt = (dim * tp_size) / 64;
  long remainder = (dim * tp_size) % 64;
  if (remainder == 0)
    return dim;
  else
    return ((qt + 1) * 64) / tp_size;
}

template<typename T, class Point>
struct PointRange {
  using pT = Point;

  long dimension() { return dims; }
  long aligned_dimension() { return aligned_dims; }

  PointRange() : values(std::shared_ptr<T[]>(nullptr, std::free)) { n = 0; }

  // data is a 2D array-type (n x d)
  // data[i][j] must be accessible
  template<typename Seq>
  PointRange(const Seq& data, unsigned _d) :
      values(std::shared_ptr<T[]>(nullptr, std::free)),
      dims(_d),
      aligned_dims(dim_round_up(dims, sizeof(T))),
      n(data.size()) {
    values =
        std::shared_ptr<T[]>((T*)aligned_alloc(64, (n + 1) * aligned_dims * sizeof(T)), std::free);
    // The last vector is always the 0 vector (starting_point of vamana)
    // std::memset(values.get() + n * aligned_dims, T(0), aligned_dims * sizeof(T));
    parlay::parallel_for(0, n, [&](size_t i) {
      T* val = values.get() + i * aligned_dims;
      for (size_t j = 0; j < dims; j++) {
        val[j] = data[i][j];
      }
    });
    std::fill(values.get() + n * aligned_dims, values.get() + (n + 1) * aligned_dims, T(0));
    // n++; // Added only via AddZero
  }

  size_t size() const { return n; }

  unsigned int get_dims() const { return dims; }

  unsigned int get_aligned_dims() const { return aligned_dims; }

  Point operator[](long i) const {
    return Point(values.get() + i * aligned_dims, dims, aligned_dims, i);
  }

  Point operator[](long i) { return Point(values.get() + i * aligned_dims, dims, aligned_dims, i); }

  PointRange<T, Point> make_copy() const {
    PointRange<T, Point> copy(n, dims);
    parlay::parallel_for(0, n, [&](size_t i) {
      std::memcpy(copy.values.get() + i * aligned_dims, values.get() + i * aligned_dims,
                  aligned_dims * sizeof(T));
    });
    return copy;
  }

  void AddZero() { n++; }

  void centroid(long i, long j, size_t sz1, size_t sz2) {
    size_t total_size = sz1 + sz2;
    T* val = values.get() + i * aligned_dims;
    T* val2 = values.get() + j * aligned_dims;
    for (size_t k = 0; k < aligned_dims; k++) {
      val[k] = (val[k] * sz1 + val2[k] * sz2) / total_size;
    }
  }

 private:
  std::shared_ptr<T[]> values;
  unsigned int dims;
  unsigned int aligned_dims;
  size_t n;
};

}  // namespace dyn