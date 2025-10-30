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

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <algorithm>
#include <iostream>

#include "parlay/internal/file_map.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"

#include "NSGDist.h"
#include "mvsic/core/distance_measures/one_to_one.h"

namespace mvsic {

template<typename T>
struct IP_Point {
 private:
  T *values;
  unsigned int d;
  unsigned int aligned_d;
  long id_;

 public:
  using distanceType = T;

  IP_Point() : values(nullptr), d(0), aligned_d(0), id_(-1) {}
  // Non-owning version
  IP_Point(T *values, unsigned int d, unsigned int ad, long id = -1) :
      values(values), d(d), aligned_d(ad), id_(id) {}

  static distanceType d_min() { return -std::numeric_limits<float>::max(); }
  static bool is_metric() { return false; }
  T operator[](long i) const { return *(values + i); }
  T &operator[](long i) { return *(values + i); }
  float distance(const IP_Point<T> &x) const { return ip_distance(values, x.values, d); }
  std::pair<float, size_t> distance_w_cmps(const IP_Point<T> &x) const {
    return std::make_pair(ip_distance(values, x.values, d), 2 * d);
  }
  void prefetch() const {
    int l = (aligned_d * sizeof(T)) / 64;
    for (int i = 0; i < l; i++)
      __builtin_prefetch((char *)values + i * 64);
  }
  long id() const { return id_; }
  bool operator==(const IP_Point<T> &q) const {
    for (int i = 0; i < d; i++) {
      if (values[i] != q.values[i]) {
        return false;
      }
    }
    return true;
  }
  inline unsigned int get_dims() const { return d; }
  bool same_as(const IP_Point<T> &q) { return values == q.values; }
  inline auto get_slice() const { return parlay::make_slice(values, values + d); }
  inline T *data() const noexcept { return values; }
};

}  // namespace mvsic