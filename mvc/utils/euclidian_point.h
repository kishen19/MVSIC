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

#include <algorithm>
#include <iostream>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/internal/file_map.h"
#include "NSGDist.h"


float euclidian_distance(const uint8_t *p, const uint8_t *q, unsigned d) {
  int result = 0;
  for (int i = 0; i < d; i++) {
    result += ((int32_t)((int16_t)q[i] - (int16_t)p[i])) *
      ((int32_t)((int16_t)q[i] - (int16_t)p[i]));
  }
  return (float)result;
}

float euclidian_distance(const int8_t *p, const int8_t *q, unsigned d) {
  int result = 0;
  for (int i = 0; i < d; i++) {
    result += ((int32_t)((int16_t)q[i] - (int16_t)p[i])) *
      ((int32_t)((int16_t)q[i] - (int16_t)p[i]));
  }
  return (float)result;
}

float euclidian_distance(const float *p, const float *q, unsigned d) {
  efanna2e::DistanceL2 distfunc;
  return distfunc.compare(p, q, d);
}

template<typename T>
struct Euclidian_Point {
private:
  T* values;
  unsigned int d;
  unsigned int aligned_d;
  long id_;
  bool owns;

public:
  using distanceType = T;

  Euclidian_Point()
    : values(nullptr), d(0), aligned_d(0), id_(-1), owns(false) {}
  // Non-owning version
  Euclidian_Point(T* values, unsigned int d, unsigned int ad, long id)
    : values(values), d(d), aligned_d(ad), id_(id), owns(false) {}
  // Owning version, creates a copy. Typically no id associated
  Euclidian_Point(T* values_, unsigned int d, unsigned int ad)
    : values(nullptr), d(d), aligned_d(ad), id_(-1), owns(true) {
    values = static_cast<float*>(parlay::p_malloc(d * sizeof(T)));
    std::memcpy(values, values_, d * sizeof(T));
  }
  // Copy Constructor
  Euclidian_Point(const Euclidian_Point& p)
    : values(nullptr), d(p.d), aligned_d(p.aligned_d), id_(p.id_), owns(p.owns) {
    if (owns) {
      values = static_cast<float*>(parlay::p_malloc(d * sizeof(T)));
      std::memcpy(values, p.values, d * sizeof(T));
    } else {
      values = p.values;
    }
  }
  // Move Constructor
  Euclidian_Point(Euclidian_Point&& p)
    : values(p.values), d(p.d), aligned_d(p.aligned_d), id_(p.id_), 
      owns(p.owns) {
    p.values = nullptr;
    p.d = 0;
    p.aligned_d = 0;
    p.id_ = -1;
    p.owns = false;
  }
  // Copy Assignment Operator: creates owning copy of values 
  Euclidian_Point& operator=(const Euclidian_Point& p) {
    if (this != &p) {
      if (owns) {
        parlay::p_free(values);
        owns = false;
      }
      d = p.d;
      aligned_d = p.aligned_d;
      id_ = p.id_;
      if (p.values == nullptr) {
        values = nullptr;
        owns = false;
      } else {
        values = static_cast<float*>(parlay::p_malloc(d * sizeof(T)));
        std::memcpy(values, p.values, d * sizeof(T));
        owns = true;
      }
    }
    return *this;
  }
  // Move Assignment Operator
  Euclidian_Point& operator=(Euclidian_Point&& p) {
    if (this != &p) {
      if (owns) {
        parlay::p_free(values);
        owns = false;
      }
      values = p.values;
      d = p.d;
      aligned_d = p.aligned_d;
      id_ = p.id_;
      owns = p.owns;
      p.values = nullptr;
      p.d = 0;
      p.aligned_d = 0;
      p.id_ = -1;
      p.owns = false;
    }
    return *this;
  }
  ~Euclidian_Point() {
    if (owns && (values != nullptr)) {
      parlay::p_free(values);
      values = nullptr;
      owns = false;
    }
  }
  
  static distanceType d_min() {return 0;}
  static bool is_metric() {return true;}
  T operator[](long i) const {return *(values + i);}
  T &operator[](long i) {return *(values + i);}
  float distance(const Euclidian_Point<T>& x) const {
    return euclidian_distance(this->values, x.values, d);}
  void prefetch() const {
    int l = (aligned_d * sizeof(T))/64;
    for (int i=0; i < l; i++)
      __builtin_prefetch((char*) values + i* 64);
  }
  long id() const {return id_;}
  bool operator==(const Euclidian_Point<T>& q) const {
    for (int i = 0; i < d; i++) {
      if (values[i] != q.values[i]) {
        return false;
      }
    }
    return true;
  }
  inline unsigned int get_dims() const {return d;}
  bool same_as(const Euclidian_Point<T>& q){
    return values == q.values;
  }
  inline auto get_slice() const {
    return parlay::make_slice(values, values+d);
  }
};
