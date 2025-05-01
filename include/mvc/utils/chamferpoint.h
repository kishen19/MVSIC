#pragma once

// #include <Eigen/Core>
// include eigen/Core (header-only including a recent copy of the
// library should work just fine).

template <typename T>
struct ChamferPoint {
  using distance_type = T;

  ChamferPoint() : id(0), values(nullptr), d(0), n(0), owns(false) {}
  ChamferPoint(long id_, T* values_, size_t num_, unsigned int dims_)
      : id(id_), values(values_), d(dims_), n(num_), owns(false) {}
  ChamferPoint(T* values_, size_t num_, unsigned int dims_)
      : id(0), d(dims_), n(num_), owns(true) {
    values = static_cast<T*>(malloc(n * d * sizeof(T)));
    std::memcpy(values, values_, n * d * sizeof(T));
  }
  ~ChamferPoint() {
    if (owns && (values != nullptr)) { 
      // std::cout << "Freeing values of ChamferPoint with id: " << id 
      //           << ", n: " << n << ", d: " << d << std::endl;
      free(values);
      values = nullptr;
      owns = false;
    }
  }
  
  auto operator[](long i) const {return parlay::make_slice(values + i*d, values + i*d + d);}
  ChamferPoint& operator=(const ChamferPoint& other) {
    if (this != &other) {
      if (owns && (values != nullptr)) {
        // std::cout << "Freeing values of ChamferPoint with id: " << id 
        //         << ", n: " << n << ", d: " << d << std::endl;
        free(values);
        values = nullptr;
        owns = false;
      }
      id = other.id;
      n = other.n;
      d = other.d;
      aligned_d = other.aligned_d;
      values = static_cast<T*>(malloc(n * d * sizeof(T)));
      std::memcpy(values, other.values, n * d * sizeof(T));
      owns = true;
    }
    return *this;
  }

  float distance(const ChamferPoint& x) const {
    // return 1 - (chamfer_sim(*this, x, d)); // + chamfer_sim(x, *this, dim));
    return chamfer_dis(*this, x, d);
  }

  void prefetch() {}
  long get_id() const { return id; }
  size_t size() const { return n; }
  auto Coords() const { return parlay::make_slice(values, values + n*d); }

  static float chamfer_dis(const ChamferPoint& a, const ChamferPoint& b,
    unsigned int dim) {
    auto our_coords = a.Coords();
    auto their_coords = b.Coords();
    auto a_dists = parlay::sequence<T>::from_function(a.size(), [&](size_t i) {
      auto dists_b = parlay::sequence<T>::from_function(b.size(), [&](size_t j) {
        float curr_dis = 0;
        for (unsigned int k = 0; k < dim; ++k) {
          curr_dis += (our_coords[i * dim + k] - their_coords[j * dim + k]) *
                      (our_coords[i * dim + k] - their_coords[j * dim + k]);
        }
        return curr_dis;
      });
      return parlay::reduce(dists_b, parlay::minm<T>());
    });
    auto dis = parlay::reduce(a_dists);
    return dis / a.size();
  }

private:
  T* values;
  size_t n = 0;
  unsigned int d;
  unsigned int aligned_d;
  long id;
  bool owns = false;
};



  // bool operator==(const ChamferPoint& q) const {
  //   if (n != q.size()) return false;
  //   for (int i = 0; i < n; i++) {
  //     if (values[i] != q.values[i]) {
  //       return false;
  //     }
  //   }
  //   return true;
  // }

  // bool same_as(const ChamferPoint& q) {
  //   return values == q.values;
  // }

  // // Asymmetric chamfer similarity from a-->b
  // static float chamfer_sim(const ChamferPoint& a, const ChamferPoint& b,
  //                           unsigned int dim) {
  //   auto our_coords = a.GetSpan();
  //   auto their_coords = b.GetSpan();
  //   Eigen::Map<MatrixRowMajor> a_matrix(const_cast<float*>(our_coords.begin()),
  //                                       our_coords.size() / dim, dim);
  //   Eigen::Map<MatrixRowMajor> b_matrix(
  //       const_cast<float*>(their_coords.begin()), their_coords.size() / dim,
  //       dim);
  //   Eigen::MatrixXf all_similarities = a_matrix * b_matrix.transpose();
  //   return all_similarities.rowwise().maxCoeff().mean();
  // }

  // static float chamfer_sim(const ChamferPoint& a, const ChamferPoint& b,
  //   unsigned int dim) {
  //   auto our_coords = a.Coords();
  //   auto their_coords = b.Coords();
  //   float sim = 0;
  //   // TODO: parallel
  //   for (size_t i = 0; i < a.size(); ++i) {
  //     float max_sim = -1e9;
  //     for (size_t j = 0; j < b.size(); ++j) {
  //       float curr_sim = 0;
  //       for (unsigned int k = 0; k < dim; ++k) {
  //         curr_sim += our_coords[i * dim + k] * their_coords[j * dim + k];
  //       }
  //       max_sim = std::max(max_sim, curr_sim);
  //     }
  //     sim += max_sim;
  //   }
  //   return sim / (a.size());
  // }