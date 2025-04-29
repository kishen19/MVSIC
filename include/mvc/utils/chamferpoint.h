#pragma once

// #include <Eigen/Core>
// include eigen/Core (header-only including a recent copy of the
// library should work just fine).

template <typename T>
struct ChamferPoint {
  using distance_type = T;
  static bool is_metric() { return false; }
  auto operator[](long i) const {return parlay::make_slice(values + i*d, values + i*d + d);}

  float distance(const ChamferPoint& x) const {
    return 1 - (chamfer_sim(*this, x, d)); // + chamfer_sim(x, *this, dim));
  }

  void prefetch() {}

  long id() const { return id_; }
  
  size_t size() const { return n; }

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

  auto Coords() const { return parlay::make_slice(values, values + n*d); }
  
  ChamferPoint(long id, T* values_, size_t num_, unsigned int dims_)
      : id_(id), values(values_), d(dims_), n(num_) {}

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

  static float chamfer_sim(const ChamferPoint& a, const ChamferPoint& b,
    unsigned int dim) {
    auto our_coords = a.Coords();
    auto their_coords = b.Coords();
    float sim = 0;
    // TODO: parallel
    for (size_t i = 0; i < a.size(); ++i) {
      float max_sim = -1e9;
      for (size_t j = 0; j < b.size(); ++j) {
        float curr_sim = 0;
        for (unsigned int k = 0; k < dim; ++k) {
          curr_sim += our_coords[i * dim + k] * their_coords[j * dim + k];
        }
        max_sim = std::max(max_sim, curr_sim);
      }
      sim += max_sim;
    }
    return sim / (a.size());
  }

private:
  T* values;
  size_t n = 0;
  unsigned int d;
  unsigned int aligned_d;
  long id_;
};
