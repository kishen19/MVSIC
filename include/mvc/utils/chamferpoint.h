#include <Eigen/Core>
// include eigen/Core (header-only including a recent copy of the
// library should work just fine).

template <typename T>
struct ChamferPoint {
  using distanceType = T;
  static bool is_metric() { return false; }

  // Asymmetric chamfer similarity from a-->b
  // static float chamfer_sim(const ChamferPoint& a, const ChamferPoint& b,
  //                           unsigned int dim) {
  //   auto our_coords = a.GetSpan();
  //   auto their_coords = b.GetSpan();
  //   Eigen::Map<Eigen::MatrixRowMajor> a_matrix(const_cast<float*>(our_coords.begin()),
  //                                       our_coords.size() / dim, dim);
  //   Eigen::Map<Eigen::MatrixRowMajor> b_matrix(
  //       const_cast<float*>(their_coords.begin()), their_coords.size() / dim,
  //       dim);
  //   Eigen::MatrixXf all_similarities = a_matrix * b_matrix.transpose();
  //   return all_similarities.rowwise().maxCoeff().mean();
  // }

  // Trying symmetric chamfer distance.
  float distance(const ChamferPoint& x) {
    return 1 - (chamfer_sim(*this, x, dim) + chamfer_sim(x, *this, dim));
//    return 1 - (chamfer_sim(*this, x, dim)); // + chamfer_sim(x, *this, dim));
  }

  void prefetch() {}

  long id() { return id_; }

  double norm() {
    double acc = 0.0;
    for (size_t i=0; i < values.size(); ++i) {
      acc += values[i] * values[i];
    }
    return acc;
  }

  bool operator==(const ChamferPoint& q) const {
    if (values.size() != q.values.size()) return false;
    for (int i = 0; i < values.size(); i++) {
      if (values[i] != q.values[i]) {
        return false;
      }
    }
    return true;
  }

  bool same_as(const ChamferPoint& q) {
    return values.begin() == q.values.begin();
  }

  auto GetSpan() const { return values; }

  ChamferPoint(long id, T* values_, unsigned int dim_)
      : id_(id), values(values_), dim(dim_) {}

  long id_;
  T* values;
  unsigned int dim;
};
