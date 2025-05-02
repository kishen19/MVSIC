#pragma once
#include <Eigen/Core>

template <typename T>
T chamfer_euclidian_distance(const T*, size_t, const T*, size_t, unsigned int);

template <typename T>
T chamfer_mips_distance(const T*, size_t, const T*, size_t, unsigned int);

template <typename T>
struct Chamfer_Euclidian_Point {
  using distance_type = T;

  Chamfer_Euclidian_Point() : id(0), values(nullptr), d(0), n(0), owns(false) {}
  Chamfer_Euclidian_Point(long id_, T* values_, size_t num_, unsigned int dims_)
      : id(id_), values(values_), d(dims_), n(num_), owns(false) {}
      Chamfer_Euclidian_Point(T* values_, size_t num_, unsigned int dims_)
      : id(0), d(dims_), n(num_), owns(true) {
    values = static_cast<T*>(malloc(n * d * sizeof(T)));
    std::memcpy(values, values_, n * d * sizeof(T));
  }
  ~Chamfer_Euclidian_Point() {
    if (owns && (values != nullptr)) {
      free(values);
      values = nullptr;
      owns = false;
    }
  }
  
  auto operator[](long i) const {return parlay::make_slice(values + i*d, values + i*d + d);}

  Chamfer_Euclidian_Point& operator=(const Chamfer_Euclidian_Point& other) {
    if (this != &other) {
      if (owns && (values != nullptr)) {
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

  float distance(const Chamfer_Euclidian_Point& x) const {
    return chamfer_euclidian_distance(values, n, x.values, x.n, d);
  }

  void prefetch() {}
  long get_id() const { return id; }
  size_t size() const { return n; }
  auto Coords() const { return parlay::make_slice(values, values + n*d); }

private:
  T* values;
  size_t n = 0;
  unsigned int d;
  unsigned int aligned_d;
  long id;
  bool owns = false;
};

template <typename T>
struct Chamfer_Mips_Point {
  using distance_type = T;

  Chamfer_Mips_Point() : id(0), values(nullptr), d(0), n(0), owns(false) {}
  Chamfer_Mips_Point(long id_, T* values_, size_t num_, unsigned int dims_)
      : id(id_), values(values_), d(dims_), n(num_), owns(false) {}
  Chamfer_Mips_Point(T* values_, size_t num_, unsigned int dims_)
      : id(0), d(dims_), n(num_), owns(true) {
    values = static_cast<T*>(malloc(n * d * sizeof(T)));
    std::memcpy(values, values_, n * d * sizeof(T));
  }
  ~Chamfer_Mips_Point() {
    if (owns && (values != nullptr)) { 
      free(values);
      values = nullptr;
      owns = false;
    }
  }
  
  auto operator[](long i) const {return parlay::make_slice(values + i*d, values + i*d + d);}
  Chamfer_Mips_Point& operator=(const Chamfer_Mips_Point& other) {
    if (this != &other) {
      if (owns && (values != nullptr)) {
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

  float distance(const Chamfer_Mips_Point& x) const {
    return chamfer_mips_distance(values, n, x.values, x.n, d);
  }

  void prefetch() {}
  long get_id() const { return id; }
  size_t size() const { return n; }
  auto Coords() const { return parlay::make_slice(values, values + n*d); }

private:
  T* values;
  size_t n = 0;
  unsigned int d;
  unsigned int aligned_d;
  long id;
  bool owns = false;
};

template <typename T>
T chamfer_euclidian_distance(const T* a, size_t n_a,
                             const T* b, size_t n_b, unsigned int dim) {
  Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      mat_a(a, n_a, dim);
  Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      mat_b(b, n_b, dim);
  // n_a x 1
  Eigen::Matrix<T, Eigen::Dynamic, 1> sq_norms_a = mat_a.rowwise().squaredNorm(); // Column vector, 
  // 1 x n_b
  Eigen::Matrix<T, 1, Eigen::Dynamic> sq_norms_b = mat_b.rowwise().squaredNorm().transpose();
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> sq_dist_matrix =
      sq_norms_a.replicate(1, n_b)     // n_a x n_b
    - 2 * (mat_a * mat_b.transpose())  // n_a x n_b
    + sq_norms_b.replicate(n_a, 1);    // n_a x n_b

  sq_dist_matrix = sq_dist_matrix.cwiseMax(static_cast<T>(0));
  // n_a x 1
  Eigen::Matrix<T, Eigen::Dynamic, 1> min_dists_A_to_B = sq_dist_matrix.rowwise().minCoeff();
  T sum_min_dists = min_dists_A_to_B.sum();
  return sum_min_dists / static_cast<T>(n_a);
}


template <typename T>
T chamfer_mips_distance(const T* a, size_t n_a,
                         const T* b, size_t n_b, unsigned int dim) {
  Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      mat_a(a, n_a, dim);
  Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      mat_b(b, n_b, dim);
  // (n_a x dim) * (dim x n_b) = (n_a x n_b)
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> inner_product_matrix =
      mat_a * mat_b.transpose();
  // (n_a x 1)
  Eigen::Matrix<T, Eigen::Dynamic, 1> max_ips_A_to_B =
      inner_product_matrix.rowwise().maxCoeff();
  T sum_max_ips = max_ips_A_to_B.sum();
  return - sum_max_ips / static_cast<T>(n_a);
}