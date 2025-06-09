#pragma once

#include "parlay/primitives.h"
#include <Eigen/Core>

float chamfer_l2_distance(const float *a, size_t n_a, const float *b,
													size_t n_b, size_t dim);

struct ChamferL2_Point {
private:
	size_t n = 0;
	size_t dims = 0;
	size_t aligned_dims = 0;
	float *values = nullptr;
	size_t id = std::numeric_limits<size_t>::max();
	bool owns = false;

public:
	ChamferL2_Point() noexcept {}
	// Non-owning version
	ChamferL2_Point(size_t n, size_t dims, float *values, size_t id) noexcept
			: n(n), dims(dims), values(values), id(id), owns(false) {}
	// Owning version, creates a copy. Typically no id associated.
	ChamferL2_Point(size_t n, size_t dims, const float *values_)
			: n(n), dims(dims), owns(true) {
		values = static_cast<float *>(parlay::p_malloc(n * dims * sizeof(float)));
		std::memcpy(values, values_, n * dims * sizeof(float));
	}
	// Move Assignment Operator: does exactly what the input does
	ChamferL2_Point &operator=(ChamferL2_Point &&other) noexcept;
	// *Copy Assignment Operator: creates owning copy of values regardless of
	// input
	ChamferL2_Point &operator=(const ChamferL2_Point &other);
	// Move Constructor: does exactly what the input does
	ChamferL2_Point(ChamferL2_Point &&other) noexcept;
	// Copy Constructor: does exactly what the input does
	ChamferL2_Point(const ChamferL2_Point &other);
	// Destructor: free if owns values
	~ChamferL2_Point() noexcept {
		if (owns && (values != nullptr)) {
			parlay::p_free(values);
			values = nullptr;
			owns = false;
		}
	}

	// Returns number of embeddings in the point cloud
	inline size_t size() const noexcept { return n; }

	// Returns embedding dimension
	inline size_t get_dims() const noexcept { return dims; }

	// Returns id of the pointcloud
	inline size_t get_id() const noexcept { return id; }

	// Returns non-owning view of the i-th embedding
	inline auto operator[](size_t i) const noexcept {
		return parlay::make_slice(values + i * dims, values + (i + 1) * dims);
	}

	// Returns pointer to i-th embedding
	inline float *get_coords(size_t i) const noexcept {
		return values + i * dims;
	}

	// Returns True since L2 is a metric
	constexpr inline bool is_metric() const noexcept { return true; }

	// Computes the (asymmetric) distance from the current point cloud
	// to the given point cloud
	float distance(const ChamferL2_Point &x) const {
		return chamfer_l2_distance(values, n, x.values, x.n, dims);
	}

	// Returns non-owning view of all coordinates
	inline auto get_slice() const noexcept {
		return parlay::make_slice(values, values + n * dims);
	}
};

float chamfer_l2_distance(const float *a, size_t n_a, const float *b,
													size_t n_b, size_t dim) {
	Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
																 Eigen::RowMajor>>
			mat_a(a, n_a, dim);
	Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
																 Eigen::RowMajor>>
			mat_b(b, n_b, dim);
	// n_a x 1
	Eigen::Matrix<float, Eigen::Dynamic, 1> sq_norms_a =
			mat_a.rowwise().squaredNorm(); // Column vector,
	// 1 x n_b
	Eigen::Matrix<float, 1, Eigen::Dynamic> sq_norms_b =
			mat_b.rowwise().squaredNorm().transpose();
	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> sq_dist_matrix =
			sq_norms_a.replicate(1, n_b)			// n_a x n_b
			- 2 * (mat_a * mat_b.transpose()) // n_a x n_b
			+ sq_norms_b.replicate(n_a, 1);		// n_a x n_b

	sq_dist_matrix = sq_dist_matrix.cwiseMax(0.0);
	// n_a x 1
	Eigen::Matrix<float, Eigen::Dynamic, 1> min_dists_A_to_B =
			sq_dist_matrix.rowwise().minCoeff();
	float sum_min_dists = min_dists_A_to_B.sum();
	return sum_min_dists / static_cast<float>(n_a);
}

// Move Assignment Operator
ChamferL2_Point &ChamferL2_Point::operator=(ChamferL2_Point &&other) noexcept {
	if (this != &other) {
		if (owns && (values != nullptr)) {
			parlay::p_free(values);
			owns = false;
		}
		n = other.n;
		dims = other.dims;
		aligned_dims = other.aligned_dims;
		values = other.values;
		id = other.id;
		owns = other.owns;

		other.n = 0;
		other.dims = 0;
		other.aligned_dims = 0;
		other.values = nullptr;
		other.id = std::numeric_limits<size_t>::max();
		other.owns = false;
	}
	return *this;
}

// Copy Assignment Operator: creates owning copy of values regardless
// of input
ChamferL2_Point &ChamferL2_Point::operator=(const ChamferL2_Point &other) {
	if (this != &other) {
		if (owns && (values != nullptr)) {
			parlay::p_free(values);
			owns = false;
		}
		n = other.n;
		dims = other.dims;
		aligned_dims = other.aligned_dims;
		id = other.id;
		if (other.values == nullptr) {
			values = nullptr;
			owns = false;
		} else {
			values = static_cast<float *>(parlay::p_malloc(n * dims * sizeof(float)));
			std::memcpy(values, other.values, n * dims * sizeof(float));
			owns = true;
		}
	}
	return *this;
}

// Move Constructor
ChamferL2_Point::ChamferL2_Point(ChamferL2_Point &&other) noexcept
		: n(other.n), dims(other.dims), aligned_dims(other.aligned_dims),
			values(other.values), id(other.id), owns(other.owns) {
	other.n = 0;
	other.dims = 0;
	other.aligned_dims = 0;
	other.values = nullptr;
	other.id = std::numeric_limits<size_t>::max();
	other.owns = false;
}

// Copy Constructor: does exactly what the input does
ChamferL2_Point::ChamferL2_Point(const ChamferL2_Point &other)
		: n(other.n), dims(other.dims), aligned_dims(other.aligned_dims),
			id(other.id), owns(other.owns) {
	if (owns) {
		values = static_cast<float *>(parlay::p_malloc(n * dims * sizeof(float)));
		std::memcpy(values, other.values, n * dims * sizeof(float));
	} else {
		values = other.values;
	}
}