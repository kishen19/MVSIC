#pragma once

#include <queue>
#include <Eigen/Dense>
#include "parlay/primitives.h"

namespace mvsic {

template<typename ChPoint, typename PCS>
class OneToMany {
 public:
  static parlay::sequence<float> AllDistances(const ChPoint& q, const PCS& B) {
    parlay::sequence<float> results = parlay::sequence<float>::uninitialized(B.size());
    AllDistancesIntoUninitialized(q, B, results.data());
    return results;
  }

  static std::pair<uint32_t, float> Top(const ChPoint& q, const PCS& B) {
    std::pair<uint32_t, float> result;
    TopIntoUninitialized(q, B, &result);
    return result;
  }

  static parlay::sequence<std::pair<uint32_t, float>> TopK(const ChPoint& q, const PCS& B,
                                                           uint32_t k) {
    parlay::sequence<std::pair<uint32_t, float>> results =
        parlay::sequence<std::pair<uint32_t, float>>::uninitialized(k);
    TopKIntoUninitialized(q, B, k, results.data());
    return results;
  }

  static void AllDistancesIntoUninitialized(const ChPoint& q, const PCS& B, float* results) {
    const size_t m = B.size();
    const size_t d = B.get_dims();
    const size_t q_size = q.size();
    const float s_b = B.average_size();
    auto offsetsB = B.get_offsets();

    if (q_size == 0) {
      for (size_t j = 0; j < m; ++j) {
        results[j] = std::numeric_limits<float>::max();
      }
      return;
    }

    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        q_matrix(q.data(), q_size, d);
    Eigen::Matrix<float, Eigen::Dynamic, 1> q_sq_norms;
    if constexpr (PCS::is_metric()) {
      q_sq_norms = q_matrix.rowwise().squaredNorm();
    }

    const size_t BLOCK_SIZE_B = static_cast<size_t>(std::ceil(128.0 / s_b));

    parlay::parallel_for(0, (m + BLOCK_SIZE_B - 1) / BLOCK_SIZE_B, [&](size_t block_idx_B) {
      const size_t start_idx_B = block_idx_B * BLOCK_SIZE_B;
      const size_t num_in_block_B = std::min(BLOCK_SIZE_B, m - start_idx_B);
      const size_t end_idx_B = start_idx_B + num_in_block_B;
      const size_t float_start_offset_B = offsetsB[start_idx_B];
      const size_t num_vectors_in_block_B = (offsetsB[end_idx_B] - float_start_offset_B) / d;

      if (num_vectors_in_block_B == 0) {
        for (size_t j = 0; j < num_in_block_B; ++j) {
          results[start_idx_B + j] = std::numeric_limits<float>::max();
        }
        return;
      }

      Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
          block_B_matrix(B.data() + float_start_offset_B, num_vectors_in_block_B, d);

      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> dist_matrix(q_size,
                                                                       num_vectors_in_block_B);

      if constexpr (PCS::is_metric()) {
        dist_matrix.noalias() = -2 * (q_matrix * block_B_matrix.transpose());
        dist_matrix.colwise() += q_sq_norms;
        Eigen::Matrix<float, 1, Eigen::Dynamic> block_B_sq_norms =
            block_B_matrix.rowwise().squaredNorm().transpose();
        dist_matrix.rowwise() += block_B_sq_norms;
        dist_matrix = dist_matrix.cwiseMax(0.0);
      } else {
        dist_matrix.noalias() = -1 * (q_matrix * block_B_matrix.transpose());
      }

      for (size_t j = 0; j < num_in_block_B; ++j) {
        const size_t vec_offset_B = (offsetsB[start_idx_B + j] - float_start_offset_B) / d;
        const size_t vec_count_B = (offsetsB[start_idx_B + j + 1] - offsetsB[start_idx_B + j]) / d;
        if (vec_count_B == 0) {
          results[start_idx_B + j] = std::numeric_limits<float>::max();
          continue;
        }
        auto final_slice = dist_matrix.middleCols(vec_offset_B, vec_count_B);
        results[start_idx_B + j] = final_slice.rowwise().minCoeff().mean();
      }
    });
  }

  static void TopIntoUninitialized(const ChPoint& q, const PCS& B,
                                   std::pair<uint32_t, float>* result) {
    const size_t m = B.size();
    const size_t d = B.get_dims();
    const size_t q_size = q.size();
    const float s_b = B.average_size();
    auto offsetsB = B.get_offsets();

    *result = {0, std::numeric_limits<float>::max()};
    if (q_size == 0) return;

    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        q_matrix(q.data(), q_size, d);
    Eigen::Matrix<float, Eigen::Dynamic, 1> q_sq_norms;
    if constexpr (PCS::is_metric()) {
      q_sq_norms = q_matrix.rowwise().squaredNorm();
    }

    const size_t BLOCK_SIZE_B = static_cast<size_t>(std::ceil(128.0 / s_b));
    const size_t num_blocks = (m + BLOCK_SIZE_B - 1) / BLOCK_SIZE_B;

    auto block_results = parlay::sequence<std::pair<uint32_t, float>>::uninitialized(num_blocks);

    parlay::parallel_for(0, num_blocks, [&](size_t block_idx_B) {
      const size_t start_idx_B = block_idx_B * BLOCK_SIZE_B;
      const size_t num_in_block_B = std::min(BLOCK_SIZE_B, m - start_idx_B);
      const size_t end_idx_B = start_idx_B + num_in_block_B;
      const size_t float_start_offset_B = offsetsB[start_idx_B];
      const size_t num_vectors_in_block_B = (offsetsB[end_idx_B] - float_start_offset_B) / d;

      block_results[block_idx_B] = {0, std::numeric_limits<float>::max()};
      if (num_vectors_in_block_B == 0) return;

      Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
          block_B_matrix(B.data() + float_start_offset_B, num_vectors_in_block_B, d);

      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> dist_matrix(q_size,
                                                                       num_vectors_in_block_B);

      if constexpr (PCS::is_metric()) {
        dist_matrix.noalias() = -2 * (q_matrix * block_B_matrix.transpose());
        dist_matrix.colwise() += q_sq_norms;
        Eigen::Matrix<float, 1, Eigen::Dynamic> block_B_sq_norms =
            block_B_matrix.rowwise().squaredNorm().transpose();
        dist_matrix.rowwise() += block_B_sq_norms;
        dist_matrix = dist_matrix.cwiseMax(0.0);
      } else {
        dist_matrix.noalias() = -1 * (q_matrix * block_B_matrix.transpose());
      }

      for (size_t j_idx = 0; j_idx < num_in_block_B; ++j_idx) {
        const size_t j = start_idx_B + j_idx;
        const size_t vec_offset_B = (offsetsB[j] - float_start_offset_B) / d;
        const size_t vec_count_B = (offsetsB[j + 1] - offsetsB[j]) / d;
        if (vec_count_B == 0) continue;

        auto final_slice = dist_matrix.middleCols(vec_offset_B, vec_count_B);
        float dist = final_slice.rowwise().minCoeff().mean();
        if (dist < block_results[block_idx_B].second) {
          block_results[block_idx_B] = {static_cast<uint32_t>(j), dist};
        }
      }
    });

    for (size_t i = 0; i < num_blocks; ++i) {
      if (block_results[i].second < result->second) {
        *result = block_results[i];
      }
    }
  }

  static void TopKIntoUninitialized(const ChPoint& q, const PCS& B, uint32_t k,
                                    std::pair<uint32_t, float>* results) {
    const size_t m = B.size();
    const size_t d = B.get_dims();
    const size_t q_size = q.size();
    const float s_b = B.average_size();
    auto offsetsB = B.get_offsets();

    if (q_size == 0) {
      for (size_t ki = 0; ki < k; ++ki)
        results[ki] = {0, std::numeric_limits<float>::max()};
      return;
    }

    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        q_matrix(q.data(), q_size, d);
    Eigen::Matrix<float, Eigen::Dynamic, 1> q_sq_norms;
    if constexpr (PCS::is_metric()) {
      q_sq_norms = q_matrix.rowwise().squaredNorm();
    }

    const size_t BLOCK_SIZE_B = static_cast<size_t>(std::ceil(128.0 / s_b));
    const size_t num_blocks = (m + BLOCK_SIZE_B - 1) / BLOCK_SIZE_B;

    using T = std::pair<float, uint32_t>;
    auto block_results = parlay::sequence<std::priority_queue<T>>(num_blocks);

    parlay::parallel_for(0, num_blocks, [&](size_t block_idx_B) {
      const size_t start_idx_B = block_idx_B * BLOCK_SIZE_B;
      const size_t num_in_block_B = std::min(BLOCK_SIZE_B, m - start_idx_B);
      const size_t end_idx_B = start_idx_B + num_in_block_B;
      const size_t float_start_offset_B = offsetsB[start_idx_B];
      const size_t num_vectors_in_block_B = (offsetsB[end_idx_B] - float_start_offset_B) / d;

      if (num_vectors_in_block_B == 0) return;

      Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
          block_B_matrix(B.data() + float_start_offset_B, num_vectors_in_block_B, d);

      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> dist_matrix(q_size,
                                                                       num_vectors_in_block_B);

      if constexpr (PCS::is_metric()) {
        dist_matrix.noalias() = -2 * (q_matrix * block_B_matrix.transpose());
        dist_matrix.colwise() += q_sq_norms;
        Eigen::Matrix<float, 1, Eigen::Dynamic> block_B_sq_norms =
            block_B_matrix.rowwise().squaredNorm().transpose();
        dist_matrix.rowwise() += block_B_sq_norms;
        dist_matrix = dist_matrix.cwiseMax(0.0);
      } else {
        dist_matrix.noalias() = -1 * (q_matrix * block_B_matrix.transpose());
      }

      for (size_t j_idx = 0; j_idx < num_in_block_B; ++j_idx) {
        const size_t j = start_idx_B + j_idx;
        const size_t vec_offset_B = (offsetsB[j] - float_start_offset_B) / d;
        const size_t vec_count_B = (offsetsB[j + 1] - offsetsB[j]) / d;
        if (vec_count_B == 0) continue;

        auto final_slice = dist_matrix.middleCols(vec_offset_B, vec_count_B);
        float dist = final_slice.rowwise().minCoeff().mean();

        if (block_results[block_idx_B].size() < k) {
          block_results[block_idx_B].push({dist, static_cast<uint32_t>(j)});
        } else if (dist < block_results[block_idx_B].top().first) {
          block_results[block_idx_B].pop();
          block_results[block_idx_B].push({dist, static_cast<uint32_t>(j)});
        }
      }
    });

    std::priority_queue<T> heap;
    for (size_t i = 0; i < num_blocks; ++i) {
      while (!block_results[i].empty()) {
        auto val = block_results[i].top();
        block_results[i].pop();
        if (heap.size() < k) {
          heap.push(val);
        } else if (val.first < heap.top().first) {
          heap.pop();
          heap.push(val);
        }
      }
    }

    size_t count = heap.size();
    for (size_t ki = 0; ki < count; ++ki) {
      results[count - 1 - ki] = {heap.top().second, heap.top().first};
      heap.pop();
    }
    for (size_t ki = count; ki < k; ++ki) {
      results[ki] = {0, std::numeric_limits<float>::max()};
    }
  }
};

}  // namespace mvsic