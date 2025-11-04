#pragma once

#include <queue>
#include "parlay/primitives.h"

namespace mvsic {

template<typename PCS>
class ManyToMany {
public:
    static parlay::sequence<float> AllPairs(const PCS& A, const PCS& B) {
        parlay::sequence<float> results = parlay::sequence<float>::uninitialized(A.size() * B.size());
        AllPairsIntoUninitialized(A, B, results.data());
        return results;
    }

    static parlay::sequence<std::pair<uint32_t, float>> Top(const PCS& A, const PCS& B) {
        parlay::sequence<std::pair<uint32_t, float>> results = parlay::sequence<std::pair<uint32_t, float>>::uninitialized(A.size());
        TopIntoUninitialized(A, B, results.data());
        return results;
    }

    static parlay::sequence<std::pair<uint32_t, float>> TopK(const PCS& A, const PCS& B, uint32_t k) {
        parlay::sequence<std::pair<uint32_t, float>> results = parlay::sequence<std::pair<uint32_t, float>>::uninitialized(A.size() * k);
        TopKIntoUninitialized(A, B, k, results.data());
        return results;
    }

    static void AllPairsIntoUninitialized(const PCS& A, const PCS& B, float* results) {
        const size_t n = A.size();
        const size_t m = B.size();
        const size_t d = A.get_dims();
        const float s_a = A.average_size();
        const float s_b = B.average_size();
        auto offsetsA = A.get_offsets();
        auto offsetsB = B.get_offsets();

        const size_t BLOCK_SIZE_A = static_cast<size_t>(std::ceil(128.0 / s_a));
        const size_t BLOCK_SIZE_B = static_cast<size_t>(std::ceil(128.0 / s_b));

        parlay::parallel_for(0, (n + BLOCK_SIZE_A - 1) / BLOCK_SIZE_A, [&](size_t block_idx_A) {
            const size_t start_idx_A = block_idx_A * BLOCK_SIZE_A;
            const size_t num_in_block_A = std::min(BLOCK_SIZE_A, n - start_idx_A);
            const size_t end_idx_A = start_idx_A + num_in_block_A;
            const size_t float_start_offset_A = offsetsA[start_idx_A];
            const size_t num_vectors_in_block_A = (offsetsA[end_idx_A] - float_start_offset_A) / d;

            if (num_vectors_in_block_A == 0) {
                for (size_t i = 0; i < num_in_block_A; ++i) {
                    for (size_t j = 0; j < m; ++j) {
                        results[(start_idx_A + i) * m + j] = std::numeric_limits<float>::max();
                    }
                }
                return;
            }

            Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> block_A_matrix(
                A.data() + float_start_offset_A, num_vectors_in_block_A, d);
            Eigen::Matrix<float, Eigen::Dynamic, 1> block_A_sq_norms;
            if constexpr (PCS::is_metric()) {
                block_A_sq_norms = block_A_matrix.rowwise().squaredNorm();
            }

            for (size_t block_idx_B = 0; block_idx_B < (m + BLOCK_SIZE_B - 1) / BLOCK_SIZE_B; ++block_idx_B) {
                const size_t start_idx_B = block_idx_B * BLOCK_SIZE_B;
                const size_t num_in_block_B = std::min(BLOCK_SIZE_B, m - start_idx_B);
                const size_t end_idx_B = start_idx_B + num_in_block_B;
                const size_t float_start_offset_B = offsetsB[start_idx_B];
                const size_t num_vectors_in_block_B = (offsetsB[end_idx_B] - float_start_offset_B) / d;

                if (num_vectors_in_block_B == 0) {
                    for (size_t i = 0; i < num_in_block_A; ++i) {
                        for (size_t j = 0; j < num_in_block_B; ++j) {
                            results[(start_idx_A + i) * m + (start_idx_B + j)] = std::numeric_limits<float>::max();
                        }
                    }
                    continue;
                }

                Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> block_B_matrix(
                    B.data() + float_start_offset_B, num_vectors_in_block_B, d);
                
                Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> dist_matrix(
                    num_vectors_in_block_A, num_vectors_in_block_B);

                if constexpr (PCS::is_metric()) {
                    dist_matrix.noalias() = -2 * (block_A_matrix * block_B_matrix.transpose());
                    dist_matrix.colwise() += block_A_sq_norms;
                    Eigen::Matrix<float, 1, Eigen::Dynamic> block_B_sq_norms = block_B_matrix.rowwise().squaredNorm().transpose();
                    dist_matrix.rowwise() += block_B_sq_norms;
                    dist_matrix = dist_matrix.cwiseMax(0.0);
                } else {
                    dist_matrix.noalias() = -1 * (block_A_matrix * block_B_matrix.transpose());
                }

                for (size_t i = 0; i < num_in_block_A; ++i) {
                    const size_t vec_offset_A = (offsetsA[start_idx_A + i] - float_start_offset_A) / d;
                    const size_t vec_count_A = (offsetsA[start_idx_A + i + 1] - offsetsA[start_idx_A + i]) / d;
                    if (vec_count_A == 0) {
                        for (size_t j = 0; j < num_in_block_B; ++j) {
                            results[(start_idx_A + i) * m + (start_idx_B + j)] = std::numeric_limits<float>::max();
                        }
                        continue;
                    }
                    auto dist_slice = dist_matrix.middleRows(vec_offset_A, vec_count_A);
                    
                    for (size_t j = 0; j < num_in_block_B; ++j) {
                         const size_t vec_offset_B = (offsetsB[start_idx_B + j] - float_start_offset_B) / d;
                         const size_t vec_count_B = (offsetsB[start_idx_B + j + 1] - offsetsB[start_idx_B + j]) / d;
                        if (vec_count_B == 0) {
                            results[(start_idx_A + i) * m + (start_idx_B + j)] = std::numeric_limits<float>::max();
                            continue;
                        }
                        auto final_slice = dist_slice.middleCols(vec_offset_B, vec_count_B);
                        results[(start_idx_A + i) * m + (start_idx_B + j)] = final_slice.rowwise().minCoeff().mean();
                    }
                }
            }
        });
    }

    static void TopIntoUninitialized(const PCS& A, const PCS& B, std::pair<uint32_t, float>* results) {
        // Implementation with dual-block and updating top-1
        const size_t n = A.size();
        const size_t m = B.size();
        const size_t d = A.get_dims();
        const float s_a = A.average_size();
        const float s_b = B.average_size();
        auto offsetsA = A.get_offsets();
        auto offsetsB = B.get_offsets();

        const size_t BLOCK_SIZE_A = static_cast<size_t>(std::ceil(128.0 / s_a));
        const size_t BLOCK_SIZE_B = static_cast<size_t>(std::ceil(128.0 / s_b));

        parlay::parallel_for(0, n, [&](size_t i) {
            results[i] = {0, std::numeric_limits<float>::max()};
            const size_t vec_count_A = (offsetsA[i + 1] - offsetsA[i]) / d;
            if (vec_count_A == 0) return;

            Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> a_matrix(
                A.data() + offsetsA[i], vec_count_A, d);
            Eigen::Matrix<float, Eigen::Dynamic, 1> a_sq_norms;
            if constexpr (PCS::is_metric()) {
                a_sq_norms = a_matrix.rowwise().squaredNorm();
            }

            for (size_t block_idx_B = 0; block_idx_B < (m + BLOCK_SIZE_B - 1) / BLOCK_SIZE_B; ++block_idx_B) {
                const size_t start_idx_B = block_idx_B * BLOCK_SIZE_B;
                const size_t num_in_block_B = std::min(BLOCK_SIZE_B, m - start_idx_B);
                const size_t end_idx_B = start_idx_B + num_in_block_B;
                const size_t float_start_offset_B = offsetsB[start_idx_B];
                const size_t num_vectors_in_block_B = (offsetsB[end_idx_B] - float_start_offset_B) / d;

                if (num_vectors_in_block_B == 0) continue;

                Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> block_B_matrix(
                    B.data() + float_start_offset_B, num_vectors_in_block_B, d);

                Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> dist_matrix(
                    vec_count_A, num_vectors_in_block_B);

                if constexpr (PCS::is_metric()) {
                    dist_matrix.noalias() = -2 * (a_matrix * block_B_matrix.transpose());
                    dist_matrix.colwise() += a_sq_norms;
                    Eigen::Matrix<float, 1, Eigen::Dynamic> block_B_sq_norms = block_B_matrix.rowwise().squaredNorm().transpose();
                    dist_matrix.rowwise() += block_B_sq_norms;
                    dist_matrix = dist_matrix.cwiseMax(0.0);
                } else {
                    dist_matrix.noalias() = -1 * (a_matrix * block_B_matrix.transpose());
                }

                for (size_t j_idx = 0; j_idx < num_in_block_B; ++j_idx) {
                    const size_t j = start_idx_B + j_idx;
                    const size_t vec_offset_B = (offsetsB[j] - float_start_offset_B) / d;
                    const size_t vec_count_B = (offsetsB[j+1] - offsetsB[j]) / d;
                    if(vec_count_B == 0) continue;

                    auto final_slice = dist_matrix.middleCols(vec_offset_B, vec_count_B);
                    float dist = final_slice.rowwise().minCoeff().mean();
                    if (dist < results[i].second) {
                        results[i] = {static_cast<uint32_t>(j), dist};
                    }
                }
            }
        });
    }

    static void TopKIntoUninitialized(const PCS& A, const PCS& B, uint32_t k, std::pair<uint32_t, float>* results) {
        const size_t n = A.size();
        const size_t m = B.size();
        const size_t d = A.get_dims();
        const float s_a = A.average_size();
        const float s_b = B.average_size();
        auto offsetsA = A.get_offsets();
        auto offsetsB = B.get_offsets();

        const size_t BLOCK_SIZE_A = static_cast<size_t>(std::ceil(128.0 / s_a));
        const size_t BLOCK_SIZE_B = static_cast<size_t>(std::ceil(128.0 / s_b));

        parlay::parallel_for(0, n, [&](size_t i) {
            using T = std::pair<float, uint32_t>;
            std::priority_queue<T> heap;

            const size_t vec_count_A = (offsetsA[i + 1] - offsetsA[i]) / d;
            if (vec_count_A == 0) {
                for(size_t ki=0; ki<k; ++ki) results[i*k+ki] = {0, std::numeric_limits<float>::max()};
                return;
            }

            Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> a_matrix(
                A.data() + offsetsA[i], vec_count_A, d);
            Eigen::Matrix<float, Eigen::Dynamic, 1> a_sq_norms;
            if constexpr (PCS::is_metric()) {
                a_sq_norms = a_matrix.rowwise().squaredNorm();
            }

            for (size_t block_idx_B = 0; block_idx_B < (m + BLOCK_SIZE_B - 1) / BLOCK_SIZE_B; ++block_idx_B) {
                const size_t start_idx_B = block_idx_B * BLOCK_SIZE_B;
                const size_t num_in_block_B = std::min(BLOCK_SIZE_B, m - start_idx_B);
                const size_t end_idx_B = start_idx_B + num_in_block_B;
                const size_t float_start_offset_B = offsetsB[start_idx_B];
                const size_t num_vectors_in_block_B = (offsetsB[end_idx_B] - float_start_offset_B) / d;

                if (num_vectors_in_block_B == 0) continue;

                Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> block_B_matrix(
                    B.data() + float_start_offset_B, num_vectors_in_block_B, d);

                Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> dist_matrix(
                    vec_count_A, num_vectors_in_block_B);

                if constexpr (PCS::is_metric()) {
                    dist_matrix.noalias() = -2 * (a_matrix * block_B_matrix.transpose());
                    dist_matrix.colwise() += a_sq_norms;
                    Eigen::Matrix<float, 1, Eigen::Dynamic> block_B_sq_norms = block_B_matrix.rowwise().squaredNorm().transpose();
                    dist_matrix.rowwise() += block_B_sq_norms;
                    dist_matrix = dist_matrix.cwiseMax(0.0);
                } else {
                    dist_matrix.noalias() = -1 * (a_matrix * block_B_matrix.transpose());
                }

                for (size_t j_idx = 0; j_idx < num_in_block_B; ++j_idx) {
                    const size_t j = start_idx_B + j_idx;
                    const size_t vec_offset_B = (offsetsB[j] - float_start_offset_B) / d;
                    const size_t vec_count_B = (offsetsB[j+1] - offsetsB[j]) / d;
                    if(vec_count_B == 0) continue;

                    auto final_slice = dist_matrix.middleCols(vec_offset_B, vec_count_B);
                    float dist = final_slice.rowwise().minCoeff().mean();
                    
                    if (heap.size() < k) {
                        heap.push({dist, static_cast<uint32_t>(j)});
                    } else if (dist < heap.top().first) {
                        heap.pop();
                        heap.push({dist, static_cast<uint32_t>(j)});
                    }
                }
            }
            
            size_t count = heap.size();
            for(size_t ki=0; ki<count; ++ki) {
                results[i*k + (count - 1 - ki)] = {heap.top().second, heap.top().first};
                heap.pop();
            }
            for(size_t ki=count; ki<k; ++ki) {
                results[i*k + ki] = {0, std::numeric_limits<float>::max()};
            }
        });
    }
};

}  // namespace mvsic