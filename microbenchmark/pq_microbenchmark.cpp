#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <atomic>
#include <algorithm>
#include <unordered_set>
#include <utility>

#include "parlay/parallel.h"

#include "mvsic/core/quantization/pq.h"
#include "mvsic/core/quantization/rabitq.h"
#include "mvsic/core/quantization/scann.h"
#include "mvsic/core/quantization/wrapper.h"

#include "mvsic/core/types/chamfer_ip_point.h"
#include "mvsic/core/types/chamfer_l2_point.h"
#include "mvsic/core/types/point_cloud_set.h"
#include "mvsic/core/utils/parse_command_line.h"

using namespace mvsic;

template<typename PC>
void check_normalized(const PC& points) {
    std::cout << "Checking if all points are normalized..." << std::endl;
    uint32_t n = points.size();
    uint32_t dims = points.get_dims();
    std::atomic<bool> all_normalized(true);
    std::atomic<size_t> non_normalized_count(0);

    parlay::parallel_for(0, n, [&](uint32_t i) {
        float* data = points.data(i);
        uint32_t num_points = points.get_size(i);
        for (uint32_t j = 0; j < num_points; ++j) {
            float norm_sq = 0.0f;
            for (uint32_t d = 0; d < dims; ++d) {
                float val = data[j * dims + d];
                norm_sq += val * val;
            }
            if (std::abs(norm_sq - 1.0f) > 1e-4) {
                all_normalized = false;
                non_normalized_count++;
            }
        }
    });

    if (all_normalized) {
        std::cout << "All points are normalized." << std::endl;
    } else {
        std::cout << "Warning: " << non_normalized_count << " points are NOT normalized." << std::endl;
    }
}

template<typename ChPoint>
void run_microbenchmark(mvsic::commandLine &P) {
    using PC = PointCloudSet<ChPoint>;

    char *inFile = P.getOptionValue("-i");
    if (inFile == nullptr) {
        std::cout << "Error: Input file not specified. Use -i <file>" << std::endl;
        return;
    }
    bool is_mmap = P.getOption("-mm");

    std::cout << "Loading PointCloud from: " << inFile << " (mmap: " << is_mmap << ")" << std::endl;
    auto points = PC(inFile, is_mmap);
    std::cout << "Loaded " << points.size() << " point clouds with dim " << points.get_dims() << std::endl;

    check_normalized(points);

    uint32_t m = P.getOptionIntValue("-m", 8);
    uint32_t k = P.getOptionIntValue("-k", 256);
    uint32_t s = P.getOptionIntValue("-s", 20);

    constexpr bool Metric = ChPoint::is_metric();
    using PQ_Range = pq::Quantized_Point_Range<FlattenedPCRange<PC>, Metric>;
    using PQ_PCSet = Quantized_Point_Cloud_Set<PQ_Range, Metric>;

    std::cout << "Training PQ with m=" << m << ", k=" << k << ", subsample_mult=" << s << std::endl;
    parlay::internal::timer t;
    t.start();
    PQ_PCSet q_db(points, m, k, s);
    std::cout << "PQ Training and Encoding completed in " << t.next_time() << "s" << std::endl;

    char *qFile = P.getOptionValue("-q");
    if (qFile != nullptr) {
        std::cout << "Loading Query PointCloud from: " << qFile << std::endl;
        auto queries = PC(qFile);
        std::cout << "Loaded " << queries.size() << " query point clouds" << std::endl;

        if (queries.size() > 0 && points.size() > 0) {
             std::cout << "Demonstrating single vector-vector distance computation:" << std::endl;
             auto q_query_0 = q_db.quantize_query(queries[0]);
             if (q_query_0.vec_queries.size() > 0) {
                 auto q_vec = q_query_0.vec_queries[0];
                 auto db_cloud = q_db[0];
                 if (db_cloud.size() > 0) {
                     auto db_vec = db_cloud[0];
                     float dist = q_vec.distance(db_vec);
                     std::cout << "  Quantized Distance (Query[0].Vec[0] <-> DB[0].Vec[0]): " << dist << std::endl;
                     auto true_q_vec = queries[0][0];
                     auto true_db_vec = points[0][0];
                     float true_dist = true_q_vec.distance(true_db_vec);
                     std::cout << "  True Distance      (Query[0].Vec[0] <-> DB[0].Vec[0]): " << true_dist << std::endl;
                 }
             }
        }

        // K' Metric Calculation
        size_t top_k = P.getOptionIntValue("-top_k", 10);
        size_t num_eval = P.getOptionIntValue("-num_eval", 100);

        if (queries.size() > 0 && points.size() > 0) {
            std::cout << "\nCalculating Average K' for top " << top_k
                      << " neighbors (evaluating first " << num_eval << " query vectors)..." << std::endl;

            size_t num_db_vecs = points.total_size();
            float* db_data = points.data();
            uint32_t dims = points.get_dims();

            using VecType = decltype(std::declval<ChPoint>()[0]);

            std::atomic<size_t> total_k_prime(0);
            size_t processed_queries = 0;

            for (size_t i = 0; i < queries.size(); ++i) {
                if (processed_queries >= num_eval) break;

                auto q_query_cloud = q_db.quantize_query(queries[i]);
                size_t num_vecs = queries[i].size();

                for (size_t j = 0; j < num_vecs; ++j) {
                    if (processed_queries >= num_eval) break;
                    processed_queries++;

                    // 1. True Distances
                    auto true_q_vec = queries[i][j];
                    std::vector<std::pair<float, size_t>> true_dists(num_db_vecs);

                    parlay::parallel_for(0, num_db_vecs, [&](size_t k) {
                        VecType db_vec(db_data + k * dims, dims, dims, k);
                        true_dists[k] = {true_q_vec.distance(db_vec), k};
                    });

                    std::nth_element(true_dists.begin(), true_dists.begin() + top_k, true_dists.end());
                    std::unordered_set<size_t> true_nn_indices;
                    for(size_t k=0; k<top_k; ++k) {
                      true_nn_indices.insert(true_dists[k].second);
                    }

                    // 2. Quantized Distances
                    auto q_vec_lut = q_query_cloud.vec_queries[j];
                    std::vector<std::pair<float, size_t>> q_dists(num_db_vecs);

                    parlay::parallel_for(0, num_db_vecs, [&](size_t k) {
                        auto q_db_point = q_db.vec_quantizer[k];
                        q_dists[k] = {q_vec_lut.distance(q_db_point), k};
                    });

                    std::sort(q_dists.begin(), q_dists.end());

                    // 3. Find K'
                    size_t current_k_prime = 0;
                    size_t found_count = 0;
                    for (size_t k = 0; k < num_db_vecs; ++k) {
                        if (true_nn_indices.count(q_dists[k].second)) {
                            current_k_prime = k + 1;
                            found_count++;
                            if (found_count == top_k) break;
                        }
                    }
                    total_k_prime = total_k_prime + current_k_prime;

                    if (processed_queries % 10 == 0) {
                         std::cout << "Processed " << processed_queries << "/" << num_eval << " vectors..." << std::endl;
                    }
                }
            }
            if (processed_queries > 0) {
                double avg_k_prime = (double)total_k_prime / processed_queries;
                std::cout << "Average K': " << avg_k_prime << std::endl;
            }
        }

        std::cout << "Total vectors: " << points.total_size() << std::endl;

        std::cout << "Microbenchmarking PQ query quantization..." << std::endl;
        t.start();
        for (size_t i = 0; i < queries.size(); ++i) {
            auto q_query = q_db.quantize_query(queries[i]);
            if (i == 0) {
                std::cout << "First quantized query has " << q_query.vec_queries.size() << " vectors." << std::endl;
            }
        }
        double total_time = t.next_time();
        std::cout << "Quantized " << queries.size() << " queries in " << total_time << "s ("
                  << (queries.size() / total_time) << " queries/s)" << std::endl;
    }
}

int main(int argc, char** argv) {
    mvsic::commandLine P(argc, argv,
                       "[-i <inFile>] [-q <qFile>] [-mm] [-dist_func <IP|L2>] "
                       "[-m <blocks>] [-k <clusters>] [-s <subsample>] "
                       "[-top_k <K>] [-num_eval <N>]");

    std::string df = P.getOptionValue("-dist_func", "IP");

    if (df == "L2") {
        std::cout << "Using L2 distance" << std::endl;
        using ChPoint = ChamferL2_Point;
        run_microbenchmark<ChPoint>(P);
    } else if (df == "IP") {
        std::cout << "Using IP distance" << std::endl;
        using ChPoint = ChamferIP_Point;
        run_microbenchmark<ChPoint>(P);
    } else {
        std::cout << "Unknown distance function: " << df << ". using IP." << std::endl;
        using ChPoint = ChamferIP_Point;
        run_microbenchmark<ChPoint>(P);
    }

    return 0;
}
