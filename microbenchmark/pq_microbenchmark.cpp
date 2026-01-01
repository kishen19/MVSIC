#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <atomic>

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
             // Quantize the first query point cloud
             auto q_query_0 = q_db.quantize_query(queries[0]);
             
             // Get the first quantized vector from this query
             if (q_query_0.vec_queries.size() > 0) {
                 auto q_vec = q_query_0.vec_queries[0];
                 
                 // Get the first database point cloud
                 auto db_cloud = q_db[0];
                 
                 // Get the first quantized vector from this DB cloud
                 if (db_cloud.size() > 0) {
                     auto db_vec = db_cloud[0];
                     
                     // Compute distance
                     float dist = q_vec.distance(db_vec);
                     std::cout << "  Quantized Distance (Query[0].Vec[0] <-> DB[0].Vec[0]): " << dist << std::endl;

                     // Compute true distance
                     auto true_q_vec = queries[0][0];
                     auto true_db_vec = points[0][0];
                     float true_dist = true_q_vec.distance(true_db_vec);
                     std::cout << "  True Distance      (Query[0].Vec[0] <-> DB[0].Vec[0]): " << true_dist << std::endl;
                 } else {
                     std::cout << "  DB Cloud[0] is empty." << std::endl;
                 }
             } else {
                 std::cout << "  Query Cloud[0] is empty." << std::endl;
             }
        }

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
                       "[-m <blocks>] [-k <clusters>] [-s <subsample>]");

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