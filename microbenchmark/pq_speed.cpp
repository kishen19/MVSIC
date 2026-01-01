#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <atomic>
#include <algorithm>
#include <unordered_set>
#include <utility>
#include <type_traits>
#include <iomanip>

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

template<typename QuantizedSet, typename PC, typename ChPoint>
void run_speed_bench(mvsic::commandLine &P, const QuantizedSet& q_db, const PC& points, const PC& queries) {
    size_t n_queries = queries.total_size();
    size_t n_db = points.total_size();
    uint32_t dims = points.get_dims();
    
    std::cout << "\nPreparing for Speed Benchmark..." << std::endl;
    std::cout << "Total Database Vectors: " << n_db << std::endl;
    std::cout << "Total Query Vectors: " << n_queries << std::endl;

    // 1. Flatten Queries (Unquantized)
    // We use a vector of the specific Point type (L2_Point or IP_Point)
    using VecType = decltype(std::declval<ChPoint>()[0]);
    std::vector<VecType> flat_queries;
    flat_queries.reserve(n_queries);
    for(size_t i=0; i<queries.size(); ++i) {
        for(size_t j=0; j<queries[i].size(); ++j) {
            flat_queries.push_back(queries[i][j]);
        }
    }
    
    // 2. Flatten Queries (Quantized)
    using QQueryType = typename std::remove_reference<decltype(q_db.quantize_query(queries[0]).vec_queries[0])>::type;
    std::vector<QQueryType> flat_q_queries;
    flat_q_queries.reserve(n_queries);
    
    // Pre-quantize queries
    for(size_t i=0; i<queries.size(); ++i) {
        auto q_query_cloud = q_db.quantize_query(queries[i]);
        for(const auto& q_vec : q_query_cloud.vec_queries) {
            flat_q_queries.push_back(q_vec);
        }
    }

    // Benchmark Parameters
    // We want at least ~100M comparisons to get a stable reading
    size_t target_ops = 200000000; 
    size_t limit_q = flat_queries.size();
    
    // Adjust limit_q if n_db is large enough to avoid running too long
    if (n_db > 0) {
        size_t needed_q = std::max((size_t)1, target_ops / n_db);
        if (needed_q < limit_q) limit_q = needed_q;
    }
    
    std::cout << "Benchmarking with " << limit_q << " queries against " << n_db << " database vectors." << std::endl;
    std::cout << "Total comparisons per run: " << (limit_q * n_db) << std::endl;

    // --- 1. Unquantized Speed ---
    std::cout << "\n--- Unquantized Distance Speed ---" << std::endl;
    {
        float* db_data = points.data();
        parlay::internal::timer t;
        
        // Warmup
        for(size_t i=0; i<std::min(limit_q, (size_t)10); ++i) {
             const auto& q = flat_queries[i];
             float sum = 0;
             for(size_t j=0; j<std::min(n_db, (size_t)1000); ++j) {
                 VecType db_vec(db_data + j * dims, dims, dims, j);
                 sum += q.distance(db_vec);
             }
             volatile float v = sum; (void)v;
        }

        t.start();
        std::atomic<size_t> dummy_counter(0);
        
        for(size_t i=0; i<limit_q; ++i) {
             const auto& q = flat_queries[i];
             float local_sum = 0;
             for(size_t j=0; j<n_db; ++j) {
                 VecType db_vec(db_data + j * dims, dims, dims, j);
                 local_sum += q.distance(db_vec);
             }
             if (local_sum > 1e10) dummy_counter++; 
        }
        
        double elapsed = t.next_time();
        double ops = (double)limit_q * n_db;
        double qps = ops / elapsed;
        
        std::cout << "Time: " << elapsed << " s" << std::endl;
        std::cout << "Throughput: " << std::fixed << std::setprecision(2) << (qps / 1e6) << " M ops/sec" << std::endl;
        std::cout << "Latency: " << (elapsed * 1e9 / ops) << " ns/op" << std::endl;
    }

    // --- 2. Quantized Speed ---
    std::cout << "\n--- Quantized Distance Speed ---" << std::endl;
    {
        auto& quantizer = q_db.vec_quantizer;
        parlay::internal::timer t;
        
        // Warmup
        for(size_t i=0; i<std::min(limit_q, (size_t)10); ++i) {
             const auto& q = flat_q_queries[i];
             float sum = 0;
             for(size_t j=0; j<std::min(n_db, (size_t)1000); ++j) {
                 sum += q.distance(quantizer[j]);
             }
             volatile float v = sum; (void)v;
        }

        t.start();
        std::atomic<size_t> dummy_counter(0);

        for(size_t i=0; i<limit_q; ++i) {
             const auto& q = flat_q_queries[i];
             float local_sum = 0;
             for(size_t j=0; j<n_db; ++j) {
                 local_sum += q.distance(quantizer[j]);
             }
             if (local_sum > 1e10) dummy_counter++;
        }

        double elapsed = t.next_time();
        double ops = (double)limit_q * n_db;
        double qps = ops / elapsed;
        
        std::cout << "Time: " << elapsed << " s" << std::endl;
        std::cout << "Throughput: " << std::fixed << std::setprecision(2) << (qps / 1e6) << " M ops/sec" << std::endl;
        std::cout << "Latency: " << (elapsed * 1e9 / ops) << " ns/op" << std::endl;
        
        // Calculate Speedup
        // Note: Can't calculate speedup relative to unquantized easily here without saving previous result
        // but user can see it.
    }
}

template<typename ChPoint>
void run_speed_main(mvsic::commandLine &P) {
    using PC = PointCloudSet<ChPoint>;

    char *inFile = P.getOptionValue("-i");
    if (inFile == nullptr) {
        std::cout << "Error: Input file not specified. Use -i <file>" << std::endl;
        return;
    }
    bool is_mmap = P.getOption("-mm");

    std::cout << "Loading PointCloud from: " << inFile << " (mmap: " << is_mmap << ")" << std::endl;
    auto points = PC(inFile, is_mmap);
    
    PC queries;
    char *qFile = P.getOptionValue("-q");
    if (qFile == nullptr) {
        std::cout << "Error: Query file not specified for speed bench. Use -q <file>" << std::endl;
        return;
    }
    std::cout << "Loading Query PointCloud from: " << qFile << std::endl;
    queries = PC(qFile);

    bool use_rabitq = P.getOption("-rabitq");
    constexpr bool Metric = ChPoint::is_metric();

    if (use_rabitq) {
        uint32_t rbits = P.getOptionIntValue("-rbits", 8);
        std::cout << "Training RaBitQ with bits=" << rbits << std::endl;
        
        using RaBitQ_Range = rabitq::Quantized_Point_Range<FlattenedPCRange<PC>, Metric>;
        using RaBitQ_Set = Quantized_Point_Cloud_Set<RaBitQ_Range, Metric>;
        
        RaBitQ_Set q_db(points, rbits);
        run_speed_bench<RaBitQ_Set, PC, ChPoint>(P, q_db, points, queries);

    } else {
        uint32_t m = P.getOptionIntValue("-m", 8);
        uint32_t k = P.getOptionIntValue("-k", 256);
        uint32_t s = P.getOptionIntValue("-s", 20);
        std::cout << "Training PQ with m=" << m << ", k=" << k << ", subsample_mult=" << s << std::endl;

        using PQ_Range = pq::Quantized_Point_Range<FlattenedPCRange<PC>, Metric>;
        using PQ_PCSet = Quantized_Point_Cloud_Set<PQ_Range, Metric>;

        PQ_PCSet q_db(points, m, k, s);
        run_speed_bench<PQ_PCSet, PC, ChPoint>(P, q_db, points, queries);
    }
}

int main(int argc, char** argv) {
    mvsic::commandLine P(argc, argv,
                       "[-i <inFile>] [-q <qFile>] [-mm] [-dist_func <IP|L2>] "
                       "[-m <blocks>] [-k <clusters>] [-s <subsample>] "
                       "[-rabitq] [-rbits <bits>]");

    std::string df = P.getOptionValue("-dist_func", "IP");

    if (df == "L2") {
        std::cout << "Using L2 distance" << std::endl;
        using ChPoint = ChamferL2_Point;
        run_speed_main<ChPoint>(P);
    } else if (df == "IP") {
        std::cout << "Using IP distance" << std::endl;
        using ChPoint = ChamferIP_Point;
        run_speed_main<ChPoint>(P);
    } else {
        std::cout << "Unknown distance function: " << df << ". using IP." << std::endl;
        using ChPoint = ChamferIP_Point;
        run_speed_main<ChPoint>(P);
    }
    
    return 0;
}
