#pragma once

#include <iostream>

#include <Eigen/Dense>
#include "scann/data_format/dataset.h"
#include "scann/hashes/asymmetric_hashing2/indexing.h"
#include "scann/hashes/asymmetric_hashing2/querying.h"
#include "scann/hashes/asymmetric_hashing2/training.h"
#include "scann/distance_measures/distance_measure_factory.h"
#include "scann/proto/scann.pb.h"
#include "scann/utils/types.h"
#include "scann/oss_wrappers/scann_threadpool.h"
#include "algorithms/utils/point_range.h"

#include "point_cloud_set.h"

namespace mvsic {
namespace scann_pq {

// Type aliases for convenience
using ScannDenseDataset = research_scann::DenseDataset<float>;           // Scann's float dataset
using ScannPQModel = research_scann::asymmetric_hashing2::Model<float>;  // PQ training model
using ScannPQTrainingOptions =
    research_scann::asymmetric_hashing2::TrainingOptions<float>;             // PQ Training options
using ScannPQConfig = research_scann::AsymmetricHasherConfig;                // PQ configuration
using ScannPQIndexer = research_scann::asymmetric_hashing2::Indexer<float>;  // PQ Indexer
using ScannHashedDataset =
    research_scann::DenseDataset<uint8_t>;  // Hashed dataset for storing the encodings
using ScannPQQueryer = research_scann::asymmetric_hashing2::AsymmetricQueryer<float>;  // PQ Queryer
using ScannLookupTable =
    research_scann::asymmetric_hashing2::LookupTable;  // Lookup table for fast distance computation

// Holds the results of PQ training and encoding.
struct PQResult {
  // The trained PQ model, which includes the codebooks.
  std::shared_ptr<const ScannPQModel> model;
  // The dataset, with each vector compressed into a sequence of codes.
  ScannHashedDataset hashed_dataset;
};

// --- Workflow -- -
// 1. Create a `ScannPQConfig` object. A key parameter is `num_blocks`,
//    which is the number of subspaces to split the vectors into. The vector dimension
//    must be divisible by `num_blocks`.
// 2. Call `train_and_encode_pq` with a `PointCloudSet` and the config.
// 3. The returned `PQResult` struct contains the trained model and the
//    encoded dataset.

// Converts a PointCloudSet to a Scann DenseDataset by flattening all vectors.
template<typename ChPoint>
std::unique_ptr<ScannDenseDataset> point_cloud_set_to_scann_dataset(
    const PointCloudSet<ChPoint>& pcs) {
  const size_t total_vectors = pcs.total_size();
  const size_t dims = pcs.get_dims();
  const float* all_vectors = pcs.data();

  // Create a std::vector and copy the data in parallel
  std::vector<float> data_vec(total_vectors * dims);
  parlay::parallel_for(0, total_vectors * dims,
                       [&](size_t i) { data_vec[i] = all_vectors[i]; });

  // Use the DenseDataset constructor that takes an rvalue reference to a vector
  auto dataset = std::make_unique<ScannDenseDataset>(std::move(data_vec), total_vectors);
  dataset->set_dimensionality(dims);
  return dataset;
}

// Trains a PQ model on the given dataset.
// Returns a shared_ptr to the model for use in subsequent steps.
research_scann::StatusOr<std::shared_ptr<const ScannPQModel>> train_pq(
    const ScannDenseDataset& dataset, const ScannPQConfig& config) {
  // Get the quantization distance measure
  SCANN_ASSIGN_OR_RETURN(
      std::shared_ptr<const research_scann::DistanceMeasure> quantization_distance,
      research_scann::GetDistanceMeasure(config.quantization_distance()));
  // Set up training options
  ScannPQTrainingOptions opts(config, quantization_distance, dataset);

  // Create a thread pool for parallel training if num_cpus > 1
  std::shared_ptr<research_scann::ThreadPool> pool = nullptr;
  if (config.num_cpus() > 1) {
    pool = std::make_shared<research_scann::ThreadPool>("pq-training", config.num_cpus());
  }

  // Train the model
  SCANN_ASSIGN_OR_RETURN(
      auto model_unique_ptr,
      research_scann::asymmetric_hashing2::TrainSingleMachine(dataset, opts, pool));
  // Return as shared_ptr
  return std::shared_ptr<const ScannPQModel>(std::move(model_unique_ptr));
}

// Hashes the dataset using the trained PQ model.
research_scann::StatusOr<ScannHashedDataset> hash_dataset(const ScannDenseDataset& dataset,
                                                          std::shared_ptr<const ScannPQModel> model,
                                                          const ScannPQConfig& config) {
  // Get the projection proto from the config
  SCANN_ASSIGN_OR_RETURN(auto projector, model->GetProjection(config.projection()));
  // Get the quantization distance measure proto from the config
  SCANN_ASSIGN_OR_RETURN(
      std::shared_ptr<const research_scann::DistanceMeasure> quantization_distance,
      research_scann::GetDistanceMeasure(config.quantization_distance()));
  // Create the indexer
  ScannPQIndexer indexer(projector, quantization_distance, model);

  // Parallelize the hashing process
  const size_t num_datapoints = dataset.size();
  const size_t hashed_dimensionality = indexer.hashed_space_bytes();
  parlay::sequence<uint8_t> hashed_data(num_datapoints * hashed_dimensionality);

  parlay::parallel_for(
      0, num_datapoints,
      [&](size_t i) {
        auto dest_span =
            absl::MakeSpan(hashed_data.data() + i * hashed_dimensionality, hashed_dimensionality);
        auto status = indexer.Hash(dataset[i], dest_span);
        if (!status.ok()) {
          // This might not be perfectly thread-safe, but it's for error reporting
          std::cerr << "Hashing failed for datapoint " << i << ": " << status << std::endl;
        }
      },
      16);

  // Create the hashed dataset from the parallel-computed data
  std::vector<uint8_t> hashed_vector(hashed_data.begin(), hashed_data.end());
  ScannHashedDataset hashed_dataset(std::move(hashed_vector), num_datapoints);
  hashed_dataset.set_dimensionality(hashed_dimensionality);

  return hashed_dataset;
}

// Wrapper function to train and encode a point_cloud_set using PQ.
template<typename ChPoint>
research_scann::StatusOr<PQResult> train_and_encode_pq(const PointCloudSet<ChPoint>& pcs,
                                                       const ScannPQConfig& config) {
  // 1. Convert PointCloudSet to Scann's dataset format
  std::unique_ptr<ScannDenseDataset> dataset = point_cloud_set_to_scann_dataset(pcs);

  // 2. Train the PQ model
  auto model_or = train_pq(*dataset, config);
  if (!model_or.ok()) {
    std::cerr << "train_pq failed: " << model_or.status() << std::endl;
    return model_or.status();
  }
  auto model = std::move(model_or).value();

  // 3. Hash the dataset with the trained model
  auto hashed_dataset_or = hash_dataset(*dataset, model, config);
  if (!hashed_dataset_or.ok()) {
    std::cerr << "hash_dataset failed: " << hashed_dataset_or.status() << std::endl;
    return hashed_dataset_or.status();
  }
  auto hashed_dataset = std::move(hashed_dataset_or).value();

  // 4. Return the results
  return PQResult{model, std::move(hashed_dataset)};
}

// Converts a PointRange to a Scann DenseDataset.
template<typename Point>
std::unique_ptr<ScannDenseDataset> to_dense_dataset(const parlayANN::PointRange<Point>& pr) {
  const size_t num_points = pr.size();
  const size_t dims = pr.dimension();
  std::vector<float> data_vec(num_points * dims);
  parlay::parallel_for(0, num_points, [&](size_t i) {
    auto point = pr[i];
    for (size_t j = 0; j < dims; ++j) {
      data_vec[i * dims + j] = point[j];
    }
  });
  auto dataset = std::make_unique<ScannDenseDataset>(std::move(data_vec), num_points);
  dataset->set_dimensionality(dims);
  return dataset;
}

// Wrapper function to train and encode a point_range using PQ.
template<typename Point>
research_scann::StatusOr<PQResult> train_and_encode_pq(const parlayANN::PointRange<Point>& pr,
                                                       const ScannPQConfig& config) {
  // 1. Convert PointRange to Scann's dataset format
  std::unique_ptr<ScannDenseDataset> dataset = to_dense_dataset(pr);

  // 2. Train the PQ model
  auto model_or = train_pq(*dataset, config);
  if (!model_or.ok()) {
    std::cerr << "train_pq failed: " << model_or.status() << std::endl;
    return model_or.status();
  }
  auto model = std::move(model_or).value();

  // 3. Hash the dataset with the trained model
  auto hashed_dataset_or = hash_dataset(*dataset, model, config);
  if (!hashed_dataset_or.ok()) {
    std::cerr << "hash_dataset failed: " << hashed_dataset_or.status() << std::endl;
    return hashed_dataset_or.status();
  }
  auto hashed_dataset = std::move(hashed_dataset_or).value();

  // 4. Return the results
  return PQResult{model, std::move(hashed_dataset)};
}

// Creates a lookup table for a given query vector for distance calculations.
template<typename Point>
research_scann::StatusOr<ScannLookupTable> create_lookup_table(const Point& query,
                                                               const ScannPQQueryer& queryer) {
  auto query_dptr = research_scann::MakeDatapointPtr(query.data(), query.get_dims());
  return queryer.CreateLookupTable<float>(query_dptr);
}

// Creates lookup tables for a batch of query *point clouds* using Eigen for optimization.
template<typename ChPoint>
inline parlay::sequence<parlay::sequence<float>> create_lookup_tables_batched_eigen(
    const ChPoint& query,
    std::shared_ptr<const research_scann::ChunkingProjection<float>> projector,
    research_scann::ConstSpan<research_scann::DenseDataset<float>> codebooks) {
  const size_t num_points_in_query = query.size();
  if (num_points_in_query == 0) {
    return parlay::sequence<parlay::sequence<float>>();
  }

  // Project all query points
  parlay::sequence<research_scann::ChunkedDatapoint<float>> projected_points(num_points_in_query);
  parlay::parallel_for(0, num_points_in_query, [&](size_t i) {
    auto point_dptr = research_scann::MakeDatapointPtr(query[i].data(), query[i].get_dims());
    projector->ProjectInput(point_dptr, &projected_points[i]);
  });

  const size_t num_blocks = codebooks.size();
  const size_t num_centers_per_block = codebooks[0].size();

  parlay::sequence<parlay::sequence<float>> lookup_tables(num_points_in_query);
  parlay::parallel_for(0, num_points_in_query, [&](size_t i) {
    lookup_tables[i].resize(num_blocks * num_centers_per_block);
  });

  // Iterate over each block (subspace) in parallel
  parlay::parallel_for(0, num_blocks, [&](size_t block_idx) {
    // Get the codebook for the current block
    const auto& centers_ds = codebooks[block_idx];
    const size_t sub_dimensionality = centers_ds.dimensionality();

    // Create an Eigen matrix to hold the centers (codebook entries) for this block
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> centers_mat(
        num_centers_per_block, sub_dimensionality);
    // Create an Eigen vector to hold the squared norms of the centers, if using metric distance
    Eigen::VectorXf centers_sq_norms(num_centers_per_block);

    // Populate centers_mat and centers_sq_norms
    for (size_t i = 0; i < num_centers_per_block; ++i) {
      Eigen::Map<const Eigen::VectorXf> center_vec(centers_ds[i].values(), sub_dimensionality);
      centers_mat.row(i) = center_vec;
      if (ChPoint::is_metric()) {
        centers_sq_norms(i) = center_vec.squaredNorm();
      }
    }

    // Create an Eigen matrix to hold the projected query chunks for this block
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> query_chunks_mat(
        num_points_in_query, sub_dimensionality);
    // Create an Eigen vector to hold the squared norms of the query chunks, if using metric
    // distance
    Eigen::VectorXf query_chunks_sq_norms(num_points_in_query);

    // Populate query_chunks_mat and query_chunks_sq_norms
    for (size_t point_idx = 0; point_idx < num_points_in_query; ++point_idx) {
      const auto& query_chunk = projected_points[point_idx][block_idx];
      Eigen::Map<const Eigen::VectorXf> query_vec(query_chunk.values(), sub_dimensionality);
      query_chunks_mat.row(point_idx) = query_vec;
      if (ChPoint::is_metric()) {
        query_chunks_sq_norms(point_idx) = query_vec.squaredNorm();
      }
    }

    // Calculate block distances based on the distance metric
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> block_distances;
    if (ChPoint::is_metric()) {
      // For L2 distance: (a-b)^2 = a^2 - 2ab + b^2
      // Here, we compute -2ab, then add a^2 and b^2
      block_distances = -2 * query_chunks_mat * centers_mat.transpose();
      block_distances.rowwise() += centers_sq_norms.transpose();
      block_distances.colwise() += query_chunks_sq_norms;
    } else {
      // For Inner Product distance: -IP(a,b)
      block_distances = -1 * (query_chunks_mat * centers_mat.transpose());
    }

    // Populate the lookup tables for each query point with the calculated block distances
    parlay::parallel_for(0, num_points_in_query, [&](size_t point_idx) {
      // Map a segment of the lookup_tables sequence to an Eigen vector
      // and assign the row of block_distances corresponding to the current query point.
      // This effectively copies the distances for this block and query point into the lookup table.
      Eigen::Map<Eigen::VectorXf>(&lookup_tables[point_idx][block_idx * num_centers_per_block],
                                  num_centers_per_block) = block_distances.row(point_idx);
    });
  });
  // End of parallel loop over blocks

  return lookup_tables;
}

}  // namespace scann_pq
}  // namespace mvsic
