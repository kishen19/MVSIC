#pragma once

#include "mvsic/core/utils/point_cloud_set.h"
#include "scann/data_format/dataset.h"
#include "scann/hashes/asymmetric_hashing2/indexing.h"
#include "scann/hashes/asymmetric_hashing2/querying.h"
#include "scann/hashes/asymmetric_hashing2/training.h"
#include "scann/distance_measures/distance_measure_factory.h"
#include "scann/proto/scann.pb.h"
#include "scann/utils/types.h"
#include <iostream>

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
// 1. Create a `ScannPQConfig` object. A key parameter is `num_blocks`, which
//    is the number of subspaces to split the vectors into. The vector dimension
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

  // Create a std::vector and copy the data
  std::vector<float> data_vec(all_vectors, all_vectors + total_vectors * dims);

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
  // Train the model
  SCANN_ASSIGN_OR_RETURN(auto model_unique_ptr,
                         research_scann::asymmetric_hashing2::TrainSingleMachine(dataset, opts));
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
  // Index the dataset and return
  return indexer.HashDataset(dataset);
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

// Creates a lookup table for a given query vector for distance calculations.
template<typename Point>
research_scann::StatusOr<ScannLookupTable> create_lookup_table(const Point& query,
                                                               const ScannPQQueryer& queryer) {
  auto query_dptr = research_scann::MakeDatapointPtr(query.data(), query.get_dims());
  return queryer.CreateLookupTable<float>(query_dptr);
}

}  // namespace scann_pq
}  // namespace mvsic
