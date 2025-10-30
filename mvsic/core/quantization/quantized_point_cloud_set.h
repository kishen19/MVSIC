#pragma once

#include <fstream>
#include <iostream>
#include <string>

#include "parlay/primitives.h"
#include "scann/proto/centers.pb.h"
#include "mmap.h"
#include "mvsic/core/utils/point_cloud_set.h"
#include "mvsic/core/utils/pq_helper.h"

namespace mvsic {

/* ==============================Product Quantized Point Clouds========================== */

template<typename ChPoint>
class QuantizedPointCloudSet {
 private:
  uint32_t n_;
  uint32_t dims_;
  parlay::sequence<size_t> offsets_;
  parlay::sequence<uint32_t> ids_;
  scann_pq::ScannPQConfig config_;
  scann_pq::PQResult pq_result_;

 public:
  QuantizedPointCloudSet() noexcept : n_(0), dims_(0) {}

  // Constructor: Trains and encodes a PointCloudSet
  QuantizedPointCloudSet(const PointCloudSet<ChPoint> &pcs, uint32_t num_blocks,
                         uint32_t num_clusters_per_block, uint32_t sample_size = 100000);

  // Constructor: Loads a quantized point cloud set from a file
  explicit QuantizedPointCloudSet(const std::string &filename);

  // Constructor: Loads a quantized point cloud set from a stream
  explicit QuantizedPointCloudSet(std::istream &in);

  // Copy constructor
  QuantizedPointCloudSet(const QuantizedPointCloudSet &other) :
      n_(other.n_),
      dims_(other.dims_),
      offsets_(other.offsets_),
      ids_(other.ids_),
      config_(other.config_) {
    pq_result_.model = other.pq_result_.model;
    if (other.pq_result_.hashed_dataset.size() > 0) {
      pq_result_.hashed_dataset = other.pq_result_.hashed_dataset.Copy();
    }
  }

  // Copy assignment operator
  QuantizedPointCloudSet &operator=(const QuantizedPointCloudSet &other) {
    if (this == &other) {
      return *this;
    }
    n_ = other.n_;
    dims_ = other.dims_;
    offsets_ = other.offsets_;
    ids_ = other.ids_;
    config_ = other.config_;
    pq_result_.model = other.pq_result_.model;
    pq_result_.hashed_dataset = other.pq_result_.hashed_dataset.Copy();
    return *this;
  }

  // Save the quantized point cloud set to a file
  void save(const std::string &filename) const;

  // Save the quantized point cloud set to a stream
  void save(std::ostream &out) const;

  // Returns number of point clouds
  inline uint32_t size() const noexcept { return n_; }
  // Returns total number of individual embeddings
  inline size_t total_size() const noexcept { return offsets_[n_] / dims_; }
  // Returns embedding dimension
  inline uint32_t get_dims() const noexcept { return dims_; }
  // Returns number of embeddings of pointcloud i
  inline uint32_t get_size(size_t i) const noexcept {
    return (offsets_[i + 1] - offsets_[i]) / dims_;
  }
  // Returns id of pointcloud i
  inline uint32_t get_id(size_t i) const noexcept { return (ids_.size() > 0) ? ids_[i] : i; }
  // Get the PQ config
  inline const scann_pq::ScannPQConfig &get_config() const { return config_; }
  // Returns the PQResult Object
  inline const scann_pq::PQResult &get_pq_result() const { return pq_result_; }
  // Returns non-owning sequence of offsets
  inline auto get_offsets() const noexcept {
    return parlay::make_slice(offsets_.begin(), offsets_.end());
  }
  // Returns ChPoint type object on the embeddings of point cloud i
  // inline ChPoint operator[](size_t i) const {
  //   return ChPoint(get_size(i), dims, data(i), get_id(i));
  // }
  // inline constexpr bool is_metric() const noexcept { return ChPoint::is_metric(); }
  static inline constexpr bool is_metric() noexcept { return ChPoint::is_metric(); }

  // Returns approximated distances from a query point cloud to all point clouds in the set
  inline size_t distances_old(const ChPoint &query, std::pair<uint32_t, float> *results) const;

  inline std::pair<parlay::sequence<std::pair<uint32_t, float>>, size_t> distances_old(
      const ChPoint &query) const {
    auto results = parlay::sequence<std::pair<uint32_t, float>>::uninitialized(n_);
    auto cmps = distances_old(query, results.data());
    return std::make_pair(results, cmps);
  }

  // Returns approximated distances from a query point cloud to all point clouds in the set
  inline size_t distances(const ChPoint &query, std::pair<uint32_t, float> *results) const;

  inline std::pair<parlay::sequence<std::pair<uint32_t, float>>, size_t> distances(
      const ChPoint &query) const {
    auto results = parlay::sequence<std::pair<uint32_t, float>>::uninitialized(n_);
    auto cmps = distances(query, results.data());
    return std::make_pair(results, cmps);
  }
};

/* =======================================Implementation======================================= */

template<typename ChPoint>
QuantizedPointCloudSet<ChPoint>::QuantizedPointCloudSet(const PointCloudSet<ChPoint> &pcs,
                                                        uint32_t num_blocks,
                                                        uint32_t num_clusters_per_block,
                                                        uint32_t sample_size) :
    n_(pcs.size()),
    dims_(pcs.get_dims()),
    ids_(parlay::tabulate(pcs.size(), [&](size_t i) { return pcs.get_id(i); })) {
  auto pcs_offsets = pcs.get_offsets();
  offsets_ = parlay::tabulate(pcs_offsets.size(), [&](size_t i) { return pcs_offsets[i]; });

  scann_pq::ScannPQConfig config;
  config.mutable_projection()->set_projection_type(
      research_scann::ProjectionConfig::IDENTITY_CHUNK);
  config.mutable_projection()->set_num_blocks(num_blocks);
  config.mutable_projection()->set_input_dim(pcs.get_dims());
  config.set_num_clusters_per_block(num_clusters_per_block);
  config.set_expected_sample_size(sample_size);

  if (ChPoint::is_metric()) {
    config.mutable_quantization_distance()->set_distance_measure("SquaredL2Distance");
  } else {
    config.mutable_quantization_distance()->set_distance_measure("DotProductDistance");
  }

  auto pq_result_or = scann_pq::train_and_encode_pq(pcs, config);
  if (!pq_result_or.ok()) {
    std::cerr << "Failed to train and encode PQ model: " << pq_result_or.status() << std::endl;
    throw std::runtime_error("Failed to train and encode PQ model.");
  }
  pq_result_ = std::move(pq_result_or).value();
  config_ = config;  // Store the constructed config
}

template<typename ChPoint>
QuantizedPointCloudSet<ChPoint>::QuantizedPointCloudSet(const std::string &filename) {
  std::ifstream in(filename, std::ios::binary);
  if (!in) {
    throw std::runtime_error("Failed to open file for loading: " + filename);
  }
  *this = QuantizedPointCloudSet<ChPoint>(in);
}

template<typename ChPoint>
QuantizedPointCloudSet<ChPoint>::QuantizedPointCloudSet(std::istream &in) {
  in.read(reinterpret_cast<char *>(&n_), sizeof(n_));
  in.read(reinterpret_cast<char *>(&dims_), sizeof(dims_));
  size_t offsets_size;
  in.read(reinterpret_cast<char *>(&offsets_size), sizeof(offsets_size));
  offsets_.resize(offsets_size);
  in.read(reinterpret_cast<char *>(offsets_.data()), offsets_size * sizeof(size_t));
  size_t ids_size;
  in.read(reinterpret_cast<char *>(&ids_size), sizeof(ids_size));
  ids_.resize(ids_size);
  in.read(reinterpret_cast<char *>(ids_.data()), ids_size * sizeof(uint32_t));

  // Load config
  size_t config_size;
  in.read(reinterpret_cast<char *>(&config_size), sizeof(config_size));
  std::string config_str(config_size, '\0');
  in.read(&config_str[0], config_size);
  if (!config_.ParseFromString(config_str)) {
    throw std::runtime_error("Failed to parse ScannPQConfig from file.");
  }

  // Load model
  size_t model_proto_size;
  in.read(reinterpret_cast<char *>(&model_proto_size), sizeof(model_proto_size));
  std::string model_proto_str(model_proto_size, '\0');
  in.read(&model_proto_str[0], model_proto_size);
  research_scann::CentersForAllSubspaces model_proto;
  if (!model_proto.ParseFromString(model_proto_str)) {
    throw std::runtime_error("Failed to parse CentersForAllSubspaces from file.");
  }
  auto model_or = scann_pq::ScannPQModel::FromProto(model_proto);
  if (!model_or.ok()) {
    throw std::runtime_error("Failed to deserialize ScannPQModel from proto.");
  }
  pq_result_.model = std::shared_ptr<const scann_pq::ScannPQModel>(std::move(model_or.value()));

  // Load hashed dataset
  size_t num_points, num_dims;
  in.read(reinterpret_cast<char *>(&num_points), sizeof(num_points));
  in.read(reinterpret_cast<char *>(&num_dims), sizeof(num_dims));
  std::vector<uint8_t> hashed_data(num_points * num_dims);
  in.read(reinterpret_cast<char *>(hashed_data.data()), hashed_data.size() * sizeof(uint8_t));
  pq_result_.hashed_dataset = scann_pq::ScannHashedDataset(std::move(hashed_data), num_points);
  pq_result_.hashed_dataset.set_dimensionality(num_dims);
}

template<typename ChPoint>
void QuantizedPointCloudSet<ChPoint>::save(const std::string &filename) const {
  std::ofstream out(filename, std::ios::binary);
  if (!out) {
    throw std::runtime_error("Failed to open file for saving: " + filename);
  }
  save(out);
}

template<typename ChPoint>
void QuantizedPointCloudSet<ChPoint>::save(std::ostream &out) const {
  out.write(reinterpret_cast<const char *>(&n_), sizeof(n_));
  out.write(reinterpret_cast<const char *>(&dims_), sizeof(dims_));
  size_t offsets_size = offsets_.size();
  out.write(reinterpret_cast<const char *>(&offsets_size), sizeof(offsets_size));
  out.write(reinterpret_cast<const char *>(offsets_.data()), offsets_size * sizeof(size_t));
  size_t ids_size = ids_.size();
  out.write(reinterpret_cast<const char *>(&ids_size), sizeof(ids_size));
  out.write(reinterpret_cast<const char *>(ids_.data()), ids_size * sizeof(uint32_t));

  // Save config
  std::string config_str;
  if (!config_.SerializeToString(&config_str)) {
    throw std::runtime_error("Failed to serialize ScannPQConfig.");
  }
  size_t config_size = config_str.size();
  out.write(reinterpret_cast<const char *>(&config_size), sizeof(config_size));
  out.write(config_str.c_str(), config_size);

  // Save model
  research_scann::CentersForAllSubspaces model_proto = pq_result_.model->ToProto();
  std::string model_proto_str;
  if (!model_proto.SerializeToString(&model_proto_str)) {
    throw std::runtime_error("Failed to serialize model proto.");
  }
  size_t model_proto_size = model_proto_str.size();
  out.write(reinterpret_cast<const char *>(&model_proto_size), sizeof(model_proto_size));
  out.write(model_proto_str.c_str(), model_proto_size);

  // Save hashed dataset
  size_t num_points = pq_result_.hashed_dataset.size();
  size_t num_dims = pq_result_.hashed_dataset.dimensionality();
  out.write(reinterpret_cast<const char *>(&num_points), sizeof(num_points));
  out.write(reinterpret_cast<const char *>(&num_dims), sizeof(num_dims));
  out.write(reinterpret_cast<const char *>(pq_result_.hashed_dataset.data().data()),
            num_points * num_dims * sizeof(uint8_t));
}

// Returns approximated distances from a query point cloud to all point clouds in the set
template<typename ChPoint>
inline size_t QuantizedPointCloudSet<ChPoint>::distances_old(
    const ChPoint &query, std::pair<uint32_t, float> *results) const {
  // 1. Create the Scann queryer from the trained model
  auto projector_or = pq_result_.model->GetProjection(config_.projection());
  if (!projector_or.ok()) {
    throw std::runtime_error("Failed to get projector from model.");
  }
  auto projector = projector_or.value();
  // Get the quantization distance measure
  auto quantization_distance_or =
      research_scann::GetDistanceMeasure(config_.quantization_distance());
  if (!quantization_distance_or.ok()) {
    throw std::runtime_error("Failed to get quantization distance measure.");
  }
  auto quantization_distance = quantization_distance_or.value();
  // Create the queryer
  scann_pq::ScannPQQueryer queryer(projector, quantization_distance, pq_result_.model);

  // 2. Compute Lookup tables per query point in parallel
  auto luts = parlay::tabulate(query.size(), [&](size_t j) {
    const auto &query_point = query[j];
    auto lut_or = scann_pq::create_lookup_table(query_point, queryer);
    if (!lut_or.ok()) {
      throw std::runtime_error("Failed to create lookup table for a query point.");
    }
    return std::move(lut_or).value();
  });
  size_t cmps = query.size() * dims_;                // Size of query
  cmps += queryer.num_clusters_per_block() * dims_;  // Size of lookup tables

  // 3. For each point cloud in the set, compute the approximate Chamfer distance
  const uint8_t *encoded_data = pq_result_.hashed_dataset.data().data();
  const size_t num_blocks = pq_result_.model->num_blocks();
  parlay::parallel_for(0, n_, [&](uint32_t i) {
    // Get the slice of encoded vectors for point cloud `i`
    size_t start_offset = offsets_[i] / dims_;
    size_t end_offset = offsets_[i + 1] / dims_;
    auto num_vectors_in_pc = end_offset - start_offset;

    float total_dist = 0.0f;

    // For each point in the query cloud...
    for (size_t j = 0; j < query.size(); ++j) {
      const auto &lut = luts[j];
      // 4. Find the min distance from this query point to the encoded db point cloud
      float min_dist_for_query_point = std::numeric_limits<float>::max();
      for (size_t k = 0; k < num_vectors_in_pc; ++k) {
        const uint8_t *vec_start = encoded_data + (start_offset + k) * num_blocks;
        float dist = 0.0f;
        const float *lookup_table = lut.float_lookup_table.data();
        const size_t num_clusters = queryer.num_clusters_per_block();
        for (size_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
          const uint8_t code = vec_start[block_idx];
          dist += lookup_table[block_idx * num_clusters + code];
        }
        min_dist_for_query_point = std::min(min_dist_for_query_point, dist);
      }
      total_dist += min_dist_for_query_point;
    }
    float avg_dist = total_dist / query.size();
    results[i] = std::make_pair(this->ids_[i], avg_dist);
  });
  return cmps;
}

template<typename ChPoint>
inline size_t QuantizedPointCloudSet<ChPoint>::distances(
    const ChPoint &query, std::pair<uint32_t, float> *results) const {
  // 1. Get projector from the trained model
  auto projector_or = pq_result_.model->GetProjection(config_.projection());
  if (!projector_or.ok()) {
    throw std::runtime_error("Failed to get projector from model.");
  }
  auto projector = projector_or.value();

  // 2. Compute Lookup tables for all query points in a batched fashion
  auto lookup_tables =
      scann_pq::create_lookup_tables_batched_eigen(query, projector, pq_result_.model->centers());

  size_t cmps = query.size() * dims_;                          // Size of query
  cmps += pq_result_.model->num_clusters_per_block() * dims_;  // Size of lookup tables

  // 3. For each point cloud in the set, compute the approximate Chamfer distance
  const uint8_t *encoded_data = pq_result_.hashed_dataset.data().data();
  const size_t num_blocks = pq_result_.model->num_blocks();
  const size_t num_clusters_per_block = pq_result_.model->num_clusters_per_block();

  parlay::parallel_for(
      0, n_,
      [&](uint32_t i) {
        // Get the slice of encoded vectors for point cloud `i`
        size_t start_offset = offsets_[i] / dims_;
        size_t end_offset = offsets_[i + 1] / dims_;
        auto num_vectors_in_pc = end_offset - start_offset;

        float total_dist = 0.0f;

        // For each point in the query cloud...
        // for (size_t j = 0; j < query.size(); ++j) {
        auto dists_query = parlay::sequence<float>::uninitialized(query.size());
        parlay::parallel_for(0, query.size(), [&](size_t j) {
          const auto &lut = lookup_tables[j];  // this is a sequence of floats
          // 4. Find the min distance from this query point to the encoded db point cloud
          float min_dist_for_query_point = std::numeric_limits<float>::max();
          for (size_t k = 0; k < num_vectors_in_pc; ++k) {
            const uint8_t *vec_start = encoded_data + (start_offset + k) * num_blocks;
            float dist = 0.0f;
            const float *lookup_table_ptr = lut.data();
            for (size_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
              const uint8_t code = vec_start[block_idx];
              dist += lookup_table_ptr[block_idx * num_clusters_per_block + code];
            }
            min_dist_for_query_point = std::min(min_dist_for_query_point, dist);
          }
          dists_query[j] = min_dist_for_query_point;
        });
        float avg_dist = parlay::reduce(dists_query) / query.size();
        results[i] = std::make_pair(this->ids_[i], avg_dist);
      },
      1);
  return cmps;
}

}  // namespace mvsic