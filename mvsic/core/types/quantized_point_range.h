#pragma once

#include <fstream>
#include <iostream>
#include <string>

#include "parlay/primitives.h"
// #include "scann/proto/centers.pb.h"
#include "mvsic/core/utils/mmap.h"

#include "mvsic/core/types/pq_helper.h"
#include "algorithms/utils/point_range.h"

namespace mvsic {

// Forward declaration
template<typename Point>
class QuantizedPoint;

// Proxy object for a single quantized vector in the database
struct QuantizedVectorRef {
  const uint8_t* codes;
  size_t num_blocks;
  bool metric;

  QuantizedVectorRef(const uint8_t* c, size_t nb, bool is_metric);

  template<typename Point>
  float distance(const QuantizedPoint<Point>& query) const;

  // Functions used by beam_search
  template<typename Point>
  bool same_as(const QuantizedPoint<Point>& q) const {
    return false;
  }
  bool same_as(const QuantizedVectorRef& q) const { return false; }
  void prefetch() const {}
  bool is_metric() const { return metric; }
};

/* ==============================Product Quantized Point Range========================== */

template<typename Point>
class QuantizedPointRange {
 private:
  uint32_t n_;
  uint32_t dims_;
  scann_pq::ScannPQConfig config_;
  scann_pq::PQResult pq_result_;

 public:
  QuantizedPointRange() noexcept : n_(0), dims_(0) {}

  // Constructor: Trains and encodes a PointRange
  QuantizedPointRange(const parlayANN::PointRange<Point>& pr, uint32_t num_blocks,
                      uint32_t num_clusters_per_block, uint32_t sample_size = 100000);

  // Constructor: Loads a quantized point range from a file
  explicit QuantizedPointRange(const std::string& filename);

  // Constructor: Loads a quantized point range from a stream
  explicit QuantizedPointRange(std::istream& in);

  // Save the quantized point range to a file
  void save(const std::string& filename) const;

  // Save the quantized point range to a stream
  void save(std::ostream& out) const;

  // Returns number of points
  inline uint32_t size() const noexcept { return n_; }
  // Returns embedding dimension
  inline uint32_t get_dims() const noexcept { return dims_; }
  // Get the PQ config
  inline const scann_pq::ScannPQConfig& get_config() const { return config_; }
  // Returns the PQResult Object
  inline const scann_pq::PQResult& get_pq_result() const { return pq_result_; }

  // Returns a proxy object for the i-th quantized vector
  inline QuantizedVectorRef operator[](size_t i) const {
    const size_t num_blocks = pq_result_.model->num_blocks();
    const uint8_t* vec_start = pq_result_.hashed_dataset.data().data() + i * num_blocks;
    return QuantizedVectorRef(vec_start, num_blocks, Point::is_metric());
  }

  static inline constexpr bool is_metric() noexcept { return Point::is_metric(); }
};

/* =======================================Implementation======================================= */

template<typename Point>
QuantizedPointRange<Point>::QuantizedPointRange(const parlayANN::PointRange<Point>& pr,
                                                uint32_t num_blocks,
                                                uint32_t num_clusters_per_block,
                                                uint32_t sample_size) :
    n_(pr.size()), dims_(pr.get_dims()) {

  scann_pq::ScannPQConfig config;
  config.mutable_projection()->set_projection_type(
      research_scann::ProjectionConfig::IDENTITY_CHUNK);
  config.mutable_projection()->set_num_blocks(num_blocks);
  config.mutable_projection()->set_input_dim(pr.get_dims());
  config.set_num_clusters_per_block(num_clusters_per_block);
  config.set_expected_sample_size(sample_size);

  if (Point::is_metric()) {
    config.mutable_quantization_distance()->set_distance_measure("SquaredL2Distance");
  } else {
    config.mutable_quantization_distance()->set_distance_measure("DotProductDistance");
  }

  auto pq_result_or = scann_pq::train_and_encode_pq(pr, config);
  if (!pq_result_or.ok()) {
    std::cerr << "Failed to train and encode PQ model: " << pq_result_or.status() << std::endl;
    throw std::runtime_error("Failed to train and encode PQ model.");
  }
  pq_result_ = std::move(pq_result_or).value();
  config_ = config;  // Store the constructed config
}

template<typename Point>
QuantizedPointRange<Point>::QuantizedPointRange(const std::string& filename) {
  std::ifstream in(filename, std::ios::binary);
  if (!in) {
    throw std::runtime_error("Failed to open file for loading: " + filename);
  }
  *this = QuantizedPointRange<Point>(in);
}

template<typename Point>
QuantizedPointRange<Point>::QuantizedPointRange(std::istream& in) {
  in.read(reinterpret_cast<char*>(&n_), sizeof(n_));
  in.read(reinterpret_cast<char*>(&dims_), sizeof(dims_));

  // Load config
  size_t config_size;
  in.read(reinterpret_cast<char*>(&config_size), sizeof(config_size));
  std::string config_str(config_size, '\0');
  in.read(&config_str[0], config_size);
  if (!config_.ParseFromString(config_str)) {
    throw std::runtime_error("Failed to parse ScannPQConfig from file.");
  }

  // Load model
  size_t model_proto_size;
  in.read(reinterpret_cast<char*>(&model_proto_size), sizeof(model_proto_size));
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
  in.read(reinterpret_cast<char*>(&num_points), sizeof(num_points));
  in.read(reinterpret_cast<char*>(&num_dims), sizeof(num_dims));
  std::vector<uint8_t> hashed_data(num_points * num_dims);
  in.read(reinterpret_cast<char*>(hashed_data.data()), hashed_data.size() * sizeof(uint8_t));
  pq_result_.hashed_dataset = scann_pq::ScannHashedDataset(std::move(hashed_data), num_points);
  pq_result_.hashed_dataset.set_dimensionality(num_dims);
}

template<typename Point>
void QuantizedPointRange<Point>::save(const std::string& filename) const {
  std::ofstream out(filename, std::ios::binary);
  if (!out) {
    throw std::runtime_error("Failed to open file for saving: " + filename);
  }
  save(out);
}

template<typename Point>
void QuantizedPointRange<Point>::save(std::ostream& out) const {
  out.write(reinterpret_cast<const char*>(&n_), sizeof(n_));
  out.write(reinterpret_cast<const char*>(&dims_), sizeof(dims_));

  // Save config
  std::string config_str;
  if (!config_.SerializeToString(&config_str)) {
    throw std::runtime_error("Failed to serialize ScannPQConfig.");
  }
  size_t config_size = config_str.size();
  out.write(reinterpret_cast<const char*>(&config_size), sizeof(config_size));
  out.write(config_str.c_str(), config_size);

  // Save model
  research_scann::CentersForAllSubspaces model_proto = pq_result_.model->ToProto();
  std::string model_proto_str;
  if (!model_proto.SerializeToString(&model_proto_str)) {
    throw std::runtime_error("Failed to serialize model proto.");
  }
  size_t model_proto_size = model_proto_str.size();
  out.write(reinterpret_cast<const char*>(&model_proto_size), sizeof(model_proto_size));
  out.write(model_proto_str.c_str(), model_proto_size);

  // Save hashed dataset
  size_t num_points = pq_result_.hashed_dataset.size();
  size_t num_dims = pq_result_.hashed_dataset.dimensionality();
  out.write(reinterpret_cast<const char*>(&num_points), sizeof(num_points));
  out.write(reinterpret_cast<const char*>(&num_dims), sizeof(num_dims));
  out.write(reinterpret_cast<const char*>(pq_result_.hashed_dataset.data().data()),
            num_points * num_dims * sizeof(uint8_t));
}

}  // namespace mvsic
