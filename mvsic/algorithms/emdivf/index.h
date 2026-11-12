#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

#include "mvsic/core/types/chamfer_ip_point.h"
#include "mvsic/core/types/point_cloud_set.h"

namespace emdivf {

using CloudSet = mvsic::PointCloudSet<mvsic::ChamferIP_Point>;

struct Neighbor {
  uint32_t document;
  float distance;
};

struct Index {
  uint32_t dimension = 0;
  uint32_t corpus_size = 0;
  uint32_t train_size = 0;
  uint32_t max_bucket_size = 0;
  uint64_t seed = 0;
  bool random_centers = false;
  std::vector<uint32_t> centers;
  std::vector<std::vector<Neighbor>> buckets;
  // Timings are cumulative, indexed by the number of selected centers minus one.
  std::vector<double> selection_seconds;
  std::vector<double> bucket_seconds;
  std::vector<double> cover_radius;
  std::vector<uint32_t> train_nearest_center;
  std::vector<double> train_distance;
};

struct AssignmentStats {
  std::vector<uint32_t> assigned_center;
  std::vector<double> distance;
  std::vector<uint32_t> cluster_size;
  std::vector<double> cluster_radius;
  double radius = 0;
  double mean_distance = 0;
  double median_distance = 0;
};

Index Build(const CloudSet& train, const CloudSet& corpus, uint32_t m,
            uint32_t max_bucket_size, uint64_t seed, bool random_centers);
AssignmentStats AssignTrainingQueries(const CloudSet& train, const Index& index, uint32_t m);
void Save(const Index& index, const std::filesystem::path& path);
Index Load(const std::filesystem::path& path);

}  // namespace emdivf
