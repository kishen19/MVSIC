#include <unistd.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "mvsic/algorithms/stats.h"
#include "mvsic/core/types/chamfer_ip_point.h"
#include "mvsic/core/types/point_cloud_set.h"

namespace {

static_assert(sizeof(size_t) == sizeof(uint64_t));
static_assert(sizeof(std::pair<float, uint32_t>) == 8);

template<typename T>
void Write(std::ofstream& output, const T& value) {
  output.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

class TemporaryFiles {
 public:
  TemporaryFiles() {
    auto base = std::filesystem::temp_directory_path() /
                ("mvsic_embed_format_" + std::to_string(getpid()));
    pcs = base.string() + ".pcs";
    gt = base.string() + ".gt";
    gold = base.string() + ".csr";
  }

  ~TemporaryFiles() {
    std::filesystem::remove(pcs);
    std::filesystem::remove(gt);
    std::filesystem::remove(gold);
  }

  std::string pcs;
  std::string gt;
  std::string gold;
};

TEST(EmbedFormats, NativeReadersAcceptDocumentedLayouts) {
  TemporaryFiles files;

  // Same byte sequence emitted by embed.formats.StreamingPCSWriter:
  // two 2-D clouds, with 2 vectors and 1 vector respectively. Offsets count
  // floats, so they are [0, 4, 6].
  {
    std::ofstream output(files.pcs, std::ios::binary);
    const uint64_t dim = 2, count = 2, num_vectors = 3;
    const float values[] = {1, 0, 0, 1, -1, 0};
    const uint64_t num_offsets = 3;
    const uint64_t offsets[] = {0, 4, 6};
    Write(output, dim);
    Write(output, count);
    Write(output, num_vectors);
    output.write(reinterpret_cast<const char*>(values), sizeof(values));
    Write(output, num_offsets);
    output.write(reinterpret_cast<const char*>(offsets), sizeof(offsets));
  }

  mvsic::PointCloudSet<mvsic::ChamferIP_Point> points(files.pcs.c_str());
  EXPECT_EQ(points.get_dims(), 2);
  EXPECT_EQ(points.size(), 2);
  EXPECT_EQ(points.total_size(), 3);
  EXPECT_EQ(points.get_size(0), 2);
  EXPECT_EQ(points.get_size(1), 1);

  // Exact GT: int32 k followed by (float32 distance, uint32 ID) records.
  {
    std::ofstream output(files.gt, std::ios::binary);
    const int32_t k = 2;
    const std::pair<float, uint32_t> rows[] = {
        {-1.0f, 0}, {0.0f, 1}, {-1.0f, 1}, {0.0f, 0}};
    Write(output, k);
    output.write(reinterpret_cast<const char*>(rows), sizeof(rows));
  }
  std::string gt_path = files.gt;
  auto gt = ReadGT(gt_path, 2);
  ASSERT_EQ(gt.size(), 2);
  EXPECT_EQ(gt[0][0].first, 0);
  EXPECT_FLOAT_EQ(gt[0][0].second, -1.0f);
  EXPECT_EQ(gt[1][0].first, 1);

  // Gold GT: size_t counts and offsets, followed by uint32 corpus row IDs.
  {
    std::ofstream output(files.gold, std::ios::binary);
    const uint64_t num_offsets = 3, num_entries = 3;
    const uint64_t offsets[] = {0, 2, 3};
    const uint32_t ids[] = {0, 1, 1};
    Write(output, num_offsets);
    Write(output, num_entries);
    output.write(reinterpret_cast<const char*>(offsets), sizeof(offsets));
    output.write(reinterpret_cast<const char*>(ids), sizeof(ids));
  }
  std::string gold_path = files.gold;
  auto gold = ReadGoldGT(gold_path, 2);
  ASSERT_EQ(gold.size(), 2);
  EXPECT_EQ(gold[0].size(), 2);
  EXPECT_EQ(gold[1].size(), 1);
  EXPECT_EQ(gold[0][1].first, 1);
}

}  // namespace
