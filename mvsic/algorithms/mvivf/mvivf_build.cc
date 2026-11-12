#include "mvsic/algorithms/mvivf/mvivf_internal.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "mvsic/algorithms/mvivf/mvclustering/mvclustering_internal.h"
#include "parlay/parallel.h"

namespace mvsic {
namespace {

using mvivf_internal::Node;
constexpr uint32_t kMagic = 0x4d564946;
constexpr uint32_t kVersion = 4;

// Keep recursive construction private; flatten once after all parallel work.
// Search nodes and the rows borrowed by encoded leaves then never move.
struct BuildNode {
  Node node;
  std::vector<BuildNode> children;
};

BuildNode recursive_build(const PointCloudSetView& points, std::vector<uint32_t> rows,
                         uint32_t depth, const ClusteringContext* clustering,
                         const IndexParams& params) {
  BuildNode result;
  const size_t n = rows.size();
  if (n <= params.max_leaf_size || (params.max_depth && depth >= params.max_depth)) {
    result.node.rows = std::move(rows);
    return result;
  }
  const size_t auto_nc = params.k_per_level ? params.k_per_level : size_t(std::ceil(std::sqrt(double(n))));
  const size_t small_nc = 4 * (n + params.max_leaf_size - 1) / params.max_leaf_size;
  const uint32_t k = uint32_t(std::min(n, std::max(size_t(4), std::min(auto_nc, small_nc))));
  // The root already has contiguous full-corpus storage. Only recursive
  // subsets need gathering; copying the root can otherwise double peak RAM.
  auto local = depth ? gather_point_clouds(points, rows, false) : points;
  local.ids = nullptr;
  auto clustered = clustering->cluster(local, rows, k, params.s);
  std::vector<std::vector<uint32_t>> groups(k);
  for (size_t i = 0; i < n; ++i) groups[clustered.assignments[i]].push_back(rows[i]);
  std::vector<uint32_t> active;
  for (uint32_t i = 0; i < k; ++i) if (!groups[i].empty()) active.push_back(i);
  // Identical documents can all choose the same center; another recursion
  // would not reduce the problem, even with max_depth=0.
  if (active.size() <= 1) {
    result.node.rows = std::move(rows);
    return result;
  }
  result.node.centers = gather_point_clouds(clustered.centers, active, false);
  result.children.resize(active.size());
  local = {};  // float training scratch is not part of the searchable index
  parlay::parallel_for(0, active.size(), [&](size_t i) {
    result.children[i] = recursive_build(points, std::move(groups[active[i]]), depth + 1, clustering, params);
  }, 1);
  return result;
}

uint32_t flatten(BuildNode& source, std::vector<Node>& nodes) {
  if (nodes.size() >= UINT32_MAX) throw std::length_error("MVIVF has too many nodes");
  const uint32_t id = uint32_t(nodes.size());
  nodes.push_back(std::move(source.node));
  std::vector<uint32_t> children;
  children.reserve(source.children.size());
  for (auto& child : source.children) children.push_back(flatten(child, nodes));
  nodes[id].children = std::move(children);
  return id;
}

// Check lengths against the actual file before allocating any persisted array.
class Reader {
  std::ifstream file_;
  size_t remaining_;
 public:
  explicit Reader(const std::string& filename) : file_(filename, std::ios::binary | std::ios::ate) {
    if (!file_) throw std::runtime_error("Cannot open MVIVF index: " + filename);
    auto length = file_.tellg();
    if (length < 0) throw std::runtime_error("Cannot size MVIVF index");
    remaining_ = size_t(length);
    file_.seekg(0);
  }
  template<class T> std::vector<T> array(size_t count) {
    if (count > remaining_ / sizeof(T)) throw std::runtime_error("Truncated MVIVF index");
    std::vector<T> result(count);
    const size_t bytes = count * sizeof(T);
    if (bytes && !file_.read(reinterpret_cast<char*>(result.data()), bytes))
      throw std::runtime_error("Cannot read MVIVF index");
    remaining_ -= bytes;
    return result;
  }
  template<class T> T value() { return array<T>(1)[0]; }
  void finish() const { if (remaining_) throw std::runtime_error("Trailing data in MVIVF index"); }
};

void validate_offsets(const std::vector<size_t>& offsets) {
  if (offsets.empty() || offsets.front() || !std::is_sorted(offsets.begin(), offsets.end()))
    throw std::runtime_error("Invalid MVIVF offsets");
}

}  // namespace

IndexMVIVF::Impl::Impl(uint32_t dimension, const IndexParams& configuration)
    : d(dimension), params(configuration) {
  if (!d || d > 65535) throw std::invalid_argument("MVIVF dimension must be in [1, 65535]");
  if (params.method != "mvivf" && !params.method.empty())
    throw std::invalid_argument("IndexMVIVF requires mvivf parameters");
  if (!params.max_leaf_size) throw std::invalid_argument("max_leaf_size must be positive");
  validate_clustering_config(params.mvclus);
  if (params.build_with_8btq || params.mvclus.build_with_8btq) {
    if (params.mvclus.assignment_quantizer != ClusteringQuantizer::None &&
        params.mvclus.assignment_quantizer != ClusteringQuantizer::TQ8)
      throw std::invalid_argument("build_with_8btq conflicts with assignment_quantizer");
    params.mvclus.assignment_quantizer = ClusteringQuantizer::TQ8;
  }
  if (params.compress_input) throw std::invalid_argument("MVIVF input compression is unsupported; compress the corpus explicitly");
  switch (params.pq.method) {
    case IndexParams::QuantizerType::None:
    case IndexParams::QuantizerType::OneBitTQ:
    case IndexParams::QuantizerType::TurboQuant:
    case IndexParams::QuantizerType::EightBitTQ: break;
    default: throw std::invalid_argument("Unsupported MVIVF leaf quantizer");
  }
}

void IndexMVIVF::Impl::validate_corpus(const PointCloudSetView& corpus) const {
  corpus.validate();
  if (corpus.d != d) throw std::invalid_argument("MVIVF corpus dimension mismatch");
  if (corpus.ids) {
    std::unordered_set<uint32_t> ids;
    ids.reserve(corpus.n);
    for (uint32_t i = 0; i < corpus.n; ++i) {
      if (corpus.ids[i] == UINT32_MAX) throw std::invalid_argument("UINT32_MAX is reserved for missing neighbors");
      if (!ids.insert(corpus.ids[i]).second) throw std::invalid_argument("Corpus document IDs must be unique");
    }
  }
  const bool unit = params.pq.method == IndexParams::QuantizerType::OneBitTQ;
  mvivf_internal::validate_values(corpus.values, corpus.total_size(), d, unit, "corpus");
}

void IndexMVIVF::Impl::validate_points(const PointCloudSetView& corpus) const {
  if (!kernels) throw std::logic_error("MVIVF must be built or loaded before search");
  if (corpus.n != points.n || corpus.d != d || corpus.values != points.values ||
      corpus.offsets != points.offsets || corpus.ids != points.ids)
    throw std::invalid_argument("Search corpus must be the storage bound by build or load");
}

void IndexMVIVF::Impl::prepare() {
  leaves.clear();
  height = 0;
  std::vector<std::pair<uint32_t, size_t>> stack{{0, 1}};
  while (!stack.empty()) {
    const auto [id, depth] = stack.back();
    stack.pop_back();
    height = std::max(height, depth);
    if (nodes[id].is_leaf()) leaves.push_back(id);
    else for (auto child : nodes[id].children) stack.emplace_back(child, depth + 1);
  }
  kernels = mvivf_internal::make_kernels(params, points);
  kernels->prepare_nodes(nodes, encoded);
}

IndexMVIVF::IndexMVIVF(uint32_t d, Metric metric, const IndexParams& params) {
  if (metric != Metric::IP)
    throw std::invalid_argument("MVIVF retrieval supports only IP/MaxSim; L2 is available for clustering");
  impl_ = std::make_unique<Impl>(d, params);
}
IndexMVIVF::~IndexMVIVF() = default;
IndexMVIVF::IndexMVIVF(IndexMVIVF&&) noexcept = default;
IndexMVIVF& IndexMVIVF::operator=(IndexMVIVF&&) noexcept = default;

void IndexMVIVF::build(const PointCloudSetView& points) {
  auto next = std::make_unique<Impl>(impl_->d, impl_->params);
  next->validate_corpus(points);
  next->points = points;
  std::vector<uint32_t> rows(points.n);
  std::iota(rows.begin(), rows.end(), 0);
  std::unique_ptr<ClusteringContext> clustering;
  if (points.n > next->params.max_leaf_size)
    clustering = std::make_unique<ClusteringContext>(points, Metric::IP, next->params.mvclus);
  auto root = recursive_build(points, std::move(rows), 0, clustering.get(), next->params);
  flatten(root, next->nodes);
  root = {};
  clustering.reset();
  next->prepare();
  impl_ = std::move(next);
}

void IndexMVIVF::save(const std::string& filename) const {
  if (!impl_->kernels) throw std::logic_error("MVIVF must be built before save");
  std::ofstream file(filename, std::ios::binary | std::ios::trunc);
  if (!file) throw std::runtime_error("Cannot write MVIVF index: " + filename);
  auto write = [&](const auto& value) {
    file.write(reinterpret_cast<const char*>(&value), sizeof(value));
  };
  auto array = [&](const auto& values) {
    if (!values.empty()) file.write(reinterpret_cast<const char*>(values.data()),
        values.size() * sizeof(values[0]));
  };
  write(kMagic); write(kVersion); write(uint32_t(0));
  const auto& p = impl_->params;
  write(p.k_per_level); write(p.max_leaf_size); write(p.max_depth); write(p.num_spill);
  write(p.s); write(p.mvclus.niters); write(p.mvclus.max_point_clouds_per_cluster);
  write(p.mvclus.max_points_per_centroid_inner_kmeans);
  std::vector<size_t> centers{0}, children{0}, points{0}, child_ids;
  std::vector<uint32_t> point_ids;
  for (const auto& node : impl_->nodes) {
    for (size_t j = 0; j < node.children.size(); ++j)
      centers.push_back(centers.back() + node.centers.get_size(j) * size_t(impl_->d));
    children.push_back(children.back() + node.children.size());
    child_ids.insert(child_ids.end(), node.children.begin(), node.children.end());
    points.push_back(points.back() + node.rows.size());
    for (auto row : node.rows) point_ids.push_back(impl_->points.get_id(row));
  }
  write(impl_->nodes.size()); write(centers.size()); array(centers);
  for (const auto& node : impl_->nodes)
    if (!node.is_leaf()) file.write(reinterpret_cast<const char*>(node.centers.values), node.centers.num_bytes());
  array(children); array(child_ids); array(points); array(point_ids);
  file.close();
  if (!file) throw std::runtime_error("Failed to write MVIVF index: " + filename);
}

void IndexMVIVF::load(const std::string& filename, const PointCloudSetView& points) {
  auto next = std::make_unique<Impl>(impl_->d, impl_->params);
  next->validate_corpus(points);
  next->points = points;
  Reader file(filename);
  if (file.value<uint32_t>() != kMagic || file.value<uint32_t>() != kVersion || file.value<uint32_t>())
    throw std::runtime_error("Expected an MVIVF version-4 skeleton");
  auto& p = next->params;
  p.k_per_level = file.value<uint32_t>(); p.max_leaf_size = file.value<uint32_t>();
  p.max_depth = file.value<uint32_t>(); p.num_spill = file.value<uint32_t>();
  p.s = file.value<uint32_t>(); p.mvclus.niters = file.value<uint32_t>();
  p.mvclus.max_point_clouds_per_cluster = file.value<uint32_t>();
  p.mvclus.max_points_per_centroid_inner_kmeans = file.value<uint32_t>();
  if (!p.max_leaf_size) throw std::runtime_error("Invalid persisted max_leaf_size");
  validate_clustering_config(p.mvclus);
  const size_t count = file.value<size_t>(), center_count = file.value<size_t>();
  if (!count || count > UINT32_MAX || center_count != count)
    throw std::runtime_error("Invalid MVIVF tree size");
  auto centers = file.array<size_t>(center_count);
  validate_offsets(centers);
  auto values = file.array<float>(centers.back());
  auto children = file.array<size_t>(count + 1);
  validate_offsets(children);
  if (children.back() != count - 1) throw std::runtime_error("MVIVF skeleton is not a tree");
  auto child_ids = file.array<size_t>(children.back());
  auto point_offsets = file.array<size_t>(count + 1);
  validate_offsets(point_offsets);
  if (point_offsets.back() != points.n) throw std::runtime_error("MVIVF corpus size mismatch");
  auto point_ids = file.array<uint32_t>(point_offsets.back());
  file.finish();
  std::unordered_map<uint32_t, uint32_t> id_to_row;
  if (points.ids) {
    id_to_row.reserve(points.n);
    for (uint32_t i = 0; i < points.n; ++i) id_to_row.emplace(points.get_id(i), i);
  }
  std::vector<bool> seen_rows(points.n, false), seen_nodes(count, false);
  next->nodes.resize(count);
  for (size_t i = 0; i < count; ++i) {
    auto& node = next->nodes[i];
    const size_t begin = children[i], end = children[i + 1];
    if (begin != end) {
      if (point_offsets[i + 1] != point_offsets[i]) throw std::runtime_error("Internal MVIVF node contains documents");
      std::vector<size_t> offsets(end - begin + 1);
      for (size_t j = begin; j <= end; ++j) {
        offsets[j - begin] = centers[j] - centers[begin];
        if (j < end && (centers[j + 1] <= centers[j] || (centers[j + 1] - centers[j]) % next->d))
          throw std::runtime_error("Invalid MVIVF center dimensions");
      }
      node.centers = own_point_clouds(std::vector<float>(values.begin() + centers[begin], values.begin() + centers[end]),
                                      std::move(offsets), next->d);
      mvivf_internal::validate_values(node.centers.values, node.centers.total_size(), next->d, false, "center");
      for (size_t j = begin; j < end; ++j) {
        const size_t child = child_ids[j];
        // The persisted preorder invariant also rules out cycles without a
        // recursive verifier that can overflow on a hostile file.
        if (child <= i || child >= count || seen_nodes[child]) throw std::runtime_error("Invalid MVIVF child topology");
        seen_nodes[child] = true;
        node.children.push_back(uint32_t(child));
      }
    } else {
      if (point_offsets[i + 1] == point_offsets[i]) throw std::runtime_error("Empty MVIVF leaf");
      for (size_t j = point_offsets[i]; j < point_offsets[i + 1]; ++j) {
        const uint32_t id = point_ids[j];
        uint32_t row = id;
        if (points.ids) {
          auto found = id_to_row.find(id);
          if (found == id_to_row.end()) throw std::runtime_error("MVIVF document ID is absent from corpus");
          row = found->second;
        }
        if (row >= points.n || seen_rows[row]) throw std::runtime_error("Invalid or duplicate MVIVF document ID");
        seen_rows[row] = true;
        node.rows.push_back(row);
      }
    }
  }
  next->prepare();
  impl_ = std::move(next);
}

Metric IndexMVIVF::metric() const noexcept { return Metric::IP; }
uint32_t IndexMVIVF::get_dims() const noexcept { return impl_->d; }
size_t IndexMVIVF::get_height() const noexcept { return impl_->height; }
const IndexParams& IndexMVIVF::params() const noexcept { return impl_->params; }
size_t IndexMVIVF::mean_cluster_size() const noexcept {
  return impl_->leaves.empty() ? 0 : impl_->points.n / impl_->leaves.size();
}
size_t IndexMVIVF::max_cluster_size() const noexcept {
  size_t result = 0;
  for (auto id : impl_->leaves) result = std::max(result, impl_->nodes[id].rows.size());
  return result;
}
std::vector<uint32_t> IndexMVIVF::get_flat_clustering() const {
  std::vector<uint32_t> result(impl_->points.n);
  for (uint32_t cluster = 0; cluster < impl_->leaves.size(); ++cluster)
    for (auto row : impl_->nodes[impl_->leaves[cluster]].rows) result[row] = cluster;
  return result;
}

}  // namespace mvsic
