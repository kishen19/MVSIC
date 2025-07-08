#pragma once

#include <queue>
#include <set>
#include "algorithms/utils/graph.h"
#include "algorithms/utils/stats.h"
#include "algorithms/utils/types.h"
#include "algorithms/vamana/index.h"
#include "algorithms/utils/point_range.h"
#include "algorithms/utils/euclidian_point.h"
#include "algorithms/utils/mips_point.h"
#include "src/common/index.h"
#include "fde/fixed_dimensional_encoding.h"

namespace mvivf {

/* =========================================Params Type======================================== */
struct IndexMUVERAParams {
  int num_repetitions = 40;         // Number of independent repetitions for FDE generation
  int num_simhash_projections = 6;  // Number of SimHash projections used to partition space
  int seed = 1;                     // Seed for the FDE generation process
  int projection_dimension = 128;   // Dimension to which points are reduced via random projections
  bool fill_empty_partitions = false;  // Fill empty partitions with nearest point coordinates
  int final_projection_dimension = 0;  // Dimension to which the final FDE is projected
  size_t R = 200;                      // Max Outdegree of the routing graph
  size_t L = 64;                       // Beam length
  double alpha = 1.2;                  // Robust pruning parameter
  bool two_pass = false;               // Two-pass graph construction
  bool verbose = false;                // Print debug statements
};
/* ===================================MUVERA Index Class=================================== */
template<bool metric>
class IndexMUVERA : Index<metric>, IndexMUVERAParams {
 public:
  using ChPoint = Index<metric>::ChPoint;
  using Point = std::conditional_t<metric, Euclidian_Point<float>, Mips_Point<float>>;
  using Range = PointRange<float, Point>;
  using pid = std::pair<size_t, float>;
  using Index<metric>::d;  // Embedding dimension

  Range points_fdes;  // FDEs
  Graph<size_t> G;    // Vamana graph
  BuildParams BP;
  knn_index<Point, Range, size_t> I;  // Vamana index

  IndexMUVERA(size_t d_) noexcept :
      BP(BuildParams(R, 600, alpha, two_pass)), I(knn_index<Point, Range, size_t>(BP)) {
    d = d_;
  }
  IndexMUVERA(size_t d_, const IndexMUVERAParams &params) noexcept :
      IndexMUVERAParams(params),
      BP(BuildParams(R, 600, alpha, two_pass)),
      I(knn_index<Point, Range, size_t>(BP)) {
    d = d_;
  }
  ~IndexMUVERA();
  /* ----------------------------Overridden Functions---------------------------- */
  void build(const PointCloudSet<ChPoint> &points) override;
  std::pair<parlay::sequence<std::pair<size_t, float>>, size_t> search(
      const ChPoint &query, const PointCloudSet<ChPoint> &points,
      const SearchParams &params) override;
  void save(const std::string &filename) override {}
  void load(const std::string &filename, const PointCloudSet<ChPoint> &points) override {}

  /* ------------------------------Helper Functions------------------------------ */
};

/* =======================================Implementation======================================= */
// Builds the index given a point cloud set.
template<bool metric>
void IndexMUVERA<metric>::build(const PointCloudSet<ChPoint> &points) {
  std::cout << "Building index..." << std::endl;
  // Step 1: Compute FDEs of data point clouds
  auto fdes = parlay::sequence<std::vector<float>>::uninitialized(points.size());
  graph_mining::FixedDimensionalEncodingConfig fde_config(
      d, num_repetitions, num_simhash_projections, seed,
      graph_mining::FixedDimensionalEncodingConfig::AVERAGE, projection_dimension,
      graph_mining::FixedDimensionalEncodingConfig::DEFAULT_IDENTITY, fill_empty_partitions,
      final_projection_dimension);
  parlay::parallel_for(0, points.size(), [&](size_t i) {
    size_t point_size = points.get_size(i);
    float *coords = points.get_coords(i);
    std::vector<float> point_data(coords, coords + point_size * d);
    fdes[i] = graph_mining::GenerateDocumentFixedDimensionalEncoding(point_data, fde_config);
  });
  points_fdes = Range(fdes, d);
  // Step 2: Build Vamana index on the FDEs
  BuildParams BP(R, 600, alpha, two_pass);
  G = Graph<size_t>(BP.R, points.size());
  stats<size_t> BuildStats(G.size());
  I.build_index(G, points_fdes, BuildStats);
  if (verbose) std::cout << "FDE Dimension: " << points_fdes.get_dims() << std::endl;
}

// Returns the top-k point clouds for the query point cloud
// Output format: < [<id, distance>, ...], No. of distance computations >
template<bool metric>
std::pair<parlay::sequence<std::pair<size_t, float>>, size_t> IndexMUVERA<metric>::search(
    const ChPoint &query, const PointCloudSet<ChPoint> &points, const SearchParams &params) {
  size_t k = params.k;
  size_t nprobes = params.nprobes;
  size_t beam_length = params.beam_length;
  // Step 1: Compute FDE of the query point cloud
  graph_mining::FixedDimensionalEncodingConfig fde_config(
      d, num_repetitions, num_simhash_projections, seed,
      graph_mining::FixedDimensionalEncodingConfig::DEFAULT_SUM, projection_dimension,
      graph_mining::FixedDimensionalEncodingConfig::DEFAULT_IDENTITY, fill_empty_partitions,
      final_projection_dimension);
  std::vector<float> query_vec(query.data(), query.data() + query.size() * d);
  std::vector<float> query_fde =
      graph_mining::GenerateQueryFixedDimensionalEncoding(query_vec, fde_config);
  // Step 2: Run beam search and collect top cand neighbors
  double cut = 1.35;
  size_t start_point = I.get_start();
  auto QP = QueryParams(1, L, cut, (long)G.size(), (long)G.max_degree());
  auto query_point = Point(query_fde.data(), query_fde.size(), query_fde.size(), -1);
  auto [result, dist_cmps] =
      beam_search<Point, Range, size_t>(query_point, G, points_fdes, start_point, QP);
  parlay::sequence<pid> visited = result.second;
  // Step 3: Re-rank the candidates and return top k
  auto cmp_rerank = parlay::sequence<size_t>::uninitialized(std::min(2 * k, visited.size()));
  auto results_rerank = parlay::sequence<std::pair<float, size_t>>::from_function(
      std::min(2 * k, visited.size()), [&](size_t i) {
        size_t id = visited[i].first;
        auto [dist, d_c] = query.distance_w_cmps(points[id]);
        cmp_rerank[i] = d_c;
        return std::make_pair(dist, id);
      });
  dist_cmps += parlay::reduce(cmp_rerank);
  parlay::sort_inplace(results_rerank);
  auto final_results = parlay::sequence<std::pair<size_t, float>>::from_function(
      std::min(k, results_rerank.size()),
      [&](size_t i) { return std::make_pair(results_rerank[i].second, visited[i].first); });
  return std::make_pair(final_results, dist_cmps);
}

// template<bool metric>
// void IndexMUVERA<metric>::save(const std::string &filename) {
//   std::ofstream outfile(filename, std::ios::binary);
//   std::cout << "Saving index to " << filename << std::endl;
//   if (!outfile.is_open()) {
//     std::cerr << "Error opening file for writing: " << filename << std::endl;
//     return;
//   }
//   // Collect data
//   parlay::sequence<node_t *> ind_to_node;
//   std::unordered_map<node_t *, size_t> node_to_ind;
//   parlay::sequence<size_t> center_offsets;
//   parlay::sequence<size_t> children_offsets;
//   parlay::sequence<size_t> point_offsets;

//   size_t height = traverse_tree(root, ind_to_node, node_to_ind, center_offsets, children_offsets,
//                                 point_offsets, 0);
//   std::cout << "Height of tree: " << height << std::endl;

//   size_t total_center_sizes = parlay::scan_inplace(center_offsets);
//   center_offsets.push_back(total_center_sizes);
//   size_t total_children_sizes = parlay::scan_inplace(children_offsets);
//   children_offsets.push_back(total_children_sizes);
//   size_t total_point_sizes = parlay::scan_inplace(point_offsets);
//   point_offsets.push_back(total_point_sizes);

//   // Write num
//   size_t num = ind_to_node.size();
//   outfile.write(reinterpret_cast<const char *>(&num), sizeof(size_t));
//   // Write center offsets
//   size_t num_center_offsets = center_offsets.size();
//   outfile.write(reinterpret_cast<const char *>(&num_center_offsets), sizeof(size_t));
//   outfile.write(reinterpret_cast<const char *>(center_offsets.begin()),
//                 center_offsets.size() * sizeof(size_t));
//   // Write centers values
//   for (size_t i = 0; i < num; ++i) {  // TODO: make parallel
//     node_t *node = ind_to_node[i];
//     if (node->children.size() > 0) {  // Internal Nodes only
//       auto coords = node->data.data();
//       size_t num_entries = (node->data.total_size()) * (node->data.get_dims());
//       outfile.write(reinterpret_cast<const char *>(coords), num_entries * sizeof(float));
//     }
//   }
//   // Write children offsets
//   outfile.write(reinterpret_cast<const char *>(children_offsets.begin()),
//                 children_offsets.size() * sizeof(size_t));
//   // Write children values
//   for (size_t i = 0; i < num; ++i) {
//     node_t *node = ind_to_node[i];
//     parlay::sequence<node_t *> children = node->children;
//     for (size_t j = 0; j < children.size(); ++j) {  // TODO: make parallel
//       node_t *child = children[j];
//       size_t child_id = node_to_ind[child];
//       outfile.write(reinterpret_cast<const char *>(&child_id), sizeof(size_t));
//     }
//   }
//   // Write point offsets
//   outfile.write(reinterpret_cast<const char *>(point_offsets.begin()),
//                 point_offsets.size() * sizeof(size_t));
//   // Write point values
//   for (size_t i = 0; i < num; ++i) {
//     node_t *node = ind_to_node[i];
//     if (node->children.size() == 0) {  // leaves only
//       PointCloudSet<ChPoint> points = node->data;
//       for (size_t j = 0; j < points.size(); ++j) {  // TODO: make parallel
//         size_t point_id = points.get_id(j);
//         outfile.write(reinterpret_cast<const char *>(&point_id), sizeof(size_t));
//       }
//     }
//   }
//   outfile.close();
// }

// template<bool metric>
// void IndexMUVERA<metric>::load(const std::string &filename, const PointCloudSet<ChPoint> &points)
// {
//   std::ifstream infile(filename, std::ios::binary);
//   std::cout << "Loading index from " << filename << std::endl;
//   if (!infile.is_open()) {
//     std::cerr << "Error opening file for reading: " << filename << std::endl;
//     return;
//   }
//   // Read number of nodes
//   size_t num = 0;
//   infile.read(reinterpret_cast<char *>(&num), sizeof(size_t));
//   // Read center offsets
//   size_t num_center_offsets = 0;
//   infile.read(reinterpret_cast<char *>(&num_center_offsets), sizeof(size_t));
//   parlay::sequence<size_t> center_offsets(num_center_offsets);
//   infile.read(reinterpret_cast<char *>(center_offsets.begin()),
//               center_offsets.size() * sizeof(size_t));
//   // Read centers values
//   parlay::sequence<float> center_values(center_offsets[center_offsets.size() - 1]);
//   infile.read(reinterpret_cast<char *>(center_values.begin()),
//               center_values.size() * sizeof(float));
//   // Read children offsets
//   parlay::sequence<size_t> children_offsets(num + 1);
//   infile.read(reinterpret_cast<char *>(children_offsets.begin()),
//               children_offsets.size() * sizeof(size_t));
//   // Read children values
//   parlay::sequence<size_t> children_values(children_offsets[children_offsets.size() - 1]);
//   infile.read(reinterpret_cast<char *>(children_values.begin()),
//               children_values.size() * sizeof(size_t));
//   // Read point offsets
//   parlay::sequence<size_t> point_offsets(num + 1);
//   infile.read(reinterpret_cast<char *>(point_offsets.begin()),
//               point_offsets.size() * sizeof(size_t));
//   // Read point values
//   parlay::sequence<size_t> point_values(point_offsets[point_offsets.size() - 1]);
//   infile.read(reinterpret_cast<char *>(point_values.begin()), point_values.size() *
//   sizeof(size_t));

//   // Build the index
//   size_t dim = points.get_dims();
//   auto point_id_to_data_id = parlay::sequence<size_t>::uninitialized(points.size());
//   parlay::parallel_for(0, points.size(),
//                        [&](size_t i) { point_id_to_data_id[points.get_id(i)] = i; });
//   parlay::sequence<size_t> children_sizes = parlay::sequence<size_t>::from_function(
//       num, [&](size_t i) { return children_offsets[i + 1] - children_offsets[i]; });
//   auto [children_sizes_scan, total_children_sizes] = parlay::scan(children_sizes);
//   children_sizes_scan.push_back(total_children_sizes);
//   assert(total_children_sizes == center_offsets.);
//   parlay::sequence<size_t> point_sizes = parlay::sequence<size_t>::from_function(
//       num, [&](size_t i) { return point_offsets[i + 1] - point_offsets[i]; });
//   parlay::sequence<node_t *> ind_to_node =
//       parlay::sequence<node_t *>::from_function(num, [&](size_t i) {
//         node_t *node = new node_t();
//         if (children_sizes[i] > 0) {  // Internal Nodes
//           size_t start_offset = children_sizes_scan[i];
//           size_t end_offset = children_sizes_scan[i + 1];
//           parlay::sequence<size_t> node_center_offsets =
//               parlay::sequence<size_t>::from_function(children_sizes[i] + 1, [&](size_t j) {
//                 return center_offsets[start_offset + j] - center_offsets[start_offset];
//               });

//           node->set_data(PointCloudSet<ChPoint>(children_sizes[i], dim,
//                                                 center_values.data() +
//                                                 center_offsets[start_offset],
//                                                 node_center_offsets.data(), nullptr));
//         }
//         node->children.resize(children_sizes[i]);
//         if (point_sizes[i] > 0) {
//           parlay::sequence<size_t> point_group =
//               parlay::sequence<size_t>::from_function(point_sizes[i], [&](size_t j) {
//                 size_t point_id = point_values[point_offsets[i] + j];
//                 return point_id_to_data_id[point_id];
//               });
//           node->set_data(PointCloudSet<ChPoint>(points.filter(point_group), dim));
//         }
//         return node;
//       });

//   // Set children pointers
//   parlay::parallel_for(0, num, [&](size_t i) {
//     node_t *node = ind_to_node[i];
//     parlay::sequence<node_t *> &children = node->children;
//     parlay::parallel_for(0, children.size(), [&](size_t j) {
//       size_t child_id = children_values[children_offsets[i] + j];
//       children[j] = ind_to_node[child_id];
//     });
//   });
//   root = ind_to_node[0];
//   infile.close();
// }

template<bool metric>
IndexMUVERA<metric>::~IndexMUVERA() {
  // traverse_and_delete(root);
  // delete root;
  // root = nullptr;
}

template struct IndexMUVERA<true>;   // Instantiates for L2 metric (metric = true)
template struct IndexMUVERA<false>;  // Instantiates for MIPS      (metric = false)

}  // namespace mvivf