#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "mvsic/core/index.h"
#include "mvsic/core/index_params.h"
#include "mvsic/core/search_params.h"
#include "mvsic/core/stats.h"
#include "mvsic/core/mvclustering/mvclustering_config.h"
#include "mvsic/core/types/point_cloud_set.h"
#include "mvsic/core/types/chamfer_l2_point.h"
#include "mvsic/core/types/chamfer_ip_point.h"
#include "mvsic/mvivf/mvivf.h"
#include "mvsic/mvivf/mvivf_flat.h"
#include "mvsic/muvera/muvera.h"
#include "mvsic/vamana/vamana.h"
#include "mvsic/mpool/mpool.h"
#include "mvsic/svh/svh_ivf.h"
#include "parlay/primitives.h"

namespace py = pybind11;

// This is a type caster for parlay::sequence, which is used in the project as a replacement for
// std::vector. This allows pybind11 to automatically convert parlay::sequence to Python lists.
namespace pybind11 {
namespace detail {
template<typename T>
struct type_caster<parlay::sequence<T>> : list_caster<parlay::sequence<T>, T> {};
}  // namespace detail
}  // namespace pybind11

PYBIND11_MODULE(mvsic, m) {
  m.doc() = "Python bindings for mvsic";

  //======================================
  // Index and Search Parameters
  //======================================
  py::enum_<mvsic::IndexParams::QuantizerType>(m, "QuantizerType")
      .value("None", mvsic::IndexParams::QuantizerType::None)
      .value("PQ", mvsic::IndexParams::QuantizerType::PQ)
      .value("RaBitQ", mvsic::IndexParams::QuantizerType::RaBitQ)
      .value("FastScan", mvsic::IndexParams::QuantizerType::FastScan)
      .value("TurboQuant4Bit", mvsic::IndexParams::QuantizerType::TurboQuant4Bit)
      .value("TurboQuantPQ4Bit", mvsic::IndexParams::QuantizerType::TurboQuantPQ4Bit)
      .export_values();

  py::class_<mvsic::IndexParams::fde_config>(m, "fde_config")
      .def(py::init<>())
      .def_readwrite("num_repetitions", &mvsic::IndexParams::fde_config::num_repetitions)
      .def_readwrite("num_simhash_projections",
                     &mvsic::IndexParams::fde_config::num_simhash_projections)
      .def_readwrite("seed", &mvsic::IndexParams::fde_config::seed)
      .def_readwrite("projection_dimension", &mvsic::IndexParams::fde_config::projection_dimension)
      .def_readwrite("fill_empty_partitions",
                     &mvsic::IndexParams::fde_config::fill_empty_partitions)
      .def_readwrite("final_projection_dimension",
                     &mvsic::IndexParams::fde_config::final_projection_dimension)
      .def_readwrite("normalize", &mvsic::IndexParams::fde_config::normalize);

  py::class_<mvsic::IndexParams::ann_config>(m, "ann_config")
      .def(py::init<>())
      .def_readwrite("R", &mvsic::IndexParams::ann_config::R)
      .def_readwrite("L", &mvsic::IndexParams::ann_config::L)
      .def_readwrite("alpha", &mvsic::IndexParams::ann_config::alpha)
      .def_readwrite("num_pass", &mvsic::IndexParams::ann_config::num_pass);

  py::class_<mvsic::IndexParams::pq_config>(m, "pq_config")
      .def(py::init<>())
      .def_readwrite("method", &mvsic::IndexParams::pq_config::method)
      .def_readwrite("block_size", &mvsic::IndexParams::pq_config::block_size)
      .def_readwrite("num_clusters_per_block",
                     &mvsic::IndexParams::pq_config::num_clusters_per_block)
      .def_readwrite("num_points_per_cluster",
                     &mvsic::IndexParams::pq_config::num_points_per_cluster)
      .def_readwrite("rabitq_bits", &mvsic::IndexParams::pq_config::rabitq_bits);

  py::class_<mvsic::MVClusteringConfig>(m, "MVClusteringConfig")
      .def(py::init<>())
      .def_readwrite("niters", &mvsic::MVClusteringConfig::niters)
      .def_readwrite("max_point_clouds_per_cluster",
                     &mvsic::MVClusteringConfig::max_point_clouds_per_cluster)
      .def_readwrite("max_points_per_centroid_inner_kmeans",
                     &mvsic::MVClusteringConfig::max_points_per_centroid_inner_kmeans)
      .def_readwrite("verbose", &mvsic::MVClusteringConfig::verbose)
      .def_readwrite("init", &mvsic::MVClusteringConfig::init)
      .def_readwrite("seed", &mvsic::MVClusteringConfig::seed)
      .def_readwrite("use_weighted_inner_kmeans", &mvsic::MVClusteringConfig::use_weighted_inner_kmeans);

  py::class_<mvsic::IndexParams>(m, "IndexParams")
      .def(py::init([]() { return mvsic::IndexParams(); }))
      .def_readwrite("method", &mvsic::IndexParams::method)
      .def_readwrite("verbose", &mvsic::IndexParams::verbose)
      .def_readwrite("compress_input", &mvsic::IndexParams::compress_input)
      .def_readwrite("k_per_level", &mvsic::IndexParams::k_per_level)
      .def_readwrite("max_leaf_size", &mvsic::IndexParams::max_leaf_size)
      .def_readwrite("quantize_centers", &mvsic::IndexParams::quantize_centers)
      .def_readwrite("mvclus", &mvsic::IndexParams::mvclus)
      .def_readwrite("s", &mvsic::IndexParams::s)
      .def_readwrite("fde", &mvsic::IndexParams::fde)
      .def_readwrite("ann", &mvsic::IndexParams::ann)
      .def_readwrite("normalize", &mvsic::IndexParams::normalize)
      .def_readwrite("R", &mvsic::IndexParams::R)
      .def_readwrite("L", &mvsic::IndexParams::L)
      .def_readwrite("alpha", &mvsic::IndexParams::alpha)
      .def_readwrite("two_pass", &mvsic::IndexParams::two_pass)
      .def_readwrite("pq", &mvsic::IndexParams::pq)
      .def_readwrite("max_points_per_centroid", &mvsic::IndexParams::max_points_per_centroid)
      .def_static("mvivf", &mvsic::IndexParams::mvivf, py::arg("k_per_level") = 0,
                  py::arg("max_leaf_size") = 200, py::arg("compress_input") = false,
                  py::arg("verbose") = 0, py::arg("niters") = 5,
                  py::arg("max_point_clouds_per_cluster") = 0,
                  py::arg("max_points_per_centroid_inner_kmeans") = 20, py::arg("init") = "Random",
                  py::arg("seed") = 0, py::arg("use_weighted_inner_kmeans") = true,
                  py::arg("s") = 0, py::arg("pq_method") = 0, py::arg("block_size") = 8,
                  py::arg("num_clusters_per_block") = 256, py::arg("num_points_per_cluster") = 20,
                  py::arg("rabitq_bits") = 8, py::arg("quantize_centers") = false)
      .def_static("mvivf_flat", &mvsic::IndexParams::mvivf_flat, py::arg("k_per_level") = 0,
                  py::arg("compress_input") = false, py::arg("verbose") = 0, py::arg("niters") = 5,
                  py::arg("max_point_clouds_per_cluster") = 0,
                  py::arg("max_points_per_centroid_inner_kmeans") = 20, py::arg("init") = "Random",
                  py::arg("seed") = 0, py::arg("use_weighted_inner_kmeans") = true,
                  py::arg("s") = 0, py::arg("pq_method") = 0, py::arg("block_size") = 8,
                  py::arg("num_clusters_per_block") = 256, py::arg("num_points_per_cluster") = 20,
                  py::arg("rabitq_bits") = 8, py::arg("quantize_centers") = false)
      .def_static(
          "muvera_custom", &mvsic::IndexParams::muvera_custom, py::arg("num_repetitions") = 20,
          py::arg("num_simhash_projections") = 4, py::arg("seed") = 1,
          py::arg("projection_dimension") = 8, py::arg("fill_empty_partitions") = false,
          py::arg("final_projection_dimension") = 0, py::arg("normalize") = false,
          py::arg("R") = 200, py::arg("L") = 600, py::arg("alpha") = 1.1, py::arg("num_pass") = 1,
          py::arg("compress_input") = false, py::arg("verbose") = 0, py::arg("pq_method") = 0,
          py::arg("block_size") = 8, py::arg("num_clusters_per_block") = 256,
          py::arg("num_points_per_cluster") = 20, py::arg("rabitq_bits") = 8)
      .def_static("muvera", &mvsic::IndexParams::muvera, py::arg("d_fde") = 2560,
                  py::arg("seed") = 1, py::arg("fill_empty_partitions") = false,
                  py::arg("final_projection_dimension") = 0, py::arg("normalize") = false,
                  py::arg("R") = 200, py::arg("L") = 600, py::arg("alpha") = 1.1,
                  py::arg("num_pass") = 1, py::arg("compress_input") = false,
                  py::arg("verbose") = 0, py::arg("pq_method") = 0, py::arg("block_size") = 8,
                  py::arg("num_clusters_per_block") = 256, py::arg("num_points_per_cluster") = 20,
                  py::arg("rabitq_bits") = 8)
      .def_static("mpool", &mvsic::IndexParams::mpool, py::arg("R") = 200, py::arg("L") = 600,
                  py::arg("alpha") = 1.2, py::arg("num_pass") = 1, py::arg("normalize") = true,
                  py::arg("compress_input") = false, py::arg("verbose") = 0,
                  py::arg("pq_method") = 0, py::arg("block_size") = 8,
                  py::arg("num_clusters_per_block") = 256, py::arg("num_points_per_cluster") = 20,
                  py::arg("rabitq_bits") = 8)
      .def_static("vamana", &mvsic::IndexParams::vamana, py::arg("R") = 200, py::arg("L") = 600,
                  py::arg("alpha") = 1.2, py::arg("two_pass") = false,
                  py::arg("compress_input") = false, py::arg("verbose") = 0,
                  py::arg("pq_method") = 0, py::arg("block_size") = 8,
                  py::arg("num_clusters_per_block") = 256, py::arg("num_points_per_cluster") = 20,
                  py::arg("rabitq_bits") = 8)
      .def_static("svh_ivf", &mvsic::IndexParams::svh_ivf, py::arg("k_per_level") = 0,
                  py::arg("max_leaf_size") = 500, py::arg("compress_input") = false,
                  py::arg("verbose") = 0, py::arg("max_points_per_centroid") = 100,
                  py::arg("pq_method") = 0, py::arg("block_size") = 8,
                  py::arg("num_clusters_per_block") = 256, py::arg("num_points_per_cluster") = 20,
                  py::arg("rabitq_bits") = 8, py::arg("quantize_centers") = false);

  py::class_<mvsic::SearchParams>(m, "SearchParams")
      .def(py::init([]() { return mvsic::SearchParams(); }))
      .def_readwrite("method", &mvsic::SearchParams::method)
      .def_readwrite("k", &mvsic::SearchParams::k)
      .def_readwrite("num_rerank", &mvsic::SearchParams::num_rerank)
      .def_readwrite("nprobes", &mvsic::SearchParams::nprobes)
      .def_readwrite("L", &mvsic::SearchParams::L)
      .def_readwrite("cut", &mvsic::SearchParams::cut)
      .def_readwrite("norerank", &mvsic::SearchParams::norerank)
      .def_static("mvivf", &mvsic::SearchParams::mvivf, py::arg("k"), py::arg("nprobes"),
                  py::arg("num_rerank") = 0)
      .def_static("mvivf_flat", &mvsic::SearchParams::mvivf_flat, py::arg("k"), py::arg("nprobes"),
                  py::arg("num_rerank") = 0)
      .def_static("vamana", &mvsic::SearchParams::vamana, py::arg("k"), py::arg("L"),
                  py::arg("cut") = 1.35, py::arg("num_rerank") = 0)
      .def_static("muvera", &mvsic::SearchParams::muvera, py::arg("k"), py::arg("L"),
                  py::arg("num_rerank"), py::arg("cut") = 1.35, py::arg("norerank") = false)
      .def_static("mpool", &mvsic::SearchParams::mpool, py::arg("k"), py::arg("L"),
                  py::arg("num_rerank"), py::arg("cut") = 1.35, py::arg("norerank") = false)
      .def_static("svh_ivf", &mvsic::SearchParams::svh_ivf, py::arg("k"), py::arg("nprobes"),
                  py::arg("num_rerank"), py::arg("norerank") = false);

  //======================================
  // Point Cloud and Point Types
  //======================================

  py::class_<mvsic::ChamferL2_Point>(m, "ChamferL2_Point")
      .def("size", &mvsic::ChamferL2_Point::size)
      .def("get_dims", &mvsic::ChamferL2_Point::get_dims)
      .def("get_id", &mvsic::ChamferL2_Point::get_id);

  py::class_<mvsic::ChamferIP_Point>(m, "ChamferIP_Point")
      .def("size", &mvsic::ChamferIP_Point::size)
      .def("get_dims", &mvsic::ChamferIP_Point::get_dims)
      .def("get_id", &mvsic::ChamferIP_Point::get_id);

  py::class_<mvsic::PointCloudSet<mvsic::ChamferL2_Point>>(m, "PointCloudSetL2")
      .def(py::init<const char *, bool>(), py::arg("filename"), py::arg("is_mmap") = false)
      .def(py::init([](py::array_t<float> data, py::array_t<size_t> offsets,
                       py::array_t<uint32_t> ids, uint32_t dim) {
             py::buffer_info data_buf = data.request();
             py::buffer_info offsets_buf = offsets.request();
             py::buffer_info ids_buf = ids.request();

             if (data_buf.ndim != 1 || offsets_buf.ndim != 1 || ids_buf.ndim != 1) {
               throw std::runtime_error("NumPy arrays must be 1-dimensional");
             }

             return new mvsic::PointCloudSet<mvsic::ChamferL2_Point>(
                 ids_buf.shape[0], dim, static_cast<const float *>(data_buf.ptr),
                 static_cast<const size_t *>(offsets_buf.ptr),
                 static_cast<const uint32_t *>(ids_buf.ptr));
           }),
           py::arg("data"), py::arg("offsets"), py::arg("ids"), py::arg("dim"))
      .def("__getitem__", [](const mvsic::PointCloudSet<mvsic::ChamferL2_Point> &self,
                             size_t i) { return self[i]; })
      .def("size", &mvsic::PointCloudSet<mvsic::ChamferL2_Point>::size);

  py::class_<mvsic::PointCloudSet<mvsic::ChamferIP_Point>>(m, "PointCloudSetIP")
      .def(py::init<const char *, bool>(), py::arg("filename"), py::arg("is_mmap") = false)
      .def(py::init([](py::array_t<float> data, py::array_t<size_t> offsets,
                       py::array_t<uint32_t> ids, uint32_t dim) {
             py::buffer_info data_buf = data.request();
             py::buffer_info offsets_buf = offsets.request();
             py::buffer_info ids_buf = ids.request();

             if (data_buf.ndim != 1 || offsets_buf.ndim != 1 || ids_buf.ndim != 1) {
               throw std::runtime_error("NumPy arrays must be 1-dimensional");
             }

             return new mvsic::PointCloudSet<mvsic::ChamferIP_Point>(
                 ids_buf.shape[0], dim, static_cast<const float *>(data_buf.ptr),
                 static_cast<const size_t *>(offsets_buf.ptr),
                 static_cast<const uint32_t *>(ids_buf.ptr));
           }),
           py::arg("data"), py::arg("offsets"), py::arg("ids"), py::arg("dim"))
      .def("__getitem__", [](const mvsic::PointCloudSet<mvsic::ChamferIP_Point> &self,
                             size_t i) { return self[i]; })
      .def("size", &mvsic::PointCloudSet<mvsic::ChamferIP_Point>::size);

  //======================================
  // Index Types
  //======================================

#define BIND_INDEX(index_type, point_type, class_name_str)                                         \
  py::class_<mvsic::index_type>(m, class_name_str)                                                 \
      .def(py::init<size_t, const mvsic::IndexParams &>(), py::arg("dim"), py::arg("params"))      \
      .def("build", &mvsic::index_type::build, "Build the index.", py::arg("points"))              \
      .def(                                                                                        \
          "search",                                                                                \
          [](mvsic::index_type &index, const mvsic::point_type &query_point,                       \
             const mvsic::PointCloudSet<mvsic::point_type> &points,                                \
             const mvsic::SearchParams &params) {                                                  \
            auto result = index.search(query_point, points, params);                               \
            return result;                                                                         \
          },                                                                                       \
          "Search the index.", py::arg("query_point"), py::arg("points"), py::arg("params"))       \
      .def(                                                                                        \
          "search_with_stats",                                                                     \
          [](mvsic::index_type &index, const mvsic::point_type &query_point,                       \
             const mvsic::PointCloudSet<mvsic::point_type> &points,                                \
             const mvsic::SearchParams &params) {                                                  \
            auto result = index.search_with_stats(query_point, points, params);                    \
            return result;                                                                         \
          },                                                                                       \
          "Returns some stats.", py::arg("query_point"), py::arg("points"), py::arg("params"))     \
      .def(                                                                                        \
          "search_all",                                                                            \
          [](mvsic::index_type &index,                                                             \
             const mvsic::PointCloudSet<mvsic::point_type> &query_points,                          \
             const mvsic::PointCloudSet<mvsic::point_type> &points,                                \
             const mvsic::SearchParams &params) {                                                  \
            auto result = index.search_all(query_points, points, params);                          \
            return result;                                                                         \
          },                                                                                       \
          "Search all queries.", py::arg("query_points"), py::arg("points"), py::arg("params"))    \
      .def("save", &mvsic::index_type::save, "Save the index to a file.")                          \
      .def("load", &mvsic::index_type::load, "Load the index from a file.", py::arg("filename"),   \
           py::arg("points"))                                                                      \
      .def("get_height", &mvsic::index_type::get_height,                                           \
           "Returns height of kmeans tree, if applicable.");

  BIND_INDEX(IndexMVIVFL2, ChamferL2_Point, "IndexMVIVFL2")
  BIND_INDEX(IndexMVIVFIP, ChamferIP_Point, "IndexMVIVFIP")

  // MVIVF-only: tree stats as module function (avoids re-opening class which can
  // trigger "object with that name is already defined" in some pybind11 builds).
  m.def(
      "get_mvivf_tree_stats",
      [](mvsic::IndexMVIVFL2 &index) {
        auto s = index.get_tree_stats();
        py::dict d;
        d["num_internal_nodes"] = s.num_internal_nodes;
        d["num_leaves"] = s.num_leaves;
        d["avg_leaf_size"] = s.avg_leaf_size;
        d["avg_internal_node_size"] = s.avg_internal_node_size;
        d["total_point_clouds_internal"] = s.total_point_clouds_internal;
        d["height"] = s.height;
        return d;
      },
      py::arg("index"), "Returns tree stats for an MVIVF L2 index.");
  m.def(
      "get_mvivf_tree_stats",
      [](mvsic::IndexMVIVFIP &index) {
        auto s = index.get_tree_stats();
        py::dict d;
        d["num_internal_nodes"] = s.num_internal_nodes;
        d["num_leaves"] = s.num_leaves;
        d["avg_leaf_size"] = s.avg_leaf_size;
        d["avg_internal_node_size"] = s.avg_internal_node_size;
        d["total_point_clouds_internal"] = s.total_point_clouds_internal;
        d["height"] = s.height;
        return d;
      },
      py::arg("index"), "Returns tree stats for an MVIVF IP index.");

  BIND_INDEX(IndexMVIVFFlatL2, ChamferL2_Point, "IndexMVIVFFlatL2")
  BIND_INDEX(IndexMVIVFFlatIP, ChamferIP_Point, "IndexMVIVFFlatIP")
  BIND_INDEX(IndexMUVERAL2, ChamferL2_Point, "IndexMUVERAL2")
  BIND_INDEX(IndexMUVERAIP, ChamferIP_Point, "IndexMUVERAIP")
  BIND_INDEX(IndexVamanaL2, ChamferL2_Point, "IndexVamanaL2")
  BIND_INDEX(IndexVamanaIP, ChamferIP_Point, "IndexVamanaIP")
  BIND_INDEX(IndexMPoolL2, ChamferL2_Point, "IndexMPoolL2")
  BIND_INDEX(IndexMPoolIP, ChamferIP_Point, "IndexMPoolIP")
  BIND_INDEX(IndexSVHIVFL2, ChamferL2_Point, "IndexSVHIVFL2")
  BIND_INDEX(IndexSVHIVFIP, ChamferIP_Point, "IndexSVHIVFIP")

  //======================================
  // Stats Functions
  //======================================

  py::class_<mvsic::Stats>(m, "Stats")
      .def(py::init<>())
      .def_readwrite("QPS_seq", &mvsic::Stats::QPS_seq)
      .def_readwrite("QPS_par", &mvsic::Stats::QPS_par)
      .def_readwrite("avg_cmps", &mvsic::Stats::avg_cmps)
      .def_readwrite("recall_1_k", &mvsic::Stats::recall_1_k)
      .def_readwrite("recall_k_k", &mvsic::Stats::recall_k_k);

  py::class_<mvsic::StatsExtended>(m, "StatsExtended")
      .def(py::init<>())
      .def_readwrite("QPS_seq", &mvsic::StatsExtended::QPS_seq)
      .def_readwrite("QPS_par", &mvsic::StatsExtended::QPS_par)
      .def_readwrite("avg_cmps", &mvsic::StatsExtended::avg_cmps)
      .def_readwrite("recall_1_k", &mvsic::StatsExtended::recall_1_k)
      .def_readwrite("recall_k_k", &mvsic::StatsExtended::recall_k_k)
      .def_readwrite("avg_timings", &mvsic::StatsExtended::avg_timings);

  m.def("ReadGT", &ReadGT, "Read ground truth file", py::arg("file_path"), py::arg("num_points"));

  m.def("compute_recall", &compute_recall, "Compute recall", py::arg("pred"), py::arg("gt"),
        py::arg("k"), py::arg("k_gt"));

  m.def(
      "compute_scores",
      [](const parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>> &pred,
         const parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>> &gt,
         size_t k) { return compute_scores(pred, gt, k); },
      "Compute scores given predictions");

#define BIND_COMPUTE_STATS(index_type, point_type, index_name_str)                                 \
  m.def(                                                                                           \
      "compute_stats",                                                                             \
      [](mvsic::index_type &index, const mvsic::PointCloudSet<mvsic::point_type> &points,          \
         const mvsic::PointCloudSet<mvsic::point_type> &query_points,                              \
         const parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>> &gt,                 \
         const mvsic::SearchParams &params) {                                                      \
        return compute_stats(index, points, query_points, gt, params);                             \
      },                                                                                           \
      "Compute stats for " index_name_str " index", py::arg("index"), py::arg("points"),           \
      py::arg("query_points"), py::arg("gt"), py::arg("params"));                                  \
                                                                                                   \
  m.def(                                                                                           \
      "compute_stats",                                                                             \
      [](mvsic::index_type &index, const mvsic::PointCloudSet<mvsic::point_type> &points,          \
         const mvsic::PointCloudSet<mvsic::point_type> &query_points,                              \
         const parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>> &gt,                 \
         const parlay::sequence<mvsic::SearchParams> &params) {                                    \
        return compute_stats(index, points, query_points, gt, params);                             \
      },                                                                                           \
      "Compute stats for " index_name_str " index for a sequence of params", py::arg("index"),     \
      py::arg("points"), py::arg("query_points"), py::arg("gt"), py::arg("params"));               \
                                                                                                   \
  m.def(                                                                                           \
      "compute_stats_extended",                                                                    \
      [](mvsic::index_type &index, const mvsic::PointCloudSet<mvsic::point_type> &points,          \
         const mvsic::PointCloudSet<mvsic::point_type> &query_points,                              \
         const parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>> &gt,                 \
         const mvsic::SearchParams &params) {                                                      \
        return compute_stats_extended(index, points, query_points, gt, params);                    \
      },                                                                                           \
      "Compute extended stats for " index_name_str " index", py::arg("index"), py::arg("points"),  \
      py::arg("query_points"), py::arg("gt"), py::arg("params"));                                  \
                                                                                                   \
  m.def(                                                                                           \
      "compute_stats_extended",                                                                    \
      [](mvsic::index_type &index, const mvsic::PointCloudSet<mvsic::point_type> &points,          \
         const mvsic::PointCloudSet<mvsic::point_type> &query_points,                              \
         const parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>> &gt,                 \
         const parlay::sequence<mvsic::SearchParams> &params) {                                    \
        return compute_stats_extended(index, points, query_points, gt, params);                    \
      },                                                                                           \
      "Compute extended stats for " index_name_str " index for a sequence of params",              \
      py::arg("index"), py::arg("points"), py::arg("query_points"), py::arg("gt"),                 \
      py::arg("params"));                                                                          \
  m.def(                                                                                           \
      "compute_stats_extended_p_threaded",                                                         \
      [](mvsic::index_type &index, const mvsic::PointCloudSet<mvsic::point_type> &points,          \
         const mvsic::PointCloudSet<mvsic::point_type> &query_points,                              \
         const parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>> &gt,                 \
         const mvsic::SearchParams &params, size_t num_threads) {                                  \
        return compute_stats_extended_p_threaded(index, points, query_points, gt, params,          \
                                                 num_threads);                                     \
      },                                                                                           \
      "Compute extended stats for " index_name_str " index, using p threads per query",            \
      py::arg("index"), py::arg("points"), py::arg("query_points"), py::arg("gt"),                 \
      py::arg("params"), py::arg("num_threads"));                                                  \
                                                                                                   \
  m.def(                                                                                           \
      "compute_stats_extended_p_threaded",                                                         \
      [](mvsic::index_type &index, const mvsic::PointCloudSet<mvsic::point_type> &points,          \
         const mvsic::PointCloudSet<mvsic::point_type> &query_points,                              \
         const parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>> &gt,                 \
         const parlay::sequence<mvsic::SearchParams> &params, size_t num_threads) {                \
        return compute_stats_extended_p_threaded(index, points, query_points, gt, params,          \
                                                 num_threads);                                     \
      },                                                                                           \
      "Compute extended stats for " index_name_str                                                 \
      " index, using p threads per query, for a sequence of params",                               \
      py::arg("index"), py::arg("points"), py::arg("query_points"), py::arg("gt"),                 \
      py::arg("params"), py::arg("num_threads"));

  BIND_COMPUTE_STATS(IndexMVIVFIP, ChamferIP_Point, "MVIVFIP")
  BIND_COMPUTE_STATS(IndexMVIVFL2, ChamferL2_Point, "MVIVFL2")
  BIND_COMPUTE_STATS(IndexMVIVFFlatIP, ChamferIP_Point, "MVIVFFlatIP")
  BIND_COMPUTE_STATS(IndexMVIVFFlatL2, ChamferL2_Point, "MVIVFFlatL2")
  BIND_COMPUTE_STATS(IndexMUVERAIP, ChamferIP_Point, "MUVERAIP")
  BIND_COMPUTE_STATS(IndexMUVERAL2, ChamferL2_Point, "MUVERAL2")
  BIND_COMPUTE_STATS(IndexVamanaIP, ChamferIP_Point, "VamanaIP")
  BIND_COMPUTE_STATS(IndexVamanaL2, ChamferL2_Point, "VamanaL2")
  BIND_COMPUTE_STATS(IndexMPoolIP, ChamferIP_Point, "MPoolIP")
  BIND_COMPUTE_STATS(IndexMPoolL2, ChamferL2_Point, "MPoolL2")
  BIND_COMPUTE_STATS(IndexSVHIVFIP, ChamferIP_Point, "SVHIVFIP")
  BIND_COMPUTE_STATS(IndexSVHIVFL2, ChamferL2_Point, "SVHIVFL2")
}