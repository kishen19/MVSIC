#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "mvsic/core/index.h"
#include "mvsic/core/index_params.h"
#include "mvsic/core/search_params.h"
#include "mvsic/core/stats.h"
#include "mvsic/core/utils/point_cloud_set.h"
#include "mvsic/core/utils/chamfer_l2_point.h"
#include "mvsic/core/utils/chamfer_ip_point.h"
#include "mvsic/mvivf/mvivf.h"
#include "mvsic/muvera/muvera.h"
#include "mvsic/vamana/vamana.h"
// #include "mvsic/mpool/mpool.h"
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

  py::class_<mvsic::IndexParams::vamana_config>(m, "vamana_config")
      .def(py::init<>())
      .def_readwrite("R", &mvsic::IndexParams::vamana_config::R)
      .def_readwrite("L", &mvsic::IndexParams::vamana_config::L)
      .def_readwrite("alpha", &mvsic::IndexParams::vamana_config::alpha)
      .def_readwrite("two_pass", &mvsic::IndexParams::vamana_config::two_pass);

  py::class_<mvsic::IndexParams::pq_config>(m, "pq_config")
      .def(py::init<>())
      .def_readwrite("enabled", &mvsic::IndexParams::pq_config::enabled)
      .def_readwrite("num_blocks", &mvsic::IndexParams::pq_config::num_blocks)
      .def_readwrite("num_clusters_per_block",
                     &mvsic::IndexParams::pq_config::num_clusters_per_block)
      .def_readwrite("sample_size", &mvsic::IndexParams::pq_config::sample_size);

  py::class_<mvsic::IndexParams>(m, "IndexParams")
      .def(py::init([]() { return mvsic::IndexParams(); }))
      .def_readwrite("method", &mvsic::IndexParams::method)
      .def_readwrite("verbose", &mvsic::IndexParams::verbose)
      .def_readwrite("compress_input", &mvsic::IndexParams::compress_input)
      .def_readwrite("use_PQ", &mvsic::IndexParams::use_PQ)
      .def_readwrite("k_per_level", &mvsic::IndexParams::k_per_level)
      .def_readwrite("max_leaf_size", &mvsic::IndexParams::max_leaf_size)
      .def_readwrite("s", &mvsic::IndexParams::s)
      .def_readwrite("fde", &mvsic::IndexParams::fde)
      .def_readwrite("vamana", &mvsic::IndexParams::vamana)
      .def_readwrite("pq", &mvsic::IndexParams::pq)
      .def_readwrite("normalize", &mvsic::IndexParams::normalize)
      .def_static("mvivf", &mvsic::IndexParams::mvivf, py::arg("k_per_level") = 0,
                  py::arg("max_leaf_size") = 200, py::arg("compress_input") = false,
                  py::arg("verbose") = 0, py::arg("niters") = 5,
                  py::arg("max_points_per_centroid_inner_kmeans") = 20, py::arg("init") = "Random",
                  py::arg("seed") = 0, py::arg("use_weighted_inner_kmeans") = false,
                  py::arg("s") = 0, py::arg("pq_enabled") = false, py::arg("num_blocks") = 8,
                  py::arg("num_clusters_per_block") = 256, py::arg("sample_size") = 100000)
      .def_static("muvera", &mvsic::IndexParams::muvera, py::arg("num_repetitions") = 20,
                  py::arg("num_simhash_projections") = 4, py::arg("seed") = 1,
                  py::arg("projection_dimension") = 8, py::arg("fill_empty_partitions") = false,
                  py::arg("final_projection_dimension") = 0, py::arg("normalize") = false,
                  py::arg("R") = 200, py::arg("L") = 600, py::arg("alpha") = 1.2,
                  py::arg("two_pass") = false, py::arg("compress_input") = false,
                  py::arg("apply_PQ") = false, py::arg("verbose") = 0,
                  py::arg("pq_enabled") = false, py::arg("num_blocks") = 8,
                  py::arg("num_clusters_per_block") = 256, py::arg("sample_size") = 100000)
      //   .def_static("mpool", &mvsic::IndexParams::mpool, py::arg("R") = 200, py::arg("L") = 600,
      //   py::arg("alpha") = 1.2, py::arg("two_pass") = false,
      //               py::arg("normalize") = true, py::arg("compress_input") = false,
      //               py::arg("apply_PQ") = false, py::arg("verbose") = 0, py::arg("pq_enabled") =
      //               false, py::arg("num_blocks") = 8, py::arg("num_clusters_per_block") = 256,
      //               py::arg("sample_size") = 100000)
      .def_static("mvvamana", &mvsic::IndexParams::mvvamana, py::arg("R") = 200, py::arg("L") = 600,
                  py::arg("alpha") = 1.2, py::arg("two_pass") = false,
                  py::arg("compress_input") = false, py::arg("apply_PQ") = false,
                  py::arg("verbose") = 0, py::arg("pq_enabled") = false, py::arg("num_blocks") = 8,
                  py::arg("num_clusters_per_block") = 256, py::arg("sample_size") = 100000);

  py::class_<mvsic::SearchParams>(m, "SearchParams")
      .def(py::init([]() { return mvsic::SearchParams(); }))
      .def_readwrite("method", &mvsic::SearchParams::method)
      .def_readwrite("k", &mvsic::SearchParams::k)
      .def_readwrite("num_rerank", &mvsic::SearchParams::num_rerank)
      .def_readwrite("nprobes", &mvsic::SearchParams::nprobes)
      .def_readwrite("L", &mvsic::SearchParams::L)
      .def_readwrite("cut", &mvsic::SearchParams::cut)
      .def_readwrite("limit", &mvsic::SearchParams::limit)
      .def_readwrite("degree_limit", &mvsic::SearchParams::degree_limit)
      .def_readwrite("norerank", &mvsic::SearchParams::norerank)
      .def_static("mvivf", &mvsic::SearchParams::mvivf, py::arg("k"), py::arg("nprobes"),
                  py::arg("num_rerank"))
      .def_static("mvvamana", &mvsic::SearchParams::mvvamana, py::arg("k"), py::arg("L"),
                  py::arg("cut"), py::arg("limit"), py::arg("degree_limit"))
      .def_static("muvera", &mvsic::SearchParams::muvera, py::arg("k"), py::arg("L"),
                  py::arg("cut"), py::arg("limit"), py::arg("degree_limit"), py::arg("num_rerank"),
                  py::arg("norerank") = false);
  //   .def_static("mpool", &mvsic::SearchParams::mpool, py::arg("k"), py::arg("L"),
  //   py::arg("cut"), py::arg("limit"), py::arg("degree_limit"), py::arg("num_rerank"),
  //   py::arg("norerank") = false);

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

  py::class_<mvsic::IndexMVIVFL2>(m, "IndexMVIVFL2")
      .def(py::init<size_t, const mvsic::IndexParams &>(), py::arg("dim"), py::arg("params"))
      .def("build", &mvsic::IndexMVIVFL2::build, "Build the index.", py::arg("points"))
      .def(
          "search",
          [](mvsic::IndexMVIVFL2 &index, const mvsic::ChamferL2_Point &query_point,
             const mvsic::PointCloudSet<mvsic::ChamferL2_Point> &points,
             const mvsic::SearchParams &params) {
            auto result = index.search(query_point, points, params);
            return result;
          },
          "Search the index.", py::arg("query_point"), py::arg("points"), py::arg("params"))
      .def(
          "search_all",
          [](mvsic::IndexMVIVFL2 &index,
             const mvsic::PointCloudSet<mvsic::ChamferL2_Point> &query_points,
             const mvsic::PointCloudSet<mvsic::ChamferL2_Point> &points,
             const mvsic::SearchParams &params) {
            auto result = index.search_all(query_points, points, params);
            return result;
          },
          "Search all queries.", py::arg("query_points"), py::arg("points"), py::arg("params"))
      .def("save", &mvsic::IndexMVIVFL2::save, "Save the index to a file.")
      .def("load", &mvsic::IndexMVIVFL2::load, "Load the index from a file.", py::arg("filename"),
           py::arg("points"));

  py::class_<mvsic::IndexMVIVFIP>(m, "IndexMVIVFIP")
      .def(py::init<size_t, const mvsic::IndexParams &>(), py::arg("dim"), py::arg("params"))
      .def("build", &mvsic::IndexMVIVFIP::build, "Build the index.", py::arg("points"))
      .def(
          "search",
          [](mvsic::IndexMVIVFIP &index, const mvsic::ChamferIP_Point &query_point,
             const mvsic::PointCloudSet<mvsic::ChamferIP_Point> &points,
             const mvsic::SearchParams &params) {
            auto result = index.search(query_point, points, params);
            return result;
          },
          "Search the index.", py::arg("query_point"), py::arg("points"), py::arg("params"))
      .def(
          "search_all",
          [](mvsic::IndexMVIVFIP &index,
             const mvsic::PointCloudSet<mvsic::ChamferIP_Point> &query_points,
             const mvsic::PointCloudSet<mvsic::ChamferIP_Point> &points,
             const mvsic::SearchParams &params) {
            auto result = index.search_all(query_points, points, params);
            return result;
          },
          "Search all queries.", py::arg("query_points"), py::arg("points"), py::arg("params"))
      .def("save", &mvsic::IndexMVIVFIP::save, "Save the index to a file.")
      .def("load", &mvsic::IndexMVIVFIP::load, "Load the index from a file.", py::arg("filename"),
           py::arg("points"));

  py::class_<mvsic::IndexMUVERAL2>(m, "IndexMUVERAL2")
      .def(py::init<size_t, const mvsic::IndexParams &>(), py::arg("dim"), py::arg("params"))
      .def("build", &mvsic::IndexMUVERAL2::build, "Build the index.", py::arg("points"))
      .def(
          "search",
          [](mvsic::IndexMUVERAL2 &index, const mvsic::ChamferL2_Point &query_point,
             const mvsic::PointCloudSet<mvsic::ChamferL2_Point> &points,
             const mvsic::SearchParams &params) {
            auto result = index.search(query_point, points, params);
            return result;
          },
          "Search the index.", py::arg("query_point"), py::arg("points"), py::arg("params"))
      .def(
          "search_all",
          [](mvsic::IndexMUVERAL2 &index,
             const mvsic::PointCloudSet<mvsic::ChamferL2_Point> &query_points,
             const mvsic::PointCloudSet<mvsic::ChamferL2_Point> &points,
             const mvsic::SearchParams &params) {
            auto result = index.search_all(query_points, points, params);
            return result;
          },
          "Search all queries.", py::arg("query_points"), py::arg("points"), py::arg("params"))
      .def("save", &mvsic::IndexMUVERAL2::save, "Save the index to a file.")
      .def("load", &mvsic::IndexMUVERAL2::load, "Load the index from a file.", py::arg("filename"),
           py::arg("points"));

  py::class_<mvsic::IndexMUVERAIP>(m, "IndexMUVERAIP")
      .def(py::init<size_t, const mvsic::IndexParams &>(), py::arg("dim"), py::arg("params"))
      .def("build", &mvsic::IndexMUVERAIP::build, "Build the index.", py::arg("points"))
      .def(
          "search",
          [](mvsic::IndexMUVERAIP &index, const mvsic::ChamferIP_Point &query_point,
             const mvsic::PointCloudSet<mvsic::ChamferIP_Point> &points,
             const mvsic::SearchParams &params) {
            auto result = index.search(query_point, points, params);
            return result;
          },
          "Search the index.", py::arg("query_point"), py::arg("points"), py::arg("params"))
      .def(
          "search_all",
          [](mvsic::IndexMUVERAIP &index,
             const mvsic::PointCloudSet<mvsic::ChamferIP_Point> &query_points,
             const mvsic::PointCloudSet<mvsic::ChamferIP_Point> &points,
             const mvsic::SearchParams &params) {
            auto result = index.search_all(query_points, points, params);
            return result;
          },
          "Search all queries.", py::arg("query_points"), py::arg("points"), py::arg("params"))
      .def("save", &mvsic::IndexMUVERAIP::save, "Save the index to a file.")
      .def("load", &mvsic::IndexMUVERAIP::load, "Load the index from a file.", py::arg("filename"),
           py::arg("points"));

  py::class_<mvsic::IndexVamanaL2>(m, "IndexVamanaL2")
      .def(py::init<size_t, const mvsic::IndexParams &>(), py::arg("dim"), py::arg("params"))
      .def("build", &mvsic::IndexVamanaL2::build, "Build the index.", py::arg("points"))
      .def(
          "search",
          [](mvsic::IndexVamanaL2 &index, const mvsic::ChamferL2_Point &query_point,
             const mvsic::PointCloudSet<mvsic::ChamferL2_Point> &points,
             const mvsic::SearchParams &params) {
            auto result = index.search(query_point, points, params);
            return result;
          },
          "Search the index.", py::arg("query_point"), py::arg("points"), py::arg("params"))
      .def(
          "search_all",
          [](mvsic::IndexVamanaL2 &index,
             const mvsic::PointCloudSet<mvsic::ChamferL2_Point> &query_points,
             const mvsic::PointCloudSet<mvsic::ChamferL2_Point> &points,
             const mvsic::SearchParams &params) {
            auto result = index.search_all(query_points, points, params);
            return result;
          },
          "Search all queries.", py::arg("query_points"), py::arg("points"), py::arg("params"))
      .def("save", &mvsic::IndexVamanaL2::save, "Save the index to a file.")
      .def("load", &mvsic::IndexVamanaL2::load, "Load the index from a file.", py::arg("filename"),
           py::arg("points"));

  py::class_<mvsic::IndexVamanaIP>(m, "IndexVamanaIP")
      .def(py::init<size_t, const mvsic::IndexParams &>(), py::arg("dim"), py::arg("params"))
      .def("build", &mvsic::IndexVamanaIP::build, "Build the index.", py::arg("points"))
      .def(
          "search",
          [](mvsic::IndexVamanaIP &index, const mvsic::ChamferIP_Point &query_point,
             const mvsic::PointCloudSet<mvsic::ChamferIP_Point> &points,
             const mvsic::SearchParams &params) {
            auto result = index.search(query_point, points, params);
            return result;
          },
          "Search the index.", py::arg("query_point"), py::arg("points"), py::arg("params"))
      .def(
          "search_all",
          [](mvsic::IndexVamanaIP &index,
             const mvsic::PointCloudSet<mvsic::ChamferIP_Point> &query_points,
             const mvsic::PointCloudSet<mvsic::ChamferIP_Point> &points,
             const mvsic::SearchParams &params) {
            auto result = index.search_all(query_points, points, params);
            return result;
          },
          "Search all queries.", py::arg("query_points"), py::arg("points"), py::arg("params"))
      .def("save", &mvsic::IndexVamanaIP::save, "Save the index to a file.")
      .def("load", &mvsic::IndexVamanaIP::load, "Load the index from a file.", py::arg("filename"),
           py::arg("points"));

  //   py::class_<mvsic::IndexMPoolL2>(m, "IndexMPoolL2")
  //       .def(py::init<size_t, const mvsic::IndexParams &>(), py::arg("dim"), py::arg("params"))
  //       .def("build", &mvsic::IndexMPoolL2::build, "Build the index.", py::arg("points"))
  //       .def(
  //           "search",
  //           [](mvsic::IndexMPoolL2 &index,
  //              const mvsic::ChamferL2_Point &query_point,
  //              const mvsic::PointCloudSet<mvsic::ChamferL2_Point> &points,
  //              const mvsic::SearchParams &params) {
  //             auto result = index.search(query_point, points, params);
  //             return result;
  //           },
  //           "Search the index.", py::arg("query_point"), py::arg("points"), py::arg("params"))
  //       .def(
  //           "search_all",
  //           [](mvsic::IndexMPoolL2 &index,
  //              const mvsic::PointCloudSet<mvsic::ChamferL2_Point> &query_points,
  //              const mvsic::PointCloudSet<mvsic::ChamferL2_Point> &points,
  //              const mvsic::SearchParams &params) {
  //             auto result = index.search_all(query_points, points, params);
  //             return result;
  //           },
  //           "Search all queries.", py::arg("query_points"), py::arg("points"), py::arg("params"))
  //       .def("save", &mvsic::IndexMPoolL2::save, "Save the index to a file.")
  //       .def("load", &mvsic::IndexMPoolL2::load, "Load the index from a file.",
  //       py::arg("filename"),
  //            py::arg("points"));

  //   py::class_<mvsic::IndexMPoolIP>(m, "IndexMPoolIP")
  //       .def(py::init<size_t, const mvsic::IndexParams &>(), py::arg("dim"), py::arg("params"))
  //       .def("build", &mvsic::IndexMPoolIP::build, "Build the index.", py::arg("points"))
  //       .def(
  //           "search",
  //           [](mvsic::IndexMPoolIP &index,
  //              const mvsic::ChamferIP_Point &query_point,
  //              const mvsic::PointCloudSet<mvsic::ChamferIP_Point> &points,
  //              const mvsic::SearchParams &params) {
  //             auto result = index.search(query_point, points, params);
  //             return result;
  //           },
  //           "Search the index.", py::arg("query_point"), py::arg("points"), py::arg("params"))
  //       .def(
  //           "search_all",
  //           [](mvsic::IndexMPoolIP &index,
  //              const mvsic::PointCloudSet<mvsic::ChamferIP_Point> &query_points,
  //              const mvsic::PointCloudSet<mvsic::ChamferIP_Point> &points,
  //              const mvsic::SearchParams &params) {
  //             auto result = index.search_all(query_points, points, params);
  //             return result;
  //           },
  //           "Search all queries.", py::arg("query_points"), py::arg("points"), py::arg("params"))
  //       .def("save", &mvsic::IndexMPoolIP::save, "Save the index to a file.")
  //       .def("load", &mvsic::IndexMPoolIP::load, "Load the index from a file.",
  //       py::arg("filename"),
  //            py::arg("points"));

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

  m.def("ReadGT", &ReadGT, "Read ground truth file", py::arg("file_path"), py::arg("num_points"));

  m.def("compute_recall", &compute_recall, "Compute recall", py::arg("pred"), py::arg("gt"),
        py::arg("k"), py::arg("k_gt"));

  m.def(
      "compute_stats",
      [](const parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>> &pred,
         const parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>> &gt,
         size_t k) { return compute_stats(pred, gt, k); },
      "Compute stats given predictions");

  m.def(
      "compute_stats",
      [](mvsic::IndexMVIVFIP &index, const mvsic::PointCloudSet<mvsic::ChamferIP_Point> &points,
         const mvsic::PointCloudSet<mvsic::ChamferIP_Point> &query_points,
         const parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>> &gt,
         const mvsic::SearchParams &params) {
        return compute_stats(index, points, query_points, gt, params);
      },
      "Compute stats for MVIVFIP index");

  m.def(
      "compute_stats",
      [](mvsic::IndexMVIVFL2 &index, const mvsic::PointCloudSet<mvsic::ChamferL2_Point> &points,
         const mvsic::PointCloudSet<mvsic::ChamferL2_Point> &query_points,
         const parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>> &gt,
         const mvsic::SearchParams &params) {
        return compute_stats(index, points, query_points, gt, params);
      },
      "Compute stats for MVIVFL2 index");

  m.def(
      "compute_stats",
      [](mvsic::IndexMUVERAIP &index, const mvsic::PointCloudSet<mvsic::ChamferIP_Point> &points,
         const mvsic::PointCloudSet<mvsic::ChamferIP_Point> &query_points,
         const parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>> &gt,
         const mvsic::SearchParams &params) {
        return compute_stats(index, points, query_points, gt, params);
      },
      "Compute stats for MUVERAIP index");

  m.def(
      "compute_stats",
      [](mvsic::IndexMUVERAL2 &index, const mvsic::PointCloudSet<mvsic::ChamferL2_Point> &points,
         const mvsic::PointCloudSet<mvsic::ChamferL2_Point> &query_points,
         const parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>> &gt,
         const mvsic::SearchParams &params) {
        return compute_stats(index, points, query_points, gt, params);
      },
      "Compute stats for MUVERAL2 index");

  m.def(
      "compute_stats",
      [](mvsic::IndexVamanaIP &index, const mvsic::PointCloudSet<mvsic::ChamferIP_Point> &points,
         const mvsic::PointCloudSet<mvsic::ChamferIP_Point> &query_points,
         const parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>> &gt,
         const mvsic::SearchParams &params) {
        return compute_stats(index, points, query_points, gt, params);
      },
      "Compute stats for VamanaIP index");

  m.def(
      "compute_stats",
      [](mvsic::IndexVamanaL2 &index, const mvsic::PointCloudSet<mvsic::ChamferL2_Point> &points,
         const mvsic::PointCloudSet<mvsic::ChamferL2_Point> &query_points,
         const parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>> &gt,
         const mvsic::SearchParams &params) {
        return compute_stats(index, points, query_points, gt, params);
      },
      "Compute stats for VamanaL2 index");
}