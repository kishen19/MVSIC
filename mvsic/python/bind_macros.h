#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "parlay/primitives.h"

namespace py = pybind11;

// Type caster for parlay::sequence. Must be visible in every TU that registers
// a function whose signature mentions parlay::sequence (otherwise pybind11 will
// fail to resolve the type at registration time).
namespace pybind11 {
namespace detail {
template<typename T>
struct type_caster<parlay::sequence<T>> : list_caster<parlay::sequence<T>, T> {};
}  // namespace detail
}  // namespace pybind11

// ---------- Per-family registration entry points ----------
// register_common must run before any of the per-family registers so that
// IndexParams / SearchParams / PointCloudSet{L2,IP} are known to pybind11.
void register_common(py::module_& m);

void register_mvivf_l2_a(py::module_& m);
void register_mvivf_l2_b(py::module_& m);
void register_mvivf_ip_a(py::module_& m);
void register_mvivf_ip_b(py::module_& m);

void register_mvivf_flat_l2_a(py::module_& m);
void register_mvivf_flat_l2_b(py::module_& m);
void register_mvivf_flat_ip_a(py::module_& m);
void register_mvivf_flat_ip_b(py::module_& m);

void register_mvivf_spill_l2_a(py::module_& m);
void register_mvivf_spill_l2_b(py::module_& m);
void register_mvivf_spill_ip_a(py::module_& m);
void register_mvivf_spill_ip_b(py::module_& m);

void register_muvera_l2(py::module_& m);
void register_muvera_ip(py::module_& m);

void register_vamana_l2(py::module_& m);
void register_vamana_ip(py::module_& m);

void register_mpool_l2(py::module_& m);
void register_mpool_ip(py::module_& m);

void register_svh_ivf_l2_a(py::module_& m);
void register_svh_ivf_l2_b(py::module_& m);
void register_svh_ivf_ip_a(py::module_& m);
void register_svh_ivf_ip_b(py::module_& m);

void register_svh_graph_l2(py::module_& m);
void register_svh_graph_ip(py::module_& m);

// ---------- Macros (extracted verbatim from the original bindings.cpp) ----------

#define BIND_INDEX(index_type, point_type, class_name_str)                                         \
  py::class_<mvsic::index_type>(m, class_name_str)                                                 \
      .def(py::init<size_t, const mvsic::IndexParams&>(), py::arg("dim"), py::arg("params"))       \
      .def("build", &mvsic::index_type::build, "Build the index.", py::arg("points"))              \
      .def(                                                                                        \
          "search",                                                                                \
          [](mvsic::index_type& index, const mvsic::point_type& query_point,                       \
             const mvsic::PointCloudSet<mvsic::point_type>& points,                                \
             const mvsic::SearchParams& params) {                                                  \
            auto result = index.search(query_point, points, params);                               \
            return result;                                                                         \
          },                                                                                       \
          "Search the index.", py::arg("query_point"), py::arg("points"), py::arg("params"))       \
      .def(                                                                                        \
          "search_with_stats",                                                                     \
          [](mvsic::index_type& index, const mvsic::point_type& query_point,                       \
             const mvsic::PointCloudSet<mvsic::point_type>& points,                                \
             const mvsic::SearchParams& params) {                                                  \
            auto result = index.search_with_stats(query_point, points, params);                    \
            return result;                                                                         \
          },                                                                                       \
          "Returns some stats.", py::arg("query_point"), py::arg("points"), py::arg("params"))     \
      .def(                                                                                        \
          "search_all",                                                                            \
          [](mvsic::index_type& index,                                                             \
             const mvsic::PointCloudSet<mvsic::point_type>& query_points,                          \
             const mvsic::PointCloudSet<mvsic::point_type>& points,                                \
             const mvsic::SearchParams& params) {                                                  \
            auto result = index.search_all(query_points, points, params);                          \
            return result;                                                                         \
          },                                                                                       \
          "Search all queries.", py::arg("query_points"), py::arg("points"), py::arg("params"))    \
      .def("save", &mvsic::index_type::save, "Save the index to a file.")                          \
      .def("load", &mvsic::index_type::load, "Load the index from a file.", py::arg("filename"),   \
           py::arg("points"))                                                                      \
      .def("get_height", &mvsic::index_type::get_height,                                           \
           "Returns height of kmeans tree, if applicable.");

#define BIND_COMPUTE_STATS(index_type, point_type, index_name_str)                                 \
  m.def(                                                                                           \
      "compute_stats",                                                                             \
      [](mvsic::index_type& index, const mvsic::PointCloudSet<mvsic::point_type>& points,          \
         const mvsic::PointCloudSet<mvsic::point_type>& query_points,                              \
         const parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>>& gt,                 \
         const mvsic::SearchParams& params) {                                                      \
        return compute_stats(index, points, query_points, gt, params);                             \
      },                                                                                           \
      "Compute stats for " index_name_str " index", py::arg("index"), py::arg("points"),           \
      py::arg("query_points"), py::arg("gt"), py::arg("params"));                                  \
                                                                                                   \
  m.def(                                                                                           \
      "compute_stats",                                                                             \
      [](mvsic::index_type& index, const mvsic::PointCloudSet<mvsic::point_type>& points,          \
         const mvsic::PointCloudSet<mvsic::point_type>& query_points,                              \
         const parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>>& gt,                 \
         const parlay::sequence<mvsic::SearchParams>& params) {                                    \
        return compute_stats(index, points, query_points, gt, params);                             \
      },                                                                                           \
      "Compute stats for " index_name_str " index for a sequence of params", py::arg("index"),     \
      py::arg("points"), py::arg("query_points"), py::arg("gt"), py::arg("params"));               \
                                                                                                   \
  m.def(                                                                                           \
      "compute_stats_extended",                                                                    \
      [](mvsic::index_type& index, const mvsic::PointCloudSet<mvsic::point_type>& points,          \
         const mvsic::PointCloudSet<mvsic::point_type>& query_points,                              \
         const parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>>& gt,                 \
         const mvsic::SearchParams& params) {                                                      \
        return compute_stats_extended(index, points, query_points, gt, params);                    \
      },                                                                                           \
      "Compute extended stats for " index_name_str " index", py::arg("index"), py::arg("points"),  \
      py::arg("query_points"), py::arg("gt"), py::arg("params"));                                  \
                                                                                                   \
  m.def(                                                                                           \
      "compute_stats_extended",                                                                    \
      [](mvsic::index_type& index, const mvsic::PointCloudSet<mvsic::point_type>& points,          \
         const mvsic::PointCloudSet<mvsic::point_type>& query_points,                              \
         const parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>>& gt,                 \
         const parlay::sequence<mvsic::SearchParams>& params) {                                    \
        return compute_stats_extended(index, points, query_points, gt, params);                    \
      },                                                                                           \
      "Compute extended stats for " index_name_str " index for a sequence of params",              \
      py::arg("index"), py::arg("points"), py::arg("query_points"), py::arg("gt"),                 \
      py::arg("params"));                                                                          \
  m.def(                                                                                           \
      "compute_stats_extended_p_threaded",                                                         \
      [](mvsic::index_type& index, const mvsic::PointCloudSet<mvsic::point_type>& points,          \
         const mvsic::PointCloudSet<mvsic::point_type>& query_points,                              \
         const parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>>& gt,                 \
         const mvsic::SearchParams& params, size_t num_threads) {                                  \
        return compute_stats_extended_p_threaded(index, points, query_points, gt, params,          \
                                                 num_threads);                                     \
      },                                                                                           \
      "Compute extended stats for " index_name_str " index, using p threads per query",            \
      py::arg("index"), py::arg("points"), py::arg("query_points"), py::arg("gt"),                 \
      py::arg("params"), py::arg("num_threads"));                                                  \
                                                                                                   \
  m.def(                                                                                           \
      "compute_stats_extended_p_threaded",                                                         \
      [](mvsic::index_type& index, const mvsic::PointCloudSet<mvsic::point_type>& points,          \
         const mvsic::PointCloudSet<mvsic::point_type>& query_points,                              \
         const parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>>& gt,                 \
         const parlay::sequence<mvsic::SearchParams>& params, size_t num_threads) {                \
        return compute_stats_extended_p_threaded(index, points, query_points, gt, params,          \
                                                 num_threads);                                     \
      },                                                                                           \
      "Compute extended stats for " index_name_str                                                 \
      " index, using p threads per query, for a sequence of params",                               \
      py::arg("index"), py::arg("points"), py::arg("query_points"), py::arg("gt"),                 \
      py::arg("params"), py::arg("num_threads"));                                                  \
  m.def(                                                                                           \
      "compute_stats_latency",                                                                     \
      [](mvsic::index_type& index, const mvsic::PointCloudSet<mvsic::point_type>& points,          \
         const mvsic::PointCloudSet<mvsic::point_type>& query_points,                              \
         const parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>>& gt,                 \
         const mvsic::SearchParams& params) {                                                      \
        return compute_stats_latency(index, points, query_points, gt, params);                     \
      },                                                                                           \
      "Per-query single-thread latency for " index_name_str " index.", py::arg("index"),           \
      py::arg("points"), py::arg("query_points"), py::arg("gt"), py::arg("params"));               \
  m.def(                                                                                           \
      "compute_stats_latency",                                                                     \
      [](mvsic::index_type& index, const mvsic::PointCloudSet<mvsic::point_type>& points,          \
         const mvsic::PointCloudSet<mvsic::point_type>& query_points,                              \
         const parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>>& gt,                 \
         const parlay::sequence<mvsic::SearchParams>& params) {                                    \
        return compute_stats_latency(index, points, query_points, gt, params);                     \
      },                                                                                           \
      "Per-query single-thread latency for " index_name_str " index, sequence of params.",         \
      py::arg("index"), py::arg("points"), py::arg("query_points"), py::arg("gt"),                 \
      py::arg("params"));                                                                          \
  m.def(                                                                                           \
      "compute_stats_multi_latency",                                                               \
      [](mvsic::index_type& index, const mvsic::PointCloudSet<mvsic::point_type>& points,          \
         const mvsic::PointCloudSet<mvsic::point_type>& query_points,                              \
         const parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>>& gt,                 \
         const mvsic::SearchParams& params) {                                                      \
        return compute_stats_multi_latency(index, points, query_points, gt, params);               \
      },                                                                                           \
      "Per-query multi-thread latency for " index_name_str " index (uses ambient parlay pool).",   \
      py::arg("index"), py::arg("points"), py::arg("query_points"), py::arg("gt"),                 \
      py::arg("params"));                                                                          \
  m.def(                                                                                           \
      "compute_stats_multi_latency",                                                               \
      [](mvsic::index_type& index, const mvsic::PointCloudSet<mvsic::point_type>& points,          \
         const mvsic::PointCloudSet<mvsic::point_type>& query_points,                              \
         const parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>>& gt,                 \
         const parlay::sequence<mvsic::SearchParams>& params) {                                    \
        return compute_stats_multi_latency(index, points, query_points, gt, params);               \
      },                                                                                           \
      "Per-query multi-thread latency for " index_name_str " index, sequence of params.",          \
      py::arg("index"), py::arg("points"), py::arg("query_points"), py::arg("gt"),                 \
      py::arg("params"));                                                                          \
  m.def(                                                                                           \
      "compute_stats_batch",                                                                       \
      [](mvsic::index_type& index, const mvsic::PointCloudSet<mvsic::point_type>& points,          \
         const mvsic::PointCloudSet<mvsic::point_type>& query_points,                              \
         const parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>>& gt,                 \
         const mvsic::SearchParams& params) {                                                      \
        return compute_stats_batch(index, points, query_points, gt, params);                       \
      },                                                                                           \
      "Batch (search_all) throughput for " index_name_str " index.", py::arg("index"),             \
      py::arg("points"), py::arg("query_points"), py::arg("gt"), py::arg("params"));               \
  m.def(                                                                                           \
      "compute_stats_batch",                                                                       \
      [](mvsic::index_type& index, const mvsic::PointCloudSet<mvsic::point_type>& points,          \
         const mvsic::PointCloudSet<mvsic::point_type>& query_points,                              \
         const parlay::sequence<parlay::sequence<std::pair<uint32_t, float>>>& gt,                 \
         const parlay::sequence<mvsic::SearchParams>& params) {                                    \
        return compute_stats_batch(index, points, query_points, gt, params);                       \
      },                                                                                           \
      "Batch (search_all) throughput for " index_name_str " index, sequence of params.",           \
      py::arg("index"), py::arg("points"), py::arg("query_points"), py::arg("gt"),                 \
      py::arg("params"));
