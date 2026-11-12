#include <algorithm>
#include <limits>
#include <memory>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mvsic/algorithms/mvivf/mvivf.h"
#include "mvsic/algorithms/mvivf/mvclustering/mvclustering.h"
#include "mvsic/algorithms/benchmark.h"
#include "mvsic/core/kernels/capabilities.h"

namespace py = pybind11;
namespace {
using namespace mvsic;

struct ArrayStorage {
  py::array values;
  std::vector<size_t> offsets;
  std::vector<uint32_t> ids;
};

PointCloudSetView FromArray(py::array values, py::object offsets, py::object ids, bool copy) {
  if (copy) {
    auto converted = py::array_t<float, py::array::c_style | py::array::forcecast>(values);
    values = converted.attr("copy")("C");
  }
  if (!values.dtype().is(py::dtype::of<float>()) || !(values.flags() & py::array::c_style))
    throw py::value_error("values must be C-contiguous float32; use copy=True to convert");
  if (values.ndim() != 2 && values.ndim() != 3)
    throw py::value_error("values must have shape (vectors, dim) or (clouds, vectors, dim)");
  const auto d = values.shape(values.ndim() - 1);
  if (d <= 0 || d > UINT32_MAX) throw py::value_error("invalid vector dimension");
  auto storage = std::unique_ptr<ArrayStorage>(new ArrayStorage{values, {}, {}});
  if (offsets.is_none()) {
    const auto n = values.shape(0);
    if (n <= 0 || uint64_t(n) > UINT32_MAX || (values.ndim() == 3 && values.shape(1) <= 0))
      throw py::value_error("values must contain between 1 and UINT32_MAX nonempty point clouds");
    const size_t width = values.ndim() == 3 ? size_t(values.shape(1)) * d : d;
    storage->offsets.resize(n + 1);
    for (ssize_t i = 0; i <= n; ++i)
      storage->offsets[i] = size_t(i) * width;
  } else {
    if (values.ndim() != 2)
      throw py::value_error("explicit scalar offsets require a 2D values array");
    if (!py::isinstance<py::array>(offsets))
      throw py::value_error("offsets must be a NumPy uint64 array");
    const auto array = py::cast<py::array>(offsets);
    if (!array.dtype().is(py::dtype::of<size_t>()) || !(array.flags() & py::array::c_style) ||
        array.ndim() != 1)
      throw py::value_error("offsets must be a contiguous uint64 array");
    const auto input = py::reinterpret_borrow<py::array_t<size_t>>(array);
    if (input.size() < 2 || uint64_t(input.size()) > uint64_t(UINT32_MAX) + 1)
      throw py::value_error("offsets must describe between 1 and UINT32_MAX point clouds");
    storage->offsets.assign(input.data(), input.data() + input.size());
  }
  if (storage->offsets.empty() || storage->offsets.back() != size_t(values.size()))
    throw py::value_error("offsets must cover all scalar coordinates in values");
  const auto n = storage->offsets.size() - 1;
  if (n > UINT32_MAX) throw py::value_error("too many point clouds");
  if (!ids.is_none()) {
    if (!py::isinstance<py::array>(ids)) throw py::value_error("ids must be a NumPy uint32 array");
    const auto array = py::cast<py::array>(ids);
    if (!array.dtype().is(py::dtype::of<uint32_t>()) || !(array.flags() & py::array::c_style) ||
        array.ndim() != 1 || size_t(array.size()) != n)
      throw py::value_error("ids must be a contiguous uint32 array with one ID per cloud");
    const auto input = py::reinterpret_borrow<py::array_t<uint32_t>>(array);
    storage->ids.assign(input.data(), input.data() + input.size());
  }
  const auto* raw = storage.release();
  std::shared_ptr<const void> owner(raw, [](const void* object) {
    py::gil_scoped_acquire gil;
    delete static_cast<const ArrayStorage*>(object);
  });
  PointCloudSetView view{uint32_t(n),
                         uint32_t(d),
                         static_cast<const float*>(raw->values.data()),
                         raw->offsets.data(),
                         raw->ids.empty() ? nullptr : raw->ids.data(),
                         owner};
  view.validate();
  return view;
}

PointCloud Query(py::array array, uint32_t d) {
  if (!array.dtype().is(py::dtype::of<float>()) || !(array.flags() & py::array::c_style) ||
      array.ndim() != 2 || array.shape(1) != d || array.shape(0) <= 0 ||
      array.shape(0) > UINT32_MAX)
    throw py::value_error(
        "query must be a nonempty C-contiguous float32 array of shape (vectors, dim)");
  return {static_cast<const float*>(array.data()), uint32_t(array.shape(0)), d, 0};
}

py::tuple Arrays(const Neighbors& values) {
  py::array_t<uint32_t> ids(values.size());
  py::array_t<float> distances(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    ids.mutable_data()[i] = values[i].first;
    distances.mutable_data()[i] = values[i].second;
  }
  return py::make_tuple(std::move(ids), std::move(distances));
}

py::tuple Arrays(const BatchNeighbors& values, size_t k) {
  if (k > size_t(PY_SSIZE_T_MAX) || (k && values.size() > size_t(PY_SSIZE_T_MAX) / k))
    throw py::value_error("result shape is too large");
  py::array_t<uint32_t> ids({py::ssize_t(values.size()), py::ssize_t(k)});
  py::array_t<float> distances({py::ssize_t(values.size()), py::ssize_t(k)});
  std::fill_n(ids.mutable_data(), values.size() * k, UINT32_MAX);
  std::fill_n(distances.mutable_data(), values.size() * k, std::numeric_limits<float>::infinity());
  for (size_t i = 0; i < values.size(); ++i)
    for (size_t j = 0; j < values[i].size(); ++j) {
      ids.mutable_data()[i * k + j] = values[i][j].first;
      distances.mutable_data()[i * k + j] = values[i][j].second;
    }
  return py::make_tuple(std::move(ids), std::move(distances));
}
}  // namespace

PYBIND11_MODULE(_mvsic, m) {
  using namespace mvsic;
  m.doc() = "Compiled MaxSim MVIVF with independent IP/L2 clustering";
  py::enum_<Metric>(m, "Metric").value("IP", Metric::IP).value("L2", Metric::L2);
  py::enum_<ClusteringQuantizer>(m, "ClusteringQuantizer")
      .value("None_", ClusteringQuantizer::None)
      .value("TQ8", ClusteringQuantizer::TQ8)
      .value("TQ4", ClusteringQuantizer::TQ4)
      .value("TQ1", ClusteringQuantizer::TQ1);
  using Quantizer = IndexParams::QuantizerType;
  py::enum_<Quantizer>(m, "QuantizerType")
      .value("None_", Quantizer::None)
      .value("TQ", Quantizer::TurboQuant)
      .value("OneBitTQ", Quantizer::OneBitTQ)
      .value("EightBitTQ", Quantizer::EightBitTQ);
  using Compression = SearchParams::QueryCompression;
  py::enum_<Compression>(m, "QueryCompression")
      .value("None_", Compression::None)
      .value("Carve", Compression::Carve)
      .value("Wards", Compression::Wards);

  py::class_<PointCloudSetView>(m, "PointCloudSet")
      .def(py::init(&FromArray), py::arg("values"), py::arg("offsets") = py::none(),
           py::arg("ids") = py::none(), py::arg("copy") = false)
      .def_static(
          "from_file",
          [](const std::string& filename, bool mmap) {
            py::gil_scoped_release gil;
            return load_point_clouds(filename, mmap);
          },
          py::arg("filename"), py::arg("mmap") = true)
      .def("__len__", &PointCloudSetView::size)
      .def_property_readonly("dims", &PointCloudSetView::get_dims)
      .def_property_readonly("total_size", &PointCloudSetView::total_size)
      .def("__getitem__", [](const PointCloudSetView& points, py::ssize_t row) {
        if (row < 0) row += points.size();
        if (row < 0 || size_t(row) >= points.size()) throw py::index_error();
        auto cloud = points[row];
        auto array =
            py::array_t<float>({py::ssize_t(cloud.n), py::ssize_t(cloud.d)},
                               {py::ssize_t(cloud.d * sizeof(float)), py::ssize_t(sizeof(float))},
                               cloud.values, py::cast(points));
        array.attr("setflags")(false);
        return array;
      });

  auto mvclus = py::class_<MVClusteringConfig>(m, "MVClusteringConfig").def(py::init<>());
#define FIELD(cls, name) cls.def_readwrite(#name, &MVClusteringConfig::name)
  FIELD(mvclus, niters);
  FIELD(mvclus, max_point_clouds_per_cluster);
  FIELD(mvclus, max_points_per_centroid_inner_kmeans);
  FIELD(mvclus, verbose);
  FIELD(mvclus, init);
  FIELD(mvclus, seed);
  FIELD(mvclus, use_weighted_inner_kmeans);
  FIELD(mvclus, assignment_quantizer);
  FIELD(mvclus, build_with_8btq);
  FIELD(mvclus, inner_metric);
  FIELD(mvclus, inner_init);
  FIELD(mvclus, inner_backend);
#undef FIELD
  auto ip = py::class_<IndexParams>(m, "IndexParams").def(py::init<>());
#define FIELD(cls, name) cls.def_readwrite(#name, &IndexParams::name)
  FIELD(ip, method);
  FIELD(ip, verbose);
  FIELD(ip, compress_input);
  FIELD(ip, k_per_level);
  FIELD(ip, max_leaf_size);
  FIELD(ip, max_depth);
  FIELD(ip, quantize_centers);
  FIELD(ip, s);
  FIELD(ip, build_with_8btq);
  FIELD(ip, mvclus);
#undef FIELD
  ip.def_property(
      "quantizer", [](const IndexParams& p) { return p.pq.method; },
      [](IndexParams& p, Quantizer q) { p.pq.method = q; });
  ip.def_static("mvivf", &IndexParams::mvivf, py::arg("k_per_level") = 0,
                py::arg("max_leaf_size") = 500, py::arg("compress_input") = false,
                py::arg("verbose") = 0, py::arg("niters") = 5,
                py::arg("max_point_clouds_per_cluster") = 100,
                py::arg("max_points_per_centroid_inner_kmeans") = 20, py::arg("init") = "Random",
                py::arg("seed") = 0, py::arg("use_weighted_inner_kmeans") = true, py::arg("s") = 0,
                py::arg("max_depth") = 0, py::arg("build_with_8btq") = false);

  auto sp = py::class_<SearchParams>(m, "SearchParams").def(py::init<>());
#define FIELD(cls, name) cls.def_readwrite(#name, &SearchParams::name)
  FIELD(sp, method);
  FIELD(sp, k);
  FIELD(sp, nprobes);
  FIELD(sp, num_rerank);
  FIELD(sp, tq8_rerank);
  FIELD(sp, query_compression);
  FIELD(sp, query_compression_threshold);
  FIELD(sp, compress_rerank);
  FIELD(sp, query_alignment);
  FIELD(sp, query_alignment_strict);
  FIELD(sp, root_m2m);
#undef FIELD
  sp.def_static("mvivf", &SearchParams::mvivf, py::arg("k"), py::arg("nprobes"),
                py::arg("num_rerank") = 0, py::arg("tq8_rerank") = true,
                py::arg("query_compression") = Compression::None,
                py::arg("query_compression_threshold") = .7f, py::arg("compress_rerank") = false);

  py::class_<Index, std::unique_ptr<Index>>(m, "Index");
  py::class_<IndexMVIVF, Index>(m, "IndexMVIVF")
      .def(py::init<uint32_t, Metric, const IndexParams&>(), py::arg("d"),
           py::arg("metric") = Metric::IP, py::arg("params") = IndexParams::mvivf())
      .def("build", &IndexMVIVF::build, py::call_guard<py::gil_scoped_release>())
      .def("load", &IndexMVIVF::load, py::call_guard<py::gil_scoped_release>())
      .def("save", &IndexMVIVF::save, py::call_guard<py::gil_scoped_release>())
      .def("search",
           [](const IndexMVIVF& index, py::array query, const SearchParams& params) {
             auto cloud = Query(query, index.get_dims());
             Neighbors result;
             {
               py::gil_scoped_release gil;
               result = index.search(cloud, params);
             }
             return Arrays(result);
           })
      .def("search_with_stats",
           [](const IndexMVIVF& index, py::array query, const PointCloudSetView& points,
              const SearchParams& params) {
             auto cloud = Query(query, index.get_dims());
             SearchResultWithStats result;
             {
               py::gil_scoped_release gil;
               result = index.search_with_stats(cloud, points, params);
             }
             return py::make_tuple(Arrays(std::get<0>(result)), std::get<1>(result),
                                   std::get<2>(result));
           })
      .def("search_all",
           [](const IndexMVIVF& index, const PointCloudSetView& queries,
              const SearchParams& params) {
             BatchNeighbors result;
             {
               py::gil_scoped_release gil;
               result = index.search_all(queries, params);
             }
             return Arrays(result, params.k);
           })
      .def("get_dims", &IndexMVIVF::get_dims)
      .def("get_height", &IndexMVIVF::get_height)
      .def("mean_cluster_size", &IndexMVIVF::mean_cluster_size)
      .def("max_cluster_size", &IndexMVIVF::max_cluster_size)
      .def("get_flat_clustering", &IndexMVIVF::get_flat_clustering)
      .def_property_readonly("metric", &IndexMVIVF::metric);

  m.def(
      "benchmark",
      [](const Index& index, const PointCloudSetView& queries, const SearchParams& params,
         const std::string& mode, size_t warmup, size_t repeats) {
        BenchmarkResult result;
        {
          py::gil_scoped_release gil;
          result = benchmark(index, queries, params, mode, warmup, repeats);
        }
        auto arrays = Arrays(result.neighbors, params.k);
        py::dict out;
        out["ids"] = arrays[0];
        out["distances"] = arrays[1];
        out["seconds"] = result.seconds;
        out["query_seconds"] =
            result.query_seconds.empty() ? py::none() : py::cast(result.query_seconds);
        out["selected_repeat"] = result.selected_repeat;
        out["warmup_seconds"] = result.warmup_seconds;
        out["first_call_seconds"] = result.first_call_seconds;
        out["timing_scope"] = "native";
        return out;
      },
      py::arg("index"), py::arg("queries"), py::arg("params"), py::arg("mode"),
      py::arg("warmup") = 10, py::arg("repeats") = 3);

  m.def("capabilities", [] {
    const auto caps = kernels::cpu_capabilities();
    py::dict result, hardware, usable;
    hardware["avx2"] = caps.avx2;
    hardware["avx512"] = caps.avx512f && caps.avx512bw && caps.avx512dq && caps.avx512vl;
    hardware["vnni"] = caps.avx512vnni;
    hardware["amx_int8"] = caps.amx_int8;
    usable["avx2"] = caps.usable_avx2();
    usable["avx512"] = caps.usable_avx512();
    usable["vnni"] = caps.usable_vnni();
    usable["amx_int8"] = caps.usable_amx_int8();
    usable["x86_64_v3"] = caps.x86_64_v3;
    result["hardware"] = hardware;
    result["usable"] = usable;
#ifdef MVSIC_RUNTIME_ISA_DISPATCH
    result["compiled_serving"] = py::make_tuple("avx2", "avx512", "vnni");
    result["runtime_dispatch"] = true;
#else
    result["compiled_serving"] = py::make_tuple("build_target");
    result["runtime_dispatch"] = false;
#endif
    result["amx_implemented"] = false;
    result["minimum_cpu"] = "x86-64-v3";
    result["version"] = "0.1.0";
    result["compiler"] = __VERSION__;
    return result;
  });

  m.def(
      "cluster_point_clouds",
      [](const PointCloudSetView& points, uint32_t k, uint32_t s, Metric metric,
         const MVClusteringConfig& config) {
        ClusteringResult result;
        {
          py::gil_scoped_release gil;
          result = cluster_point_clouds(points, k, s, metric, config);
        }
        py::array_t<uint32_t> assignments(result.assignments.size());
        std::copy(result.assignments.begin(), result.assignments.end(), assignments.mutable_data());
        py::dict stats;
        stats["assignment_implementation"] = result.assignment_implementation;
        stats["assignment_quantizer"] = result.assignment_quantizer;
        py::list iterations;
        for (const auto& iteration : result.iterations) {
          py::dict row;
          row["assignment_seconds"] = iteration.assignment_seconds;
          row["centroid_update_seconds"] = iteration.centroid_update_seconds;
          row["float_cost"] = iteration.float_cost;
          iterations.append(row);
        }
        stats["iterations"] = iterations;
        return py::make_tuple(result.centers, assignments, stats);
      },
      py::arg("points"), py::arg("k"), py::arg("s"), py::arg("metric") = Metric::IP,
      py::arg("config") = MVClusteringConfig{});
}
