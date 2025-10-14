load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

pybind_extension(
    name = "mvsic",
    srcs = ["bindings.cpp"],
    deps = [
        "//mvsic/core:index",
        "//mvsic/core:index_params",
        "//mvsic/core:search_params",
        "//mvsic/core/utils:chamfer_ip_point",
        "//mvsic/core/utils:chamfer_l2_point",
        "//mvsic/core/utils:point_cloud_set",
        # "//mvsic/mpool",
        "//mvsic/muvera",
        "//mvsic/mvivf",
        "//mvsic/vamana",
        "@parlaylib//parlay:primitives",
    ],
)

py_library(
    name = "python_pkg",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
)
