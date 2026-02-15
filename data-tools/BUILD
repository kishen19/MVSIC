load("@rules_cc//cc:defs.bzl", "cc_binary")
load("@rules_python//python:defs.bzl", "py_binary", "py_library")

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "compute_ground_truth",
    srcs = ["compute_ground_truth.cpp"],
    deps = [
        "//mvsic/core/types:chamfer_ip_point",
        "//mvsic/core/types:chamfer_l2_point",
        "//mvsic/core/types:point_cloud_set",
        "//mvsic/core/utils:parse_command_line",
        "//mvsic/core/utils:top_neighbors",
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
)

py_binary(
    name = "data_checker",
    srcs = ["data_checker.py"],
    deps = ["utils"],
)

py_binary(
    name = "reduce_queries",
    srcs = ["reduce_queries.py"],
    deps = ["utils"],
)
