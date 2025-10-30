package(default_visibility = ["//visibility:public"])

# load("@rules_python//python:defs.bzl", "py_binary", "py_library")

cc_binary(
    name = "compute_ground_truth",
    srcs = ["compute_ground_truth.cpp"],
    deps = [
        "//mvsic/core/utils:point_cloud_set",
        "//mvsic/core/utils:parse_command_line",
        "//mvsic/core/utils:top_neighbors",
        "//mvsic/core/utils:chamfer_l2_point",
        "//mvsic/core/utils:chamfer_ip_point",
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
