load("@rules_python//python:defs.bzl", "py_binary", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "utils",
    srcs = ["utils.py"],
)

py_binary(
    name = "point_cloud_spread_stats",
    srcs = ["point_cloud_spread_stats.py"],
    deps = ["utils"],
)
