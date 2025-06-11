# load("@rules_python//python:defs.bzl", "py_binary", "py_library")

package(default_visibility = ["//visibility:public"])

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
