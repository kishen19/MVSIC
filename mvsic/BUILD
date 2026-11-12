load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

# The installed wheel is built by PEP 517/CMake. Bazel remains useful for native
# development; it does not label a CPython extension as a pure-Python wheel.
pybind_extension(
    name = "_mvsic",
    srcs = ["//mvsic/python:bindings.cpp"],
    deps = [
        "//mvsic/algorithms/mvivf:mvivf",
        "//mvsic/algorithms/mvivf/mvclustering:mvclustering",
        "//mvsic/algorithms:benchmark",
        "//mvsic/core/kernels:capabilities",
    ],
)

py_library(
    name = "mvsic_py",
    srcs = ["__init__.py", "cli.py"],
    deps = [":_mvsic", "//mvsic/benchmark:python"],
)
