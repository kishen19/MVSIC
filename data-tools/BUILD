load("@rules_cc//cc:defs.bzl", "cc_binary")
load("@rules_python//python:defs.bzl", "py_binary", "py_library")

package(default_visibility = ["//visibility:public"])

# Direct-MKL ground-truth tool.  MKL is system-installed via
# intel-oneapi-mkl-devel; we link against /opt/intel/oneapi/mkl/latest/* with
# rpath embedded so the binary runs without LD_LIBRARY_PATH.  Threading uses
# Intel OpenMP runtime (libiomp5) which is independent of parlay's homegrown
# scheduler — no -fopenmp / clang-libomp/libgomp interplay required.
cc_binary(
    name = "compute_ground_truth",
    srcs = ["compute_ground_truth.cpp"],
    linkopts = [
        "-Wl,--no-as-needed",
        "-L/opt/intel/oneapi/mkl/latest/lib",
        "-L/opt/intel/oneapi/compiler/latest/lib",
        "-Wl,-rpath,/opt/intel/oneapi/mkl/latest/lib",
        "-Wl,-rpath,/opt/intel/oneapi/compiler/latest/lib",
        "-lmkl_intel_lp64",
        "-lmkl_intel_thread",
        "-lmkl_core",
        "-liomp5",
        "-lpthread",
        "-lm",
        "-ldl",
    ],
    deps = [
        "@mkl//:mkl_headers",
        "@parlaylib//parlay:primitives",
    ],
)

cc_binary(
    name = "convert_pointcloud_format",
    srcs = ["convert_pointcloud_format.cpp"],
    deps = [
        "//mvsic/core/utils:mmap",
        "//mvsic/core/utils:parse_command_line",
        "@parlaylib//parlay:primitives",
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
