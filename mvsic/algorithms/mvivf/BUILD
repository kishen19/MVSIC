load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

package(default_visibility = ["//visibility:public"])

# Bazel builds the selected compile-time ISA; CMake additionally installs
# isolated AVX2/AVX-512/VNNI backends with runtime capability selection.
cc_library(
    name = "mvivf",
    srcs = ["mvivf_build.cc", "mvivf_search.cc", "mvivf_kernels.cc"],
    hdrs = ["mvivf.h", "mvivf_internal.h"],
    deps = [
        "//mvsic/algorithms/mvivf/mvclustering:mvclustering",
        "//mvsic/core:index",
        "//mvsic/core:index_params",
        "//mvsic/core:query_compression",
        "//mvsic/core/distance_measures:one_to_many",
        "//mvsic/core/distance_measures:many_to_many",
        "//mvsic/core/quantization:turboquant_mv",
        "//mvsic/core/quantization:turboquant_1bit_mv",
        "//mvsic/core/quantization:turboquant_8bit_mv",
        "//mvsic/core/types:chamfer_ip_point",
        "//mvsic/core/utils:interval_heap",
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:sequence",
    ],
)

cc_test(
    name = "mvivf_test",
    srcs = ["mvivf_test.cc"],
    copts = ["-UNDEBUG"],
    deps = [
        ":mvivf",
        "@googletest//:gtest_main",
        "@parlaylib//parlay:parallel",
    ],
)
