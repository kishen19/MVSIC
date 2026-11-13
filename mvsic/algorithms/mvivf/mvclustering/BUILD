load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "evals",
    hdrs = ["evals.h"],
    deps = [],
)

cc_library(
    name = "mvclustering_config",
    hdrs = ["mvclustering_config.h"],
    deps = ["//mvsic/core:types"],
)

cc_library(
    name = "mvclustering_impl",
    hdrs = [
        "mvclustering_impl.h",
        "mvclustering_8bit.h",
    ],
    deps = [
        ":mvclustering_config",
        "//mvsic/core/distance_measures:many_to_many",
        "//mvsic/algorithms/mvivf/mvclustering/seeding:uniformlyrandom",
        "//mvsic/core/quantization:turboquant_8bit_mv",
        "//mvsic/core/types:chamfer_ip_point",
        "//mvsic/core/types:point_cloud_set",
        "//mvsic/core/kmeans:kmeans_util",
        "//mvsic/core/kmeans:lloyds_kmeans",
        "//mvsic/core/utils:util",
        "@parlaylib//parlay:primitives",
    ],
)

cc_binary(
    name = "bench",
    srcs = ["bench.cpp"],
    deps = [
        ":evals",
        "//mvsic/algorithms/mvivf/mvclustering:mvclustering_impl",
        "//mvsic/core/utils:parse_command_line",
    ],
)

cc_library(
    name = "mvclustering",
    srcs = ["mvclustering.cc", "mvclustering_kernels.cc"],
    hdrs = ["mvclustering.h", "mvclustering_internal.h"],
    deps = [
        "//mvsic/core:data",
        "//mvsic/core:types",
        ":mvclustering_config",
        "//mvsic/algorithms/mvivf/mvclustering:mvclustering_impl",
        "//mvsic/core/quantization:turboquant_mv",
        "//mvsic/core/quantization:turboquant_8bit_mv",
        "//mvsic/core/quantization:turboquant_1bit",
        "//mvsic/core/quantization:turboquant_1bit_mv",
        "//mvsic/core/kernels:panels_avx2",
    ],
)

cc_test(
    name = "mvclustering_test",
    srcs = ["mvclustering_test.cc"],
    copts = ["-UNDEBUG"],
    linkstatic = True,
    deps = [
        ":mvclustering",
        "//mvsic/core/quantization:turboquant_1bit",
        "//mvsic/algorithms/mvivf/mvclustering:mvclustering_impl",
        "@googletest//:gtest_main",
    ],
)
