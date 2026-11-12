load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "emd",
    srcs = ["emd.cc"],
    hdrs = ["emd.h"],
    linkstatic = True,
    deps = ["@eigen"],
)

cc_library(
    name = "index",
    srcs = ["index.cc"],
    hdrs = ["index.h"],
    linkstatic = True,
    deps = [
        ":emd",
        "//mvsic/core/distance_measures:one_to_many",
        "//mvsic/core/types:chamfer_ip_point",
        "//mvsic/core/types:point_cloud_set",
        "@parlaylib//parlay:primitives",
    ],
)

cc_library(
    name = "emdivf",
    srcs = ["emdivf.cc"],
    hdrs = ["emdivf.h"],
    linkstatic = True,
    deps = [
        ":index",
        "//mvsic/core:query_compression",
        "//mvsic/core:search_params",
        "//mvsic/core/quantization:turboquant_1bit_mv",
        "//mvsic/core/quantization:turboquant_8bit_mv",
        "//mvsic/core/quantization:turboquant_mv",
        "//mvsic/core/types:chamfer_ip_point",
        "//mvsic/algorithms/vamana:beam_search",
        "//mvsic/algorithms/vamana:graph",
        "@parlaylib//parlay:primitives",
        "@parlaylib//parlay:random",
    ],
)

cc_binary(
    name = "bench_build",
    srcs = ["bench_build.cc"],
    deps = [
        ":emd",
        ":index",
        "//mvsic/core/distance_measures:one_to_many",
        "//mvsic/core/quantization:turboquant_8bit_mv",
        "//mvsic/core/types:chamfer_ip_point",
        "//mvsic/core/types:point_cloud_set",
        "@parlaylib//parlay:primitives",
    ],
)

cc_binary(
    name = "bench_search",
    srcs = ["bench_search.cc"],
    deps = [
        ":emdivf",
        ":index",
        "@parlaylib//parlay:primitives",
    ],
)

cc_test(
    name = "emd_test",
    srcs = ["emd_test.cc"],
    deps = [":emd", "@googletest//:gtest_main"],
)
