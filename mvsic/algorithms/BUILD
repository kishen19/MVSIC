load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "benchmark",
    srcs = ["benchmark.cc"],
    hdrs = ["benchmark.h"],
    deps = ["//mvsic/core:index", "@parlaylib//parlay:parallel"],
)

cc_library(
    name = "stats",
    hdrs = ["stats.h"],
    deps = [
        "//mvsic/core:search_params",
        "//mvsic/core/utils:csvfile",
        "@parlaylib//parlay:primitives",
    ],
)

cc_library(
    name = "bench_utils",
    hdrs = ["bench_utils.h"],
    deps = [
        "//mvsic/core:query_compression",
        "//mvsic/core:search_params",
        ":stats",
        "//mvsic/core/types:chamfer_ip_point",
        "//mvsic/core/types:chamfer_l2_point",
        "//mvsic/core/types:point_cloud_set",
        "//mvsic/core/utils:parse_command_line",
        "@eigen",
        "@parlaylib//parlay:primitives",
    ],
)
