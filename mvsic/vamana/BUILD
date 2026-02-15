load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "beam_search",
    hdrs = ["beam_search.h"],
    deps = [
        "//mvsic/core:search_params",
        "@parlaylib//parlay:io",
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:primitives",
        "@parlaylib//parlay:random",
    ],
)

cc_library(
    name = "graph",
    hdrs = ["graph.h"],
    deps = [
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:primitives",
        "@parlaylib//parlay/internal:file_map",
    ],
)

cc_library(
    name = "vamana",
    hdrs = ["vamana.h"],
    deps = [
        ":beam_search",
        ":graph",
        "//mvsic/core:index",
        "//mvsic/core/quantization:fastscan",
        "//mvsic/core/quantization:pq",
        "//mvsic/core/quantization:rabitq",
        "//mvsic/core/quantization:wrapper",
    ],
)

cc_binary(
    name = "bench",
    srcs = ["bench.cpp"],
    deps = [
        ":vamana",
        "//mvsic/core:stats",
        "//mvsic/core/types:chamfer_ip_point",
        "//mvsic/core/types:chamfer_l2_point",
        "//mvsic/core/types:point_cloud_set",
        "//mvsic/core/utils:parse_command_line",
    ],
)
