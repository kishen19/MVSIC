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
        "@parlaylib//parlay:primitives",
    ],
)

cc_binary(
    name = "bench",
    srcs = ["bench.cpp"],
    deps = [
        ":vamana",
        "//mvsic/core:stats",
        "//mvsic/core/utils:chamfer_ip_point",
        "//mvsic/core/utils:chamfer_l2_point",
        "//mvsic/core/utils:parse_command_line",
        "//mvsic/core/utils:point_cloud_set",
    ],
)
