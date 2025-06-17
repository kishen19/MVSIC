package(default_visibility = ["//visibility:public"])

cc_library(
    name = "beam_search",
    hdrs = ["beam_search.h"],
    deps = [
        "//src/common:search_params",
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
        "//src/common:index",
        "//src/utils:top_neighbors",
        "@parlaylib//parlay:primitives",
    ],
)

cc_binary(
    name = "bench",
    srcs = ["bench.cpp"],
    deps = [
        ":vamana",
        "//src/utils:chamfer_ip_point",
        "//src/utils:chamfer_l2_point",
        "//src/utils:parse_command_line",
        "//src/utils:point_cloud_set",
        "//src/utils:stats",
    ],
)
