package(default_visibility = ["//visibility:public"])

cc_library(
    name = "svheuristic",
    hdrs = ["svheuristic.h"],
    deps = [
        "//src/common:index",
        "//src/utils:ip_point",
        "//src/utils:kmeans_util",
        "//src/utils:l2_point",
        "//src/utils:point_range",
        "//src/utils:top_neighbors",
        "@parlaylib//parlay:primitives",
    ],
)

cc_binary(
    name = "bench",
    srcs = ["bench.cpp"],
    deps = [
        ":svheuristic",
        "//src/utils:chamfer_ip_point",
        "//src/utils:chamfer_l2_point",
        "//src/utils:ip_point",
        "//src/utils:l2_point",
        "//src/utils:parse_command_line",
        "//src/utils:point_cloud_set",
        "//src/utils:point_range",
        "//src/utils:stats",
    ],
)
