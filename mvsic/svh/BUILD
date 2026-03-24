load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "svh_ivf",
    hdrs = ["svh_ivf.h"],
    deps = [
        "//mvsic/core:index",
        "//mvsic/core/types:ip_point",
        "//mvsic/core/types:l2_point",
        "//mvsic/core/types:point_range",
        "//mvsic/core/utils:kmeans_util",
        "//mvsic/core/utils:util",
    ],
)

cc_library(
    name = "svh_graph",
    hdrs = ["svh_graph.h"],
    deps = [
        "//mvsic/core:index",
        "//mvsic/core/types:io",
        "//mvsic/core/utils:util",
        "@parlayann//algorithms/utils:graph",
        "@parlayann//algorithms/utils:stats",
        "@parlayann//algorithms/utils:types",
        "@parlayann//algorithms/vamana:index",
    ],
)

cc_binary(
    name = "bench",
    srcs = ["bench.cpp"],
    deps = [
        ":svh_graph",
        ":svh_ivf",
        "//mvsic/core:stats",
        "//mvsic/core/types:chamfer_ip_point",
        "//mvsic/core/types:chamfer_l2_point",
        "//mvsic/core/types:point_cloud_set",
        "//mvsic/core/utils:parse_command_line",
    ],
)
