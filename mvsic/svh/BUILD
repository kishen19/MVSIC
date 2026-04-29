load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "svh_ivf",
    hdrs = ["svh_ivf.h"],
    deps = [
        "//mvsic/core:index",
        "//mvsic/core:query_compression",
        "//mvsic/core/quantization:variant_io",
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
        "//mvsic/core:query_compression",
        "//mvsic/core/quantization:variant_io",
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

cc_binary(
    name = "quick_bench",
    srcs = ["quick_bench.cpp"],
    deps = [
        ":svh_graph",
        ":svh_ivf",
        "//mvsic/core:bench_utils",
    ],
)

cc_binary(
    name = "bench_build",
    srcs = ["bench_build.cpp"],
    deps = [
        ":svh_graph",
        ":svh_ivf",
        "//mvsic/core:bench_utils",
    ],
)

cc_binary(
    name = "bench_search",
    srcs = ["bench_search.cpp"],
    deps = [
        ":svh_graph",
        ":svh_ivf",
        "//mvsic/core:bench_utils",
    ],
)

cc_binary(
    name = "bench_search_all",
    srcs = ["bench_search_all.cpp"],
    deps = [
        ":svh_graph",
        ":svh_ivf",
        "//mvsic/core:bench_utils",
    ],
)

cc_binary(
    name = "bench_search_seq",
    srcs = ["bench_search_seq.cpp"],
    defines = ["PARLAY_SEQUENTIAL"],
    deps = [
        ":svh_graph",
        ":svh_ivf",
        "//mvsic/core:bench_utils",
    ],
)
