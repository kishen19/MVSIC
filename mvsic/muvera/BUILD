load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "muvera",
    hdrs = ["muvera.h"],
    deps = [
        "//mvsic/core:index",
        "//mvsic/core:query_compression",
        "//mvsic/core/quantization:fastscan",
        "//mvsic/core/quantization:pq",
        "//mvsic/core/quantization:rabitq",
        "//mvsic/core/quantization:variant_io",
        "//mvsic/core/types:io",
        "//mvsic/muvera/fde:fixed_dimensional_encoding",
        "@parlayann//algorithms/utils:beamSearch",
        "@parlayann//algorithms/utils:euclidian_point",
        "@parlayann//algorithms/utils:graph",
        "@parlayann//algorithms/utils:mips_point",
        "@parlayann//algorithms/utils:point_range",
        "@parlayann//algorithms/utils:stats",
        "@parlayann//algorithms/utils:types",
        "@parlayann//algorithms/vamana:index",
    ],
)

cc_binary(
    name = "bench",
    srcs = ["bench.cpp"],
    deps = [
        ":muvera",
        "//mvsic/core:stats",
        "//mvsic/core/types:chamfer_ip_point",
        "//mvsic/core/types:chamfer_l2_point",
        "//mvsic/core/types:point_cloud_set",
        "//mvsic/core/utils:parse_command_line",
        "@eigen",
    ],
)

cc_binary(
    name = "bench_build",
    srcs = ["bench_build.cpp"],
    deps = [
        ":muvera",
        "//mvsic/core:bench_utils",
    ],
)

cc_binary(
    name = "bench_search",
    srcs = ["bench_search.cpp"],
    deps = [
        ":muvera",
        "//mvsic/core:bench_utils",
    ],
)

cc_binary(
    name = "bench_search_all",
    srcs = ["bench_search_all.cpp"],
    deps = [
        ":muvera",
        "//mvsic/core:bench_utils",
    ],
)

cc_binary(
    name = "bench_search_seq",
    srcs = ["bench_search_seq.cpp"],
    defines = ["PARLAY_SEQUENTIAL"],
    deps = [
        ":muvera",
        "//mvsic/core:bench_utils",
    ],
)
