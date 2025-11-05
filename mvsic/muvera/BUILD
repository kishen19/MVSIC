package(default_visibility = ["//visibility:public"])

cc_library(
    name = "muvera",
    hdrs = ["muvera.h"],
    deps = [
        "//mvsic/core:index",
        "//mvsic/core/types:point_range",
        "//mvsic/muvera/fde:fixed_dimensional_encoding",
        "@parlayann//algorithms/utils:euclidian_point",
        "@parlayann//algorithms/utils:graph",
        "@parlayann//algorithms/utils:mips_point",
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
