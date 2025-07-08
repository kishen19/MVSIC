package(default_visibility = ["//visibility:public"])

cc_library(
    name = "muvera",
    hdrs = ["muvera.h"],
    deps = [
        "//src/common:index",
        "//src/muvera/fde:fixed_dimensional_encoding",
        "//src/utils:point_range",
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
        "//src/utils:chamfer_ip_point",
        "//src/utils:chamfer_l2_point",
        "//src/utils:parse_command_line",
        "//src/utils:point_cloud_set",
        "//src/utils:stats",
        "@eigen",
    ],
)
