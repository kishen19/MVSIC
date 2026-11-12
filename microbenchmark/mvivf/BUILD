load("@rules_cc//cc:defs.bzl", "cc_binary")

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "bench_dedup",
    srcs = ["bench_dedup.cpp"],
    deps = [
        "//mvsic/core/utils:parse_command_line",
        "//mvsic/core/utils:util",
        "@parlaylib//parlay:primitives",
    ],
)

cc_binary(
    name = "bench_sort",
    srcs = ["bench_sort.cpp"],
    deps = [
        "//mvsic/core/utils:parse_command_line",
        "//mvsic/core/utils:util",
        "@parlaylib//parlay:primitives",
    ],
)

cc_binary(
    name = "bench_fastscan_m2m",
    srcs = ["bench_fastscan_m2m.cpp"],
    deps = [
        "//mvsic/core/quantization:fastscan_mv",
        "//mvsic/core/types:chamfer_ip_point",
        "//mvsic/core/types:point_cloud_set",
        "//mvsic/core/utils:parse_command_line",
        "@parlaylib//parlay:primitives",
    ],
)

cc_binary(
    name = "bench_1btq_harley_seal",
    srcs = ["bench_1btq_harley_seal.cpp"],
    deps = [],
)

cc_binary(
    name = "bench_1btq_vbmi",
    srcs = ["bench_1btq_vbmi.cpp"],
    deps = [],
)
