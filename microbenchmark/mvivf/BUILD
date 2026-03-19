load("@rules_cc//cc:defs.bzl", "cc_binary")

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "bench_mvivf_leaf_diversity",
    srcs = ["bench_mvivf_leaf_diversity.cpp"],
    deps = [
        "//mvsic/mvivf:mvivf",
        "//mvsic/core:stats",
        "//mvsic/core/types:chamfer_ip_point",
        "//mvsic/core/types:chamfer_l2_point",
        "//mvsic/core/types:point_cloud_set",
        "//mvsic/core/utils:parse_command_line",
        "@parlaylib//parlay:primitives",
    ],
)

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

