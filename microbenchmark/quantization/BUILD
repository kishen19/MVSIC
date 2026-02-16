load("@rules_cc//cc:defs.bzl", "cc_binary")

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "bench_pq_fastscan",
    srcs = ["bench_pq_fastscan.cpp"],
    deps = [
        "//mvsic/core/distance_measures:one_to_one",
        "//mvsic/core/quantization:fastscan",
        "//mvsic/core/quantization:pq",
        "//mvsic/core/quantization:rabitq",
        "@eigen",
        "@parlaylib//parlay:primitives",
    ],
)

cc_binary(
    name = "bench_chamfer_pq_fastscan",
    srcs = ["bench_chamfer_pq_fastscan.cpp"],
    deps = [
        "//mvsic/core/quantization:fastscan",
        "//mvsic/core/quantization:pq",
        "//mvsic/core/quantization:rabitq",
        "//mvsic/core/quantization:wrapper",
        "//mvsic/core/types:chamfer_ip_point",
        "//mvsic/core/types:chamfer_l2_point",
        "//mvsic/core/types:point_cloud_set",
        "//mvsic/core/utils:parse_command_line",
        "@eigen",
        "@parlaylib//parlay:primitives",
    ],
)
