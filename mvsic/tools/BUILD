load("@rules_cc//cc:defs.bzl", "cc_binary")

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "measure_codebook_time",
    srcs = ["measure_codebook_time.cpp"],
    deps = [
        "//mvsic/core/quantization:fastscan_mv",
        "//mvsic/core/quantization:pq_mv",
        "//mvsic/core/quantization:pqtq_mv",
        "//mvsic/core/quantization:rabitq_mv",
        "//mvsic/core/quantization:turboquant_1bit_mv",
        "//mvsic/core/quantization:turboquant_mv",
        "//mvsic/core/types:chamfer_ip_point",
        "//mvsic/core/types:chamfer_l2_point",
        "//mvsic/core/types:point_cloud_set",
        "//mvsic/core/utils:parse_command_line",
        "@parlaylib//parlay:primitives",
    ],
)
