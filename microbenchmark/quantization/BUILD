load("@rules_cc//cc:defs.bzl", "cc_binary")

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "bench_pq_fastscan",
    srcs = ["bench_pq_fastscan.cpp"],
    deps = [
        "//mvsic/core/distance_measures:one_to_one",
        "//mvsic/core/quantization:fastscan",
        "//mvsic/core/quantization:rabitq",
        "//mvsic/core/quantization:turboquant_4bit",
        "//mvsic/core/quantization:turboquant_pq_4bit",
        "//mvsic/core/quantization:turboquant_byte",
        "//mvsic/core/types:ip_point",
        "//mvsic/core/types:point_range",
        "//mvsic/core/utils:parse_command_line",
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
        "//mvsic/core/quantization:turboquant_4bit",
        "//mvsic/core/quantization:turboquant_pq_4bit",
        "//mvsic/core/quantization:turboquant_byte",
        "//mvsic/core/quantization:wrapper",
        "//mvsic/core/types:chamfer_ip_point",
        "//mvsic/core/types:chamfer_l2_point",
        "//mvsic/core/types:point_cloud_set",
        "//mvsic/core/utils:parse_command_line",
        "@eigen",
        "@parlaylib//parlay:primitives",
    ],
)

cc_binary(
    name = "bench_chamfer_overretrieve",
    srcs = ["bench_chamfer_overretrieve.cpp"],
    deps = [
        "//mvsic/core/quantization:fastscan",
        "//mvsic/core/quantization:pq",
        "//mvsic/core/quantization:rabitq",
        "//mvsic/core/quantization:turboquant_4bit",
        "//mvsic/core/quantization:turboquant_pq_4bit",
        "//mvsic/core/quantization:turboquant_byte",
        "//mvsic/core/quantization:wrapper",
        "//mvsic/core/types:chamfer_ip_point",
        "//mvsic/core/types:chamfer_l2_point",
        "//mvsic/core/types:point_cloud_set",
        "//mvsic/core/utils:parse_command_line",
        "@eigen",
        "@parlaylib//parlay:primitives",
    ],
)

cc_binary(
    name = "bench_overretrieve",
    srcs = ["bench_overretrieve.cpp"],
    deps = [
        "//mvsic/core/distance_measures:one_to_one",
        "//mvsic/core/quantization:fastscan",
        "//mvsic/core/quantization:rabitq",
        "//mvsic/core/quantization:turboquant_4bit",
        "//mvsic/core/quantization:turboquant_byte",
        "//mvsic/core/quantization:turboquant_pq_4bit",
        "//mvsic/core/quantization:turboquant_pq_4bit_scalar",
        "//mvsic/core/types:ip_point",
        "//mvsic/core/types:l2_point",
        "//mvsic/core/types:point_range",
        "//mvsic/core/utils:parse_command_line",
        "@eigen",
        "@parlaylib//parlay:primitives",
        "@parlaylib//parlay:sequence",
    ],
)

cc_binary(
    name = "bench_turboquant_4bit",
    srcs = ["bench_turboquant_4bit.cpp"],
    deps = [
        "//mvsic/core/quantization:turboquant_4bit",
        "@google_benchmark//:benchmark_main",
    ],
)

cc_binary(
    name = "bench_chamfer_leaf",
    srcs = ["bench_chamfer_leaf.cpp"],
    deps = [
        "//mvsic/core/quantization:fastscan",
        "//mvsic/core/quantization:pq",
        "//mvsic/core/quantization:rabitq",
        "//mvsic/core/quantization:turboquant_4bit",
        "//mvsic/core/quantization:turboquant_pq_4bit",
        "//mvsic/core/quantization:turboquant_byte",
        "//mvsic/core/quantization:wrapper",
        "//mvsic/core/types:chamfer_ip_point",
        "//mvsic/core/types:chamfer_l2_point",
        "//mvsic/core/types:point_cloud_set",
        "//mvsic/core/utils:parse_command_line",
        "@parlaylib//parlay:primitives",
    ],
)

cc_binary(
    name = "bench_chamfer_topk",
    srcs = ["bench_chamfer_topk.cpp"],
    deps = [
        "//mvsic/core/quantization:fastscan",
        "//mvsic/core/quantization:turboquant_4bit",
        "//mvsic/core/quantization:turboquant_pq_4bit",
        "//mvsic/core/quantization:wrapper",
        "//mvsic/core/types:chamfer_ip_point",
        "//mvsic/core/types:chamfer_l2_point",
        "//mvsic/core/types:point_cloud_set",
        "//mvsic/core/utils:parse_command_line",
        "@parlaylib//parlay:primitives",
    ],
)

cc_binary(
    name = "test_vnni_chamfer",
    srcs = ["test_vnni_chamfer.cpp"],
    deps = [
        "//mvsic/core/quantization:turboquant_4bit",
    ],
)

cc_binary(
    name = "measure_stretch_tq",
    srcs = ["measure_stretch_tq.cpp"],
    deps = [
        "//mvsic/core:stats",
        "//mvsic/core/quantization:rabitq",
        "//mvsic/core/quantization:turboquant_4bit",
        "//mvsic/core/quantization:turboquant_pq_4bit",
        "//mvsic/core/quantization:turboquant_byte",
        "//mvsic/core/quantization:turboquant_centered",
        "//mvsic/core/quantization:turboquant_low_bit",
        "//mvsic/core/types:ip_point",
        "//mvsic/core/types:l2_point",
        "//mvsic/core/types:point_range",
        "//mvsic/core/utils:parse_command_line",
        "@eigen",
        "@parlaylib//parlay:primitives",
    ],
)
