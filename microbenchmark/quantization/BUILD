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
        "//mvsic/core/quantization:turboquant",
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
        "//mvsic/core/quantization:turboquant",
        "//mvsic/core/quantization:one_to_many_turboquant",
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
    name = "bench_chamfer_rabitq",
    srcs = ["bench_chamfer_rabitq.cpp"],
    deps = [
        "//mvsic/core/quantization:rabitq",
        "//mvsic/core/quantization:wrapper",
        "//mvsic/core/types:chamfer_ip_point",
        "//mvsic/core/types:chamfer_l2_point",
        "//mvsic/core/types:point_cloud_set",
        "//mvsic/core/utils:parse_command_line",
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
        "//mvsic/core/quantization:turboquant",
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
        "//mvsic/core/quantization:pq",
        "//mvsic/core/quantization:rabitq",
        "//mvsic/core/quantization:turboquant",
        "//mvsic/core/utils:parse_command_line",
        "@eigen",
        "@parlaylib//parlay:primitives",
    ],
)

cc_binary(
    name = "bench_one_to_many_turboquant",
    srcs = ["bench_one_to_many_turboquant.cc"],
    copts = [
        "-mavx512f",
        "-mavx512bw",
        "-mavx512vl",
    ],
    deps = [
        "//mvsic/core/quantization:one_to_many_turboquant",
        "@google_benchmark//:benchmark_main",
    ],
)

cc_binary(
    name = "test_vnni_chamfer",
    srcs = ["test_vnni_chamfer.cc"],
    deps = [
        "//mvsic/core/quantization:one_to_many_turboquant",
    ],
)
