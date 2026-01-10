package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "pq_microbenchmark",
    srcs = ["pq_microbenchmark.cpp"],
    copts = [
        "-march=native",
        "-mavx512f",
        "-mavx512vpopcntdq",
    ],
    linkopts = [
        "-pthread",
        "-fopenmp",
    ],
    deps = [
        "//mvsic/core/quantization:pq",
        "//mvsic/core/quantization:rabitq",
        "//mvsic/core/quantization:scann",
        "//mvsic/core/quantization:wrapper",
        "//mvsic/core/types:chamfer_ip_point",
        "//mvsic/core/types:chamfer_l2_point",
        "//mvsic/core/types:point_cloud_set",
        "//mvsic/core/utils:parse_command_line",
    ],
)

cc_binary(
    name = "pq_speed",
    srcs = ["pq_speed.cpp"],
    copts = [
        "-march=native",
        "-mavx512f",
        "-mavx512bw",
        "-mavx512dq",
        "-mavx512vl",
        "-mavx512vpopcntdq",
    ],
    linkopts = [
        "-pthread",
        "-fopenmp",
    ],
    deps = [
        "//mvsic/core/quantization:fastscan",
        "//mvsic/core/quantization:pq",
        "//mvsic/core/quantization:rabitq",
        "//mvsic/core/quantization:scann",
        "//mvsic/core/quantization:wrapper",
        "//mvsic/core/types:chamfer_ip_point",
        "//mvsic/core/types:chamfer_l2_point",
        "//mvsic/core/types:point_cloud_set",
        "//mvsic/core/utils:parse_command_line",
    ],
)

cc_binary(
    name = "bench_pq_fastscan",
    srcs = ["bench_pq_fastscan.cpp"],
    deps = [
        "//mvsic/core/distance_measures:one_to_one",
        "//mvsic/core/quantization:fastscan",
        "//mvsic/core/quantization:pq",
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
        "//mvsic/core/quantization:wrapper",
        "//mvsic/core/types:chamfer_ip_point",
        "//mvsic/core/types:chamfer_l2_point",
        "//mvsic/core/types:point_cloud_set",
        "//mvsic/core/utils:parse_command_line",
        "@eigen",
        "@parlaylib//parlay:primitives",
    ],
)
