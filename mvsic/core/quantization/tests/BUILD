load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_test")

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "fastscan_test",
    srcs = ["fastscan_test.cpp"],
    linkopts = [
        "-pthread",
        "-fopenmp",
    ],
    deps = [
        "//mvsic/core/quantization/other_methods:fastscan",
        "//mvsic/core/quantization/other_methods:wrapper",
        "//mvsic/core/types:point_cloud_set",
    ],
)

cc_test(
    name = "correctness_fastscan",
    srcs = ["correctness_fastscan.cpp"],
    deps = [
        "//mvsic/core/quantization:fastscan",
        "@googletest//:gtest_main",
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:sequence",
    ],
)

cc_test(
    name = "turboquant_4bit_test",
    srcs = ["turboquant_4bit_test.cpp"],
    deps = [
        "//mvsic/core/quantization/other_methods:turboquant_4bit",
        "@googletest//:gtest_main",
        "@parlaylib//parlay:parallel",
    ],
)

cc_test(
    name = "bench_turboquant_mv",
    srcs = ["bench_turboquant_mv.cpp"],
    linkstatic = 1,
    deps = [
        "//mvsic/core/quantization:turboquant_mv",
        "//mvsic/core/quantization/other_methods:turboquant_4bit",
        "//mvsic/core/quantization/other_methods:wrapper",
        "//mvsic/core/types:chamfer_ip_point",
        "//mvsic/core/types:chamfer_l2_point",
        "//mvsic/core/types:point_cloud_set",
        "@google_benchmark//:benchmark_main",
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:sequence",
    ],
)

cc_test(
    name = "bench_fastscan_mv",
    srcs = ["bench_fastscan_mv.cpp"],
    linkstatic = 1,
    deps = [
        "//mvsic/core/quantization:fastscan_mv",
        "//mvsic/core/quantization/other_methods:fastscan",
        "//mvsic/core/quantization/other_methods:wrapper",
        "//mvsic/core/types:chamfer_ip_point",
        "//mvsic/core/types:chamfer_l2_point",
        "//mvsic/core/types:point_cloud_set",
        "@google_benchmark//:benchmark_main",
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:sequence",
    ],
)

cc_test(
    name = "bench_pq_mv",
    srcs = ["bench_pq_mv.cpp"],
    linkstatic = 1,
    deps = [
        "//mvsic/core/quantization:pq",
        "//mvsic/core/quantization:pq_mv",
        "//mvsic/core/quantization/other_methods:wrapper",
        "//mvsic/core/types:chamfer_ip_point",
        "//mvsic/core/types:chamfer_l2_point",
        "//mvsic/core/types:point_cloud_set",
        "@google_benchmark//:benchmark_main",
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:sequence",
    ],
)

cc_test(
    name = "bench_rabitq_mv",
    srcs = ["bench_rabitq_mv.cpp"],
    linkstatic = 1,
    deps = [
        "//mvsic/core/quantization:rabitq",
        "//mvsic/core/quantization:rabitq_mv",
        "//mvsic/core/quantization/other_methods:wrapper",
        "//mvsic/core/types:chamfer_ip_point",
        "//mvsic/core/types:chamfer_l2_point",
        "//mvsic/core/types:point_cloud_set",
        "@google_benchmark//:benchmark_main",
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:sequence",
    ],
)

cc_binary(
    name = "bench_turboquant_single",
    srcs = ["bench_turboquant_single.cpp"],
    deps = [
        "//mvsic/core/quantization:turboquant",
        "//mvsic/core/quantization/other_methods:turboquant_4bit",
        "@google_benchmark//:benchmark_main",
    ],
)

cc_test(
    name = "fastscan_microbench",
    srcs = ["fastscan_microbench.cc"],
    deps = [
        "//mvsic/core/quantization:fastscan",
        "@google_benchmark//:benchmark",
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:sequence",
    ],
)

cc_test(
    name = "fastscan_mv_microbench",
    srcs = ["fastscan_mv_microbench.cc"],
    deps = [
        "//mvsic/core/quantization:fastscan_mv",
        "@google_benchmark//:benchmark",
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:sequence",
    ],
)

cc_test(
    name = "fastscan_kernel_bench",
    srcs = ["fastscan_kernel_bench.cpp"],
    deps = [
        "@google_benchmark//:benchmark",
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:sequence",
    ],
)

cc_test(
    name = "turboquant_mv_microbench",
    srcs = ["turboquant_mv_microbench.cc"],
    deps = [
        "@google_benchmark//:benchmark",
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:sequence",
    ],
)
