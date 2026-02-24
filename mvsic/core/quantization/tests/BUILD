load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_test")

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "fastscan_test",
    srcs = ["fastscan_test.cpp"],
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
        "//mvsic/core/quantization:wrapper",
        "//mvsic/core/types:point_cloud_set",
    ],
)

cc_test(
    name = "turboquant_4bit_test",
    srcs = ["turboquant_4bit_test.cpp"],
    copts = ["-march=native"],
    deps = [
        "//mvsic/core/quantization:turboquant_4bit",
        "@googletest//:gtest_main",
        "@parlaylib//parlay:parallel",
    ],
)
