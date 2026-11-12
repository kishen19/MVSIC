load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "capabilities",
    srcs = ["capabilities.cc"],
    hdrs = ["capabilities.h"],
)

cc_library(
    name = "dispatch",
    srcs = ["dispatch.cc", "dot_intel.cc"],
    hdrs = ["dispatch.h"],
    deps = [":capabilities"],
)

cc_test(
    name = "dispatch_test",
    srcs = ["dispatch_test.cc"],
    copts = ["-UNDEBUG"],
    deps = [":dispatch", "@googletest//:gtest_main"],
)

cc_library(
    name = "panels_avx2",
    hdrs = ["panels_avx2.h"],
)

cc_test(
    name = "panels_test",
    srcs = ["panels_test.cc"],
    copts = ["-mavx2", "-UNDEBUG"],
    deps = [
        ":capabilities",
        ":panels_avx2",
        "//mvsic/algorithms/mvivf:mvivf",
        "//mvsic/core:data",
        "//mvsic/core/quantization:turboquant_mv",
        "//mvsic/core/quantization:turboquant_1bit_mv",
        "//mvsic/core/quantization:turboquant_8bit_mv",
    ],
)

cc_test(
    name = "panels_avx512_test",
    srcs = ["panels_test.cc"],
    copts = ["-march=skylake-avx512", "-UNDEBUG"],
    tags = ["manual", "requires-avx512"],
    deps = [
        ":capabilities",
        ":panels_avx2",
        "//mvsic/algorithms/mvivf:mvivf",
        "//mvsic/core:data",
        "//mvsic/core/quantization:turboquant_mv",
        "//mvsic/core/quantization:turboquant_1bit_mv",
        "//mvsic/core/quantization:turboquant_8bit_mv",
    ],
)
