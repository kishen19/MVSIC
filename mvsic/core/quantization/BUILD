load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "pq",
    hdrs = ["pq.h"],
    deps = [
        "//mvsic/core/utils:kmeans_util",
        "@eigen",
        "@parlayann//algorithms/utils:point_range",
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:primitives",
    ],
)

cc_library(
    name = "fastscan",
    hdrs = ["fastscan.h"],
    deps = [
        "//mvsic/core/utils:kmeans_util",
        "@eigen",
        "@parlayann//algorithms/utils:point_range",
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:primitives",
    ],
)

cc_library(
    name = "turboquant",
    hdrs = ["turboquant.h"],
    deps = [
        "@eigen",
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:sequence",
    ],
)

cc_library(
    name = "one_to_many_turboquant",
    hdrs = ["one_to_many_turboquant.h"],
    deps = [
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:sequence",
        "@rabitq//:rabitqlib",
    ],
)

cc_library(
    name = "byte_turboquant",
    hdrs = ["byte_turboquant.h"],
    deps = [
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:sequence",
        "@rabitq//:rabitqlib",
    ],
)

cc_library(
    name = "scann",
    hdrs = ["scann.h"],
    deps = [
        ":pq",
        "//mvsic/core/utils:kmeans_util",
        "@eigen",
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:primitives",
    ],
)

cc_library(
    name = "rabitq",
    hdrs = ["rabitq.h"],
    # ENABLE AVX512 FLAGS HERE
    copts = [
        "-mavx512f",
        "-mavx512vpopcntdq",
    ],
    deps = [
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:primitives",
        "@rabitq//:rabitqlib",
    ],
)

cc_library(
    name = "wrapper",
    hdrs = ["wrapper.h"],
    deps = [
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:sequence",
    ],
)

cc_test(
    name = "one_to_many_turboquant_test",
    srcs = ["one_to_many_turboquant_test.cpp"],
    copts = ["-march=native"],
    deps = [
        ":one_to_many_turboquant",
        "@googletest//:gtest_main",
        "@parlaylib//parlay:parallel",
    ],
)
