load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "flat",
    hdrs = ["flat.h"],
    deps = [
        "//mvsic/core:index",
        "//mvsic/core:query_compression",
        "//mvsic/core/utils:util",
    ],
)

cc_binary(
    name = "bench",
    srcs = ["bench.cc"],
    deps = [
        ":flat",
        "//mvsic/core:bench_utils",
        "//mvsic/core:stats",
    ],
)
