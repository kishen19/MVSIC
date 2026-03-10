load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "rabitq_fast",
    hdrs = ["rabitq_fast.h"],
    deps = [
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:primitives",
        "@rabitq//:rabitqlib",
    ],
)

cc_library(
    name = "rabitq_sym",
    hdrs = ["rabitq_sym.h"],
    deps = [
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:primitives",
        "@rabitq//:rabitqlib",
    ],
)

cc_library(
    name = "turboquant_naive",
    hdrs = ["turboquant_naive.h"],
    deps = [
        "@eigen",
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:sequence",
    ],
)

cc_library(
    name = "turboquant_pq_4bit_dd",
    hdrs = ["turboquant_pq_4bit_dd.h"],
    deps = [
        ":turboquant_pq_4bit",
        "//mvsic/core/utils:kmeans_util",
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:sequence",
    ],
)