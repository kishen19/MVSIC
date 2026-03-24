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

cc_library(
    name = "fastscan",
    hdrs = ["fastscan.h"],
    deps = [
        "//mvsic/core/utils:kmeans_util",
        "@eigen",
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:primitives",
    ],
)


cc_library(
    name = "turboquant_4bit",
    hdrs = ["turboquant_4bit.h"],
    deps = [
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:sequence",
        "@rabitq//:rabitqlib",
    ],
)


cc_library(
    name = "turboquant_pq_4bit",
    hdrs = [
        "turboquant_pq_4bit.h",
        "turboquant_pq_codebooks.h",
    ],
    deps = [
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:sequence",
        "@rabitq//:rabitqlib",
    ],
)


cc_library(
    name = "turboquant_low_bit",
    hdrs = ["turboquant_low_bit.h"],
    deps = [],
)


cc_library(
    name = "turboquant_pq_4bit_scalar",
    hdrs = [
        "turboquant_pq_4bit_scalar.h",
        "turboquant_pq_codebooks.h",
    ],
    deps = ["@rabitq//:rabitqlib"],
)


cc_library(
    name = "turboquant_pq_sym_scalar",
    hdrs = [
        "turboquant_pq_codebooks.h",
        "turboquant_pq_sym_scalar.h",
    ],
    deps = ["@rabitq//:rabitqlib"],
)


cc_library(
    name = "turboquant_byte",
    hdrs = ["turboquant_byte.h"],
    deps = [
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:sequence",
        "@rabitq//:rabitqlib",
    ],
)


cc_library(
    name = "turboquant_centered",
    hdrs = ["turboquant_centered.h"],
    deps = [
        ":turboquant_4bit",
    ],
)


cc_library(
    name = "wrapper",
    hdrs = ["wrapper.h"],
    deps = [
        ":turboquant_4bit",
        ":turboquant_byte",
        ":turboquant_pq_4bit",
        "@parlaylib//parlay:primitives",
    ],
)
