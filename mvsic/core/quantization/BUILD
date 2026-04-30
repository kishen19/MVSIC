load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "fastscan",
    hdrs = ["fastscan.h"],
    deps = [
        "//mvsic/core/utils:kmeans_util",
        "@eigen",
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:sequence",
    ],
)

cc_library(
    name = "fastscan_mv",
    hdrs = ["fastscan_mv.h"],
    deps = [
        "//mvsic/core/utils:kmeans_util",
        "@eigen",
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:sequence",
    ],
)

cc_library(
    name = "pq",
    hdrs = ["pq.h"],
    deps = [
        "//mvsic/core/utils:kmeans_util",
        "@eigen",
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:sequence",
    ],
)

cc_library(
    name = "pq_mv",
    hdrs = ["pq_mv.h"],
    deps = [
        "//mvsic/core/utils:kmeans_util",
        "@eigen",
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:sequence",
    ],
)

cc_library(
    name = "rabitq",
    hdrs = ["rabitq.h"],
    deps = [
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:sequence",
        "@rabitq//:rabitqlib",
    ],
)

cc_library(
    name = "rabitq_mv",
    hdrs = ["rabitq_mv.h"],
    deps = [
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:sequence",
        "@rabitq//:rabitqlib",
    ],
)

cc_library(
    name = "turboquant",
    hdrs = [
        "turboquant.h",
        "turboquant_utils.h",
    ],
    deps = [
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:sequence",
        "@rabitq//:rabitqlib",
    ],
)

cc_library(
    name = "turboquant_mv",
    hdrs = [
        "turboquant_mv.h",
        "turboquant_utils.h",
    ],
    deps = [
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:sequence",
    ],
)

cc_library(
    name = "pqtq",
    hdrs = [
        "pqtq.h",
        "pqtq_codebooks.h",
    ],
    deps = [
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:sequence",
        "@rabitq//:rabitqlib",
    ],
)

cc_library(
    name = "pqtq_mv",
    hdrs = [
        "pqtq_mv.h",
    ],
    deps = [
        ":pqtq",
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:sequence",
    ],
)

cc_library(
    name = "turboquant_1bit_mv",
    hdrs = [
        "turboquant_1bit_mv.h",
        "turboquant_utils.h",
    ],
    deps = [
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:sequence",
        "@rabitq//:rabitqlib",
    ],
)

cc_library(
    name = "turboquant_8bit_mv",
    hdrs = [
        "turboquant_8bit_mv.h",
        "turboquant_utils.h",
    ],
    deps = [
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:sequence",
        "@rabitq//:rabitqlib",
    ],
)

cc_library(
    name = "variant_io",
    hdrs = ["variant_io.h"],
    deps = [
        "//mvsic/core:index_params",
    ],
)
