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
