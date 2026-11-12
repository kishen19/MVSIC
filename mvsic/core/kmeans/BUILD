load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

# The compatibility include root preserves upstream includes such as
# "seeding/kmeans_utils.h", including in the untouched baseline checkout.
cc_library(
    name = "kmeans_utils",
    hdrs = ["seeding/kmeans_utils.h"],
    includes = ["."],
    deps = [
        "@parlaylib//parlay:delayed_sequence",
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:primitives",
        "@parlaylib//parlay:sequence",
    ],
)

# Graph-based and weighted trainers retained for the existing algorithms.
cc_library(
    name = "legacy",
    hdrs = glob([
        "lloyds/*.h",
        "lloyds_weighted/*.h",
        "seeding/*.h",
    ], exclude = ["seeding/kmeans_utils.h"]),
    includes = ["."],
    deps = [
        ":kmeans_utils",
        "@eigen",
        "@parlayann//algorithms/utils:graph",
        "@parlayann//algorithms/utils:point_range",
        "@parlayann//algorithms/utils:stats",
        "@parlayann//algorithms/utils:types",
        "@parlayann//algorithms/vamana:index",
        "@parlaylib//parlay:delayed_sequence",
        "@parlaylib//parlay:io",
        "@parlaylib//parlay:monoid",
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:primitives",
        "@parlaylib//parlay:sequence",
    ],
)

cc_library(
    name = "lloyds_kmeans",
    hdrs = ["lloyds_kmeans.h"],
    deps = [
        "//mvsic/core/quantization:turboquant_8bit_mv",
        "@eigen",
        ":kmeans_utils",
        "@parlayann//algorithms/utils:euclidian_point",
        "@parlayann//algorithms/utils:mips_point",
        "@parlayann//algorithms/utils:point_range",
        "@parlaylib//parlay:delayed_sequence",
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:primitives",
        "@parlaylib//parlay:sequence",
    ],
)

cc_library(
    name = "kmeans_util",
    hdrs = ["kmeans_util.h"],
    deps = [
        ":lloyds_kmeans",
        ":legacy",
        "@parlayann//algorithms/utils:euclidian_point",
        "@parlayann//algorithms/utils:mips_point",
        "@parlayann//algorithms/utils:point_range",
        "@parlaylib//parlay:primitives",
    ],
)
