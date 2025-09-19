package(default_visibility = ["//visibility:public"])

cc_library(
    name = "anns",
    hdrs = ["anns.h"],
    deps = [
        "@parlayann//algorithms/utils:stats",
        "@parlayann//algorithms/utils:types",
        "@parlayann//algorithms/vamana:index",
        "@parlaylib//parlay:delayed_sequence",
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:primitives",
        "@parlaylib//parlay:sequence",
    ],
)

cc_library(
    name = "kmeans",
    hdrs = ["kmeans.h"],
    deps = [
        ":anns",
        ":multi-swap",
        ":pairwise",
        "//seeding:kmeansparallel",
        "//seeding:kmeansplusplus",
        "//seeding:ksetcover",
        "//seeding:prefixdoubling",
        "//seeding:uniformlyrandom",
        "@parlayann//algorithms/utils:point_range",
        "@parlayann//algorithms/utils:types",
        "@parlaylib//parlay:io",
        "@parlaylib//parlay:sequence",
    ],
)

cc_library(
    name = "multi-swap",
    hdrs = ["multi-swap.h"],
    deps = [
        "//seeding:kmeans_utils",
        "//seeding:utils",
        "@parlaylib//parlay:monoid",
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:primitives",
        "@parlaylib//parlay:sequence",
    ],
)

cc_library(
    name = "pairwise",
    hdrs = ["pairwise.h"],
    deps = [
        "@eigen",
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:primitives",
        "@parlaylib//parlay:sequence",
    ],
)

cc_binary(
    name = "bench",
    srcs = ["bench.cpp"],
    deps = [
        ":kmeans",
        "@parlayann//algorithms/bench:parse_command_line",
        "@parlayann//algorithms/utils:euclidian_point",
        "@parlayann//algorithms/utils:mips_point",
        "@parlayann//algorithms/utils:point_range",
        "@parlaylib//parlay/internal:get_time",
    ],
)
