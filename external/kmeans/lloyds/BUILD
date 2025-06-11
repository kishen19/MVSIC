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
        ":pairwise",
        ":anns",
        ":multi-swap",
        "//seeding:kmeansplusplus",
        "//seeding:prefixdoubling",
        "//seeding:uniformlyrandom",
        "//seeding:kmeansparallel",
        "//seeding:ksetcover",
        # "//seeding:wards",
        #"//utils:evals",
        "@parlaylib//parlay:io",
        "@parlaylib//parlay:sequence",
        "@parlayann//algorithms/utils:point_range",
        "@parlayann//algorithms/utils:types",
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
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:primitives",
        "@parlaylib//parlay:sequence",
    ],
)
