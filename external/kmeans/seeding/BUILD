package(default_visibility = ["//visibility:public"])

cc_library(
    name = "utils",
    hdrs = ["utils.h"],
)

cc_library(
    name = "kmeans_utils",
    hdrs = ["kmeans_utils.h"],
    deps = [
        "@parlaylib//parlay:delayed_sequence",
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:primitives",
        "@parlaylib//parlay:sequence",
    ],
)

cc_library(
    name = "kmeansplusplus",
    hdrs = ["kmeansplusplus.h"],
    deps = [
        "//seeding:kmeans_utils",
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:primitives",
        "@parlaylib//parlay:sequence",
    ],
)

cc_library(
    name = "prefixdoubling",
    hdrs = ["prefixdoubling.h"],
    deps = [
        "//seeding:kmeans_utils",
        "@parlaylib//parlay:delayed_sequence",
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:primitives",
        "@parlaylib//parlay:sequence",
    ],
)

cc_library(
    name = "uniformlyrandom",
    hdrs = ["uniformlyrandom.h"],
    deps = [
        "@parlaylib//parlay:delayed_sequence",
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:primitives",
        "@parlaylib//parlay:sequence",
    ],
)

cc_library(
    name = "kmeansparallel",
    hdrs = ["kmeansparallel.h"],
    deps = [
        "kmeansplusplus",
        "//seeding:kmeans_utils",
        "//seeding:utils",
        "@parlaylib//parlay:delayed_sequence",
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:primitives",
        "@parlaylib//parlay:sequence",
    ],
)

cc_library(
    name = "ksetcover",
    hdrs = ["ksetcover.h"],
    deps = [
        "kmeansplusplus",
        "//seeding:kmeans_utils",
        "//seeding:utils",
        "@parlaylib//parlay:delayed_sequence",
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:primitives",
        "@parlaylib//parlay:sequence",
    ],
)
