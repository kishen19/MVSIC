package(default_visibility = ["//visibility:public"])

cc_library(
    name = "many_to_many",
    hdrs = ["many_to_many.h"],
    deps = [
        "@eigen",
        "@parlaylib//parlay:primitives",
    ],
)

cc_library(
    name = "one_to_one",
    hdrs = ["one_to_one.h"],
    deps = [
        ":NSGDist",
        "@eigen",
    ],
)

cc_library(
    name = "one_to_many",
    hdrs = ["one_to_many.h"],
    deps = [
        "@eigen",
        "@parlaylib//parlay:primitives",
    ],
)

cc_library(
    name = "NSGDist",
    hdrs = ["NSGDist.h"],
    deps = [
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:primitives",
    ],
)
