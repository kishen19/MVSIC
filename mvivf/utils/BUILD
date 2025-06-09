load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "csvfile",
    hdrs = ["csvfile.h"],
)

cc_library(
    name = "search_params",
    hdrs = ["search_params.h"],
)

cc_library(
    name = "stats",
    hdrs = ["stats.h"],
    deps = [
        ":csvfile",
        ":search_params",
        "//mvc/utils:point_cloud_set",
        "@parlaylib//parlay:primitives",
    ],
)

cc_library(
    name = "top_neighbors",
    hdrs = ["top_neighbors.h"],
    deps = [
        "//mvc/utils:point_cloud_set",
        "@parlaylib//parlay:primitives",
    ],
)
