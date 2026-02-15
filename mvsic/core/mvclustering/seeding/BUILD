load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "uniformlyrandom",
    hdrs = ["uniformlyrandom.h"],
    deps = [
        "//mvsic/core/types:point_cloud_set",
        "@parlaylib//parlay:primitives",
    ],
)
