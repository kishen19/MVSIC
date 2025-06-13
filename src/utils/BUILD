# load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "chamfer_ip_point",
    hdrs = ["chamfer_ip_point.h"],
    deps = [
        "@eigen",
        "@parlaylib//parlay:primitives",
    ],
)

cc_library(
    name = "chamfer_l2_point",
    hdrs = ["chamfer_l2_point.h"],
    deps = [
        "@eigen",
        "@parlaylib//parlay:primitives",
    ],
)

cc_library(
    name = "csvfile",
    hdrs = ["csvfile.h"],
)

cc_library(
    name = "euclidian_point",
    hdrs = ["euclidian_point.h"],
    deps = [
        ":NSGDist",
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:primitives",
        "@parlaylib//parlay/internal:file_map",
    ],
)

cc_library(
    name = "faiss_kmeans_util",
    hdrs = ["faiss_kmeans_util.h"],
    deps = [
        "@faiss//:faiss_core",
        "@parlaylib//parlay:primitives",
    ],
)

cc_library(
    name = "kmeans_util",
    hdrs = ["kmeans_util.h"],
    deps = [
        "@kmeans//lloyds:kmeans",
        "@parlaylib//parlay:primitives",
    ],
)

cc_library(
    name = "mips_point",
    hdrs = ["mips_point.h"],
    deps = [
        ":NSGDist",
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:primitives",
        "@parlaylib//parlay/internal:file_map",
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

cc_library(
    name = "parse_command_line",
    hdrs = ["parse_command_line.h"],
)

cc_library(
    name = "point_cloud_set",
    hdrs = ["point_cloud_set.h"],
    deps = [
        "@parlaylib//parlay:primitives",
    ],
)

cc_library(
    name = "point_range",
    hdrs = ["point_range.h"],
    deps = [
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:primitives",
        "@parlaylib//parlay/internal:file_map",
    ],
)

cc_library(
    name = "stats",
    hdrs = ["stats.h"],
    deps = [
        ":csvfile",
        "//src/common:search_params",
        "@parlaylib//parlay:primitives",
    ],
)

cc_library(
    name = "top_neighbors",
    hdrs = ["top_neighbors.h"],
    deps = [
        ":point_cloud_set",
        "@parlaylib//parlay:primitives",
    ],
)
