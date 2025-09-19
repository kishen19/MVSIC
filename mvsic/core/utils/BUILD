package(default_visibility = ["//visibility:public"])

cc_library(
    name = "chamfer_ip_point",
    hdrs = ["chamfer_ip_point.h"],
    deps = [
        ":ip_point",
        "@eigen",
        "@parlaylib//parlay:primitives",
    ],
)

cc_library(
    name = "chamfer_l2_point",
    hdrs = ["chamfer_l2_point.h"],
    deps = [
        ":l2_point",
        "@eigen",
        "@parlaylib//parlay:primitives",
    ],
)

cc_library(
    name = "csvfile",
    hdrs = ["csvfile.h"],
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
    name = "ip_point",
    hdrs = ["ip_point.h"],
    deps = [
        ":NSGDist",
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:primitives",
        "@parlaylib//parlay/internal:file_map",
    ],
)

cc_library(
    name = "kmeans_util",
    hdrs = ["kmeans_util.h"],
    deps = [
        "@kmeans//lloyds:kmeans",
        "@parlayann//algorithms/utils:euclidian_point",
        "@parlayann//algorithms/utils:mips_point",
        "@parlayann//algorithms/utils:point_range",
        "@parlaylib//parlay:primitives",
    ],
)

cc_library(
    name = "l2_point",
    hdrs = ["l2_point.h"],
    deps = [
        ":NSGDist",
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:primitives",
        "@parlaylib//parlay/internal:file_map",
    ],
)

cc_library(
    name = "mmap",
    hdrs = ["mmap.h"],
    deps = [],
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
        ":mmap",
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
    name = "top_neighbors",
    hdrs = ["top_neighbors.h"],
    deps = [
        ":point_cloud_set",
        ":util",
        "@parlaylib//parlay:primitives",
    ],
)

cc_library(
    name = "util",
    hdrs = ["util.h"],
    deps = [
        "@highway//hwy/contrib/sort:vqsort",
        "@parlaylib//parlay:primitives",
    ],
)

cc_binary(
    name = "test_scann",
    srcs = ["test_scann.cpp"],
    deps = [
        "@scann//scann/hashes/internal:asymmetric_hashing_impl",
    ],
)
