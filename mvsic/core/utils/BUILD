package(default_visibility = ["//visibility:public"])

cc_library(
    name = "csvfile",
    hdrs = ["csvfile.h"],
)

# cc_library(
#     name = "faiss_kmeans_util",
#     hdrs = ["faiss_kmeans_util.h"],
#     deps = [
#         "@faiss//:faiss_core",
#         "@parlaylib//parlay:primitives",
#     ],
# )

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
    name = "mmap",
    hdrs = ["mmap.h"],
    deps = [],
)

cc_library(
    name = "parse_command_line",
    hdrs = ["parse_command_line.h"],
)

cc_library(
    name = "top_neighbors",
    hdrs = ["top_neighbors.h"],
    deps = [
        ":util",
        "//mvsic/core/types:point_cloud_set",
        "@eigen",
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
