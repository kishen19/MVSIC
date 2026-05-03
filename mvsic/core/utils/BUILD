load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "csvfile",
    hdrs = ["csvfile.h"],
)

cc_library(
    name = "interval_heap",
    hdrs = ["interval_heap.h"],
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
    name = "lloyds_kmeans",
    hdrs = ["lloyds_kmeans.h"],
    deps = [
        "//mvsic/core/quantization:turboquant_8bit_mv",
        "@eigen",
        "@kmeans//lloyds:kmeans",
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
        "@kmeans//lloyds:kmeans",
        "@kmeans//lloyds_weighted:kmeans_weighted",
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
        ":interval_heap",
        "@highway//hwy/contrib/sort:vqsort",
        "@parlaylib//parlay:primitives",
    ],
)
