# load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
# load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "lower_bounds",
    hdrs = ["lower_bounds.h"],
    deps = [
        "//mvc/utils:faiss_kmeans",
        "@parlaylib//parlay:sequence",
    ],
)

cc_library(
    name = "mvkmeans",
    hdrs = ["mvkmeans.h"],
    deps = [
        "@parlaylib//parlay:primitives",
        "@kmeans//lloyds:kmeans",
        "//mvc/seeding:uniformlyrandom",
        "//mvc/utils:faiss_kmeans",
        "//mvc/utils:chamfer_l2_point",
        "//mvc/utils:chamfer_ip_point",
        # "//mvc/utils:euclidian_point",
        "@parlayann//algorithms/utils:euclidian_point",
        # "//mvc/utils:mips_point",
        "@parlayann//algorithms/utils:mips_point",
        "//mvc/utils:point_cloud_set",
        # "//mvc/utils:point_range",
        "@parlayann//algorithms/utils:point_range",
        ":lower_bounds",
    ],
)

cc_binary(
    name = "bench",
    srcs = ["bench.cpp"],
    deps = [
        ":mvkmeans",
        "//mvc/utils:chamfer_ip_point",
        "//mvc/utils:chamfer_l2_point",
        "//mvc/utils:euclidian_point",
        "//mvc/utils:mips_point",
        "//mvc/utils:parse_command_line",
        "//mvc/utils:point_cloud_set",
        "//mvc/utils:point_range",
    ],
)
