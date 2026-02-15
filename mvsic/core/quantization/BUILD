load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "pq",
    hdrs = ["pq.h"],
    deps = [
        "//mvsic/core/utils:kmeans_util",
        "@eigen",
        "@parlayann//algorithms/utils:point_range",
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:primitives",
    ],
)

cc_library(
    name = "fastscan",
    hdrs = ["fastscan.h"],
    deps = [
        "//mvsic/core/utils:kmeans_util",
        "@eigen",
        "@parlayann//algorithms/utils:point_range",
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:primitives",
    ],
)

cc_library(
    name = "scann",
    hdrs = ["scann.h"],
    deps = [
        ":pq",
        "//mvsic/core/utils:kmeans_util",
        "@eigen",
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:primitives",
    ],
)

cc_library(
    name = "rabitq",
    hdrs = ["rabitq.h"],
    # ENABLE AVX512 FLAGS HERE
    copts = [
        "-mavx512f",
        "-mavx512vpopcntdq",
    ],
    deps = [
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:primitives",
        "@rabitq//:rabitqlib",
    ],
)

cc_library(
    name = "wrapper",
    hdrs = ["wrapper.h"],
    deps = [
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:sequence",
    ],
)

# cc_binary(
#     name = "test_pq",
#     srcs = ["main.cpp"],
#     # ENABLE AVX512 FLAGS HERE
#     copts = [
#         "-march=native",
#         "-mavx512f",
#         "-mavx512vpopcntdq",
#     ],
#     # CRITICAL: RaBitQ uses OpenMP, so we must link against it.
#     linkopts = [
#         "-pthread",
#         "-fopenmp",
#     ],
#     deps = [
#         ":pq",
#         ":rabitq",
#         ":scann",
#         "//mvsic/core/utils:parse_command_line",
#         "@parlayann//algorithms/utils:euclidian_point",
#         "@parlayann//algorithms/utils:point_range",
#         "@parlaylib//parlay:parallel",
#     ],
# )

# cc_binary(
#     name = "test_pq_pc",
#     srcs = ["test_pq_pc.cpp"],
#     copts = [
#         "-march=native",
#         "-mavx512f",
#         "-mavx512vpopcntdq",
#     ],
#     linkopts = [
#         "-pthread",
#         "-fopenmp",
#     ],
#     deps = [
#         ":pq",
#         ":rabitq",
#         ":scann",
#         ":wrapper",
#         "//mvsic/core/types:chamfer_ip_point",
#         "//mvsic/core/types:chamfer_l2_point",
#         "//mvsic/core/types:point_cloud_set",
#         "//mvsic/core/utils:parse_command_line",
#         "@parlayann//algorithms/utils:euclidian_point",
#         "@parlaylib//parlay:parallel",
#     ],
# )
