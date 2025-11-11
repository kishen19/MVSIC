package(default_visibility = ["//visibility:public"])

cc_library(
    name = "chamfer_ip_point",
    hdrs = ["chamfer_ip_point.h"],
    deps = [
        ":ip_point",
        "//mvsic/core/distance_measures:one_to_one",
        "@eigen",
        "@parlaylib//parlay:primitives",
    ],
)

cc_library(
    name = "chamfer_l2_point",
    hdrs = ["chamfer_l2_point.h"],
    deps = [
        ":l2_point",
        "//mvsic/core/distance_measures:one_to_one",
        "@eigen",
        "@parlaylib//parlay:primitives",
    ],
)

cc_library(
    name = "io",
    hdrs = ["io.h"],
    deps = [
        "@parlayann//algorithms/utils:graph",
        "@parlayann//algorithms/utils:point_range",
    ],
)

cc_library(
    name = "ip_point",
    hdrs = ["ip_point.h"],
    deps = [
        "//mvsic/core/distance_measures:one_to_one",
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:primitives",
        "@parlaylib//parlay/internal:file_map",
    ],
)

cc_library(
    name = "l2_point",
    hdrs = ["l2_point.h"],
    deps = [
        "//mvsic/core/distance_measures:one_to_one",
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:primitives",
        "@parlaylib//parlay/internal:file_map",
    ],
)

cc_library(
    name = "point_cloud_set",
    hdrs = ["point_cloud_set.h"],
    deps = [
        "//mvsic/core/distance_measures:one_to_many",
        "//mvsic/core/utils:mmap",
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
    name = "pq_helper",
    hdrs = ["pq_helper.h"],
    deps = [
        ":point_cloud_set",
        "@eigen",
        "@scann//scann/data_format:dataset",
        "@scann//scann/hashes/asymmetric_hashing2:indexing",
        "@scann//scann/hashes/asymmetric_hashing2:querying",
        "@scann//scann/hashes/asymmetric_hashing2:training",
        "@scann//scann/proto:scann_cc_proto",
        "@scann//scann/utils:types",
    ],
)

cc_library(
    name = "quantized_chamfer_point",
    hdrs = ["quantized_chamfer_point.h"],
    deps = [
        ":chamfer_ip_point",
        ":chamfer_l2_point",
        ":quantized_point_cloud_set",
        "@parlaylib//parlay:primitives",
    ],
)

cc_library(
    name = "quantized_point_cloud_set",
    hdrs = ["quantized_point_cloud_set.h"],
    deps = [
        ":point_cloud_set",
        ":pq_helper",
        "//mvsic/core/utils:mmap",
        "@parlaylib//parlay:primitives",
    ],
)
