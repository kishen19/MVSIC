load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "legacy_index",
    hdrs = ["legacy_index.h"],
    deps = [
        ":index_params",
        ":search_params",
        "//mvsic/core/distance_measures:one_to_many",
        "//mvsic/core/quantization:fastscan_mv",
        "//mvsic/core/quantization:fastscan",
        "//mvsic/core/quantization:pq",
        "//mvsic/core/quantization:pq_mv",
        "//mvsic/core/quantization:rabitq",
        "//mvsic/core/quantization:rabitq_mv",
        "//mvsic/core/quantization:turboquant",
        "//mvsic/core/quantization:turboquant_mv",
        "//mvsic/core/quantization:pqtq",
        "//mvsic/core/quantization:pqtq_mv",
        "//mvsic/core/quantization:turboquant_1bit",
        "//mvsic/core/quantization:turboquant_1bit_mv",
        "//mvsic/core/quantization:turboquant_1bit_asym_mv",
        "//mvsic/core/quantization:turboquant_8bit",
        "//mvsic/core/quantization:turboquant_8bit_mv",
        "//mvsic/core/types:chamfer_ip_point",
        "//mvsic/core/types:chamfer_l2_point",
        "//mvsic/core/types:point_cloud_set",
        "@parlayann//algorithms/utils:beamSearch",
        "@parlayann//algorithms/utils:euclidian_point",
        "@parlayann//algorithms/utils:mips_point",
        "@parlayann//algorithms/utils:point_range",
        "@parlaylib//parlay:primitives",
    ],
)

cc_library(
    name = "query_compression",
    hdrs = ["query_compression.h"],
    deps = [
        ":index_params",
        ":search_params",
        "//mvsic/core/types:point_cloud_set",
        "@eigen",
        "@parlaylib//parlay:primitives",
    ],
)

cc_library(
    name = "index_params",
    hdrs = ["index_params.h"],
    deps = [
        "//mvsic/algorithms/mvivf/mvclustering:mvclustering_config",
    ],
)

cc_library(
    name = "search_params",
    hdrs = ["search_params.h"],
)



cc_library(
    name = "types",
    hdrs = ["types.h"],
)

cc_library(
    name = "data",
    srcs = ["data.cc"],
    hdrs = ["data.h"],
)

cc_library(
    name = "index",
    srcs = ["index.cc"],
    hdrs = ["index.h"],
    deps = [":data", ":types", ":index_params", ":search_params"],
)
