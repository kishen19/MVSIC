load("@rules_cc//cc:defs.bzl", "cc_binary")

cc_binary(
    name = "fiqa_parent_pool",
    srcs = ["fiqa_parent_pool.cc"],
    deps = [
        "//mvsic/algorithms/emdivf:emd",
        "//mvsic/algorithms/emdivf:index",
        "//mvsic/algorithms:stats",
        "//mvsic/core/types:point_cloud_set",
        "@parlaylib//parlay:primitives",
    ],
)

cc_binary(
    name = "document_first_audit",
    srcs = ["document_first_audit.cc"],
    deps = [
        "//mvsic/algorithms/emdivf:index",
        "//mvsic/core/quantization:turboquant_8bit_mv",
        "//mvsic/core/types:point_cloud_set",
        "@parlaylib//parlay:primitives",
    ],
)
