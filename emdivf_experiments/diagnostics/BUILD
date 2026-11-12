load("@rules_cc//cc:defs.bzl", "cc_binary")

cc_binary(
    name = "probe_curve",
    srcs = ["probe_curve.cc"],
    deps = [
        "//mvsic/algorithms/emdivf:emd",
        "//mvsic/algorithms/emdivf:index",
        "//mvsic/algorithms:stats",
        "//mvsic/core/types:point_cloud_set",
        "@parlaylib//parlay:primitives",
    ],
)

cc_binary(
    name = "routing_prefix",
    srcs = ["routing_prefix.cc"],
    deps = [
        "//mvsic/algorithms/emdivf:emd",
        "//mvsic/algorithms/emdivf:index",
        "//mvsic/algorithms:stats",
        "//mvsic/core/types:point_cloud_set",
        "@parlaylib//parlay:primitives",
    ],
)
