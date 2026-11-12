load("@rules_cc//cc:defs.bzl", "cc_binary")

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "vamana_bucket_pilot",
    srcs = ["vamana_bucket_pilot.cc"],
    deps = [
        "//mvsic/algorithms/emdivf:index",
        "//mvsic/core:search_params",
        "//mvsic/core/types:chamfer_ip_point",
        "//mvsic/core/types:point_cloud_set",
        "//mvsic/algorithms/vamana:beam_search",
        "//mvsic/algorithms/vamana:graph",
    ],
)
