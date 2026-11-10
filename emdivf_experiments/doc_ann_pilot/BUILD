load("@rules_cc//cc:defs.bzl", "cc_binary")

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "vamana_bucket_pilot",
    srcs = ["vamana_bucket_pilot.cc"],
    deps = [
        "//mvsic/emdivf:index",
        "//mvsic/core:search_params",
        "//mvsic/core/types:chamfer_ip_point",
        "//mvsic/core/types:point_cloud_set",
        "//mvsic/vamana:beam_search",
        "//mvsic/vamana:graph",
    ],
)

cc_binary(
    name = "mvivf_bucket_pilot",
    srcs = ["mvivf_bucket_pilot.cc"],
    deps = [
        "//mvsic/emdivf:index",
        "//mvsic/mvivf:mvivf",
        "//mvsic/core/types:chamfer_ip_point",
        "//mvsic/core/types:point_cloud_set",
    ],
)
