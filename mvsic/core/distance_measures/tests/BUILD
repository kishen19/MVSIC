load("@rules_cc//cc:defs.bzl", "cc_test")

package(default_visibility = ["//visibility:public"])

cc_test(
    name = "many_to_many_test",
    srcs = ["many_to_many_test.cpp"],
    deps = [
        "//mvsic/core/distance_measures:many_to_many",
        "//mvsic/core/types:chamfer_ip_point",
        "//mvsic/core/types:chamfer_l2_point",
        "//mvsic/core/types:point_cloud_set",
        "@googletest//:gtest_main",
    ],
)

cc_test(
    name = "one_to_many_test",
    srcs = ["one_to_many_test.cpp"],
    deps = [
        "//mvsic/core/distance_measures:one_to_many",
        "//mvsic/core/types:chamfer_ip_point",
        "//mvsic/core/types:chamfer_l2_point",
        "//mvsic/core/types:point_cloud_set",
        "@googletest//:gtest_main",
    ],
)
