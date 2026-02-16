load("@rules_cc//cc:defs.bzl", "cc_test")

package(default_visibility = ["//visibility:public"])

cc_test(
    name = "one_to_one_test",
    srcs = ["one_to_one_test.cpp"],
    size = "small",
    deps = [
        "//mvsic/core/distance_measures:one_to_one",
        "@googletest//:gtest_main",
    ],
)
