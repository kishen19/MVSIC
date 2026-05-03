load("@rules_cc//cc:defs.bzl", "cc_test")

package(default_visibility = ["//visibility:public"])

cc_test(
    name = "interval_heap_test",
    srcs = ["interval_heap_test.cpp"],
    deps = [
        "//mvsic/core/utils:interval_heap",
        "@googletest//:gtest_main",
    ],
)
