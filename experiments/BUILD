package(default_visibility = ["//visibility:public"])


cc_binary(
    name = "mvivf_benchmark",
    srcs = ["mvivf_benchmark.cpp"],
    deps = [
      "//mvc/utils:chamfer_ip_point",
      "//mvc/utils:chamfer_l2_point",
      "//mvc/utils:parse_command_line",
      "//mvc/utils:point_cloud_set",
      "//mvivf/utils:stats",
      "//mvivf:mvivf",
    ],
)