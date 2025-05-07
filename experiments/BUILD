package(default_visibility = ["//visibility:public"])


cc_binary(
    name = "mvivf_benchmark",
    srcs = ["mvivf_benchmark.cpp"],
    deps = [
      "@parlayann//algorithms/utils:point_range",
      "@parlayann//algorithms/utils:euclidian_point",
      "@parlayann//algorithms/utils:mips_point",
      "@parlayann//algorithms/bench:parse_command_line",
      "//mvc/utils:pointcloud",
      "//mvc/utils:chamferpoint",
      "//mvivf/utils:stats",
      "//mvivf:index",
    ],
)