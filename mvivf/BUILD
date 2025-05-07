package(default_visibility = ["//visibility:public"])


cc_library(
    name = "index",
    hdrs = ["index.h"],
    deps = [
      "@parlaylib//parlay:primitives",
      "//mvc:mvkmeans",
      "//mvivf/utils:utils",
    ],
)

cc_binary(
    name = "bench",
    srcs = ["bench.cpp"],
    deps = [
      ":index",
      "@parlayann//algorithms/utils:point_range",
      "@parlayann//algorithms/utils:euclidian_point",
      "@parlayann//algorithms/utils:mips_point",
      "@parlayann//algorithms/bench:parse_command_line",
      "//mvc/utils:pointcloud",
      "//mvc/utils:chamferpoint",
      "//mvivf/utils:stats",
    ],
)