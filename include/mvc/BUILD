package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "mvkmeans",
    srcs = ["mvkmeans.cpp"],
    deps = [
      "@parlaylib//parlay:parallel",
      "@parlaylib//parlay:primitives",
      "@parlaylib//parlay:sequence",
      "@kmeans//lloyds:kmeans",
      "@parlayann//algorithms/utils:point_range",
      "@parlayann//algorithms/utils:euclidian_point",
      "@parlayann//algorithms/utils:mips_point",
      "@parlayann//algorithms/bench:parse_command_line",
      "@parlayann//algorithms/utils:types",
      "//mvc/utils:pointcloud",
      "//mvc/seeding:uniformlyrandom",
    ],
)