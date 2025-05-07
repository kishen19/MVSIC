package(default_visibility = ["//visibility:public"])


cc_library(
  name = "mvkmeans",
  hdrs = ["mvkmeans.h"],
  deps = [
    "@parlaylib//parlay:sequence",
    "@kmeans//lloyds:kmeans",
    "//mvc/seeding:uniformlyrandom",
    "//mvc/utils:faiss_kmeans",
  ],
)

cc_binary(
  name = "mvkmeans_runner",
  srcs = ["mvkmeans.cpp"],
  deps = [
    ":mvkmeans",
    "@parlayann//algorithms/utils:point_range",
    "@parlayann//algorithms/utils:euclidian_point",
    "@parlayann//algorithms/utils:mips_point",
    "@parlayann//algorithms/bench:parse_command_line",
    "//mvc/utils:pointcloud",
    "//mvc/utils:chamferpoint",
  ],
)