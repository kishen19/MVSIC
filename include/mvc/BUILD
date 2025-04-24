package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "mvkmeans",
    srcs = ["mvkmeans.cpp"],
    deps = [
      "@parlaylib//parlay:parallel",
      "@parlaylib//parlay:primitives",
      "@parlaylib//parlay:sequence",
      "@kmeans//lloyds:kmeans",
      "//mvc/utils:pointcloud",
      "//mvc/seeding:uniformlyrandom",
    ],
)