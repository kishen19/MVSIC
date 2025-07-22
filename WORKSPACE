load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:local.bzl", "local_repository")
load("@bazel_tools//tools/cpp:cc_configure.bzl", "cc_configure")

cc_configure()

git_repository(
    name = "cpam",
    commit = "f0eaf81a4b4070a4ac56114e1eca17a308c26b02",
    remote = "https://github.com/ParAlg/CPAM.git",
    strip_prefix = "include/",
)

http_archive(
    name = "parlaylib",
    sha256 = "68c062ad116fd49d77651d7a24fb985aa66e8ec9ad05176b6af3ab5d29a16b1f",
    strip_prefix = "parlaylib-bazel/include/",
    urls = ["https://github.com/ParAlg/parlaylib/archive/refs/tags/bazel.tar.gz"],
)

http_archive(
    name = "googletest",
    sha256 = "b4870bf121ff7795ba20d20bcdd8627b8e088f2d1dab299a031c1034eddc93d5",
    strip_prefix = "googletest-release-1.11.0",
    urls = ["https://github.com/google/googletest/archive/release-1.11.0.tar.gz"],
)

git_repository(
    name = "parlayann",
    branch = "bazel",
    remote = "https://github.com/kishen19/ParlayANN.git",
    # remote = "https://github.com/cmuparlay/parlayann.git",
)

local_repository(
    name = "kmeans",
    path = "./external/kmeans/",
)

http_archive(
    name = "eigen",
    build_file_content = """
cc_library(
    name = "eigen",
    hdrs = glob(
        ["Eigen/**"],
        exclude = [
            "Eigen/src/OrderingMethods/Amd.h",
            "Eigen/src/SparseCholesky/**",
            "Eigen/Eigen",
            "Eigen/IterativeLinearSolvers",
            "Eigen/MetisSupport",
            "Eigen/Sparse",
            "Eigen/SparseCholesky",
            "Eigen/SparseLU",
        ],
    ),
    defines = [
        "EIGEN_MPL_ONLY",
        "EIGEN_NO_DEBUG",
    ],
    includes = ["."],
    visibility = ["//visibility:public"],
)
""",
    strip_prefix = "eigen-3.4-rc1",
    url = "https://gitlab.com/libeigen/eigen/-/archive/3.4-rc1/eigen-3.4-rc1.tar.gz",
)

git_repository(
    name = "faiss",
    build_file_content = """
cc_library(
    name = "faiss_core",
    srcs = glob([
        "faiss/**/*.cpp",
    ], exclude = [
        "faiss/gpu/**/*.cpp",
        "faiss/python/**/*.cpp",
    ]),
    hdrs = glob([
        "faiss/**/*.h",
    ], exclude = [
        "faiss/gpu/**/*.h",
        "faiss/python/**/*.h",
    ]),
    includes = ["."],
    copts = ["-fopenmp"],
    linkopts = ["-fopenmp", "-lopenblas"],
    visibility = ["//visibility:public"],
)
""",
    remote = "https://github.com/facebookresearch/faiss.git",
    tag = "v1.11.0",
)

http_archive(
    name = "absl",
    strip_prefix = "abseil-cpp-20240116.0",
    urls = ["https://github.com/abseil/abseil-cpp/archive/refs/tags/20240116.0.tar.gz"],
)

git_repository(
    name = "com_google_absl",
    remote = "https://github.com/abseil/abseil-cpp.git",
    tag = "20230125.2",
)

http_archive(
    name = "highway",
    strip_prefix = "highway-1.2.0",
    url = "https://github.com/google/highway/archive/refs/tags/1.2.0.zip",
)
