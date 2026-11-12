cmake_minimum_required(VERSION 3.24)
project(mvsic VERSION 0.1.0 LANGUAGES CXX)

option(MVSIC_BUILD_PYTHON "Build the Python extension" ON)
option(MVSIC_WHEEL "Install Python wheel layout" OFF)
option(BUILD_TESTING "Build correctness tests" OFF)
option(MVSIC_BUILD_COMPARISON "Build controlled comparison drivers" OFF)
set(MVSIC_BASELINE_SOURCE_DIR "" CACHE PATH "Read-only original MVSIC checkout for comparison")

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR NOT CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  message(FATAL_ERROR "This release supports Linux x86-64 (AVX2/FMA minimum)")
endif()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
include(GNUInstallDirs)
include(cmake/Dependencies.cmake)

if(MVSIC_WHEEL)
  set(CMAKE_INSTALL_LIBDIR mvsic/.libs)
  set(CMAKE_INSTALL_INCLUDEDIR mvsic/include)
endif()

function(mvsic_target target)
  target_link_libraries(${target} PRIVATE mvsic_dependencies)
  target_include_directories(${target} PRIVATE ${PROJECT_SOURCE_DIR})
  target_compile_definitions(${target} PRIVATE HOMEGROWN)
  target_compile_options(${target} PRIVATE -march=x86-64-v3 -mcx16 -pthread -fvisibility-inlines-hidden)
  set_target_properties(${target} PROPERTIES INSTALL_RPATH "$ORIGIN")
endfunction()

add_library(mvsic_core SHARED
  mvsic/core/data.cc mvsic/core/index.cc mvsic/core/kernels/runtime.cc
  mvsic/core/kernels/capabilities.cc mvsic/core/kernels/dispatch.cc
  mvsic/core/kernels/dot_intel.cc)
mvsic_target(mvsic_core)

# Isolate inline codec/Eigen symbols across ISAs. The shared Parlay runtime
# retains one scheduler across these libraries, including nested parallel work.
foreach(isa avx2 avx512 vnni)
  add_library(mvsic_${isa} SHARED
    mvsic/algorithms/mvivf/mvivf_kernels.cc
    mvsic/algorithms/mvivf/mvclustering/mvclustering_kernels.cc)
  mvsic_target(mvsic_${isa})
  target_link_libraries(mvsic_${isa} PRIVATE mvsic_core)
  target_compile_definitions(mvsic_${isa} PRIVATE
    MVSIC_KERNEL_FACTORY=make_kernels_${isa}
    MVSIC_CLUSTERING_FACTORY=make_clustering_kernels_${isa})
  target_compile_options(mvsic_${isa} PRIVATE -fvisibility=hidden)
  target_link_options(mvsic_${isa} PRIVATE -Wl,-Bsymbolic -Wl,--exclude-libs,ALL)
endforeach()
target_compile_options(mvsic_avx512 PRIVATE -march=skylake-avx512)
target_compile_options(mvsic_vnni PRIVATE -march=skylake-avx512 -mavx512vnni)

add_library(mvsic SHARED
  mvsic/algorithms/mvivf/mvivf_build.cc
  mvsic/algorithms/mvivf/mvivf_search.cc
  mvsic/algorithms/mvivf/mvivf_dispatch.cc
  mvsic/algorithms/mvivf/mvclustering/mvclustering.cc
  mvsic/algorithms/benchmark.cc)
add_library(mvsic::mvsic ALIAS mvsic)
mvsic_target(mvsic)
target_compile_features(mvsic PUBLIC cxx_std_20)
target_link_libraries(mvsic PUBLIC mvsic_core PRIVATE mvsic_avx2 mvsic_avx512 mvsic_vnni)
target_include_directories(mvsic PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

install(TARGETS mvsic mvsic_core mvsic_avx2 mvsic_avx512 mvsic_vnni
  EXPORT mvsicTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
foreach(header core/types.h core/data.h core/index.h core/index_params.h core/search_params.h
    algorithms/mvivf/mvivf.h algorithms/mvivf/mvclustering/mvclustering.h
    algorithms/mvivf/mvclustering/mvclustering_config.h algorithms/benchmark.h)
  get_filename_component(directory ${header} DIRECTORY)
  install(FILES mvsic/${header} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mvsic/${directory})
endforeach()
include(CMakePackageConfigHelpers)
configure_package_config_file(cmake/mvsicConfig.cmake.in
  ${PROJECT_BINARY_DIR}/mvsicConfig.cmake INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mvsic)
write_basic_package_version_file(${PROJECT_BINARY_DIR}/mvsicConfigVersion.cmake
  VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(EXPORT mvsicTargets NAMESPACE mvsic:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mvsic)
install(FILES ${PROJECT_BINARY_DIR}/mvsicConfig.cmake ${PROJECT_BINARY_DIR}/mvsicConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mvsic)

if(MVSIC_BUILD_PYTHON)
  find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_mvsic MODULE NO_EXTRAS mvsic/python/bindings.cpp)
  target_compile_definitions(_mvsic PRIVATE MVSIC_RUNTIME_ISA_DISPATCH=1)
  target_link_libraries(_mvsic PRIVATE mvsic)
  if(MVSIC_WHEEL)
    set_target_properties(_mvsic PROPERTIES INSTALL_RPATH "$ORIGIN/.libs")
  else()
    set_target_properties(_mvsic PROPERTIES INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
  endif()
  install(TARGETS _mvsic LIBRARY DESTINATION mvsic)
  install(FILES mvsic/__init__.py mvsic/cli.py DESTINATION mvsic)
  install(DIRECTORY mvsic/benchmark DESTINATION mvsic
    FILES_MATCHING PATTERN "*.py" PATTERN "*.json" PATTERN "*.yaml" PATTERN "*.yml" PATTERN "README.md")
endif()

if(BUILD_TESTING)
  enable_testing()
  foreach(test mvivf kernels_dispatch panels panels_avx512)
    if(test STREQUAL "mvivf")
      set(source mvsic/algorithms/mvivf/mvivf_test.cc)
    elseif(test STREQUAL "kernels_dispatch")
      set(source mvsic/core/kernels/dispatch_test.cc)
    else()
      set(source mvsic/core/kernels/panels_test.cc)
    endif()
    add_executable(${test}_test ${source})
    mvsic_target(${test}_test)
    target_compile_options(${test}_test PRIVATE -UNDEBUG)
    if(test STREQUAL "panels_avx512")
      target_compile_options(${test}_test PRIVATE -march=skylake-avx512)
    endif()
    target_link_libraries(${test}_test PRIVATE mvsic)
    add_test(NAME ${test} COMMAND ${test}_test)
    set_tests_properties(${test} PROPERTIES ENVIRONMENT "PARLAY_NUM_THREADS=4")
    if(test STREQUAL "panels_avx512")
      set_tests_properties(${test} PROPERTIES SKIP_RETURN_CODE 77)
    endif()
  endforeach()
  include(FetchContent)
  FetchContent_Declare(googletest URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  FetchContent_MakeAvailable(googletest)
  add_executable(mvclustering_test mvsic/algorithms/mvivf/mvclustering/mvclustering_test.cc)
  mvsic_target(mvclustering_test)
  target_link_libraries(mvclustering_test PRIVATE mvsic GTest::gtest_main)
  target_link_libraries(kernels_dispatch_test PRIVATE GTest::gtest_main)
  target_link_libraries(mvivf_test PRIVATE GTest::gtest_main)
  add_test(NAME mvclustering COMMAND mvclustering_test)
  set_tests_properties(mvclustering PROPERTIES ENVIRONMENT "PARLAY_NUM_THREADS=4")
  foreach(backend one two)
    add_library(runtime_test_${backend} SHARED mvsic/core/kernels/runtime_test_backend.cc)
    mvsic_target(runtime_test_${backend})
    target_compile_definitions(runtime_test_${backend} PRIVATE
      MVSIC_RUNTIME_TEST_FACTORY=mvsic_runtime_test_${backend})
    target_compile_options(runtime_test_${backend} PRIVATE -fvisibility=hidden)
    target_link_options(runtime_test_${backend} PRIVATE -Wl,-Bsymbolic)
    target_link_libraries(runtime_test_${backend} PRIVATE mvsic_core)
  endforeach()
  add_executable(runtime_test mvsic/core/kernels/runtime_test.cc)
  mvsic_target(runtime_test)
  target_link_libraries(runtime_test PRIVATE mvsic_core runtime_test_one runtime_test_two)
  add_test(NAME shared_scheduler COMMAND runtime_test)
endif()

if(MVSIC_BUILD_COMPARISON)
  foreach(isa avx2 avx512)
    add_executable(mvsic_inner_assignment_${isa} tests/refactor/inner_assignment_bench.cc)
    mvsic_target(mvsic_inner_assignment_${isa})
    target_link_libraries(mvsic_inner_assignment_${isa} PRIVATE mvsic_core)
  endforeach()
  target_compile_options(mvsic_inner_assignment_avx512 PRIVATE -march=skylake-avx512)
  add_executable(mvsic_native_bench tests/refactor/native_bench.cc)
  mvsic_target(mvsic_native_bench)
  target_link_libraries(mvsic_native_bench PRIVATE mvsic)
  if(MVSIC_BASELINE_SOURCE_DIR)
    foreach(driver mvsic_baseline_bench mvsic_baseline_avx2)
      add_executable(${driver} tests/refactor/native_bench.cc)
      target_compile_features(${driver} PRIVATE cxx_std_20)
      target_compile_definitions(${driver} PRIVATE MVSIC_BASELINE HOMEGROWN)
      target_compile_options(${driver} PRIVATE -mcx16 -pthread)
      target_include_directories(${driver} BEFORE PRIVATE ${MVSIC_BASELINE_SOURCE_DIR})
      target_link_libraries(${driver} PRIVATE mvsic_baseline_dependencies)
    endforeach()
    target_compile_options(mvsic_baseline_bench PRIVATE -march=native)
    # A correctness reference without the original AVX-512 TQ4 padding bug.
    # Performance comparisons continue to use the native baseline above.
    target_compile_options(mvsic_baseline_avx2 PRIVATE -march=x86-64-v3)
    if(MVSIC_BUILD_PYTHON)
      # Preserve the original Python conversion/GIL path for the IP serving variant.
      # This subset's build cost is not the full legacy extension's build cost.
      pybind11_add_module(_mvsic_baseline MODULE NO_EXTRAS
        tests/refactor/baseline_bindings.cpp
        ${MVSIC_BASELINE_SOURCE_DIR}/mvsic/python/register_common.cc)
      target_compile_definitions(_mvsic_baseline PRIVATE HOMEGROWN)
      target_compile_options(_mvsic_baseline PRIVATE -march=native -mcx16 -pthread)
      target_include_directories(_mvsic_baseline BEFORE PRIVATE ${MVSIC_BASELINE_SOURCE_DIR})
      target_link_libraries(_mvsic_baseline PRIVATE mvsic_baseline_dependencies)
    endif()
  endif()
endif()
