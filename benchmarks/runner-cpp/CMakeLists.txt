# Standalone runner that exercises the Apache datasketches-cpp library over the
# shared benchmark datasets and emits rows in the shared CSV schema, so the
# reporter can join C++ reference results alongside the Rust runners.

cmake_minimum_required(VERSION 3.16)
project(runner_cpp CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Disable datasketches-cpp's own tests/benchmarks before pulling it in: we only
# need its header-only INTERFACE libraries, not its test executables.
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)

add_subdirectory(
  ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/datasketches-cpp
  ${CMAKE_BINARY_DIR}/datasketches-cpp
)

add_executable(runner_cpp main.cpp)
# The datasketches INTERFACE target links hll/theta/count and others, but not
# the header-only filters library, so link it explicitly for bloom_filter.hpp.
target_link_libraries(runner_cpp PRIVATE datasketches filters)
