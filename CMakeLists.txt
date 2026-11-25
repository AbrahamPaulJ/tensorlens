cmake_minimum_required(VERSION 3.25)

# The IDA plugin needs the IDA SDK (https://github.com/HexRaysSA/ida-sdk); point IDASDK at a
# checkout. Without it, only the core library, the command-line scanner and the tests build.
option(TENSORLENS_PLUGIN "Build the IDA plugin (needs IDASDK)" ON)
option(TENSORLENS_TESTS "Build the core tests" ON)

if(TENSORLENS_PLUGIN AND (DEFINED ENV{IDASDK} OR DEFINED IDASDK))
  if(NOT IDASDK)
    set(IDASDK "$ENV{IDASDK}")
  endif()
  # Accept either the repository root or its src/ folder (the SDK's own scripts want src/).
  if(EXISTS "${IDASDK}/src/cmake/idasdk_init.cmake")
    set(IDASDK "${IDASDK}/src")
  endif()
  set(ENV{IDASDK} "${IDASDK}")
  include("${IDASDK}/cmake/idasdk_init.cmake")
  set(_tl_plugin ON)
else()
  set(_tl_plugin OFF)
endif()

project(tensorlens VERSION 0.1.0 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
if(MSVC)
  add_compile_options(/utf-8 /EHsc)
  # IDA's stub libraries are release-only: match their CRT everywhere.
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
endif()

add_library(tl_miniz STATIC third_party/miniz/miniz.c)
target_include_directories(tl_miniz PUBLIC third_party/miniz)

add_library(tensorlens_core STATIC
  core/model.cpp core/tflite.cpp core/onnx.cpp core/ncnn.cpp core/scan.cpp core/report.cpp core/inflate.cpp)
target_include_directories(tensorlens_core PUBLIC core)
target_link_libraries(tensorlens_core PUBLIC tl_miniz)

add_executable(tensorlens-scan cli/main.cpp)
target_link_libraries(tensorlens-scan PRIVATE tensorlens_core tl_miniz)

if(TENSORLENS_TESTS)
  enable_testing()
  add_executable(tensorlens_tests tests/test_core.cpp)
  target_link_libraries(tensorlens_tests PRIVATE tensorlens_core)
  add_test(NAME core COMMAND tensorlens_tests ${CMAKE_CURRENT_SOURCE_DIR}/tests/data)
endif()

if(_tl_plugin)
  find_package(idasdk REQUIRED)
  add_subdirectory(plugin)
endif()
