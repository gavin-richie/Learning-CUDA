cmake_minimum_required(VERSION 3.18)
project(lowp_quant LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Default target: RTX 4060 (Ada, sm_89). Override per GPU:
#   3090=86  4060/4090=89  H20=90  5090=120
#   e.g. cmake -DCMAKE_CUDA_ARCHITECTURES=120 ..
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(lowp_core STATIC
  src/quant_kernels.cu
  src/file_io.cpp
)
target_include_directories(lowp_core PUBLIC include)

add_executable(lowp src/main.cu)
target_link_libraries(lowp PRIVATE lowp_core)

enable_testing()

function(lowp_test name)
  add_executable(${name} tests/${name}.cu)
  target_link_libraries(${name} PRIVATE lowp_core)
  add_test(NAME ${name} COMMAND ${name})
endfunction()

lowp_test(test_formats)
lowp_test(test_io)
lowp_test(test_roundtrip)

# Optional hardware-accelerated comparison path (Transformer Engine, etc.).
# Not required for base acceptance; enable on Hopper/Blackwell machines.
option(ENABLE_TE_COMPARISON "Build Transformer Engine comparison path" OFF)
