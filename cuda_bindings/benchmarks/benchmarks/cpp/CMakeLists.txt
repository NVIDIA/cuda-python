# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.24)
project(cuda_bindings_cpp_benchmarks LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CUDA_HOME_HINT "$ENV{CUDA_HOME}")
set(CONDA_PREFIX_HINT "$ENV{CONDA_PREFIX}")

# Find cuda.h (driver API header)
find_path(
    CUDA_DRIVER_INCLUDE_DIR
    cuda.h
    HINTS
        "${CUDA_HOME_HINT}/include"
        "${CONDA_PREFIX_HINT}/targets/x86_64-linux/include"
        "${CONDA_PREFIX_HINT}/include"
)

# Find libcuda (driver API library) — lives on the system, not in toolkit
find_library(
    CUDA_DRIVER_LIBRARY
    NAMES cuda
    HINTS
        "/usr/lib/x86_64-linux-gnu"
        "/usr/lib64"
        "${CUDA_HOME_HINT}/lib64/stubs"
        "${CUDA_HOME_HINT}/lib/stubs"
        "${CONDA_PREFIX_HINT}/targets/x86_64-linux/lib/stubs"
        "${CONDA_PREFIX_HINT}/lib/stubs"
)

if(NOT CUDA_DRIVER_INCLUDE_DIR)
    message(FATAL_ERROR "Could not find cuda.h. Ensure CUDA_HOME is set or install cuda-crt-dev.")
endif()

if(NOT CUDA_DRIVER_LIBRARY)
    message(FATAL_ERROR "Could not find libcuda. Ensure the NVIDIA driver is installed.")
endif()

add_executable(bench_pointer_attributes_cpp bench_pointer_attributes.cpp)
target_include_directories(bench_pointer_attributes_cpp PRIVATE "${CUDA_DRIVER_INCLUDE_DIR}")
target_link_libraries(bench_pointer_attributes_cpp PRIVATE "${CUDA_DRIVER_LIBRARY}")

add_executable(bench_ctx_device_cpp bench_ctx_device.cpp)
target_include_directories(bench_ctx_device_cpp PRIVATE "${CUDA_DRIVER_INCLUDE_DIR}")
target_link_libraries(bench_ctx_device_cpp PRIVATE "${CUDA_DRIVER_LIBRARY}")
