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

# Find nvrtc.h and libnvrtc (for runtime compilation benchmarks)
find_path(
    NVRTC_INCLUDE_DIR
    nvrtc.h
    HINTS
        "${CUDA_HOME_HINT}/include"
        "${CONDA_PREFIX_HINT}/targets/x86_64-linux/include"
        "${CONDA_PREFIX_HINT}/include"
)

find_library(
    NVRTC_LIBRARY
    NAMES nvrtc
    HINTS
        "${CUDA_HOME_HINT}/lib64"
        "${CUDA_HOME_HINT}/lib"
        "${CONDA_PREFIX_HINT}/targets/x86_64-linux/lib"
        "${CONDA_PREFIX_HINT}/lib"
)

if(NOT CUDA_DRIVER_INCLUDE_DIR)
    message(FATAL_ERROR "Could not find cuda.h. Ensure CUDA_HOME is set or install cuda-crt-dev.")
endif()

if(NOT CUDA_DRIVER_LIBRARY)
    message(FATAL_ERROR "Could not find libcuda. Ensure the NVIDIA driver is installed.")
endif()

# Helper: add a benchmark that only needs the driver API
function(add_driver_benchmark name)
    add_executable(${name}_cpp ${name}.cpp)
    target_include_directories(${name}_cpp PRIVATE "${CUDA_DRIVER_INCLUDE_DIR}")
    target_link_libraries(${name}_cpp PRIVATE "${CUDA_DRIVER_LIBRARY}")
endfunction()

# Helper: add a benchmark that needs driver API + NVRTC
function(add_nvrtc_benchmark name)
    add_executable(${name}_cpp ${name}.cpp)
    target_include_directories(${name}_cpp PRIVATE "${CUDA_DRIVER_INCLUDE_DIR}" "${NVRTC_INCLUDE_DIR}")
    target_link_libraries(${name}_cpp PRIVATE "${CUDA_DRIVER_LIBRARY}" "${NVRTC_LIBRARY}")
endfunction()

# Driver-only benchmarks
add_driver_benchmark(bench_pointer_attributes)
add_driver_benchmark(bench_ctx_device)
add_driver_benchmark(bench_stream)
add_driver_benchmark(bench_event)

# NVRTC benchmarks (require nvrtc for kernel compilation)
if(NVRTC_INCLUDE_DIR AND NVRTC_LIBRARY)
    add_nvrtc_benchmark(bench_launch)
else()
    message(WARNING "NVRTC not found — skipping bench_launch. Install cuda-nvrtc-dev.")
endif()
