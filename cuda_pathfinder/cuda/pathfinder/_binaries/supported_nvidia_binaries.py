# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os

# Common CUDA binary utilities available on both Linux and Windows
SUPPORTED_BINARIES_ALL = (
    # Core compilation tools
    "nvcc",
    "nvdisasm",
    "cuobjdump",
    "nvprune",
    "fatbinary",
    "bin2c",
    "nvlink",
    # Runtime/debugging tools
    "cuda-gdb",
    "cuda-gdbserver",
    "compute-sanitizer",
    # Profiling tools
    "nvprof",
    "nsys",
    "nsight-sys",
    "ncu",
    "nsight-compute",
)

SUPPORTED_BINARIES = SUPPORTED_BINARIES_ALL

# Site-packages bin directories where binaries might be found
# Based on NVIDIA wheel layouts (same for Linux and Windows)
CUDA_NVCC_BIN = os.path.join("nvidia", "cuda_nvcc", "bin")
NSIGHT_SYSTEMS_BIN = os.path.join("nvidia", "nsight_systems", "bin")
NSIGHT_COMPUTE_BIN = os.path.join("nvidia", "nsight_compute", "bin")

SITE_PACKAGES_BINDIRS = {
    "nvcc": (CUDA_NVCC_BIN,),
    "nvdisasm": (CUDA_NVCC_BIN,),
    "cuobjdump": (CUDA_NVCC_BIN,),
    "nvprune": (CUDA_NVCC_BIN,),
    "fatbinary": (CUDA_NVCC_BIN,),
    "bin2c": (CUDA_NVCC_BIN,),
    "nvlink": (CUDA_NVCC_BIN,),
    "cuda-gdb": (CUDA_NVCC_BIN,),
    "cuda-gdbserver": (CUDA_NVCC_BIN,),
    "compute-sanitizer": (CUDA_NVCC_BIN,),
    "nvprof": (CUDA_NVCC_BIN,),
    "nsys": (NSIGHT_SYSTEMS_BIN,),
    "nsight-sys": (NSIGHT_SYSTEMS_BIN,),
    "ncu": (NSIGHT_COMPUTE_BIN,),
    "nsight-compute": (NSIGHT_COMPUTE_BIN,),
}

del CUDA_NVCC_BIN, NSIGHT_SYSTEMS_BIN, NSIGHT_COMPUTE_BIN, os
