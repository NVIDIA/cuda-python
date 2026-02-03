# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os

# Site-packages bin directories where binaries might be found
# Based on NVIDIA wheel layouts (same for Linux and Windows)
CUDA_NVCC_BIN = os.path.join("nvidia", "cuda_nvcc", "bin")
NSIGHT_SYSTEMS_BIN = os.path.join("nvidia", "nsight_systems", "bin")
NSIGHT_COMPUTE_BIN = os.path.join("nvidia", "nsight_compute", "bin")

# Common CUDA binary utilities available on both Linux and Windows
SITE_PACKAGES_BINDIRS = {
    # Core compilation tools
    "nvcc": (CUDA_NVCC_BIN,),
    "nvdisasm": (CUDA_NVCC_BIN,),
    "cuobjdump": (CUDA_NVCC_BIN,),
    "nvprune": (CUDA_NVCC_BIN,),
    "fatbinary": (CUDA_NVCC_BIN,),
    "bin2c": (CUDA_NVCC_BIN,),
    "nvlink": (CUDA_NVCC_BIN,),
    # Runtime/debugging tools
    "cuda-gdb": (CUDA_NVCC_BIN,),
    "cuda-gdbserver": (CUDA_NVCC_BIN,),
    "compute-sanitizer": (CUDA_NVCC_BIN,),
    # Profiling tools
    "nvprof": (CUDA_NVCC_BIN,),
    "nsys": (NSIGHT_SYSTEMS_BIN,),
    "nsight-sys": (NSIGHT_SYSTEMS_BIN,),
    "ncu": (NSIGHT_COMPUTE_BIN,),
    "nsight-compute": (NSIGHT_COMPUTE_BIN,),
}

SUPPORTED_BINARIES_ALL = SUPPORTED_BINARIES = tuple(SITE_PACKAGES_BINDIRS.keys())

del CUDA_NVCC_BIN, NSIGHT_SYSTEMS_BIN, NSIGHT_COMPUTE_BIN, os
