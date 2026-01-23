# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


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
SITE_PACKAGES_BINDIRS = {
    "nvcc": ("nvidia/cuda_nvcc/bin",),
    "nvdisasm": ("nvidia/cuda_nvcc/bin",),
    "cuobjdump": ("nvidia/cuda_nvcc/bin",),
    "nvprune": ("nvidia/cuda_nvcc/bin",),
    "fatbinary": ("nvidia/cuda_nvcc/bin",),
    "bin2c": ("nvidia/cuda_nvcc/bin",),
    "nvlink": ("nvidia/cuda_nvcc/bin",),
    "cuda-gdb": ("nvidia/cuda_nvcc/bin",),
    "cuda-gdbserver": ("nvidia/cuda_nvcc/bin",),
    "compute-sanitizer": ("nvidia/cuda_nvcc/bin",),
    "nvprof": ("nvidia/cuda_nvcc/bin",),
    "nsys": ("nvidia/nsight_systems/bin",),
    "nsight-sys": ("nvidia/nsight_systems/bin",),
    "ncu": ("nvidia/nsight_compute/bin",),
    "nsight-compute": ("nvidia/nsight_compute/bin",),
}
