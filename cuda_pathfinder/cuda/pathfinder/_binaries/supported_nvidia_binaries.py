# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os

# Site-packages bin directories where binaries might be found
# Based on NVIDIA wheel layouts (same for Linux and Windows)
_CUDA_NVCC_BIN = os.path.join("nvidia", "cuda_nvcc", "bin")
_CUDA13_BIN = os.path.join("nvidia", "cu13", "bin")
_NSIGHT_SYSTEMS_BIN = os.path.join("nvidia", "nsight_systems", "bin")
_NSIGHT_COMPUTE_BIN = os.path.join("nvidia", "nsight_compute", "bin")

# Common CUDA binary utilities available on both Linux and Windows
SITE_PACKAGES_BINDIRS = {
    # Core compilation tools
    "nvcc": (_CUDA13_BIN, _CUDA_NVCC_BIN),
    "nvdisasm": (_CUDA13_BIN, _CUDA_NVCC_BIN),
    "cuobjdump": (_CUDA_NVCC_BIN,),
    "nvprune": (_CUDA_NVCC_BIN,),
    "fatbinary": (_CUDA13_BIN, _CUDA_NVCC_BIN),
    "bin2c": (_CUDA13_BIN, _CUDA_NVCC_BIN),
    "nvlink": (_CUDA13_BIN, _CUDA_NVCC_BIN),
    # Runtime/debugging tools
    "cuda-gdb": (_CUDA_NVCC_BIN,),
    "cuda-gdbserver": (_CUDA_NVCC_BIN,),
    "compute-sanitizer": (_CUDA13_BIN, _CUDA_NVCC_BIN),
    # Profiling tools
    "nvprof": (_CUDA_NVCC_BIN,),
    "nsys": (_NSIGHT_SYSTEMS_BIN,),
    "nsight-sys": (_NSIGHT_SYSTEMS_BIN,),
    "ncu": (_NSIGHT_COMPUTE_BIN,),
    "nsight-compute": (_NSIGHT_COMPUTE_BIN,),
}

SUPPORTED_BINARIES_CTK_COMPANION_TAGS = {
    "bin2c": ("toolchain_cuda_nvcc",),
    "cuobjdump": ("toolchain_cuda_nvcc",),
    "fatbinary": ("toolchain_cuda_nvcc",),
    "nvcc": ("toolchain_cuda_nvcc",),
    "nvdisasm": ("toolchain_cuda_nvcc",),
    "nvlink": ("toolchain_cuda_nvcc",),
    "nvprune": ("toolchain_cuda_nvcc",),
}

# Nsight Systems and Nsight Compute ship in their own PyPI/Conda packages
# (`nvidia/nsight_systems`, `nvidia/nsight_compute`) and are not pinned by the
# `cuda-toolkit` distribution, so they cannot participate in CTK-coherence
# checks. They are tagged "other" so the guard rails treat them as separately
# packaged tools rather than reporting them as missing CTK metadata.
SUPPORTED_BINARIES_PACKAGED_WITH = {
    name: ("other" if name in {"nsys", "nsight-sys", "ncu", "nsight-compute"} else "ctk")
    for name in SITE_PACKAGES_BINDIRS
}

SUPPORTED_BINARIES_ALL = SUPPORTED_BINARIES = tuple(SITE_PACKAGES_BINDIRS.keys())
