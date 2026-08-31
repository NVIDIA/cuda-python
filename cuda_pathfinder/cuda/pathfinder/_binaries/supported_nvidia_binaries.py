# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os

# Site-packages bin directories where binaries might be found, in search order.
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
    # Nsight Systems/Compute ship both a CLI (nsys/ncu) and a GUI (nsys-ui/ncu-ui).
    # The GUI executable names differ between the legacy CUDA-toolkit-bundled
    # products (nsight-sys/nsight-compute) and the current standalone Nsight
    # releases (nsys-ui/ncu-ui), so all four names are kept searchable.
    "nsys": (_NSIGHT_SYSTEMS_BIN,),
    "nsys-ui": (_NSIGHT_SYSTEMS_BIN,),
    "nsight-sys": (_NSIGHT_SYSTEMS_BIN,),
    "ncu": (_NSIGHT_COMPUTE_BIN,),
    "ncu-ui": (_NSIGHT_COMPUTE_BIN,),
    "nsight-compute": (_NSIGHT_COMPUTE_BIN,),
}

SUPPORTED_BINARIES_ALL = SUPPORTED_BINARIES = tuple(SITE_PACKAGES_BINDIRS.keys())
