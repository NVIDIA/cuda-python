# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Platform-specific path configurations for CUDA artifacts.

This module centralizes all platform-specific directory structures for
different CUDA installation sources (conda, CUDA_HOME, etc.).
"""

from dataclasses import dataclass

from cuda.pathfinder._utils.platform_aware import IS_WINDOWS


@dataclass(frozen=True)
class PlatformPaths:
    """Platform-specific directory layouts for CUDA artifacts.
    
    Attributes:
        conda_bin_subdirs: Binary subdirectories under CONDA_PREFIX.
        conda_lib_subdirs: Library subdirectories under CONDA_PREFIX.
        conda_nvvm_subdirs: NVVM/libdevice subdirectories under CONDA_PREFIX.
        cuda_home_bin_subdirs: Binary subdirectories under CUDA_HOME.
        cuda_home_lib_subdirs: Library subdirectories under CUDA_HOME.
        cuda_home_nvvm_subdirs: NVVM/libdevice subdirectories under CUDA_HOME.
    """

    conda_bin_subdirs: tuple[str, ...]
    conda_lib_subdirs: tuple[str, ...]
    conda_nvvm_subdirs: tuple[str, ...]
    cuda_home_bin_subdirs: tuple[str, ...]
    cuda_home_lib_subdirs: tuple[str, ...]
    cuda_home_nvvm_subdirs: tuple[str, ...]


# Platform-specific paths (determined at import time)
if IS_WINDOWS:
    PLATFORM = PlatformPaths(
        conda_bin_subdirs=("Library/bin", "bin"),
        conda_lib_subdirs=("Library/lib", "Library/lib/x64"),
        conda_nvvm_subdirs=("Library/nvvm/libdevice",),
        cuda_home_bin_subdirs=("bin",),
        cuda_home_lib_subdirs=("lib/x64", "lib"),
        cuda_home_nvvm_subdirs=("nvvm/libdevice",),
    )
else:
    PLATFORM = PlatformPaths(
        conda_bin_subdirs=("bin",),
        conda_lib_subdirs=("lib",),
        conda_nvvm_subdirs=("nvvm/libdevice",),
        cuda_home_bin_subdirs=("bin",),
        cuda_home_lib_subdirs=("lib64", "lib"),
        cuda_home_nvvm_subdirs=("nvvm/libdevice",),
    )

# Constants for cross-compilation support (Linux only)
CUDA_TARGETS_LIB_SUBDIRS = ("lib64", "lib")
