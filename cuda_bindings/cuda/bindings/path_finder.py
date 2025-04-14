# Copyright 2024-2025 NVIDIA Corporation.  All rights reserved.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

from cuda.bindings._path_finder.cuda_paths import (
    get_conda_ctk,
    get_conda_include_dir,
    get_cuda_home,
    get_cuda_paths,
    get_current_cuda_target_name,
    get_debian_pkg_libdevice,
    get_libdevice_wheel,
    get_nvidia_cudalib_ctk,
    get_nvidia_libdevice_ctk,
    get_nvidia_nvvm_ctk,
    get_nvidia_static_cudalib_ctk,
    get_system_ctk,
)
from cuda.bindings._path_finder.find_nvidia_dynamic_library import find_nvidia_dynamic_library
from cuda.bindings._path_finder.load_nvidia_dynamic_library import load_nvidia_dynamic_library

__all__ = [
    "find_nvidia_dynamic_library",
    "load_nvidia_dynamic_library",
    "get_conda_ctk",
    "get_conda_include_dir",
    "get_cuda_home",
    "get_cuda_paths",
    "get_current_cuda_target_name",
    "get_debian_pkg_libdevice",
    "get_libdevice_wheel",
    "get_nvidia_cudalib_ctk",
    "get_nvidia_libdevice_ctk",
    "get_nvidia_nvvm_ctk",
    "get_nvidia_static_cudalib_ctk",
    "get_system_ctk",
    "SUPPORTED_LIBNAMES",
    "SUPPORTED_WINDOWS_DLLS",
]

SUPPORTED_LIBNAMES = (
    # Core CUDA Runtime and Compiler
    "cudart",
    "nvfatbin",
    "nvJitLink",
    "nvrtc",
    "nvvm",
    # Math Libraries
    "cublas",
    "cufft",
    "cufftw",
    "curand",
    "cusolver",
    "cusolverMg",
    "cusparse",
    "nppc",
    "nppial",
    "nppicc",
    "nppidei",
    "nppif",
    "nppig",
    "nppim",
    "nppist",
    "nppisu",
    "nppitc",
    "npps",
    "nvblas",
    # Other
    "cufile",
    "nvjpeg",
)

# Based on https://developer.download.nvidia.com/compute/cuda/redist/
# as of 2025-04-11 (redistrib_12.8.1.json was the newest .json file).
SUPPORTED_WINDOWS_DLLS = {
    "cublas": ("cublas64_12.dll", "cublas64_11.dll"),
    "cudart": ("cudart64_12.dll", "cudart64_110.dll", "cudart32_110.dll"),
    "cufft": ("cufft64_11.dll", "cufft64_10.dll"),
    "cufftw": ("cufftw64_10.dll", "cufftw64_11.dll"),
    "cufile": (),
    "curand": ("curand64_10.dll",),
    "cusolver": ("cusolver64_11.dll",),
    "cusolverMg": ("cusolverMg64_11.dll",),
    "cusparse": ("cusparse64_12.dll", "cusparse64_11.dll"),
    "nppc": ("nppc64_12.dll", "nppc64_11.dll"),
    "nppial": ("nppial64_12.dll", "nppial64_11.dll"),
    "nppicc": ("nppicc64_12.dll", "nppicc64_11.dll"),
    "nppidei": ("nppidei64_12.dll", "nppidei64_11.dll"),
    "nppif": ("nppif64_12.dll", "nppif64_11.dll"),
    "nppig": ("nppig64_12.dll", "nppig64_11.dll"),
    "nppim": ("nppim64_12.dll", "nppim64_11.dll"),
    "nppist": ("nppist64_12.dll", "nppist64_11.dll"),
    "nppisu": ("nppisu64_12.dll", "nppisu64_11.dll"),
    "nppitc": ("nppitc64_12.dll", "nppitc64_11.dll"),
    "npps": ("npps64_12.dll", "npps64_11.dll"),
    "nvblas": ("nvblas64_12.dll", "nvblas64_11.dll"),
    "nvfatbin": ("nvfatbin_120_0.dll",),
    "nvJitLink": ("nvJitLink_120_0.dll",),
    "nvjpeg": ("nvjpeg64_12.dll", "nvjpeg64_11.dll"),
    "nvrtc": ("nvrtc64_120_0.dll", "nvrtc64_112_0.dll"),
    "nvvm": ("nvvm64_40_0.dll",),
}
