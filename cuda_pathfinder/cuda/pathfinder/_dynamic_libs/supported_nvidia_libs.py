# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# THIS FILE NEEDS TO BE REVIEWED/UPDATED FOR EACH CTK RELEASE
# Likely candidates for updates are:
#     SUPPORTED_LIBNAMES
#     SUPPORTED_WINDOWS_DLLS
#     SUPPORTED_LINUX_SONAMES
#     EXPECTED_LIB_SYMBOLS

import sys

IS_WINDOWS = sys.platform == "win32"

SUPPORTED_LIBNAMES_COMMON = (
    # Core CUDA Runtime and Compiler
    "cudart",
    "nvfatbin",
    "nvJitLink",
    "nvrtc",
    "nvvm",
    # Math Libraries
    "cublas",
    "cublasLt",
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
    "nvjpeg",
)

# Note: The `cufile_rdma` information is intentionally retained (commented out)
# despite not being actively used in the current build. It took a nontrivial
# amount of effort to determine the SONAME, dependencies, and expected symbols
# for this special-case library, especially given its RDMA/MLX5 dependencies
# and limited availability. Keeping this as a reference avoids having to
# reconstruct the information from scratch in the future.

SUPPORTED_LIBNAMES_LINUX_ONLY = (
    "cufile",
    # "cufile_rdma",  # Requires libmlx5.so
)
SUPPORTED_LIBNAMES_LINUX = SUPPORTED_LIBNAMES_COMMON + SUPPORTED_LIBNAMES_LINUX_ONLY

SUPPORTED_LIBNAMES_WINDOWS_ONLY = ()
SUPPORTED_LIBNAMES_WINDOWS = SUPPORTED_LIBNAMES_COMMON + SUPPORTED_LIBNAMES_WINDOWS_ONLY

SUPPORTED_LIBNAMES_ALL = SUPPORTED_LIBNAMES_COMMON + SUPPORTED_LIBNAMES_LINUX_ONLY + SUPPORTED_LIBNAMES_WINDOWS_ONLY
SUPPORTED_LIBNAMES = SUPPORTED_LIBNAMES_WINDOWS if IS_WINDOWS else SUPPORTED_LIBNAMES_LINUX

# Based on ldd output for Linux x86_64 nvidia-*-cu12 wheels (12.8.1)
DIRECT_DEPENDENCIES = {
    "cublas": ("cublasLt",),
    "cufftw": ("cufft",),
    # "cufile_rdma": ("cufile",),
    "cusolver": ("nvJitLink", "cusparse", "cublasLt", "cublas"),
    "cusolverMg": ("nvJitLink", "cublasLt", "cublas"),
    "cusparse": ("nvJitLink",),
    "nppial": ("nppc",),
    "nppicc": ("nppc",),
    "nppidei": ("nppc",),
    "nppif": ("nppc",),
    "nppig": ("nppc",),
    "nppim": ("nppc",),
    "nppist": ("nppc",),
    "nppisu": ("nppc",),
    "nppitc": ("nppc",),
    "npps": ("nppc",),
    "nvblas": ("cublas", "cublasLt"),
}

# Based on these released files:
#   cuda_11.0.3_450.51.06_linux.run
#   cuda_11.1.1_455.32.00_linux.run
#   cuda_11.2.2_460.32.03_linux.run
#   cuda_11.3.1_465.19.01_linux.run
#   cuda_11.4.4_470.82.01_linux.run
#   cuda_11.5.1_495.29.05_linux.run
#   cuda_11.6.2_510.47.03_linux.run
#   cuda_11.7.1_515.65.01_linux.run
#   cuda_11.8.0_520.61.05_linux.run
#   cuda_12.0.1_525.85.12_linux.run
#   cuda_12.1.1_530.30.02_linux.run
#   cuda_12.2.2_535.104.05_linux.run
#   cuda_12.3.2_545.23.08_linux.run
#   cuda_12.4.1_550.54.15_linux.run
#   cuda_12.5.1_555.42.06_linux.run
#   cuda_12.6.2_560.35.03_linux.run
#   cuda_12.8.1_570.124.06_linux.run
#   cuda_12.9.1_575.57.08_linux.run
#   cuda_13.0.0_580.65.06_linux.run
# Generated with toolshed/build_pathfinder_sonames.py
SUPPORTED_LINUX_SONAMES = {
    "cublas": (
        "libcublas.so.11",
        "libcublas.so.12",
        "libcublas.so.13",
    ),
    "cublasLt": (
        "libcublasLt.so.11",
        "libcublasLt.so.12",
        "libcublasLt.so.13",
    ),
    "cudart": (
        "libcudart.so.11.0",
        "libcudart.so.12",
        "libcudart.so.13",
    ),
    "cufft": (
        "libcufft.so.10",
        "libcufft.so.11",
        "libcufft.so.12",
    ),
    "cufftw": (
        "libcufftw.so.10",
        "libcufftw.so.11",
        "libcufftw.so.12",
    ),
    "cufile": ("libcufile.so.0",),
    # "cufile_rdma": ("libcufile_rdma.so.1",),
    "curand": ("libcurand.so.10",),
    "cusolver": (
        "libcusolver.so.10",
        "libcusolver.so.11",
        "libcusolver.so.12",
    ),
    "cusolverMg": (
        "libcusolverMg.so.10",
        "libcusolverMg.so.11",
        "libcusolverMg.so.12",
    ),
    "cusparse": (
        "libcusparse.so.11",
        "libcusparse.so.12",
    ),
    "nppc": (
        "libnppc.so.11",
        "libnppc.so.12",
        "libnppc.so.13",
    ),
    "nppial": (
        "libnppial.so.11",
        "libnppial.so.12",
        "libnppial.so.13",
    ),
    "nppicc": (
        "libnppicc.so.11",
        "libnppicc.so.12",
        "libnppicc.so.13",
    ),
    "nppidei": (
        "libnppidei.so.11",
        "libnppidei.so.12",
        "libnppidei.so.13",
    ),
    "nppif": (
        "libnppif.so.11",
        "libnppif.so.12",
        "libnppif.so.13",
    ),
    "nppig": (
        "libnppig.so.11",
        "libnppig.so.12",
        "libnppig.so.13",
    ),
    "nppim": (
        "libnppim.so.11",
        "libnppim.so.12",
        "libnppim.so.13",
    ),
    "nppist": (
        "libnppist.so.11",
        "libnppist.so.12",
        "libnppist.so.13",
    ),
    "nppisu": (
        "libnppisu.so.11",
        "libnppisu.so.12",
        "libnppisu.so.13",
    ),
    "nppitc": (
        "libnppitc.so.11",
        "libnppitc.so.12",
        "libnppitc.so.13",
    ),
    "npps": (
        "libnpps.so.11",
        "libnpps.so.12",
        "libnpps.so.13",
    ),
    "nvJitLink": (
        "libnvJitLink.so.12",
        "libnvJitLink.so.13",
    ),
    "nvblas": (
        "libnvblas.so.11",
        "libnvblas.so.12",
        "libnvblas.so.13",
    ),
    "nvfatbin": (
        "libnvfatbin.so.12",
        "libnvfatbin.so.13",
    ),
    "nvjpeg": (
        "libnvjpeg.so.11",
        "libnvjpeg.so.12",
        "libnvjpeg.so.13",
    ),
    "nvrtc": (
        "libnvrtc.so.11.0",
        "libnvrtc.so.11.1",
        "libnvrtc.so.11.2",
        "libnvrtc.so.12",
        "libnvrtc.so.13",
    ),
    "nvvm": (
        "libnvvm.so.3",
        "libnvvm.so.4",
    ),
}

# Based on these released files:
#   cuda_11.0.3_451.82_win10.exe
#   cuda_11.1.1_456.81_win10.exe
#   cuda_11.2.2_461.33_win10.exe
#   cuda_11.3.1_465.89_win10.exe
#   cuda_11.4.4_472.50_windows.exe
#   cuda_11.5.1_496.13_windows.exe
#   cuda_11.6.2_511.65_windows.exe
#   cuda_11.7.1_516.94_windows.exe
#   cuda_11.8.0_522.06_windows.exe
#   cuda_12.0.1_528.33_windows.exe
#   cuda_12.1.1_531.14_windows.exe
#   cuda_12.2.2_537.13_windows.exe
#   cuda_12.3.2_546.12_windows.exe
#   cuda_12.4.1_551.78_windows.exe
#   cuda_12.5.1_555.85_windows.exe
#   cuda_12.6.2_560.94_windows.exe
#   cuda_12.8.1_572.61_windows.exe
#   cuda_12.9.1_576.57_windows.exe
#   cuda_13.0.0_windows.exe
# Generated with toolshed/build_pathfinder_dlls.py
SUPPORTED_WINDOWS_DLLS = {
    "cublas": (
        "cublas64_11.dll",
        "cublas64_12.dll",
        "cublas64_13.dll",
    ),
    "cublasLt": (
        "cublasLt64_11.dll",
        "cublasLt64_12.dll",
        "cublasLt64_13.dll",
    ),
    "cudart": (
        "cudart64_101.dll",
        "cudart64_110.dll",
        "cudart64_12.dll",
        "cudart64_13.dll",
        "cudart64_65.dll",
    ),
    "cufft": (
        "cufft64_10.dll",
        "cufft64_11.dll",
        "cufft64_12.dll",
    ),
    "cufftw": (
        "cufftw64_10.dll",
        "cufftw64_11.dll",
        "cufftw64_12.dll",
    ),
    "curand": ("curand64_10.dll",),
    "cusolver": (
        "cusolver64_10.dll",
        "cusolver64_11.dll",
        "cusolver64_12.dll",
    ),
    "cusolverMg": (
        "cusolverMg64_10.dll",
        "cusolverMg64_11.dll",
        "cusolverMg64_12.dll",
    ),
    "cusparse": (
        "cusparse64_11.dll",
        "cusparse64_12.dll",
    ),
    "nppc": (
        "nppc64_11.dll",
        "nppc64_12.dll",
        "nppc64_13.dll",
    ),
    "nppial": (
        "nppial64_11.dll",
        "nppial64_12.dll",
        "nppial64_13.dll",
    ),
    "nppicc": (
        "nppicc64_11.dll",
        "nppicc64_12.dll",
        "nppicc64_13.dll",
    ),
    "nppidei": (
        "nppidei64_11.dll",
        "nppidei64_12.dll",
        "nppidei64_13.dll",
    ),
    "nppif": (
        "nppif64_11.dll",
        "nppif64_12.dll",
        "nppif64_13.dll",
    ),
    "nppig": (
        "nppig64_11.dll",
        "nppig64_12.dll",
        "nppig64_13.dll",
    ),
    "nppim": (
        "nppim64_11.dll",
        "nppim64_12.dll",
        "nppim64_13.dll",
    ),
    "nppist": (
        "nppist64_11.dll",
        "nppist64_12.dll",
        "nppist64_13.dll",
    ),
    "nppisu": (
        "nppisu64_11.dll",
        "nppisu64_12.dll",
        "nppisu64_13.dll",
    ),
    "nppitc": (
        "nppitc64_11.dll",
        "nppitc64_12.dll",
        "nppitc64_13.dll",
    ),
    "npps": (
        "npps64_11.dll",
        "npps64_12.dll",
        "npps64_13.dll",
    ),
    "nvJitLink": (
        "nvJitLink_120_0.dll",
        "nvJitLink_130_0.dll",
    ),
    "nvblas": (
        "nvblas64_11.dll",
        "nvblas64_12.dll",
        "nvblas64_13.dll",
    ),
    "nvfatbin": (
        "nvfatbin_120_0.dll",
        "nvfatbin_130_0.dll",
    ),
    "nvjpeg": (
        "nvjpeg64_11.dll",
        "nvjpeg64_12.dll",
        "nvjpeg64_13.dll",
    ),
    "nvrtc": (
        "nvrtc64_110_0.dll",
        "nvrtc64_111_0.dll",
        "nvrtc64_112_0.dll",
        "nvrtc64_120_0.dll",
        "nvrtc64_130_0.dll",
    ),
    "nvvm": (
        "nvvm64.dll",
        "nvvm64_33_0.dll",
        "nvvm64_40_0.dll",
        "nvvm70.dll",
    ),
}

LIBNAMES_REQUIRING_OS_ADD_DLL_DIRECTORY = (
    "cufft",
    "nvrtc",
)


def is_suppressed_dll_file(path_basename: str) -> bool:
    if path_basename.startswith("nvrtc"):
        # nvidia_cuda_nvrtc_cu12-12.8.93-py3-none-win_amd64.whl:
        #     nvidia\cuda_nvrtc\bin\
        #         nvrtc-builtins64_128.dll
        #         nvrtc64_120_0.alt.dll
        #         nvrtc64_120_0.dll
        return path_basename.endswith(".alt.dll") or "-builtins" in path_basename
    return path_basename.startswith(("cudart32_", "nvvm32"))


# Based on `nm -D --defined-only` output for Linux x86_64 distributions.
EXPECTED_LIB_SYMBOLS = {
    "nvJitLink": (
        "__nvJitLinkCreate_12_0",  # 12.0 through 12.9
        "nvJitLinkVersion",  # 12.3 and up
    ),
    "nvrtc": ("nvrtcVersion",),
    "nvvm": ("nvvmVersion",),
    "cudart": ("cudaRuntimeGetVersion",),
    "nvfatbin": ("nvFatbinVersion",),
    "cublas": ("cublasGetVersion",),
    "cublasLt": ("cublasLtGetVersion",),
    "cufft": ("cufftGetVersion",),
    "cufftw": ("fftwf_malloc",),
    "curand": ("curandGetVersion",),
    "cusolver": ("cusolverGetVersion",),
    "cusolverMg": ("cusolverMgCreate",),
    "cusparse": ("cusparseGetVersion",),
    "nppc": ("nppGetLibVersion",),
    "nppial": ("nppiAdd_32f_C1R_Ctx",),
    "nppicc": ("nppiColorToGray_8u_C3C1R_Ctx",),
    "nppidei": ("nppiCopy_8u_C1R_Ctx",),
    "nppif": ("nppiFilterSobelHorizBorder_8u_C1R_Ctx",),
    "nppig": ("nppiResize_8u_C1R_Ctx",),
    "nppim": ("nppiErode_8u_C1R_Ctx",),
    "nppist": ("nppiMean_8u_C1R_Ctx",),
    "nppisu": ("nppiFree",),
    "nppitc": ("nppiThreshold_8u_C1R_Ctx",),
    "npps": ("nppsAdd_32f_Ctx",),
    "nvblas": ("dgemm",),
    "cufile": ("cuFileGetVersion",),
    # "cufile_rdma": ("rdma_buffer_reg",),
    "nvjpeg": ("nvjpegCreate",),
}
