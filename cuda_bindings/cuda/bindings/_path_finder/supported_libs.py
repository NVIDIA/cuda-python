# Copyright 2025 NVIDIA Corporation.  All rights reserved.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

# THIS FILE NEEDS TO BE REVIEWED/UPDATED FOR EACH CTK RELEASE

SUPPORTED_LIBNAMES = (
    # Core CUDA Runtime and Compiler
    "nvJitLink",
    "nvrtc",
    "nvvm",
)

PARTIALLY_SUPPORTED_LIBNAMES = (
    # Core CUDA Runtime and Compiler
    "cudart",
    "nvfatbin",
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
    "cufile",
    # "cufile_rdma",  # Requires libmlx5.so
    "nvjpeg",
)

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

# Based on https://developer.download.nvidia.com/compute/cuda/redist/
# as of 2025-04-11 (redistrib_12.8.1.json was the newest .json file).
# Tuples of DLLs are sorted newest-to-oldest.
SUPPORTED_WINDOWS_DLLS = {
    "cublas": ("cublas64_12.dll", "cublas64_11.dll"),
    "cublasLt": ("cublasLt64_12.dll", "cublasLt64_11.dll"),
    "cudart": ("cudart64_12.dll", "cudart64_110.dll", "cudart32_110.dll"),
    "cufft": ("cufft64_11.dll", "cufft64_10.dll"),
    "cufftw": ("cufftw64_10.dll", "cufftw64_11.dll"),
    "cufile": (),
    # "cufile_rdma": (),
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

# Based on nm output for Linux x86_64 /usr/local/cuda (12.8.1)
EXPECTED_LIB_SYMBOLS = {
    "nvJitLink": ("nvJitLinkVersion",),
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
    "nppial": ("nppiAdd_32f_C1R",),
    "nppicc": ("nppiColorToGray_8u_C3C1R",),
    "nppidei": ("nppiCopy_8u_C1R",),
    "nppif": ("nppiFilterSobelHorizBorder_8u_C1R",),
    "nppig": ("nppiResize_8u_C1R",),
    "nppim": ("nppiErode_8u_C1R",),
    "nppist": ("nppiMean_8u_C1R",),
    "nppisu": ("nppiFree",),
    "nppitc": ("nppiThreshold_8u_C1R",),
    "npps": ("nppsAdd_32f",),
    "nvblas": ("dgemm",),
    "cufile": ("cuFileGetVersion",),
    # "cufile_rdma": ("rdma_buffer_reg",),
    "nvjpeg": ("nvjpegCreate",),
}
