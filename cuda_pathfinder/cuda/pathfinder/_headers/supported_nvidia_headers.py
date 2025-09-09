# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

SUPPORTED_HEADERS_CTK = {
    "cub": "cub/cub.cuh",
    "cublas": "cublas.h",
    "cudart": "cuda_runtime.h",
    "cufft": "cufft.h",
    "cufile": "cufile.h",
    "curand": "curand.h",
    "cusolver": "cusolver_common.h",
    "cusparse": "cusparse.h",
    "libcudacxx": "cuda/std/version",
    "npp": "npp.h",
    "nvcc": "fatbinary_section.h",
    "nvfatbin": "nvFatbin.h",
    "nvjitlink": "nvJitLink.h",
    "nvjpeg": "nvjpeg.h",
    "nvrtc": "nvrtc.h",
    "nvvm": "nvvm.h",
    "thrust": "thrust/version.h",
}

SUPPORTED_SITE_PACKAGE_HEADER_DIRS_CTK = {
    "cub": ("cuda/cccl/headers/include",),
    "cublas": ("nvidia/cu13/include", "nvidia/cublas/include"),
    "cudart": ("nvidia/cu13/include", "nvidia/cuda_runtime/include"),
    "cufft": ("nvidia/cu13/include", "nvidia/cufft/include"),
    "cufile": ("nvidia/cu13/include", "nvidia/cufile/include"),
    "curand": ("nvidia/cu13/include", "nvidia/curand/include"),
    "cusolver": ("nvidia/cu13/include", "nvidia/cusolver/include"),
    "cusparse": ("nvidia/cu13/include", "nvidia/cusparse/include"),
    "libcudacxx": ("cuda/cccl/headers/include",),
    "npp": ("nvidia/cu13/include", "nvidia/npp/include"),
    "nvcc": ("nvidia/cu13/include", "nvidia/cuda_nvcc/include"),
    "nvfatbin": ("nvidia/cu13/include", "nvidia/nvfatbin/include"),
    "nvjitlink": ("nvidia/cu13/include", "nvidia/nvjitlink/include"),
    "nvjpeg": ("nvidia/cu13/include", "nvidia/nvjpeg/include"),
    "nvrtc": ("nvidia/cu13/include", "nvidia/cuda_nvrtc/include"),
    "nvvm": ("nvidia/cu13/include", "nvidia/cuda_nvcc/nvvm/include"),
    "thrust": ("cuda/cccl/headers/include",),
}

CCCL_LIBNAMES = ("cub", "libcudacxx", "thrust")
