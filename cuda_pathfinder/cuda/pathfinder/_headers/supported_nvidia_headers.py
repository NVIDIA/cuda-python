# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

SUPPORTED_HEADERS_CTK = {
    "cccl": "cuda/std/version",
    "cublas": "cublas.h",
    "cudart": "cuda_runtime.h",
    "cufft": "cufft.h",
    "cufile": "cufile.h",
    "curand": "curand.h",
    "cusolver": "cusolver_common.h",
    "cusparse": "cusparse.h",
    "npp": "npp.h",
    "nvcc": "fatbinary_section.h",
    "nvfatbin": "nvFatbin.h",
    "nvjitlink": "nvJitLink.h",
    "nvjpeg": "nvjpeg.h",
    "nvrtc": "nvrtc.h",
    "nvvm": "nvvm.h",
}

SUPPORTED_SITE_PACKAGE_HEADER_DIRS_CTK = {
    "cccl": (
        "cuda/cccl/headers/include",  # cuda-cccl
        "nvidia/cu13/include/cccl",  # cuda-toolkit[cccl]==13.*
        "nvidia/cuda_cccl/include",  # cuda-toolkit[cccl]==12.*
    ),
    "cublas": ("nvidia/cu13/include", "nvidia/cublas/include"),
    "cudart": ("nvidia/cu13/include", "nvidia/cuda_runtime/include"),
    "cufft": ("nvidia/cu13/include", "nvidia/cufft/include"),
    "cufile": ("nvidia/cu13/include", "nvidia/cufile/include"),
    "curand": ("nvidia/cu13/include", "nvidia/curand/include"),
    "cusolver": ("nvidia/cu13/include", "nvidia/cusolver/include"),
    "cusparse": ("nvidia/cu13/include", "nvidia/cusparse/include"),
    "npp": ("nvidia/cu13/include", "nvidia/npp/include"),
    "nvcc": ("nvidia/cu13/include", "nvidia/cuda_nvcc/include"),
    "nvfatbin": ("nvidia/cu13/include", "nvidia/nvfatbin/include"),
    "nvjitlink": ("nvidia/cu13/include", "nvidia/nvjitlink/include"),
    "nvjpeg": ("nvidia/cu13/include", "nvidia/nvjpeg/include"),
    "nvrtc": ("nvidia/cu13/include", "nvidia/cuda_nvrtc/include"),
    "nvvm": ("nvidia/cu13/include", "nvidia/cuda_nvcc/nvvm/include"),
}
