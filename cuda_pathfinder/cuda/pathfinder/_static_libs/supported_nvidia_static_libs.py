# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# THIS FILE NEEDS TO BE REVIEWED/UPDATED FOR EACH CTK RELEASE
# Likely candidates for updates are:
#     SUPPORTED_STATIC_LIBS
#     SITE_PACKAGES_STATIC_LIBDIRS

from cuda.pathfinder._utils.platform_aware import IS_WINDOWS

# Supported CUDA static libraries and artifacts that can be found
SUPPORTED_STATIC_LIBS_COMMON = (
    "libdevice.10.bc",
    "libcudadevrt.a",
)

SUPPORTED_STATIC_LIBS = SUPPORTED_STATIC_LIBS_COMMON

# Map from artifact name to relative paths under site-packages
SITE_PACKAGES_STATIC_LIBDIRS = {
    "libdevice.10.bc": ["nvidia/cuda_nvvm/nvvm/libdevice"],
    "libcudadevrt.a": [
        "nvidia/cuda_cudart/lib",  # Linux
        "nvidia/cuda_cudart/lib/x64",  # Windows (if present)
    ],
}
