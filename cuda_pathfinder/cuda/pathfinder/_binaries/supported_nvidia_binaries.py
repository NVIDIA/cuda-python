# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# THIS FILE NEEDS TO BE REVIEWED/UPDATED FOR EACH CTK RELEASE
# Likely candidates for updates are:
#     SUPPORTED_BINARIES
#     SITE_PACKAGES_BINDIRS

from cuda.pathfinder._utils.platform_aware import IS_WINDOWS

# Supported CUDA binaries that can be found
SUPPORTED_BINARIES_COMMON = (
    "nvdisasm",
    "cuobjdump",
)

SUPPORTED_BINARIES = SUPPORTED_BINARIES_COMMON

# Map from binary name to relative paths under site-packages
# These are typically from cuda-toolkit[nvcc] wheels
SITE_PACKAGES_BINDIRS = {
    "nvdisasm": ["nvidia/cuda_nvcc/bin"],
    "cuobjdump": ["nvidia/cuda_nvcc/bin"],
}
