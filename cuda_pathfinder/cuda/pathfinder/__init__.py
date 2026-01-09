# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cuda.pathfinder public APIs"""

from cuda.pathfinder._version import __version__  # noqa: F401

from cuda.pathfinder._binaries.find_nvidia_binaries import find_nvidia_binary as find_nvidia_binary
from cuda.pathfinder._binaries.supported_nvidia_binaries import SUPPORTED_BINARIES as SUPPORTED_NVIDIA_BINARIES
from cuda.pathfinder._dynamic_libs.load_dl_common import DynamicLibNotFoundError as DynamicLibNotFoundError
from cuda.pathfinder._dynamic_libs.load_dl_common import LoadedDL as LoadedDL
from cuda.pathfinder._dynamic_libs.load_nvidia_dynamic_lib import load_nvidia_dynamic_lib as load_nvidia_dynamic_lib
from cuda.pathfinder._dynamic_libs.supported_nvidia_libs import (
    SUPPORTED_LIBNAMES as SUPPORTED_NVIDIA_LIBNAMES,  # noqa: F401
)
from cuda.pathfinder._headers.find_nvidia_headers import find_nvidia_header_directory as find_nvidia_header_directory
from cuda.pathfinder._headers.supported_nvidia_headers import SUPPORTED_HEADERS_CTK as _SUPPORTED_HEADERS_CTK
from cuda.pathfinder._static_libs.find_nvidia_static_libs import find_nvidia_static_lib as find_nvidia_static_lib
from cuda.pathfinder._static_libs.supported_nvidia_static_libs import (
    SUPPORTED_STATIC_LIBS as SUPPORTED_NVIDIA_STATIC_LIBS,
)

# Indirections to help Sphinx find the docstrings.
#: Mapping from short CUDA Toolkit (CTK) library names to their canonical
#: header basenames (used to validate a discovered include directory).
#: Example: ``"cublas" → "cublas.h"``. The key set is platform-aware
#: (e.g., ``"cufile"`` may be Linux-only).
SUPPORTED_HEADERS_CTK = _SUPPORTED_HEADERS_CTK

# Backward compatibility: _find_nvidia_header_directory was added in release 1.2.2.
# It will be removed in release 1.2.4.
_find_nvidia_header_directory = find_nvidia_header_directory
