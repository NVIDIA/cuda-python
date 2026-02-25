# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cuda.pathfinder public APIs"""

from cuda.pathfinder._binaries.find_nvidia_binary_utility import (
    find_nvidia_binary_utility as find_nvidia_binary_utility,
)
from cuda.pathfinder._binaries.supported_nvidia_binaries import SUPPORTED_BINARIES as _SUPPORTED_BINARIES
from cuda.pathfinder._dynamic_libs.load_dl_common import (
    DynamicLibNotAvailableError as DynamicLibNotAvailableError,
)
from cuda.pathfinder._dynamic_libs.load_dl_common import DynamicLibNotFoundError as DynamicLibNotFoundError
from cuda.pathfinder._dynamic_libs.load_dl_common import (
    DynamicLibUnknownError as DynamicLibUnknownError,
)
from cuda.pathfinder._dynamic_libs.load_dl_common import LoadedDL as LoadedDL
from cuda.pathfinder._dynamic_libs.load_nvidia_dynamic_lib import load_nvidia_dynamic_lib as load_nvidia_dynamic_lib
from cuda.pathfinder._dynamic_libs.supported_nvidia_libs import (
    SUPPORTED_LIBNAMES as SUPPORTED_NVIDIA_LIBNAMES,  # noqa: F401
)
from cuda.pathfinder._headers.find_nvidia_headers import LocatedHeaderDir as LocatedHeaderDir
from cuda.pathfinder._headers.find_nvidia_headers import find_nvidia_header_directory as find_nvidia_header_directory
from cuda.pathfinder._headers.find_nvidia_headers import (
    locate_nvidia_header_directory as locate_nvidia_header_directory,
)
from cuda.pathfinder._headers.supported_nvidia_headers import SUPPORTED_HEADERS_CTK as _SUPPORTED_HEADERS_CTK
from cuda.pathfinder._static_libs.find_bitcode_lib import (
    SUPPORTED_BITCODE_LIBS as _SUPPORTED_BITCODE_LIBS,
)
from cuda.pathfinder._static_libs.find_bitcode_lib import (
    BitcodeLibNotFoundError as BitcodeLibNotFoundError,
)
from cuda.pathfinder._static_libs.find_bitcode_lib import (
    LocatedBitcodeLib as LocatedBitcodeLib,
)
from cuda.pathfinder._static_libs.find_bitcode_lib import (
    find_bitcode_lib as find_bitcode_lib,
)
from cuda.pathfinder._static_libs.find_bitcode_lib import (
    locate_bitcode_lib as locate_bitcode_lib,
)
from cuda.pathfinder._static_libs.find_static_lib import (
    SUPPORTED_STATIC_LIBS as _SUPPORTED_STATIC_LIBS,
)
from cuda.pathfinder._static_libs.find_static_lib import (
    LocatedStaticLib as LocatedStaticLib,
)
from cuda.pathfinder._static_libs.find_static_lib import (
    StaticLibNotFoundError as StaticLibNotFoundError,
)
from cuda.pathfinder._static_libs.find_static_lib import (
    find_static_lib as find_static_lib,
)
from cuda.pathfinder._static_libs.find_static_lib import (
    locate_static_lib as locate_static_lib,
)

from cuda.pathfinder._version import __version__  # isort: skip  # noqa: F401

# Indirections to help Sphinx find the docstrings.
#: Mapping from short CUDA Toolkit (CTK) library names to their canonical
#: header basenames (used to validate a discovered include directory).
#: Example: ``"cublas" → "cublas.h"``. The key set is platform-aware
#: (e.g., ``"cufile"`` may be Linux-only).
SUPPORTED_HEADERS_CTK = _SUPPORTED_HEADERS_CTK

#: Tuple of supported CUDA binary utility names that can be located
#: via ``find_nvidia_binary_utility()``. Platform-aware (e.g., some
#: utilities may be available only on Linux or Windows).
#: Example utilities: ``"nvdisasm"``, ``"cuobjdump"``, ``"nvcc"``.
SUPPORTED_BINARY_UTILITIES = _SUPPORTED_BINARIES

#: Tuple of supported bitcode library names that can be resolved
#: via ``locate_bitcode_lib()`` and ``find_bitcode_lib()``.
#: Example value: ``"device"``.
SUPPORTED_BITCODE_LIBS = _SUPPORTED_BITCODE_LIBS

#: Tuple of supported static library names that can be resolved
#: via ``locate_static_lib()`` and ``find_static_lib()``.
#: Example value: ``"cudadevrt"``.
SUPPORTED_STATIC_LIBS = _SUPPORTED_STATIC_LIBS

# Backward compatibility: _find_nvidia_header_directory was added in release 1.2.2.
# It will be removed in release 1.2.4.
_find_nvidia_header_directory = find_nvidia_header_directory
