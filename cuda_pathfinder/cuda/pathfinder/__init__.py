# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cuda.pathfinder public APIs"""

# Validate setuptools-scm version (fail loudly if git describe failed)
import os
_version_file = os.path.join(os.path.dirname(__file__), "_version.py")
if os.path.exists(_version_file):
    with open(_version_file, encoding="utf-8") as f:
        _version_content = f.read()
        # Check if version starts with "0.1" (setuptools-scm fallback)
        if '__version__ = version = \'0.1.' in _version_content:
            raise RuntimeError(
                f"setuptools-scm failed to determine version from git tags!\n"
                f"Generated version file shows fallback version '0.1.x'.\n"
                f"This usually means:\n"
                f"  1. Git tags are not fetched (run: git fetch --tags)\n"
                f"  2. Git is not available in PATH\n"
                f"  3. Running from wrong directory (setuptools_scm root='..')\n"
                f"  4. Git describe command failed\n"
                f"\n"
                f"Version file content:\n{_version_content}\n"
                f"\n"
                f"To debug, run: git describe --tags --long --match 'cuda-pathfinder-v*[0-9]*'"
            )

from cuda.pathfinder._version import __version__  # noqa: F401

from cuda.pathfinder._dynamic_libs.load_dl_common import DynamicLibNotFoundError as DynamicLibNotFoundError
from cuda.pathfinder._dynamic_libs.load_dl_common import LoadedDL as LoadedDL
from cuda.pathfinder._dynamic_libs.load_nvidia_dynamic_lib import load_nvidia_dynamic_lib as load_nvidia_dynamic_lib
from cuda.pathfinder._dynamic_libs.supported_nvidia_libs import (
    SUPPORTED_LIBNAMES as SUPPORTED_NVIDIA_LIBNAMES,  # noqa: F401
)
from cuda.pathfinder._headers.find_nvidia_headers import find_nvidia_header_directory as find_nvidia_header_directory
from cuda.pathfinder._headers.supported_nvidia_headers import SUPPORTED_HEADERS_CTK as _SUPPORTED_HEADERS_CTK

# Indirections to help Sphinx find the docstrings.
#: Mapping from short CUDA Toolkit (CTK) library names to their canonical
#: header basenames (used to validate a discovered include directory).
#: Example: ``"cublas" → "cublas.h"``. The key set is platform-aware
#: (e.g., ``"cufile"`` may be Linux-only).
SUPPORTED_HEADERS_CTK = _SUPPORTED_HEADERS_CTK

# Backward compatibility: _find_nvidia_header_directory was added in release 1.2.2.
# It will be removed in release 1.2.4.
_find_nvidia_header_directory = find_nvidia_header_directory
