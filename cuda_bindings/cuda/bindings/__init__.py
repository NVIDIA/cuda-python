# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import os

from cuda.bindings import utils
from cuda.bindings._version import __version__

# Version validation: detect setuptools-scm fallback versions (e.g., 0.1.dev...)
# This check must be kept in sync with similar checks in cuda.core and cuda.pathfinder
if not os.environ.get("CUDA_PYTHON_ALLOW_FALLBACK_VERSIONING"):
    version_parts = __version__.split(".")
    if len(version_parts) < 2:
        raise RuntimeError(
            f"Invalid version format: '{__version__}'. "
            f"The version detection system failed. "
            f"This usually means git tags are not available (e.g., shallow clone or zip archive). "
            f"To fix: ensure you have a full git checkout with tags, or set "
            f"CUDA_PYTHON_ALLOW_FALLBACK_VERSIONING=1 to disable this check."
        )
    try:
        major, minor = int(version_parts[0]), int(version_parts[1])
    except ValueError:
        raise RuntimeError(
            f"Invalid version format: '{__version__}'. "
            f"The version detection system failed. "
            f"This usually means git tags are not available (e.g., shallow clone or zip archive). "
            f"To fix: ensure you have a full git checkout with tags, or set "
            f"CUDA_PYTHON_ALLOW_FALLBACK_VERSIONING=1 to disable this check."
        ) from None
    if major == 0 and minor <= 1:
        raise RuntimeError(
            f"Invalid version detected: '{__version__}'. "
            f"The version detection system failed silently and produced a fallback version. "
            f"This usually means git tags are not available (e.g., shallow clone or zip archive). "
            f"To fix: ensure you have a full git checkout with tags, or set "
            f"CUDA_PYTHON_ALLOW_FALLBACK_VERSIONING=1 to disable this check."
        )
