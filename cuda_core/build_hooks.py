# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""PEP 517 wrapper for cuda.core's scikit-build-core backend.

The wrapper keeps CUDA-major-dependent build requirements out of the static
``build-system.requires`` list. Compilation and wheel assembly are delegated to
scikit-build-core.
"""

from __future__ import annotations

import os
import re

import scikit_build_core.build as _build_backend

build_wheel = _build_backend.build_wheel
build_editable = _build_backend.build_editable
build_sdist = _build_backend.build_sdist
prepare_metadata_for_build_wheel = _build_backend.prepare_metadata_for_build_wheel
prepare_metadata_for_build_editable = _build_backend.prepare_metadata_for_build_editable
get_requires_for_build_sdist = _build_backend.get_requires_for_build_sdist


def _get_cuda_path() -> str:
    cuda_path = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
    if cuda_path:
        return cuda_path
    raise RuntimeError("Environment variable CUDA_PATH or CUDA_HOME is not set")


def _cuda_major_from_headers(cuda_path: str) -> str:
    cuda_h = os.path.join(cuda_path, "include", "cuda.h")
    try:
        with open(cuda_h, encoding="utf-8") as f:
            for line in f:
                match = re.match(r"^#\s*define\s+CUDA_VERSION\s+(\d+)\s*$", line)
                if match:
                    # CUDA_VERSION is e.g. 12020 for 12.2.
                    return str(int(match.group(1)) // 1000)
    except OSError:
        pass

    raise RuntimeError(
        "Cannot determine CUDA major version. "
        "Set CUDA_CORE_BUILD_MAJOR, or ensure CUDA_PATH or CUDA_HOME points "
        "to a valid CUDA installation with include/cuda.h."
    )


def _determine_cuda_major_version() -> str:
    """Determine the CUDA major used for build requirements."""
    cuda_major = os.environ.get("CUDA_CORE_BUILD_MAJOR") or None
    if cuda_major is None:
        cuda_major = _cuda_major_from_headers(_get_cuda_path())

    if not re.fullmatch(r"\d+", cuda_major):
        raise RuntimeError(f"CUDA_CORE_BUILD_MAJOR must be an integer, got {cuda_major!r}")

    print("CUDA MAJOR VERSION:", cuda_major)
    return cuda_major


def _get_cuda_bindings_require() -> list[str]:
    cuda_major = _determine_cuda_major_version()
    return [f"cuda-bindings=={cuda_major}.*"]


def get_requires_for_build_wheel(config_settings=None):
    return _build_backend.get_requires_for_build_wheel(config_settings) + _get_cuda_bindings_require()


def get_requires_for_build_editable(config_settings=None):
    return _build_backend.get_requires_for_build_editable(config_settings) + _get_cuda_bindings_require()
