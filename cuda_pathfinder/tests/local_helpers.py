# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import importlib.metadata
import re
from dataclasses import dataclass
from pathlib import Path

import pytest

from cuda.pathfinder._headers.find_nvidia_headers import (
    locate_nvidia_header_directory as locate_nvidia_header_directory_raw,
)
from cuda.pathfinder._utils import driver_info
from cuda.pathfinder._utils.toolkit_info import CudaToolkitVersion, read_cuda_header_version


@dataclass(frozen=True, slots=True)
class LocatedRealCudaToolkitVersion:
    """Real-host CTK version discovered from ``cuda.h`` next to resolved ``cudart`` headers."""

    version: CudaToolkitVersion
    cuda_h_path: str
    header_dir: str
    found_via: str


@functools.cache
def have_distribution(name_pattern: str) -> bool:
    re_name_pattern = re.compile(name_pattern)
    return any(
        re_name_pattern.match(dist.metadata["Name"])
        for dist in importlib.metadata.distributions()
        if "Name" in dist.metadata
    )


@functools.cache
def locate_real_cuda_toolkit_version_from_cuda_h() -> LocatedRealCudaToolkitVersion | None:
    """Return the real-host CTK version from ``cuda.h`` if ``cudart`` headers can be located."""
    located = locate_nvidia_header_directory_raw("cudart")
    if located is None or located.abs_path is None:
        return None
    cuda_h_path = Path(located.abs_path) / "cuda.h"
    if not cuda_h_path.is_file():
        return None
    return LocatedRealCudaToolkitVersion(
        version=read_cuda_header_version(str(cuda_h_path)),
        cuda_h_path=str(cuda_h_path),
        header_dir=located.abs_path,
        found_via=located.found_via,
    )


def require_real_cuda_toolkit_version_from_cuda_h() -> LocatedRealCudaToolkitVersion:
    """Return the real-host CTK version from ``cuda.h`` or skip if it cannot be located."""
    located = locate_nvidia_header_directory_raw("cudart")
    if located is None or located.abs_path is None:
        pytest.skip("Could not locate cudart headers, so could not find cuda.h for a real CTK installation.")
    cuda_h_path = Path(located.abs_path) / "cuda.h"
    if not cuda_h_path.is_file():
        pytest.skip(
            f"Located cudart headers via {located.found_via} at {located.abs_path!r}, but could not find cuda.h."
        )
    return LocatedRealCudaToolkitVersion(
        version=read_cuda_header_version(str(cuda_h_path)),
        cuda_h_path=str(cuda_h_path),
        header_dir=located.abs_path,
        found_via=located.found_via,
    )


def require_real_driver_cuda_version() -> driver_info.DriverCudaVersion:
    """Return the real-host CUDA driver version or skip if it cannot be queried."""
    try:
        return driver_info.query_driver_cuda_version()
    except driver_info.QueryDriverCudaVersionError as exc:
        pytest.skip(f"Could not query the CUDA driver version for a real driver installation: {exc}")
