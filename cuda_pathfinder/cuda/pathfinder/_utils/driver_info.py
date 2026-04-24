# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ctypes
import functools
from collections.abc import Callable
from dataclasses import dataclass

from cuda.pathfinder._dynamic_libs.load_nvidia_dynamic_lib import (
    load_nvidia_dynamic_lib as _load_nvidia_dynamic_lib,
)
from cuda.pathfinder._utils.platform_aware import IS_WINDOWS


class QueryDriverCudaVersionError(RuntimeError):
    """Raised when ``query_driver_cuda_version()`` cannot determine the CUDA driver version."""


@dataclass(frozen=True, slots=True)
class DriverCudaVersion:
    """
    CUDA-facing driver version reported by ``cuDriverGetVersion()``.

    The name ``DriverCudaVersion`` is intentionally specific: this dataclass
    models the version shown as ``CUDA Version`` in ``nvidia-smi``, not the
    graphics driver release shown as ``Driver Version``. More specifically,
    it reflects the CUDA user-mode driver (UMD) interface version reported by
    ``cuDriverGetVersion()``, not the kernel-mode driver (KMD) package
    version.

    Example ``nvidia-smi`` output::

        +---------------------------------------------------------------------+
        | NVIDIA-SMI 595.58.03  Driver Version: 595.58.03  CUDA Version: 13.2 |
        +---------------------------------------------------------------------+

    For the example above, ``DriverCudaVersion(encoded=13020, major=13,
    minor=2)`` corresponds to ``CUDA Version: 13.2``. It does not correspond
    to ``Driver Version: 595.58.03``.
    """

    encoded: int
    major: int
    minor: int


@functools.cache
def query_driver_cuda_version() -> DriverCudaVersion:
    """Return the CUDA driver version parsed into its major/minor components."""
    try:
        encoded = _query_driver_cuda_version_int()
        return DriverCudaVersion(
            encoded=encoded,
            major=encoded // 1000,
            minor=(encoded % 1000) // 10,
        )
    except Exception as exc:
        raise QueryDriverCudaVersionError("Failed to query the CUDA driver version.") from exc


def _query_driver_cuda_version_int() -> int:
    """Return the encoded CUDA driver version from ``cuDriverGetVersion()``."""
    loaded_cuda = _load_nvidia_dynamic_lib("cuda")
    if IS_WINDOWS:
        # `ctypes.WinDLL` exists on Windows at runtime. The ignore is only for
        # Linux mypy runs, where the platform stubs do not define that attribute.
        loader_cls: Callable[[str], ctypes.CDLL] = ctypes.WinDLL  # type: ignore[attr-defined]
    else:
        loader_cls = ctypes.CDLL
    driver_lib = loader_cls(loaded_cuda.abs_path)
    cu_driver_get_version = driver_lib.cuDriverGetVersion
    cu_driver_get_version.argtypes = [ctypes.POINTER(ctypes.c_int)]
    cu_driver_get_version.restype = ctypes.c_int
    version = ctypes.c_int()
    status = cu_driver_get_version(ctypes.byref(version))
    if status != 0:
        raise RuntimeError(f"Failed to query CUDA driver version via cuDriverGetVersion() (status={status}).")
    return version.value
