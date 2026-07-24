# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ctypes
import functools
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

from cuda.pathfinder._dynamic_libs.load_nvidia_dynamic_lib import (
    load_nvidia_dynamic_lib as _load_nvidia_dynamic_lib,
)
from cuda.pathfinder._utils.platform_aware import IS_WINDOWS
from cuda.pathfinder._utils.toolkit_info import EncodedCudaVersion

_NVML_SUCCESS = 0
_NVML_SYSTEM_DRIVER_VERSION_BUFFER_LENGTH = 80
_DRIVER_RELEASE_VERSION_RE = re.compile(r"^\d+(?:\.\d+){1,2}$")


class QueryDriverCudaVersionError(RuntimeError):
    """Raised when ``query_driver_cuda_version()`` cannot determine the CUDA driver version."""


class QueryDriverReleaseVersionError(RuntimeError):
    """Raised when ``query_driver_release_version()`` cannot determine the display-driver release version."""


@dataclass(frozen=True, slots=True)
class DriverCudaVersion(EncodedCudaVersion):
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


@dataclass(frozen=True, slots=True)
class DriverReleaseVersion:
    """
    Display-driver release version shown as ``Driver Version`` in ``nvidia-smi``.

    Example ``nvidia-smi`` output::

        +---------------------------------------------------------------------+
        | NVIDIA-SMI 595.58.03  Driver Version: 595.58.03  CUDA Version: 13.2 |
        +---------------------------------------------------------------------+

    For the example above, ``DriverReleaseVersion(text="595.58.03",
    components=(595, 58, 3), branch=595)`` corresponds to ``Driver Version:
    595.58.03``. The ``branch`` field is the first numeric component because
    NVIDIA's compatibility docs publish minimum display-driver requirements in
    branch form such as ``>= 580`` for CUDA 13.x minor-version compatibility.
    """

    text: str
    components: tuple[int, ...]
    branch: int

    @classmethod
    def from_text(cls, text: str) -> DriverReleaseVersion:
        normalized_text = text.strip()
        if not _DRIVER_RELEASE_VERSION_RE.fullmatch(normalized_text):
            raise ValueError(f"Invalid driver release version text: {text!r}")
        components = tuple(int(component) for component in normalized_text.split("."))
        return cls(text=normalized_text, components=components, branch=components[0])


@functools.cache
def query_driver_cuda_version() -> DriverCudaVersion:
    """Return the CUDA driver version parsed into its major/minor components."""
    try:
        encoded = _query_driver_cuda_version_int()
        return cast(DriverCudaVersion, DriverCudaVersion.from_encoded(encoded))
    except Exception as exc:
        raise QueryDriverCudaVersionError("Failed to query the CUDA driver version.") from exc


@functools.cache
def query_driver_release_version() -> DriverReleaseVersion:
    """Return the display-driver release version parsed into branch/components."""
    try:
        return DriverReleaseVersion.from_text(_query_driver_release_version_text())
    except Exception as exc:
        raise QueryDriverReleaseVersionError("Failed to query the display-driver release version.") from exc


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


def _query_driver_release_version_text() -> str:
    """Return the display-driver release version from ``nvmlSystemGetDriverVersion()``."""
    loaded_nvml = _load_nvidia_dynamic_lib("nvml")
    nvml_lib = ctypes.CDLL(loaded_nvml.abs_path)

    nvml_init_v2 = nvml_lib.nvmlInit_v2
    nvml_init_v2.argtypes = []
    nvml_init_v2.restype = ctypes.c_int

    nvml_system_get_driver_version = nvml_lib.nvmlSystemGetDriverVersion
    nvml_system_get_driver_version.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.c_uint]
    nvml_system_get_driver_version.restype = ctypes.c_int

    nvml_shutdown = nvml_lib.nvmlShutdown
    nvml_shutdown.argtypes = []
    nvml_shutdown.restype = ctypes.c_int

    # NVML's init/shutdown pair is reference-counted (see "Initialization and
    # Cleanup" in the NVML API docs), so this balanced pair is safe even when
    # the caller has already initialized NVML elsewhere in the process.
    init_status = nvml_init_v2()
    if init_status != _NVML_SUCCESS:
        raise RuntimeError(f"Failed to initialize NVML via nvmlInit_v2() (status={init_status}).")

    try:
        version_buffer = ctypes.create_string_buffer(_NVML_SYSTEM_DRIVER_VERSION_BUFFER_LENGTH)
        status = nvml_system_get_driver_version(version_buffer, _NVML_SYSTEM_DRIVER_VERSION_BUFFER_LENGTH)
        if status != _NVML_SUCCESS:
            raise RuntimeError(
                f"Failed to query driver release version via nvmlSystemGetDriverVersion() (status={status})."
            )
        release_version = version_buffer.value.decode()
    finally:
        # Balance the init_v2() above unconditionally. If the body already
        # raised, let that error win; a non-zero shutdown status here would
        # only mask the more useful root cause (Python keeps it on
        # ``__context__`` for debugging). ``sys.exc_info()[1]`` is the
        # currently-propagating exception inside the finally, or None.
        shutdown_status = nvml_shutdown()
        if shutdown_status != _NVML_SUCCESS and sys.exc_info()[1] is None:
            raise RuntimeError(f"Failed to shut down NVML via nvmlShutdown() (status={shutdown_status}).")
    return release_version
