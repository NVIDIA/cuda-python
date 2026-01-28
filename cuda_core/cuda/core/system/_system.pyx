# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


# This file needs to either use NVML exclusively, or when `cuda.bindings._nvml`
# isn't available, fall back to non-NVML-based methods for backward
# compatibility.


CUDA_BINDINGS_NVML_IS_COMPATIBLE: bool

try:
    from cuda.bindings._version import __version_tuple__ as _BINDINGS_VERSION
except ImportError:
    CUDA_BINDINGS_NVML_IS_COMPATIBLE = False
else:
    CUDA_BINDINGS_NVML_IS_COMPATIBLE = _BINDINGS_VERSION >= (13, 1, 2) or (_BINDINGS_VERSION[0] == 12 and _BINDINGS_VERSION[1:3] >= (9, 6))


if CUDA_BINDINGS_NVML_IS_COMPATIBLE:
    from cuda.bindings import nvml
    # TODO: We need to be even more specific than version numbers for development.
    # This can be removed once we have a release including everything we need.
    for member in ["FieldId", "ClocksEventReasons"]:
        if not hasattr(nvml, member):
            CUDA_BINDINGS_NVML_IS_COMPATIBLE = False
            break

if CUDA_BINDINGS_NVML_IS_COMPATIBLE:
    from ._nvml_context import initialize
else:
    from cuda.core._utils.cuda_utils import driver, handle_return, runtime


def get_driver_version(kernel_mode: bool = False) -> tuple[int, int]:
    """
    Get the driver version.

    Parameters
    ----------
    kernel_mode: bool
        When `True`, return the kernel-mode driver version, e.g. 580.65.06.
        Otherwise, return the user-mode driver version, e.g. 13.0.1.

    Returns
    -------
    version: tuple[int, int]
        Tuple in the format `(MAJOR, MINOR)`.
    """
    return get_driver_version_full(kernel_mode)[:2]


def get_driver_version_full(kernel_mode: bool = False) -> tuple[int, int, int]:
    """
    Get the full driver version.

    Parameters
    ----------
    kernel_mode: bool
        When `True`, return the kernel-mode driver version, e.g. 580.65.06.
        Otherwise, return the user-mode driver version, e.g. 13.0.1.

    Returns
    -------
    version: tuple[int, int, int]
        Tuple in the format `(MAJOR, MINOR, PATCH)`.
    """
    cdef int v
    if kernel_mode:
        if not CUDA_BINDINGS_NVML_IS_COMPATIBLE:
            raise ValueError("Kernel-mode driver version requires NVML support")
        initialize()
        return tuple(int(v) for v in nvml.system_get_driver_version().split("."))
    else:
        if CUDA_BINDINGS_NVML_IS_COMPATIBLE:
            initialize()
            v = nvml.system_get_cuda_driver_version()
        else:
            v = handle_return(driver.cuDriverGetVersion())
        return (v // 1000, (v // 10) % 100, v % 10)


def get_nvml_version() -> tuple[int, ...]:
    """
    The version of the NVML library.
    """
    if not CUDA_BINDINGS_NVML_IS_COMPATIBLE:
        raise RuntimeError("NVML library is not available")
    return tuple(int(v) for v in nvml.system_get_nvml_version().split("."))


def get_driver_branch() -> str:
    """
    Retrieves the driver branch of the NVIDIA driver installed on the system.
    """
    if not CUDA_BINDINGS_NVML_IS_COMPATIBLE:
        raise RuntimeError("NVML library is not available")
    initialize()
    return nvml.system_get_driver_branch()


def get_num_devices() -> int:
    """
    Return the number of devices in the system.
    """
    if CUDA_BINDINGS_NVML_IS_COMPATIBLE:
        initialize()
        return nvml.device_get_count_v2()
    else:
        return handle_return(runtime.cudaGetDeviceCount())


def get_process_name(pid: int) -> str:
    """
    The name of process with given PID.

    Parameters
    ----------
    pid: int
        The PID of the process for which to get the name.

    Returns
    -------
    name: str
        The process name.
    """
    initialize()
    return nvml.system_get_process_name(pid)


__all__ = [
    "get_driver_branch",
    "get_driver_version",
    "get_driver_version_full",
    "get_nvml_version",
    "get_num_devices",
    "get_process_name",
    "CUDA_BINDINGS_NVML_IS_COMPATIBLE",
]
