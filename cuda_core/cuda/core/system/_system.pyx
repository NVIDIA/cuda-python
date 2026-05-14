# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


# This file needs to either use NVML exclusively, or when `cuda.bindings.nvml`
# isn't available, fall back to non-NVML-based methods for backward
# compatibility.


CUDA_BINDINGS_NVML_IS_COMPATIBLE: bool

try:
    from cuda.bindings._version import __version_tuple__ as _BINDINGS_VERSION
except ImportError:
    CUDA_BINDINGS_NVML_IS_COMPATIBLE = False
else:
    CUDA_BINDINGS_NVML_IS_COMPATIBLE = _BINDINGS_VERSION >= (13, 2, 0) or (_BINDINGS_VERSION[0] == 12 and _BINDINGS_VERSION[1:3] >= (9, 6))


if CUDA_BINDINGS_NVML_IS_COMPATIBLE:
    try:
        from cuda.bindings import nvml
    except ImportError:
        CUDA_BINDINGS_NVML_IS_COMPATIBLE = False

    from cuda.core.system._nvml_context import initialize
else:
    from cuda.core._utils.cuda_utils import driver, handle_return, runtime


def get_user_mode_driver_version() -> tuple[int, ...]:
    """
    Get the user-mode (UMD / CUDA) driver version.

    This is the most commonly needed version when checking CUDA driver
    compatibility.  It works with all ``cuda-bindings`` versions.

    Returns
    -------
    version : tuple[int, ...]
        A 2-tuple ``(MAJOR, MINOR)``, e.g. ``(13, 0)`` for CUDA 13.0.
    """
    cdef int v
    if CUDA_BINDINGS_NVML_IS_COMPATIBLE:
        initialize()
        v = nvml.system_get_cuda_driver_version()
    else:
        v = handle_return(driver.cuDriverGetVersion())
    return (v // 1000, (v // 10) % 100)


def get_kernel_mode_driver_version() -> tuple[int, ...]:
    """
    Get the kernel-mode (KMD / GPU) driver version, e.g. 580.65.06.

    Returns
    -------
    version : tuple[int, ...]
        Typically a 3-tuple ``(MAJOR, MINOR, PATCH)``
        (2-tuple on WSL), e.g. ``(580, 65, 6)``.

    Raises
    ------
    RuntimeError
        If the NVML library is not available.
    """
    if not CUDA_BINDINGS_NVML_IS_COMPATIBLE:
        raise RuntimeError(
            "get_kernel_mode_driver_version requires NVML support"
        )
    initialize()
    return tuple(int(x) for x in nvml.system_get_driver_version().split("."))


def get_nvml_version() -> tuple[int, ...]:
    """
    The version of the NVML library.

    Returns
    -------
    version: tuple[int, ...]
        Tuple of integers representing the NVML version components.
    """
    if not CUDA_BINDINGS_NVML_IS_COMPATIBLE:
        raise RuntimeError("NVML library is not available")
    return tuple(int(v) for v in nvml.system_get_nvml_version().split("."))


def get_driver_branch() -> str:
    """
    Retrieves the driver branch of the NVIDIA driver installed on the system.

    Returns
    -------
    branch: str
        The driver branch string (e.g., ``"560"``, ``"open"``, etc.).
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
    "get_kernel_mode_driver_version",
    "get_user_mode_driver_version",
    "get_nvml_version",
    "get_num_devices",
    "get_process_name",
    "CUDA_BINDINGS_NVML_IS_COMPATIBLE",
]
