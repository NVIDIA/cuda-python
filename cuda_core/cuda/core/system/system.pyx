# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


# This file needs to either use NVML exclusively, or when `cuda.bindings._nvml`
# isn't available, fall back to non-NVML-based methods for backward
# compatibility.

try:
    from cuda.bindings import _nvml as nvml
except ImportError:
    HAS_NVML = False
    from ._utils.cuda_utils import driver, handle_return, runtime
else:
    HAS_NVML = True


def get_driver_version() -> tuple[int, int]:
    """
    The CUDA driver version.

    Tuple in the format `(CUDA_MAJOR, CUDA_MINOR)`.
    """
    return get_driver_version_full()[:2]


def get_driver_version_full() -> tuple[int, int, int]:
    """
    The CUDA driver version.

    Tuple in the format `(CUDA_MAJOR, CUDA_MINOR, CUDA_PATCH)`.
    """
    cdef int v
    if HAS_NVML:
        v = nvml.system_get_cuda_driver_version()
    else:
        v = handle_return(driver.cuDriverGetVersion())
    return (v // 1000, (v // 10) % 100, v % 10)


def get_gpu_driver_version() -> tuple[int, ...]:
    """
    The driver version.
    """
    if not HAS_NVML:
        raise RuntimeError("NVML library is not available")
    return tuple(int(v) for v in nvml.system_get_driver_version().split("."))


def get_nvml_version() -> tuple[int, ...]:
    """
    The version of the NVML library.
    """
    if not HAS_NVML:
        raise RuntimeError("NVML library is not available")
    return tuple(int(v) for v in nvml.system_get_nvml_version().split("."))


def get_num_devices() -> int:
    """
    Return the number of devices in the system.
    """
    if HAS_NVML:
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
    return nvml.system_get_process_name(pid)


__all__ = [
    "get_driver_version",
    "get_driver_version_full",
    "get_gpu_driver_version",
    "get_nvml_version",
    "get_num_devices",
    "get_process_name",
]
