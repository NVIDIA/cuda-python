# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


# This file needs to either use NVML exclusively, or when `cuda.bindings.nvml`
# isn't available, fall back to non-NVML-based methods for backward
# compatibility.


CUDA_BINDINGS_NVML_IS_COMPATIBLE: bool


# Please keep in sync with the equivalent implementation in
# cuda_python_test_helpers/cuda_python_test_helpers/__init__.py.
cdef bint _detect_wsl():
    try:
        with open("/proc/sys/kernel/osrelease") as f:
            data = f.read().lower()
    except OSError:
        return False
    return "microsoft" in data or "wsl" in data


cdef bint _IS_WSL = _detect_wsl()


# The WSL locale guard lives in a separate module that is only compiled on
# Linux (build_hooks.py excludes it on Windows), because it relies on POSIX
# per-thread locale APIs that MSVC does not provide. On non-Linux platforms
# the import fails and we fall back to a no-op guard; _IS_WSL is then False
# so the guard is never entered anyway.
if _IS_WSL:
    from cuda.core._utils._wsl_locale import c_locale_guard
else:
    c_locale_guard = None


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
    def _get_process_name(pid) -> str:
        # NVML caches process names on a per-PID basis when queried via
        # nvmlSystemGetProcessName, and the cache is populated when enumerating
        # running processes on devices. To ensure the name is cached for the
        # requested PID, we walk all devices and query their running processes.
        for i in range(nvml.device_get_count_v2()):
            try:
                dev_h = nvml.device_get_handle_by_index_v2(i)
                nvml.device_get_compute_running_processes_v3(dev_h)
            except nvml.NvmlError:
                continue
        return nvml.system_get_process_name(pid)

    initialize()
    if not _IS_WSL:
        return _get_process_name(pid)

    # WSL workaround: nvmlSystemGetProcessName on WSL takes a wide-char
    # conversion path when the calling thread's locale is non-"C". That path
    # walks a UTF-16LE source buffer with a 4-byte stride (as if it were
    # UTF-32LE) and emits 5-byte UTF-8 sequences that look like garbage
    # preceding the trailing basename of /proc/<pid>/exe. CPython's startup
    # unconditionally calls setlocale(LC_ALL, ""), so essentially every
    # cuda.core caller hits this. The cached entry for the PID is set the
    # first time NVML resolves it (typically inside
    # nvmlDeviceGetComputeRunningProcesses_v3), so to recover a correct value
    # we re-prime the cache under the "C" locale before reading the name.
    # c_locale_guard uses POSIX per-thread locale APIs (see _wsl_locale.pyx)
    # so other threads' view of the locale is unaffected.
    with c_locale_guard():  # no-cython-lint
        return _get_process_name(pid)


__all__ = [
    "get_driver_branch",
    "get_kernel_mode_driver_version",
    "get_user_mode_driver_version",
    "get_nvml_version",
    "get_num_devices",
    "get_process_name",
    "CUDA_BINDINGS_NVML_IS_COMPATIBLE",
]
