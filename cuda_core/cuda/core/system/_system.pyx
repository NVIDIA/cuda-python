# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


# This file needs to either use NVML exclusively, or when `cuda.bindings.nvml`
# isn't available, fall back to non-NVML-based methods for backward
# compatibility.


CUDA_BINDINGS_NVML_IS_COMPATIBLE: bool


# POSIX per-thread locale APIs. We use these (rather than setlocale(3))
# so the WSL workaround in get_process_name() doesn't perturb the locale
# observed by other threads. locale_t is an opaque pointer in glibc.
cdef extern from "locale.h" nogil:
    ctypedef void *locale_t
    int LC_ALL_MASK
    locale_t LC_GLOBAL_LOCALE
    locale_t newlocale(int category_mask, const char *locale, locale_t base)
    locale_t uselocale(locale_t newloc)
    void freelocale(locale_t locobj)


cdef bint _detect_wsl():
    try:
        with open("/proc/sys/kernel/osrelease") as f:
            data = f.read().lower()
    except OSError:
        return False
    return "microsoft" in data or "wsl" in data


cdef bint _IS_WSL = _detect_wsl()

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
            dev_h = nvml.device_get_handle_by_index_v2(i)
            nvml.device_get_compute_running_processes_v3(dev_h)
        return nvml.system_get_process_name(pid)

    cdef locale_t c_locale
    cdef locale_t prev_locale

    initialize()
    if not _IS_WSL:
        return _get_process_name(pid)

    # WSL workaround: nvmlSystemGetProcessName on WSL takes a wide-char
    # conversion path when the process locale is non-"C". That path walks
    # a UTF-16LE source buffer with a 4-byte stride (as if it were UTF-32LE)
    # and emits 5-byte UTF-8 sequences that look like garbage preceding the
    # trailing basename of /proc/<pid>/exe. CPython's startup unconditionally
    # calls setlocale(LC_ALL, ""), so essentially every cuda.core caller hits
    # this. The cached entry for the PID is set the first time NVML resolves
    # it (typically inside nvmlDeviceGetComputeRunningProcesses_v3), so to
    # recover a correct value we re-prime the cache under the "C" locale
    # before reading the name. We use the POSIX per-thread locale APIs so
    # other threads' view of the locale is unaffected.
    c_locale = newlocale(LC_ALL_MASK, b"C", <locale_t>0)
    if c_locale == <locale_t>0:
        raise RuntimeError("Failed to create C locale")
    prev_locale = uselocale(c_locale)
    try:
        return _get_process_name(pid)
    finally:
        uselocale(prev_locale)
        freelocale(c_locale)


__all__ = [
    "get_driver_branch",
    "get_kernel_mode_driver_version",
    "get_user_mode_driver_version",
    "get_nvml_version",
    "get_num_devices",
    "get_process_name",
    "CUDA_BINDINGS_NVML_IS_COMPATIBLE",
]
