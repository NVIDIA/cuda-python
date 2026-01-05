# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import os
import threading

from cuda.bindings import _nvml as nvml

from . import exceptions


ctypedef enum _NVMLState:
    UNINITIALIZED = 0
    INITIALIZED = 1
    DISABLED_LIBRARY_NOT_FOUND = 2


# Initialisation must occur per-process, so an initialised state is a
# (state, pid) pair
_NVML_STATE = _NVMLState.UNINITIALIZED
# """Current initialization state"""

_NVML_OWNER_PID = 0
# """PID of process that successfully called pynvml.nvmlInit"""


_lock = threading.Lock()


cpdef initialize():
    """
    Initializes Nvidia Management Library (NVML), ensuring it only happens once per process.
    """
    global _NVML_STATE, _NVML_OWNER_PID

    with _lock:
        if _NVML_STATE == _NVMLState.DISABLED_LIBRARY_NOT_FOUND or (
            _NVML_STATE == _NVMLState.INITIALIZED and os.getpid() == _NVML_OWNER_PID
        ):
            return
        elif (
            _NVML_STATE == _NVMLState.INITIALIZED and os.getpid() != _NVML_OWNER_PID
        ) or _NVML_STATE == _NVMLState.UNINITIALIZED:
            try:
                nvml.init_v2()
            except (
                exceptions.LibraryNotFoundError,
                exceptions.DriverNotLoadedError,
                exceptions.UnknownError,
            ):
                _NVML_STATE = _NVMLState.DISABLED_LIBRARY_NOT_FOUND
                return

            # initialization was successful
            _NVML_STATE = _NVMLState.INITIALIZED
            _NVML_OWNER_PID = os.getpid()
        else:
            raise RuntimeError(f"Unhandled initialisation state ({_NVML_STATE=}, {_NVML_OWNER_PID=})")


cpdef bint is_initialized():
    """
    Check whether the NVML context is initialized on this process.

    Returns
    -------
    result: bool
        Whether the NVML context is initialized on this process.
    """
    return _NVML_STATE == _NVMLState.INITIALIZED and os.getpid() == _NVML_OWNER_PID


cpdef validate():
    """
    Validate NVML state.

    Validate that NVML is initialized, functional and that the system has at
    least one GPU available.

    Raises
    ------
    nvml.UninitializedError
        If NVML hasn't been initialized.
    nvml.LibraryNotFoundError
        If the NVML library could not be found.
    nvml.GpuNotFoundError
        If no GPUs are available.
    """
    if _NVML_STATE == _NVMLState.DISABLED_LIBRARY_NOT_FOUND:
        raise exceptions.LibraryNotFoundError("The underlying NVML library was not found")
    elif not is_initialized():
        raise exceptions.UninitializedError("NVML library is not initialized")
    elif nvml.device_get_count_v2() == 0:
        raise exceptions.GpuNotFoundError("No GPUs available")
