# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import threading

from cuda.bindings import _nvml as nvml

from . import exceptions


_NVML_STATE = _NVMLState.UNINITIALIZED


_NVML_OWNER_PID = 0


_lock = threading.Lock()


# For testing
def _get_nvml_state():
    return _NVML_STATE


cpdef _initialize():
    """
    Initializes Nvidia Management Library (NVML), ensuring it only happens once per process.
    """
    global _NVML_STATE, _NVML_OWNER_PID

    with _lock:
        # Double-check to make sure nothing has changed since acquiring the lock
        if _NVML_STATE == _NVMLState.DISABLED_LIBRARY_NOT_FOUND or (
            _NVML_STATE == _NVMLState.INITIALIZED and getpid() == _NVML_OWNER_PID
        ):
            return
        elif (
            _NVML_STATE == _NVMLState.INITIALIZED and getpid() != _NVML_OWNER_PID
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
            _NVML_OWNER_PID = getpid()
        else:
            raise RuntimeError(f"Unhandled initialisation state ({_NVML_STATE=}, {_NVML_OWNER_PID=})")


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
        raise exceptions.LibraryNotFoundError()
    elif not is_initialized():
        raise exceptions.UninitializedError()
    elif nvml.device_get_count_v2() == 0:
        raise exceptions.GpuNotFoundError()
