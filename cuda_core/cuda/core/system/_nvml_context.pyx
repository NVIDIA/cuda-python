# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import os
import threading

from cuda.bindings import _nvml as nvml


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


def initialize() -> None:
    """Idempotent (per-process) initialization of NVUtil's NVML

    Notes
    -----

    Modifies global variables _NVML_STATE and _NVML_OWNER_PID"""
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
                nvml.LibraryNotFoundError,
                nvml.DriverNotLoadedError,
                nvml.UnknownError,
            ):
                _NVML_STATE = _NVMLState.DISABLED_LIBRARY_NOT_FOUND
                return

            # initialization was successful
            _NVML_STATE = _NVMLState.INITIALIZED
            _NVML_OWNER_PID = os.getpid()
        else:
            raise RuntimeError(f"Unhandled initialisation state ({_NVML_STATE=}, {_NVML_OWNER_PID=})")


def is_initialized() -> bool:
    """
    Check whether the NVML context is initialized on this process.

    Returns
    -------
    result: bool
        Whether the NVML context is initialized on this process.
    """
    return _NVML_STATE == _NVMLState.INITIALIZED and os.getpid() == _NVML_OWNER_PID


def validate() -> None:
    """
    Validate NVML state.

    Validate that NVML is functional and that the system has at least one GPU available.

    Raises
    ------
    nvml.LibraryNotFoundError
        If the NVML library could not be found.
    nvml.GpuNotFoundError
        If no GPUs are available.
    """
    if _NVML_STATE == _NVMLState.DISABLED_LIBRARY_NOT_FOUND:
        raise nvml.LibraryNotFoundError("The underlying NVML library was not found")
    elif nvml.device_get_count_v2() == 0:
        raise nvml.GpuNotFoundError("No GPUs available")
