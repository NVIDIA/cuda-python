# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


cdef extern from *:
    """
    #if defined(_WIN32) || defined(_WIN64)
        #include <process.h>
    #else
        #include <unistd.h>
    #endif
    """
    int getpid() nogil


ctypedef enum _NVMLState:
    UNINITIALIZED = 0
    INITIALIZED = 1
    DISABLED_LIBRARY_NOT_FOUND = 2


# Initialisation must occur per-process, so an initialised state is a
# (state, pid) pair
cdef _NVMLState _NVML_STATE


cdef int _NVML_OWNER_PID


cpdef _initialize()


cpdef inline initialize():
    """
    Initializes Nvidia Management Library (NVML), ensuring it only happens once per process.
    """
    if _NVML_STATE == _NVMLState.DISABLED_LIBRARY_NOT_FOUND or (
        _NVML_STATE == _NVMLState.INITIALIZED and getpid() == _NVML_OWNER_PID
    ):
        return

    _initialize()


cpdef inline bint is_initialized():
    """
    Check whether the NVML context is initialized on this process.

    Returns
    -------
    result: bool
        Whether the NVML context is initialized on this process.
    """
    return _NVML_STATE == _NVMLState.INITIALIZED and getpid() == _NVML_OWNER_PID


cpdef validate()
