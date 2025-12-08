# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uintptr_t

from cuda.bindings import driver
from cuda.core.experimental._resource_handles cimport ContextHandle


cdef object py(ContextHandle h):
    """Convert the handle to a Python driver.CUcontext object.

    This is for use with driver (Python) API calls or returning to Python code.
    """
    return driver.CUcontext(<uintptr_t>(h.get()[0]))
