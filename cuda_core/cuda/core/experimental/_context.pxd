# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.bindings cimport cydriver

cdef class Context:
    """Cython declaration for Context class.

    This class provides access to CUDA contexts. Context objects cannot be
    instantiated directly - use factory methods or Device/Stream APIs.
    """

    cdef:
        readonly object _handle
        int _device_id

# Cython-level context operations
cdef cydriver.CUcontext get_primary_context(int dev_id) except?NULL
cdef cydriver.CUcontext get_current_context() except?NULL nogil
cdef void set_current_context(cydriver.CUcontext ctx) except *
cdef cydriver.CUcontext get_stream_context(cydriver.CUstream stream) except?NULL nogil
