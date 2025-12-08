# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.bindings cimport cydriver
from cuda.core.experimental._resource_handles cimport ContextHandle

cdef class Context:
    """Cython declaration for Context class.

    This class provides access to CUDA contexts. Context objects cannot be
    instantiated directly - use factory methods or Device/Stream APIs.
    """

    cdef:
        ContextHandle _h_context
        int _device_id

# Cython-level context operations (handle-centric API)
cdef ContextHandle get_primary_context(int dev_id) except *
cdef ContextHandle get_current_context() except * nogil
cdef void set_current_context(ContextHandle h_context) except * nogil
cdef ContextHandle get_stream_context(cydriver.CUstream stream) except * nogil
