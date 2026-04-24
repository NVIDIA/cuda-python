# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.core._resource_handles cimport ContextHandle, GreenCtxHandle

cdef class Context:
    """Cython declaration for Context class.

    This class provides access to CUDA contexts. Context objects cannot be
    instantiated directly - use factory methods or Device/Stream APIs.
    """

    cdef:
        ContextHandle _h_context
        GreenCtxHandle _h_green_ctx
        int _device_id
        bint _is_green
        object __weakref__

    @staticmethod
    cdef Context _from_handle(type cls, ContextHandle h_context, int device_id)

    @staticmethod
    cdef Context _from_green_ctx(type cls, GreenCtxHandle h_green_ctx, int device_id)

    cpdef close(self)
