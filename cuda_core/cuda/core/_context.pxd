# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.core._resource_handles cimport ContextHandle

cdef class Context:
    """Cython declaration for Context class.

    This class provides access to CUDA contexts. Context objects cannot be
    instantiated directly - use factory methods or Device/Stream APIs.
    """

    cdef:
        ContextHandle _h_context
        int _device_id

    @staticmethod
    cdef Context _from_handle(type cls, ContextHandle h_context, int device_id)
