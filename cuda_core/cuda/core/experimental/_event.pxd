# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.bindings cimport cydriver
from cuda.core.experimental._resource_handles cimport ContextHandle


cdef class Event:

    cdef:
        cydriver.CUevent _handle
        bint _timing_disabled
        bint _busy_waited
        bint _ipc_enabled
        object _ipc_descriptor
        int _device_id
        ContextHandle _h_context

    @staticmethod
    cdef Event _init(type cls, int device_id, ContextHandle h_context, options, bint is_free)

    cpdef close(self)
