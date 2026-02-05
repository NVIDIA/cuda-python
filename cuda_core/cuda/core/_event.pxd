# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.bindings cimport cydriver
from cuda.core._resource_handles cimport ContextHandle, EventHandle


cdef class Event:

    cdef:
        EventHandle _h_event
        ContextHandle _h_context
        bint _timing_disabled
        bint _busy_waited
        bint _ipc_enabled
        object _ipc_descriptor
        int _device_id
        object __weakref__

    @staticmethod
    cdef Event _init(type cls, int device_id, ContextHandle h_context, options, bint is_free)

    cpdef close(self)
