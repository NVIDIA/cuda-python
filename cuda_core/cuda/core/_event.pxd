# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.bindings cimport cydriver
from cuda.core._resource_handles cimport ContextHandle, EventHandle


cdef class Event:

    cdef:
        EventHandle _h_event
        object _ipc_descriptor
        object __weakref__

    @staticmethod
    cdef Event _init(type cls, int device_id, ContextHandle h_context, options, bint is_free)

    @staticmethod
    cdef Event _from_handle(EventHandle h_event)

    cpdef close(self)
