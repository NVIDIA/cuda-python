# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.bindings cimport cydriver
from cuda.core.experimental._resource_handles cimport StreamHandle


cdef class Stream:

    cdef:
        StreamHandle _h_stream
        object _owner
        bint _builtin
        int _nonblocking
        int _priority
        cydriver.CUdevice _device_id
        cydriver.CUcontext _ctx_handle

    cpdef close(self)
    cdef int _get_context(self) except?-1 nogil
    cdef int _get_device_and_context(self) except?-1


cpdef Stream default_stream()
cdef Stream Stream_accept(arg, bint allow_stream_protocol=*)
