# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.core.experimental._resource_handles cimport ContextHandle, StreamHandle


cdef class Stream:

    cdef:
        StreamHandle _h_stream
        ContextHandle _h_context
        int _device_id
        int _nonblocking
        int _priority

    @staticmethod
    cdef Stream _from_handle(type cls, StreamHandle h_stream)

    cpdef close(self)
    cdef int _get_context(self) except?-1 nogil
    cdef int _get_device_and_context(self) except?-1


cpdef Stream default_stream()
cdef Stream Stream_accept(arg, bint allow_stream_protocol=*)
