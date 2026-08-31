# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.core._resource_handles cimport ContextHandle, StreamHandle


cdef class Stream:

    cdef:
        StreamHandle _h_stream
        ContextHandle _h_context
        int _device_id
        int _nonblocking
        int _priority
        object __weakref__

    @staticmethod
    cdef Stream _from_handle(type cls, StreamHandle h_stream)

    cpdef close(self)


cpdef Stream default_stream()
cpdef Stream Stream_accept(arg, bint allow_stream_protocol=*)
cdef inline int Stream_check_open(Stream self) except -1:
    if not self._h_stream:
        raise RuntimeError("Stream has been closed")
    return 0
cdef bint Stream_is_default_token(Stream self) noexcept nogil
cdef bint Stream_is_legacy_default_token(Stream self) noexcept nogil
