# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uintptr_t

from cuda.bindings cimport cydriver
from cuda.core._rt cimport ContextHandle, StreamHandle, as_cu


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


cdef inline bint Stream_handle_is_default_token(cydriver.CUstream s) noexcept nogil:
    """Return True for the raw CU_STREAM_LEGACY and CU_STREAM_PER_THREAD tokens.

    These tokens carry no context of their own; they refer to whatever context
    is current, so nothing resolved from one may be cached on the object.
    """
    cdef uintptr_t h = <uintptr_t>s
    return h == <uintptr_t>cydriver.CU_STREAM_LEGACY or h == <uintptr_t>cydriver.CU_STREAM_PER_THREAD


cdef inline bint Stream_is_default_token(Stream self) noexcept nogil:
    """Return True when this stream is CU_STREAM_LEGACY or CU_STREAM_PER_THREAD."""
    return Stream_handle_is_default_token(as_cu(self._h_stream))


cdef inline bint Stream_is_legacy_default_token(Stream self) noexcept nogil:
    """Return True only for CU_STREAM_LEGACY.

    Unlike CU_STREAM_PER_THREAD, the legacy default stream token is rejected
    outright (CUDA_ERROR_INVALID_VALUE) by cuMemcpyWithAttributesAsync and
    cuMemcpyBatchAsync; CU_STREAM_PER_THREAD is a real stream to those entry
    points and is accepted normally. Use this narrower check, not
    Stream_is_default_token, wherever that distinction matters.
    """
    return <uintptr_t>as_cu(self._h_stream) == <uintptr_t>cydriver.CU_STREAM_LEGACY
