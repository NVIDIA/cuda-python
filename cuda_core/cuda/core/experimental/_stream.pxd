# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.bindings cimport cydriver


cdef cydriver.CUstream _try_to_get_stream_ptr(obj: IsStreamT) except*


cdef class Stream:

    cdef:
        cydriver.CUstream _handle
        object _owner
        bint _builtin
        int _nonblocking
        int _priority
        cydriver.CUdevice _device_id
        cydriver.CUcontext _ctx_handle

    cpdef close(self)
    cdef int _get_context(self) except?-1 nogil
    cdef int _get_device_and_context(self) except?-1


cdef Stream default_stream()
