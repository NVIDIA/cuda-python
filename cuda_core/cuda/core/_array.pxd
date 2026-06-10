# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport intptr_t
from cuda.bindings cimport cydriver


cdef class CUDAArray:

    cdef:
        cydriver.CUarray _handle
        tuple _shape                 # (w,), (w, h), or (w, h, d)
        cydriver.CUarray_format _format
        unsigned int _num_channels   # 1, 2, or 4
        int _device_id
        intptr_t _context
        bint _owning
        bint _surface_load_store
        # Optional strong reference to a parent owner (e.g. a MipmappedArray
        # whose level this CUDAArray views). When set, the parent must outlive
        # this CUDAArray because the underlying CUarray belongs to the parent.
        object _parent_ref

    cpdef close(self)
