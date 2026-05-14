# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport intptr_t
from cuda.bindings cimport cydriver


cdef class MipmappedArray:

    cdef:
        cydriver.CUmipmappedArray _handle
        tuple _shape                 # (w,), (w, h), or (w, h, d)
        int _format                  # CUarray_format value
        unsigned int _num_channels   # 1, 2, or 4
        unsigned int _num_levels
        int _device_id
        intptr_t _context
        bint _owning
        bint _surface_load_store

    cpdef close(self)
