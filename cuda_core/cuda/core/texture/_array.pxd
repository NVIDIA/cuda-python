# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.bindings cimport cydriver
from cuda.core._resource_handles cimport OpaqueArrayHandle


cdef class OpaqueArray:

    cdef:
        # Owning/non-owning + any parent (mipmap) dependency are encoded
        # structurally in the C++ box behind this handle, not in Python state.
        OpaqueArrayHandle _handle
        tuple _shape                 # (w,), (w, h), or (w, h, d)
        cydriver.CUarray_format _format
        unsigned int _num_channels   # 1, 2, or 4
        int _device_id
        bint _surface_load_store

    cpdef close(self)


# Wrap an existing OpaqueArrayHandle as a OpaqueArray, querying the driver for the
# array's shape/format/channels/surface-flag metadata. Used by get_level and
# the graphics-interop _from_handle path.
cdef OpaqueArray _array_from_handle(OpaqueArrayHandle h, int device_id)
