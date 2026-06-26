# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.bindings cimport cydriver
from cuda.core._resource_handles cimport SurfObjectHandle


cdef class SurfaceObject:

    cdef:
        # The backing OpaqueArray is kept alive structurally by the C++ box behind
        # this handle, not by _source_ref.
        SurfObjectHandle _handle
        object _source_ref      # ResourceDescriptor, retained for introspection
        int _device_id

    cpdef close(self)
