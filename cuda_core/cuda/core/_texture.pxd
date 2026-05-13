# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport intptr_t
from cuda.bindings cimport cydriver


cdef class TextureObject:

    cdef:
        cydriver.CUtexObject _handle
        object _source_ref      # keep backing Array (or other resource) alive
        object _texture_desc    # original TextureDescriptor for introspection
        int _device_id
        intptr_t _context

    cpdef close(self)
