# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.bindings cimport cydriver
from libc.stdint cimport intptr_t
from cuda.core._memoryview cimport StridedMemoryView


cdef class TensorMapDescriptor:
    cdef cydriver.CUtensorMap _tensor_map
    cdef int _device_id
    cdef intptr_t _context
    cdef object _source_ref
    cdef StridedMemoryView _view_ref
    cdef object _repr_info

    cdef int _check_context_compat(self) except -1
    cdef void* _get_data_ptr(self)
