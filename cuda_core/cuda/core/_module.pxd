# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.bindings cimport cydriver
from cuda.core._resource_handles cimport LibraryHandle, KernelHandle

cdef class ObjectCode
cdef class Kernel
cdef class KernelOccupancy
cdef class KernelAttributes


cdef class Kernel:
    cdef:
        KernelHandle _h_kernel
        KernelAttributes _attributes  # lazy
        KernelOccupancy _occupancy  # lazy
        object __weakref__

    @staticmethod
    cdef Kernel _from_obj(KernelHandle h_kernel)

    cdef tuple _get_arguments_info(self, bint param_info=*)


cdef class ObjectCode:
    cdef:
        LibraryHandle _h_library
        str _code_type
        object _module      # bytes/str source
        dict _sym_map
        str _name
        object __weakref__

    cdef int _lazy_load_module(self) except -1


cdef class KernelOccupancy:
    cdef:
        KernelHandle _h_kernel

    @staticmethod
    cdef KernelOccupancy _init(KernelHandle h_kernel)


cdef class KernelAttributes:
    cdef:
        KernelHandle _h_kernel
        dict _cache

    @staticmethod
    cdef KernelAttributes _init(KernelHandle h_kernel)

    cdef int _get_cached_attribute(self, int device_id, cydriver.CUfunction_attribute attribute) except? -1
    cdef int _resolve_device_id(self, device_id) except? -1
