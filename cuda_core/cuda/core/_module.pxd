# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

cdef class ObjectCode
cdef class Kernel
cdef class KernelOccupancy


cdef class Kernel:
    cdef:
        object _handle      # CUkernel (will become KernelHandle in phase 2c)
        ObjectCode _module  # ObjectCode reference
        object _attributes  # KernelAttributes (regular Python class)
        KernelOccupancy _occupancy  # KernelOccupancy (lazy)
        object __weakref__  # Enable weak references

    @staticmethod
    cdef Kernel _from_obj(object obj, ObjectCode mod)


cdef class ObjectCode:
    cdef:
        object _handle      # CUlibrary (will become LibraryHandle in phase 2c)
        str _code_type
        object _module      # bytes/str source
        dict _sym_map
        str _name

    @staticmethod
    cdef ObjectCode _init(object module, str code_type, str name=*, dict symbol_mapping=*)

    cdef int _lazy_load_module(self) except -1


cdef class KernelOccupancy:
    cdef:
        object _handle      # CUkernel reference

    @staticmethod
    cdef KernelOccupancy _init(object handle)
