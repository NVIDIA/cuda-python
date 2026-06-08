# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from libcpp.vector cimport vector

from cuda.bindings cimport cydriver

from ._resource_handles cimport NvJitLinkHandle, CuLinkHandle


cdef class Linker:
    cdef:
        NvJitLinkHandle _nvjitlink_handle
        CuLinkHandle _culink_handle
        # WARNING: the driver backend passes raw pointers from _drv_jit_values
        # and _drv_log_bufs to cuLinkCreate. cuLinkDestroy may still dereference
        # them, so _close_noexcept must reset _culink_handle before releasing
        # these retainers. Do not bypass or weaken that teardown order.
        vector[cydriver.CUjit_option] _drv_jit_keys
        vector[void*] _drv_jit_values
        bint _use_nvjitlink
        object _drv_log_bufs  # formatted_options list (driver); None for nvjitlink
        str _info_log         # decoded log; None until link() or pre-link get_*_log()
        str _error_log        # decoded log; None until link() or pre-link get_*_log()
        object _options       # LinkerOptions
        object __weakref__

    cdef void _close_noexcept(self) noexcept
