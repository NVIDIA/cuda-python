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
        # _drv_jit_keys/_drv_jit_values are the C arrays handed to cuLinkCreate.
        # The driver retains a reference to the optionValues array for the life
        # of the CUlinkState (it writes back log-size outputs into its slots),
        # so these must live past cuLinkCreate and outlive cuLinkDestroy.
        # Declared after _culink_handle so their C++ destructors run AFTER
        # cuLinkDestroy executes during tp_dealloc.
        vector[cydriver.CUjit_option] _drv_jit_keys
        vector[void*] _drv_jit_values
        bint _use_nvjitlink
        object _drv_log_bufs  # formatted_options list (driver); None for nvjitlink
        str _info_log         # decoded log; None until link() or pre-link get_*_log()
        str _error_log        # decoded log; None until link() or pre-link get_*_log()
        object _options       # LinkerOptions
        object __weakref__
