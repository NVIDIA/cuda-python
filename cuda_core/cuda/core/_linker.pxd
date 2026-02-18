# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from ._resource_handles cimport NvJitLinkHandle, CuLinkHandle


cdef class Linker:
    cdef:
        NvJitLinkHandle _nvjitlink_handle
        CuLinkHandle _culink_handle
        bint _use_nvjitlink
        object _drv_log_bufs  # formatted_options list (driver); None for nvjitlink; cleared in link()
        str _info_log         # decoded log; None until link() or pre-link get_*_log()
        str _error_log        # decoded log; None until link() or pre-link get_*_log()
        object _options       # LinkerOptions
        object __weakref__
