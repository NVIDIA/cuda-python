# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from ._resource_handles cimport NvJitLinkHandle, CuLinkHandle


cdef class Linker:
    cdef:
        NvJitLinkHandle _nvjitlink_handle
        CuLinkHandle _culink_handle
        bint _use_nvjitlink
        object _formatted_options  # list (both backends); driver path uses indices 0,2 for logs
        object _option_keys  # list (driver only) or None
        object _options  # LinkerOptions
        object __weakref__
