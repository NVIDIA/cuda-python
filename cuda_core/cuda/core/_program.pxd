# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from ._resource_handles cimport NvrtcProgramHandle, NvvmProgramHandle


cdef class Program:
    cdef:
        NvrtcProgramHandle _h_nvrtc
        NvvmProgramHandle _h_nvvm
        str _backend
        object _linker  # Linker
        object _options  # ProgramOptions
        object __weakref__
        object _compile_lock  # Per-instance lock for compile-time mutation
        bint _use_libdevice      # Flag for libdevice loading
        bint _libdevice_added
        bytes _nvrtc_code       # Source code for NVRTC retry (PCH auto-resize)
        str _pch_status         # PCH creation outcome after compile
