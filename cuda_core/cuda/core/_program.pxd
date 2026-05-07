# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from ._linker cimport Linker
from ._resource_handles cimport NvrtcProgramHandle, NvvmProgramHandle


cdef class Program:
    cdef:
        NvrtcProgramHandle _h_nvrtc
        NvvmProgramHandle _h_nvvm
        str _backend
        Linker _linker
        object _options  # ProgramOptions
        object __weakref__
        object _compile_lock  # Per-instance lock for compile-time mutation
        bint _use_libdevice      # Flag for libdevice loading
        bint _libdevice_added
        bytes _code             # Source code as bytes: used for key derivation and NVRTC PCH retry
        str _code_type          # Normalised code_type ("c++", "ptx", "nvvm")
        str _pch_status         # PCH creation outcome after compile
