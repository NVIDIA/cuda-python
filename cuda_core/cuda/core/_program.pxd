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
        bint _use_libdevice      # Flag for libdevice loading