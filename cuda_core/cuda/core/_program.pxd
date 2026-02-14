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
