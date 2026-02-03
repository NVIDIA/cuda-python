# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


cdef class Program:
    cdef:
        object _mnff
        str _backend
        object _linker  # Linker
        object _options  # ProgramOptions
        object __weakref__
