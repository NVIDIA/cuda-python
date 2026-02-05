# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport intptr_t
from libcpp cimport vector


cdef class ParamHolder:

    cdef:
        vector.vector[void*] data
        vector.vector[void*] data_addresses
        object kernel_args
        readonly intptr_t ptr
