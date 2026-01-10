# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uint64_t


cpdef inline list[int] _unpack_bitmask(uint64_t[:] arr):
    """
    Unpack a list of integers containing bitmasks.
    """
    cdef uint64_t i, j, idx
    cdef int mask_bits = 64

    res = []

    for i in range(len(arr)):
        cpu_offset = i * mask_bits
        idx = 1
        for j in range(mask_bits):
            if arr[i] & idx:
                res.append(cpu_offset + j)
            idx <<= 1
    return res
