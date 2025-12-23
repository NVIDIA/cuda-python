# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cpython cimport array
from libc.stdint cimport uint64_t


cpdef str format_bytes(uint64_t x):
    """Return formatted string in B, KiB, MiB, GiB or TiB"""
    if x < 1024:
        return f"{x} B"
    elif x < 1024**2:
        return f"{x / 1024:.2f} KiB"
    elif x < 1024**3:
        return f"{x / 1024**2:.2f} MiB"
    elif x < 1024**4:
        return f"{x / 1024**3:.2f} GiB"
    else:
        return f"{x / 1024**4:.2f} TiB"


cpdef list[int] unpack_bitmask(x: list[int] | array.array):
    """
    Unpack a list of integers containing bitmasks.

    Parameters
    ----------
    x: list of int
        A list of integers

    Examples
    --------
    >>> from cuda.core.experimental.system.utils import unpack_bitmask
    >>> unpack_bitmask([1 + 2 + 8])
    [0, 1, 3]
    >>> unpack_bitmask([1 + 2 + 16])
    [0, 1, 4]
    >>> unpack_bitmask([1 + 2 + 16, 2 + 4])
    [0, 1, 4, 65, 66]
    """
    cdef uint64_t[:] arr
    cdef uint64_t i, j, idx
    cdef int mask_bits = 64

    if isinstance(x, list):
        arr = array.array("Q", x)
    else:
        arr = x

    res = []

    for i in range(len(x)):
        cpu_offset = i * mask_bits
        idx = 1
        for j in range(mask_bits):
            if arr[i] & idx:
                res.append(cpu_offset + j)
            idx <<= 1
    return res
