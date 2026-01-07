# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cpython cimport array
from libc.stdint cimport uint64_t


def _unpack_bitmask(x: list[int] | array.array) -> list[int]:
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
