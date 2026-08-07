# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read this process's virtual address space layout on Windows.

Only imported on Windows -- see helpers/va_reservation.py, which guards the
import. There is deliberately no Linux counterpart: the address-space pressure
this exists to diagnose (issue #2381) is specific to the bounded per-process
budget on Windows, and on Linux the budget is large enough that the layout is
not interesting.

``cuMemAddressReserve`` probing can only report the largest single hole. That
turned out to be the wrong number: two pool reservations do not need one hole
twice their size, they need *two* holes, so a session can start with a smaller
largest-hole and still succeed. Walking the address space shows the whole hole
structure, which is what actually predicts the outcome.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes

MEM_COMMIT = 0x1000
MEM_RESERVE = 0x2000
MEM_FREE = 0x10000

# User-mode address space ceiling; walking past it wastes time and returns nothing.
_USER_SPACE_LIMIT = 1 << 47


class MEMORY_BASIC_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BaseAddress", ctypes.c_void_p),
        ("AllocationBase", ctypes.c_void_p),
        ("AllocationProtect", wintypes.DWORD),
        ("PartitionId", wintypes.WORD),
        ("__alignment", wintypes.WORD),
        ("RegionSize", ctypes.c_size_t),
        ("State", wintypes.DWORD),
        ("Protect", wintypes.DWORD),
        ("Type", wintypes.DWORD),
    ]


_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_kernel32.VirtualQuery.argtypes = [ctypes.c_void_p, ctypes.POINTER(MEMORY_BASIC_INFORMATION), ctypes.c_size_t]
_kernel32.VirtualQuery.restype = ctypes.c_size_t


def walk():
    """Yield ``(base, size, state)`` for every region in this process."""
    info = MEMORY_BASIC_INFORMATION()
    address = 0
    while address < _USER_SPACE_LIMIT:
        if not _kernel32.VirtualQuery(ctypes.c_void_p(address), ctypes.byref(info), ctypes.sizeof(info)):
            break
        size = info.RegionSize
        if size == 0:
            break
        yield address, size, info.State
        address += size


def free_regions(min_size: int = 0) -> list[tuple[int, int]]:
    """``(size, base)`` for every unallocated hole, largest first."""
    holes = [(size, base) for base, size, state in walk() if state == MEM_FREE and size >= min_size]
    holes.sort(reverse=True)
    return holes


def reserved_total() -> int:
    """Bytes reserved but not committed, i.e. address space held without memory."""
    return sum(size for _base, size, state in walk() if state == MEM_RESERVE)
