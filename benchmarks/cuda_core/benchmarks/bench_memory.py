# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import time

from runtime import ensure_device

DEV = ensure_device()
STREAM = DEV.create_stream()

# Small allocation size: we measure call overhead, not the allocator.
ALLOC_SIZE = 1024


def bench_mem_alloc_free(loops: int) -> float:
    # No-stream variant: dev.allocate() uses the default stream internally.
    _alloc = DEV.allocate
    _size = ALLOC_SIZE

    t0 = time.perf_counter()
    for _ in range(loops):
        buf = _alloc(_size)
        buf.close()
    return time.perf_counter() - t0


def bench_mem_alloc_async_free_async(loops: int) -> float:
    _alloc = DEV.allocate
    _size = ALLOC_SIZE
    _stream = STREAM

    t0 = time.perf_counter()
    for _ in range(loops):
        buf = _alloc(_size, _stream)
        buf.close(_stream)
    return time.perf_counter() - t0
