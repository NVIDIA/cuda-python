# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import ctypes
import time

import numpy as np

from runner.runtime import alloc_persistent, ensure_context

from cuda.bindings import driver as cuda

ensure_context()

# Allocation size for alloc/free benchmarks
ALLOC_SIZE = 1024

# Small transfer size (8 bytes) to measure call overhead, not bandwidth
COPY_SIZE = 8

# Pre-allocate device memory and host buffers for memcpy benchmarks
DST_DPTR = alloc_persistent(COPY_SIZE)
SRC_DPTR = alloc_persistent(COPY_SIZE)
HOST_SRC = np.zeros(COPY_SIZE, dtype=np.uint8)
HOST_DST = np.zeros(COPY_SIZE, dtype=np.uint8)

# Stream for async operations
_err, STREAM = cuda.cuStreamCreate(cuda.CUstream_flags.CU_STREAM_NON_BLOCKING.value)


def bench_mem_alloc_free(loops: int) -> float:
    _cuMemAlloc = cuda.cuMemAlloc
    _cuMemFree = cuda.cuMemFree
    _size = ALLOC_SIZE

    t0 = time.perf_counter()
    for _ in range(loops):
        _, ptr = _cuMemAlloc(_size)
        _cuMemFree(ptr)
    return time.perf_counter() - t0


def bench_mem_alloc_async_free_async(loops: int) -> float:
    _cuMemAllocAsync = cuda.cuMemAllocAsync
    _cuMemFreeAsync = cuda.cuMemFreeAsync
    _size = ALLOC_SIZE
    _stream = STREAM

    t0 = time.perf_counter()
    for _ in range(loops):
        _, ptr = _cuMemAllocAsync(_size, _stream)
        _cuMemFreeAsync(ptr, _stream)
    return time.perf_counter() - t0


def bench_memcpy_htod(loops: int) -> float:
    _cuMemcpyHtoD = cuda.cuMemcpyHtoD
    _dst = DST_DPTR
    _src = HOST_SRC
    _size = COPY_SIZE

    t0 = time.perf_counter()
    for _ in range(loops):
        _cuMemcpyHtoD(_dst, _src, _size)
    return time.perf_counter() - t0


def bench_memcpy_dtoh(loops: int) -> float:
    _cuMemcpyDtoH = cuda.cuMemcpyDtoH
    _dst = HOST_DST
    _src = SRC_DPTR
    _size = COPY_SIZE

    t0 = time.perf_counter()
    for _ in range(loops):
        _cuMemcpyDtoH(_dst, _src, _size)
    return time.perf_counter() - t0


def bench_memcpy_dtod(loops: int) -> float:
    _cuMemcpyDtoD = cuda.cuMemcpyDtoD
    _dst = DST_DPTR
    _src = SRC_DPTR
    _size = COPY_SIZE

    t0 = time.perf_counter()
    for _ in range(loops):
        _cuMemcpyDtoD(_dst, _src, _size)
    return time.perf_counter() - t0
