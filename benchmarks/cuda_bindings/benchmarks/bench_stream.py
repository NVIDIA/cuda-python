# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import time

from runner.runtime import ensure_context

from cuda.bindings import driver as cuda

ensure_context()

_err, STREAM = cuda.cuStreamCreate(cuda.CUstream_flags.CU_STREAM_NON_BLOCKING.value)


def bench_stream_create_destroy(loops: int) -> float:
    _create = cuda.cuStreamCreate
    _destroy = cuda.cuStreamDestroy
    _flags = cuda.CUstream_flags.CU_STREAM_NON_BLOCKING.value

    t0 = time.perf_counter()
    for _ in range(loops):
        _, s = _create(_flags)
        _destroy(s)
    return time.perf_counter() - t0


def bench_stream_query(loops: int) -> float:
    _fn = cuda.cuStreamQuery
    _stream = STREAM

    t0 = time.perf_counter()
    for _ in range(loops):
        _fn(_stream)
    return time.perf_counter() - t0


def bench_stream_synchronize(loops: int) -> float:
    _fn = cuda.cuStreamSynchronize
    _stream = STREAM

    t0 = time.perf_counter()
    for _ in range(loops):
        _fn(_stream)
    return time.perf_counter() - t0
