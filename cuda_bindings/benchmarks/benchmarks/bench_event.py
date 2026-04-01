# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import time

from runner.runtime import ensure_context

from cuda.bindings import driver as cuda

ensure_context()

_err, STREAM = cuda.cuStreamCreate(cuda.CUstream_flags.CU_STREAM_NON_BLOCKING.value)
_err, EVENT = cuda.cuEventCreate(cuda.CUevent_flags.CU_EVENT_DISABLE_TIMING.value)

cuda.cuEventRecord(EVENT, STREAM)
cuda.cuStreamSynchronize(STREAM)

EVENT_FLAGS = cuda.CUevent_flags.CU_EVENT_DISABLE_TIMING.value


def bench_event_create_destroy(loops: int) -> float:
    _cuEventCreate = cuda.cuEventCreate
    _cuEventDestroy = cuda.cuEventDestroy
    _flags = EVENT_FLAGS

    t0 = time.perf_counter()
    for _ in range(loops):
        _, e = _cuEventCreate(_flags)
        _cuEventDestroy(e)
    return time.perf_counter() - t0


def bench_event_record(loops: int) -> float:
    _cuEventRecord = cuda.cuEventRecord
    _event = EVENT
    _stream = STREAM

    t0 = time.perf_counter()
    for _ in range(loops):
        _cuEventRecord(_event, _stream)
    return time.perf_counter() - t0


def bench_event_query(loops: int) -> float:
    _cuEventQuery = cuda.cuEventQuery
    _event = EVENT

    t0 = time.perf_counter()
    for _ in range(loops):
        _cuEventQuery(_event)
    return time.perf_counter() - t0


def bench_event_synchronize(loops: int) -> float:
    _cuEventSynchronize = cuda.cuEventSynchronize
    _event = EVENT

    t0 = time.perf_counter()
    for _ in range(loops):
        _cuEventSynchronize(_event)
    return time.perf_counter() - t0
