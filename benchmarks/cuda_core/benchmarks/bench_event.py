# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import time

from runtime import ensure_device

DEV = ensure_device()
STREAM = DEV.create_stream()

# Pre-recorded event, completed, reused for record/query/sync benches.
EVENT = STREAM.record()
STREAM.sync()


def bench_event_create_destroy(loops: int) -> float:
    _create = DEV.create_event

    t0 = time.perf_counter()
    for _ in range(loops):
        e = _create()
        e.close()
    return time.perf_counter() - t0


def bench_event_record(loops: int) -> float:
    # Reuse EVENT so we measure cuEventRecord, not event allocation.
    _record = STREAM.record
    _event = EVENT

    t0 = time.perf_counter()
    for _ in range(loops):
        _record(_event)
    return time.perf_counter() - t0


def bench_event_query(loops: int) -> float:
    _event = EVENT

    t0 = time.perf_counter()
    for _ in range(loops):
        _event.is_done  # noqa: B018
    return time.perf_counter() - t0


def bench_event_synchronize(loops: int) -> float:
    _fn = EVENT.sync

    t0 = time.perf_counter()
    for _ in range(loops):
        _fn()
    return time.perf_counter() - t0
