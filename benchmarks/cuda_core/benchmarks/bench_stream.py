# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import time

from runtime import ensure_device

DEV = ensure_device()
STREAM = DEV.create_stream()


def bench_stream_create_destroy(loops: int) -> float:
    _create = DEV.create_stream

    t0 = time.perf_counter()
    for _ in range(loops):
        s = _create()
        s.close()
    return time.perf_counter() - t0


def bench_stream_synchronize(loops: int) -> float:
    _fn = STREAM.sync

    t0 = time.perf_counter()
    for _ in range(loops):
        _fn()
    return time.perf_counter() - t0
