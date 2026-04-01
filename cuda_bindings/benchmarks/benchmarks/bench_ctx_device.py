# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import time

from runner.runtime import ensure_context

from cuda.bindings import driver as cuda

ensure_context()


def bench_ctx_get_current(loops: int) -> float:
    _cuCtxGetCurrent = cuda.cuCtxGetCurrent

    t0 = time.perf_counter()
    for _ in range(loops):
        _cuCtxGetCurrent()
    return time.perf_counter() - t0
