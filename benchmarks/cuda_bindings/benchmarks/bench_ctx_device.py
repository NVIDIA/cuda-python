# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import time

from runner.runtime import ensure_context

from cuda.bindings import driver as cuda

CTX = ensure_context()

_, DEVICE = cuda.cuDeviceGet(0)
ATTRIBUTE = cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR


def bench_ctx_get_current(loops: int) -> float:
    _fn = cuda.cuCtxGetCurrent

    t0 = time.perf_counter()
    for _ in range(loops):
        _fn()
    return time.perf_counter() - t0


def bench_ctx_set_current(loops: int) -> float:
    _fn = cuda.cuCtxSetCurrent
    _ctx = CTX

    t0 = time.perf_counter()
    for _ in range(loops):
        _fn(_ctx)
    return time.perf_counter() - t0


def bench_ctx_get_device(loops: int) -> float:
    _fn = cuda.cuCtxGetDevice

    t0 = time.perf_counter()
    for _ in range(loops):
        _fn()
    return time.perf_counter() - t0


def bench_device_get(loops: int) -> float:
    _fn = cuda.cuDeviceGet

    t0 = time.perf_counter()
    for _ in range(loops):
        _fn(0)
    return time.perf_counter() - t0


def bench_device_get_attribute(loops: int) -> float:
    _fn = cuda.cuDeviceGetAttribute
    _attr = ATTRIBUTE
    _dev = DEVICE

    t0 = time.perf_counter()
    for _ in range(loops):
        _fn(_attr, _dev)
    return time.perf_counter() - t0
