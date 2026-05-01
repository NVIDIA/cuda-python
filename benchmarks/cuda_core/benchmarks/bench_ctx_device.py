# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import time

from runtime import ensure_device

from cuda.core import Device

DEV = ensure_device()


def bench_ctx_get_current(loops: int) -> float:
    # Device() with no args returns the TLS-cached "current" device.
    _fn = Device

    t0 = time.perf_counter()
    for _ in range(loops):
        _fn()
    return time.perf_counter() - t0


def bench_ctx_set_current(loops: int) -> float:
    _fn = DEV.set_current

    t0 = time.perf_counter()
    for _ in range(loops):
        _fn()
    return time.perf_counter() - t0


def bench_device_get(loops: int) -> float:
    # Device(id) hits the same TLS cache after the first construction.
    _fn = Device

    t0 = time.perf_counter()
    for _ in range(loops):
        _fn(0)
    return time.perf_counter() - t0


def bench_device_get_attribute(loops: int) -> float:
    # Matches the cuda.bindings bench's CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
    # call. cuda.core caches this attribute in DeviceProperties, so every
    # iteration past the first is a dict lookup rather than a driver call
    # — the bench therefore measures the user-visible cost of the public
    # API, which is legitimately faster than cuda.bindings here. See
    # BENCHMARK_PLAN.md for the rationale.
    _props = DEV.properties

    t0 = time.perf_counter()
    for _ in range(loops):
        _props.compute_capability_major  # noqa: B018
    return time.perf_counter() - t0
