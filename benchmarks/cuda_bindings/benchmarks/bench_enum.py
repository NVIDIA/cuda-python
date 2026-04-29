# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import time

from cuda.bindings import driver as cuda


def bench_curesult_construction(loops: int) -> float:
    _cls = cuda.CUresult

    t0 = time.perf_counter()
    for _ in range(loops):
        _cls(0)
    return time.perf_counter() - t0


def bench_curesult_member_access(loops: int) -> float:
    _cls = cuda.CUresult

    t0 = time.perf_counter()
    for _ in range(loops):
        _cls.CUDA_SUCCESS
    return time.perf_counter() - t0


def bench_device_attribute_construction(loops: int) -> float:
    _cls = cuda.CUdevice_attribute

    t0 = time.perf_counter()
    for _ in range(loops):
        _cls(1)
    return time.perf_counter() - t0
