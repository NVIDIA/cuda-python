# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import atexit

from cuda.bindings import driver as cuda

_ctx = None
_persistent_ptrs: list[int] = []


def assert_drv(err) -> None:
    if err != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"Cuda Error: {err}")


def ensure_context() -> int:
    global _ctx
    if _ctx is not None:
        return _ctx

    (err,) = cuda.cuInit(0)
    assert_drv(err)

    err, device = cuda.cuDeviceGet(0)
    assert_drv(err)

    err, ctx = cuda.cuCtxCreate(None, 0, device)
    assert_drv(err)
    _ctx = ctx
    return ctx


def alloc_persistent(size: int) -> int:
    ensure_context()
    err, ptr = cuda.cuMemAlloc(size)
    assert_drv(err)
    _persistent_ptrs.append(ptr)
    return ptr


def cleanup() -> None:
    global _ctx
    for ptr in reversed(_persistent_ptrs):
        (err,) = cuda.cuMemFree(ptr)
        assert_drv(err)
    _persistent_ptrs.clear()

    if _ctx is None:
        return
    (err,) = cuda.cuCtxDestroy(_ctx)
    assert_drv(err)
    _ctx = None


atexit.register(cleanup)
