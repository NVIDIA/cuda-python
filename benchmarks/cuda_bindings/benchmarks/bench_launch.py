# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import ctypes
import time

from runner.runtime import alloc_persistent, assert_drv, compile_and_load

from cuda.bindings import driver as cuda

# Compile kernels lazily so benchmark discovery does not need NVRTC.
KERNEL_SOURCE = """\
extern "C" __global__ void empty_kernel() { return; }
extern "C" __global__ void small_kernel(float *f) { *f = 0.0f; }

#define ITEM_PARAM(x, T) T x
#define REP1(x, T)   , ITEM_PARAM(x, T)
#define REP2(x, T)   REP1(x##0, T)   REP1(x##1, T)
#define REP4(x, T)   REP2(x##0, T)   REP2(x##1, T)
#define REP8(x, T)   REP4(x##0, T)   REP4(x##1, T)
#define REP16(x, T)  REP8(x##0, T)   REP8(x##1, T)

extern "C" __global__
void small_kernel_16_args(
    ITEM_PARAM(F, int*)
    REP1(A, int*)
    REP2(A, int*)
    REP4(A, int*)
    REP8(A, int*))
{ *F = 0; }
"""

MODULE = None
EMPTY_KERNEL = None
SMALL_KERNEL = None
KERNEL_16_ARGS = None
STREAM = None
FLOAT_PTR = None
INT_PTRS = None
_VAL_PS = None
PACKED_16 = None


def _ensure_launch_state() -> None:
    global MODULE, EMPTY_KERNEL, SMALL_KERNEL, KERNEL_16_ARGS, STREAM
    global FLOAT_PTR, INT_PTRS, _VAL_PS, PACKED_16

    if EMPTY_KERNEL is not None:
        return

    module = compile_and_load(KERNEL_SOURCE)

    err, empty_kernel = cuda.cuModuleGetFunction(module, b"empty_kernel")
    assert_drv(err)
    err, small_kernel = cuda.cuModuleGetFunction(module, b"small_kernel")
    assert_drv(err)
    err, kernel_16_args = cuda.cuModuleGetFunction(module, b"small_kernel_16_args")
    assert_drv(err)

    err, stream = cuda.cuStreamCreate(cuda.CUstream_flags.CU_STREAM_NON_BLOCKING.value)
    assert_drv(err)

    float_ptr = alloc_persistent(ctypes.sizeof(ctypes.c_float))
    int_ptrs = tuple(alloc_persistent(ctypes.sizeof(ctypes.c_int)) for _ in range(16))

    val_ps = [ctypes.c_void_p(int(ptr)) for ptr in int_ptrs]
    packed_16 = (ctypes.c_void_p * 16)()
    for index, value_ptr in enumerate(val_ps):
        packed_16[index] = ctypes.addressof(value_ptr)

    MODULE = module
    EMPTY_KERNEL = empty_kernel
    SMALL_KERNEL = small_kernel
    KERNEL_16_ARGS = kernel_16_args
    STREAM = stream
    FLOAT_PTR = float_ptr
    INT_PTRS = int_ptrs
    _VAL_PS = val_ps
    PACKED_16 = packed_16


def bench_launch_empty_kernel(loops: int) -> float:
    _ensure_launch_state()
    _fn = cuda.cuLaunchKernel
    _kernel = EMPTY_KERNEL
    _stream = STREAM

    t0 = time.perf_counter()
    for _ in range(loops):
        _fn(_kernel, 1, 1, 1, 1, 1, 1, 0, _stream, 0, 0)
    return time.perf_counter() - t0


def bench_launch_small_kernel(loops: int) -> float:
    _ensure_launch_state()
    _fn = cuda.cuLaunchKernel
    _kernel = SMALL_KERNEL
    _stream = STREAM
    _args = (FLOAT_PTR,)
    _arg_types = (None,)

    t0 = time.perf_counter()
    for _ in range(loops):
        _fn(_kernel, 1, 1, 1, 1, 1, 1, 0, _stream, (_args, _arg_types), 0)
    return time.perf_counter() - t0


def bench_launch_16_args(loops: int) -> float:
    _ensure_launch_state()
    _fn = cuda.cuLaunchKernel
    _kernel = KERNEL_16_ARGS
    _stream = STREAM
    _args = INT_PTRS
    _arg_types = (None,) * 16

    t0 = time.perf_counter()
    for _ in range(loops):
        _fn(_kernel, 1, 1, 1, 1, 1, 1, 0, _stream, (_args, _arg_types), 0)
    return time.perf_counter() - t0


def bench_launch_16_args_pre_packed(loops: int) -> float:
    _ensure_launch_state()
    _fn = cuda.cuLaunchKernel
    _kernel = KERNEL_16_ARGS
    _stream = STREAM
    _packed = PACKED_16

    t0 = time.perf_counter()
    for _ in range(loops):
        _fn(_kernel, 1, 1, 1, 1, 1, 1, 0, _stream, _packed, 0)
    return time.perf_counter() - t0
