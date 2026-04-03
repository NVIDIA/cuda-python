# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import ctypes
import time

from runner.runtime import alloc_persistent, compile_and_load, ensure_context

from cuda.bindings import driver as cuda

ensure_context()

# Compile kernels
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

MODULE = compile_and_load(KERNEL_SOURCE)

# Get kernel handles
_err, EMPTY_KERNEL = cuda.cuModuleGetFunction(MODULE, b"empty_kernel")
_err, SMALL_KERNEL = cuda.cuModuleGetFunction(MODULE, b"small_kernel")
_err, KERNEL_16_ARGS = cuda.cuModuleGetFunction(MODULE, b"small_kernel_16_args")

# Create a non-blocking stream for launches
_err, STREAM = cuda.cuStreamCreate(cuda.CUstream_flags.CU_STREAM_NON_BLOCKING.value)

# Allocate device memory for kernel arguments
FLOAT_PTR = alloc_persistent(ctypes.sizeof(ctypes.c_float))
INT_PTRS = [alloc_persistent(ctypes.sizeof(ctypes.c_int)) for _ in range(16)]

# Pre-pack ctypes params for the pre-packed benchmark
_val_ps = [ctypes.c_void_p(int(p)) for p in INT_PTRS]
PACKED_16 = (ctypes.c_void_p * 16)()
for _i in range(16):
    PACKED_16[_i] = ctypes.addressof(_val_ps[_i])


def bench_launch_empty_kernel(loops: int) -> float:
    _cuLaunchKernel = cuda.cuLaunchKernel
    _kernel = EMPTY_KERNEL
    _stream = STREAM

    t0 = time.perf_counter()
    for _ in range(loops):
        _cuLaunchKernel(_kernel, 1, 1, 1, 1, 1, 1, 0, _stream, 0, 0)
    return time.perf_counter() - t0


def bench_launch_small_kernel(loops: int) -> float:
    _cuLaunchKernel = cuda.cuLaunchKernel
    _kernel = SMALL_KERNEL
    _stream = STREAM
    _args = (FLOAT_PTR,)
    _arg_types = (None,)

    t0 = time.perf_counter()
    for _ in range(loops):
        _cuLaunchKernel(_kernel, 1, 1, 1, 1, 1, 1, 0, _stream, (_args, _arg_types), 0)
    return time.perf_counter() - t0


def bench_launch_16_args(loops: int) -> float:
    _cuLaunchKernel = cuda.cuLaunchKernel
    _kernel = KERNEL_16_ARGS
    _stream = STREAM
    _args = tuple(INT_PTRS)
    _arg_types = tuple([None] * 16)

    t0 = time.perf_counter()
    for _ in range(loops):
        _cuLaunchKernel(_kernel, 1, 1, 1, 1, 1, 1, 0, _stream, (_args, _arg_types), 0)
    return time.perf_counter() - t0


def bench_launch_16_args_pre_packed(loops: int) -> float:
    _cuLaunchKernel = cuda.cuLaunchKernel
    _kernel = KERNEL_16_ARGS
    _stream = STREAM
    _packed = PACKED_16

    t0 = time.perf_counter()
    for _ in range(loops):
        _cuLaunchKernel(_kernel, 1, 1, 1, 1, 1, 1, 0, _stream, _packed, 0)
    return time.perf_counter() - t0
