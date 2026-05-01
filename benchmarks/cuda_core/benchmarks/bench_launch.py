# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import time

from runtime import compile_module, ensure_device

from cuda.core import LaunchConfig, launch

# Same source as the cuda.bindings launch bench, minus the variants we
# don't need here. Pointer args are passed as Python ints (via int(handle))
# or Buffer instances; cuda.core's ParamHolder treats both as intptr_t.
KERNEL_SOURCE = """\
extern "C" __global__ void empty_kernel() { return; }
extern "C" __global__ void small_kernel(float *f) { *f = 0.0f; }

#define ITEM_PARAM(x, T) T x
#define REP1(x, T)   , ITEM_PARAM(x, T)
#define REP2(x, T)   REP1(x##0, T)   REP1(x##1, T)
#define REP4(x, T)   REP2(x##0, T)   REP2(x##1, T)
#define REP8(x, T)   REP4(x##0, T)   REP4(x##1, T)
#define REP16(x, T)  REP8(x##0, T)   REP8(x##1, T)
#define REP32(x, T)  REP16(x##0, T)  REP16(x##1, T)
#define REP64(x, T)  REP32(x##0, T)  REP32(x##1, T)
#define REP128(x, T) REP64(x##0, T)  REP64(x##1, T)
#define REP256(x, T) REP128(x##0, T) REP128(x##1, T)

extern "C" __global__
void small_kernel_16_args(
    ITEM_PARAM(F, int*)
    REP1(A, int*)
    REP2(A, int*)
    REP4(A, int*)
    REP8(A, int*))
{ *F = 0; }

extern "C" __global__
void small_kernel_256_args(
    ITEM_PARAM(F, int*)
    REP1(A, int*)
    REP2(A, int*)
    REP4(A, int*)
    REP8(A, int*)
    REP16(A, int*)
    REP32(A, int*)
    REP64(A, int*)
    REP128(A, int*))
{ *F = 0; }

extern "C" __global__
void small_kernel_512_args(
    ITEM_PARAM(F, int*)
    REP1(A, int*)
    REP2(A, int*)
    REP4(A, int*)
    REP8(A, int*)
    REP16(A, int*)
    REP32(A, int*)
    REP64(A, int*)
    REP128(A, int*)
    REP256(A, int*))
{ *F = 0; }
"""

KERNEL_NAMES = (
    "empty_kernel",
    "small_kernel",
    "small_kernel_16_args",
    "small_kernel_256_args",
    "small_kernel_512_args",
)

DEV = ensure_device()
STREAM = DEV.create_stream()
CONFIG = LaunchConfig(grid=1, block=1)

MODULE = None
EMPTY_KERNEL = None
SMALL_KERNEL = None
KERNEL_16_ARGS = None
KERNEL_256_ARGS = None
KERNEL_512_ARGS = None
FLOAT_BUF = None
INT_BUFS_512: tuple = ()
INT_PTRS_512: tuple = ()


def _ensure_launch_state() -> None:
    global MODULE, EMPTY_KERNEL, SMALL_KERNEL
    global KERNEL_16_ARGS, KERNEL_256_ARGS, KERNEL_512_ARGS
    global FLOAT_BUF, INT_BUFS_512, INT_PTRS_512

    if EMPTY_KERNEL is not None:
        return

    module = compile_module(KERNEL_SOURCE, KERNEL_NAMES)

    # Pre-allocate buffers for the kernel args. Use ints (raw pointer
    # addresses) in the launch hot path so ParamHolder skips the Buffer
    # type check and goes through its int fast-path.
    float_buf = DEV.allocate(4)
    int_bufs_512 = tuple(DEV.allocate(4) for _ in range(512))
    int_ptrs_512 = tuple(int(b.handle) for b in int_bufs_512)

    MODULE = module
    EMPTY_KERNEL = module.get_kernel("empty_kernel")
    SMALL_KERNEL = module.get_kernel("small_kernel")
    KERNEL_16_ARGS = module.get_kernel("small_kernel_16_args")
    KERNEL_256_ARGS = module.get_kernel("small_kernel_256_args")
    KERNEL_512_ARGS = module.get_kernel("small_kernel_512_args")
    FLOAT_BUF = float_buf
    INT_BUFS_512 = int_bufs_512
    INT_PTRS_512 = int_ptrs_512


def bench_launch_empty_kernel(loops: int) -> float:
    _ensure_launch_state()
    _launch = launch
    _kernel = EMPTY_KERNEL
    _stream = STREAM
    _config = CONFIG

    t0 = time.perf_counter()
    for _ in range(loops):
        _launch(_stream, _config, _kernel)
    return time.perf_counter() - t0


def bench_launch_small_kernel(loops: int) -> float:
    _ensure_launch_state()
    _launch = launch
    _kernel = SMALL_KERNEL
    _stream = STREAM
    _config = CONFIG
    _ptr = int(FLOAT_BUF.handle)

    t0 = time.perf_counter()
    for _ in range(loops):
        _launch(_stream, _config, _kernel, _ptr)
    return time.perf_counter() - t0


def bench_launch_16_args(loops: int) -> float:
    _ensure_launch_state()
    _launch = launch
    _kernel = KERNEL_16_ARGS
    _stream = STREAM
    _config = CONFIG
    _args = INT_PTRS_512[:16]

    t0 = time.perf_counter()
    for _ in range(loops):
        _launch(_stream, _config, _kernel, *_args)
    return time.perf_counter() - t0


def bench_launch_256_args(loops: int) -> float:
    _ensure_launch_state()
    _launch = launch
    _kernel = KERNEL_256_ARGS
    _stream = STREAM
    _config = CONFIG
    _args = INT_PTRS_512[:256]

    t0 = time.perf_counter()
    for _ in range(loops):
        _launch(_stream, _config, _kernel, *_args)
    return time.perf_counter() - t0


def bench_launch_512_args(loops: int) -> float:
    _ensure_launch_state()
    _launch = launch
    _kernel = KERNEL_512_ARGS
    _stream = STREAM
    _config = CONFIG
    _args = INT_PTRS_512

    t0 = time.perf_counter()
    for _ in range(loops):
        _launch(_stream, _config, _kernel, *_args)
    return time.perf_counter() - t0
