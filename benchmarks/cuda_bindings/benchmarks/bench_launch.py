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
#define REP32(x, T)  REP16(x##0, T)  REP16(x##1, T)
#define REP64(x, T)  REP32(x##0, T)  REP32(x##1, T)
#define REP128(x, T) REP64(x##0, T)  REP64(x##1, T)
#define REP256(x, T) REP128(x##0, T) REP128(x##1, T)

template<size_t maxBytes>
struct KernelFunctionParam {
   unsigned char p[maxBytes];
};

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

extern "C" __global__
void small_kernel_512_bools(
    ITEM_PARAM(F, bool)
    REP1(A, bool)
    REP2(A, bool)
    REP4(A, bool)
    REP8(A, bool)
    REP16(A, bool)
    REP32(A, bool)
    REP64(A, bool)
    REP128(A, bool)
    REP256(A, bool))
{ return; }

extern "C" __global__
void small_kernel_512_ints(
    ITEM_PARAM(F, int)
    REP1(A, int)
    REP2(A, int)
    REP4(A, int)
    REP8(A, int)
    REP16(A, int)
    REP32(A, int)
    REP64(A, int)
    REP128(A, int)
    REP256(A, int))
{ return; }

extern "C" __global__
void small_kernel_512_doubles(
    ITEM_PARAM(F, double)
    REP1(A, double)
    REP2(A, double)
    REP4(A, double)
    REP8(A, double)
    REP16(A, double)
    REP32(A, double)
    REP64(A, double)
    REP128(A, double)
    REP256(A, double))
{ return; }

extern "C" __global__
void small_kernel_512_chars(
    ITEM_PARAM(F, char)
    REP1(A, char)
    REP2(A, char)
    REP4(A, char)
    REP8(A, char)
    REP16(A, char)
    REP32(A, char)
    REP64(A, char)
    REP128(A, char)
    REP256(A, char))
{ return; }

extern "C" __global__
void small_kernel_512_longlongs(
    ITEM_PARAM(F, long long)
    REP1(A, long long)
    REP2(A, long long)
    REP4(A, long long)
    REP8(A, long long)
    REP16(A, long long)
    REP32(A, long long)
    REP64(A, long long)
    REP128(A, long long)
    REP256(A, long long))
{ return; }

extern "C" __global__
void small_kernel_2048B(KernelFunctionParam<2048> param) {
    // Do not touch param to prevent compiler from copying
    // the whole structure from const bank to lmem.
}
"""

MODULE = None
EMPTY_KERNEL = None
SMALL_KERNEL = None
KERNEL_16_ARGS = None
KERNEL_256_ARGS = None
KERNEL_512_ARGS = None
KERNEL_512_BOOLS = None
KERNEL_512_INTS = None
KERNEL_512_DOUBLES = None
KERNEL_512_CHARS = None
KERNEL_512_LONGLONGS = None
KERNEL_2048B = None
STREAM = None
FLOAT_PTR = None
INT_PTRS_512 = None
_VAL_PS_16 = None
_VAL_PS_512 = None
PACKED_16 = None
PACKED_512 = None


class _Struct2048B(ctypes.Structure):
    _fields_ = [("values", ctypes.c_uint8 * 2048)]


STRUCT_2048B = _Struct2048B()


def _ensure_launch_state() -> None:
    global MODULE, EMPTY_KERNEL, SMALL_KERNEL
    global KERNEL_16_ARGS, KERNEL_256_ARGS, KERNEL_512_ARGS
    global KERNEL_512_BOOLS, KERNEL_512_INTS, KERNEL_512_DOUBLES
    global KERNEL_512_CHARS, KERNEL_512_LONGLONGS, KERNEL_2048B
    global STREAM, FLOAT_PTR, INT_PTRS_512
    global _VAL_PS_16, _VAL_PS_512, PACKED_16, PACKED_512

    if EMPTY_KERNEL is not None:
        return

    module = compile_and_load(KERNEL_SOURCE)

    def get_func(name):
        err, func = cuda.cuModuleGetFunction(module, name.encode())
        assert_drv(err)
        return func

    err, stream = cuda.cuStreamCreate(cuda.CUstream_flags.CU_STREAM_NON_BLOCKING.value)
    assert_drv(err)

    float_ptr = alloc_persistent(ctypes.sizeof(ctypes.c_float))
    int_ptrs_512 = tuple(alloc_persistent(ctypes.sizeof(ctypes.c_int)) for _ in range(512))

    # Pre-pack 16 args
    val_ps_16 = [ctypes.c_void_p(int(ptr)) for ptr in int_ptrs_512[:16]]
    packed_16 = (ctypes.c_void_p * 16)()
    for i, vp in enumerate(val_ps_16):
        packed_16[i] = ctypes.addressof(vp)

    # Pre-pack 512 args
    val_ps_512 = [ctypes.c_void_p(int(ptr)) for ptr in int_ptrs_512]
    packed_512 = (ctypes.c_void_p * 512)()
    for i, vp in enumerate(val_ps_512):
        packed_512[i] = ctypes.addressof(vp)

    MODULE = module
    EMPTY_KERNEL = get_func("empty_kernel")
    SMALL_KERNEL = get_func("small_kernel")
    KERNEL_16_ARGS = get_func("small_kernel_16_args")
    KERNEL_256_ARGS = get_func("small_kernel_256_args")
    KERNEL_512_ARGS = get_func("small_kernel_512_args")
    KERNEL_512_BOOLS = get_func("small_kernel_512_bools")
    KERNEL_512_INTS = get_func("small_kernel_512_ints")
    KERNEL_512_DOUBLES = get_func("small_kernel_512_doubles")
    KERNEL_512_CHARS = get_func("small_kernel_512_chars")
    KERNEL_512_LONGLONGS = get_func("small_kernel_512_longlongs")
    KERNEL_2048B = get_func("small_kernel_2048B")
    STREAM = stream
    FLOAT_PTR = float_ptr
    INT_PTRS_512 = int_ptrs_512
    _VAL_PS_16 = val_ps_16
    _VAL_PS_512 = val_ps_512
    PACKED_16 = packed_16
    PACKED_512 = packed_512


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
    _args = INT_PTRS_512[:16]
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


def bench_launch_256_args(loops: int) -> float:
    _ensure_launch_state()
    _fn = cuda.cuLaunchKernel
    _kernel = KERNEL_256_ARGS
    _stream = STREAM
    _args = INT_PTRS_512[:256]
    _arg_types = (None,) * 256

    t0 = time.perf_counter()
    for _ in range(loops):
        _fn(_kernel, 1, 1, 1, 1, 1, 1, 0, _stream, (_args, _arg_types), 0)
    return time.perf_counter() - t0


def bench_launch_512_args(loops: int) -> float:
    _ensure_launch_state()
    _fn = cuda.cuLaunchKernel
    _kernel = KERNEL_512_ARGS
    _stream = STREAM
    _args = INT_PTRS_512
    _arg_types = (None,) * 512

    t0 = time.perf_counter()
    for _ in range(loops):
        _fn(_kernel, 1, 1, 1, 1, 1, 1, 0, _stream, (_args, _arg_types), 0)
    return time.perf_counter() - t0


def bench_launch_512_args_pre_packed(loops: int) -> float:
    _ensure_launch_state()
    _fn = cuda.cuLaunchKernel
    _kernel = KERNEL_512_ARGS
    _stream = STREAM
    _packed = PACKED_512

    t0 = time.perf_counter()
    for _ in range(loops):
        _fn(_kernel, 1, 1, 1, 1, 1, 1, 0, _stream, _packed, 0)
    return time.perf_counter() - t0


def bench_launch_512_bools(loops: int) -> float:
    _ensure_launch_state()
    _fn = cuda.cuLaunchKernel
    _kernel = KERNEL_512_BOOLS
    _stream = STREAM
    _args = (True,) * 512
    _arg_types = (ctypes.c_bool,) * 512

    t0 = time.perf_counter()
    for _ in range(loops):
        _fn(_kernel, 1, 1, 1, 1, 1, 1, 0, _stream, (_args, _arg_types), 0)
    return time.perf_counter() - t0


def bench_launch_512_ints(loops: int) -> float:
    _ensure_launch_state()
    _fn = cuda.cuLaunchKernel
    _kernel = KERNEL_512_INTS
    _stream = STREAM
    _args = (123,) * 512
    _arg_types = (ctypes.c_int,) * 512

    t0 = time.perf_counter()
    for _ in range(loops):
        _fn(_kernel, 1, 1, 1, 1, 1, 1, 0, _stream, (_args, _arg_types), 0)
    return time.perf_counter() - t0


def bench_launch_512_doubles(loops: int) -> float:
    _ensure_launch_state()
    _fn = cuda.cuLaunchKernel
    _kernel = KERNEL_512_DOUBLES
    _stream = STREAM
    _args = (1.2345,) * 512
    _arg_types = (ctypes.c_double,) * 512

    t0 = time.perf_counter()
    for _ in range(loops):
        _fn(_kernel, 1, 1, 1, 1, 1, 1, 0, _stream, (_args, _arg_types), 0)
    return time.perf_counter() - t0


def bench_launch_512_bytes(loops: int) -> float:
    _ensure_launch_state()
    _fn = cuda.cuLaunchKernel
    _kernel = KERNEL_512_CHARS
    _stream = STREAM
    _args = (127,) * 512
    _arg_types = (ctypes.c_byte,) * 512

    t0 = time.perf_counter()
    for _ in range(loops):
        _fn(_kernel, 1, 1, 1, 1, 1, 1, 0, _stream, (_args, _arg_types), 0)
    return time.perf_counter() - t0


def bench_launch_512_longlongs(loops: int) -> float:
    _ensure_launch_state()
    _fn = cuda.cuLaunchKernel
    _kernel = KERNEL_512_LONGLONGS
    _stream = STREAM
    _args = (9223372036854775806,) * 512
    _arg_types = (ctypes.c_longlong,) * 512

    t0 = time.perf_counter()
    for _ in range(loops):
        _fn(_kernel, 1, 1, 1, 1, 1, 1, 0, _stream, (_args, _arg_types), 0)
    return time.perf_counter() - t0


def bench_launch_2048b(loops: int) -> float:
    _ensure_launch_state()
    _fn = cuda.cuLaunchKernel
    _kernel = KERNEL_2048B
    _stream = STREAM
    _args = (STRUCT_2048B,)
    _arg_types = (None,)

    t0 = time.perf_counter()
    for _ in range(loops):
        _fn(_kernel, 1, 1, 1, 1, 1, 1, 0, _stream, (_args, _arg_types), 0)
    return time.perf_counter() - t0
