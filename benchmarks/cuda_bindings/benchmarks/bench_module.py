# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import time

from runner.runtime import assert_drv, compile_cubin, ensure_context, register_module

from cuda.bindings import driver as cuda

ensure_context()

# Compile a trivial kernel to cubin once; reuse for all benchmarks
KERNEL_SOURCE = 'extern "C" __global__ void empty_kernel() { return; }'
CUBIN = compile_cubin(KERNEL_SOURCE)

# Load a persistent module + function for the get_function / get_attribute benchmarks
_err, MODULE = cuda.cuModuleLoadData(CUBIN)
assert_drv(_err)
register_module(MODULE)
_err, FUNCTION = cuda.cuModuleGetFunction(MODULE, b"empty_kernel")
assert_drv(_err)

FUNC_ATTRIBUTE = cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK


def bench_module_load_unload(loops: int) -> float:
    _load = cuda.cuModuleLoadData
    _unload = cuda.cuModuleUnload
    _cubin = CUBIN

    t0 = time.perf_counter()
    for _ in range(loops):
        _, m = _load(_cubin)
        _unload(m)
    return time.perf_counter() - t0


def bench_module_get_function(loops: int) -> float:
    _fn = cuda.cuModuleGetFunction
    _module = MODULE

    t0 = time.perf_counter()
    for _ in range(loops):
        _fn(_module, b"empty_kernel")
    return time.perf_counter() - t0


def bench_func_get_attribute(loops: int) -> float:
    _fn = cuda.cuFuncGetAttribute
    _attr = FUNC_ATTRIBUTE
    _func = FUNCTION

    t0 = time.perf_counter()
    for _ in range(loops):
        _fn(_attr, _func)
    return time.perf_counter() - t0
