# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import time

from runner.runtime import assert_drv, ensure_context

from cuda.bindings import driver as cuda
from cuda.bindings import nvrtc

ensure_context()

KERNEL_SOURCE = b'extern "C" __global__ void empty_kernel() { return; }'
PROGRAM_NAME = b"benchmark_kernel.cu"

# Compute the arch flag once for compile benchmarks
_err, _device = cuda.cuDeviceGet(0)
assert_drv(_err)
_err, _major = cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, _device)
assert_drv(_err)
_err, _minor = cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, _device)
assert_drv(_err)
ARCH_FLAG = f"--gpu-architecture=sm_{_major}{_minor}".encode()
COMPILE_OPTIONS = [b"--fmad=false", ARCH_FLAG]

# Pre-build 100 empty headers for the headers benchmark
HEADER_NAMES = [f"header_{i}.cuh".encode() for i in range(100)]
HEADER_SOURCES = [b"// empty" for _ in range(100)]


def bench_nvrtc_create_program(loops: int) -> float:
    _create = nvrtc.nvrtcCreateProgram
    _destroy = nvrtc.nvrtcDestroyProgram
    _assert = assert_drv
    _src = KERNEL_SOURCE
    _name = PROGRAM_NAME

    t0 = time.perf_counter()
    for _ in range(loops):
        err, prog = _create(_src, _name, 0, [], [])
        _assert(err)
        (err,) = _destroy(prog)
        _assert(err)
    return time.perf_counter() - t0


def bench_nvrtc_create_program_100_headers(loops: int) -> float:
    _create = nvrtc.nvrtcCreateProgram
    _destroy = nvrtc.nvrtcDestroyProgram
    _assert = assert_drv
    _src = KERNEL_SOURCE
    _name = PROGRAM_NAME
    _headers = HEADER_SOURCES
    _header_names = HEADER_NAMES

    t0 = time.perf_counter()
    for _ in range(loops):
        err, prog = _create(_src, _name, 100, _headers, _header_names)
        _assert(err)
        (err,) = _destroy(prog)
        _assert(err)
    return time.perf_counter() - t0


def bench_nvrtc_compile_program(loops: int) -> float:
    _create = nvrtc.nvrtcCreateProgram
    _compile = nvrtc.nvrtcCompileProgram
    _destroy = nvrtc.nvrtcDestroyProgram
    _assert = assert_drv
    _src = KERNEL_SOURCE
    _name = PROGRAM_NAME
    _options = COMPILE_OPTIONS

    t0 = time.perf_counter()
    for _ in range(loops):
        err, prog = _create(_src, _name, 0, [], [])
        _assert(err)
        (err,) = _compile(prog, 2, _options)
        _assert(err)
        (err,) = _destroy(prog)
        _assert(err)
    return time.perf_counter() - t0
