# Copyright 2021-2024 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.
import pytest
import ctypes

# Always skip since cupy is not CTK 12.x yet
skip_tests = True
if not skip_tests:
    try:
        import cupy
        skip_tests = False
    except ImportError:
        skip_tests = True

from .kernels import kernel_string

def launch(kernel, args=()):
    kernel((1,), (1,), args)

# Measure launch latency with no parmaeters
@pytest.mark.skipif(skip_tests, reason="cupy is not installed")
@pytest.mark.benchmark(group="cupy")
def test_launch_latency_empty_kernel(benchmark):
    module = cupy.RawModule(code=kernel_string)
    kernel = module.get_function('empty_kernel')

    stream = cupy.cuda.stream.Stream(non_blocking=True)

    with stream:
        benchmark(launch, kernel)
        stream.synchronize()

# Measure launch latency with a single parameter
@pytest.mark.skipif(skip_tests, reason="cupy is not installed")
@pytest.mark.benchmark(group="cupy")
def test_launch_latency_small_kernel(benchmark):
    module = cupy.RawModule(code=kernel_string)
    kernel = module.get_function('small_kernel')
    cupy.cuda.set_allocator()
    arg = cupy.cuda.alloc(ctypes.sizeof(ctypes.c_float))

    stream = cupy.cuda.stream.Stream(non_blocking=True)

    with stream:
        benchmark(launch, kernel, (arg,))
        stream.synchronize()

# Measure launch latency with many parameters using builtin parameter packing
@pytest.mark.skipif(skip_tests, reason="cupy is not installed")
@pytest.mark.benchmark(group="cupy")
def test_launch_latency_small_kernel_512_args(benchmark):
    module = cupy.RawModule(code=kernel_string)
    kernel = module.get_function('small_kernel_512_args')
    cupy.cuda.set_allocator()

    args = []
    for _ in range(512):
        args.append(cupy.cuda.alloc(ctypes.sizeof(ctypes.c_int)))
    args = tuple(args)

    stream = cupy.cuda.stream.Stream(non_blocking=True)

    with stream:
        benchmark(launch, kernel, args)
        stream.synchronize()

# Measure launch latency with many parameters using builtin parameter packing
@pytest.mark.skipif(skip_tests, reason="cupy is not installed")
@pytest.mark.benchmark(group="cupy")
def test_launch_latency_small_kernel_512_bools(benchmark):
    module = cupy.RawModule(code=kernel_string)
    kernel = module.get_function('small_kernel_512_bools')
    cupy.cuda.set_allocator()

    args = [True] * 512
    args = tuple(args)

    stream = cupy.cuda.stream.Stream(non_blocking=True)

    with stream:
        benchmark(launch, kernel, args)
        stream.synchronize()

# Measure launch latency with many parameters using builtin parameter packing
@pytest.mark.skipif(skip_tests, reason="cupy is not installed")
@pytest.mark.benchmark(group="cupy")
def test_launch_latency_small_kernel_512_doubles(benchmark):
    module = cupy.RawModule(code=kernel_string)
    kernel = module.get_function('small_kernel_512_doubles')
    cupy.cuda.set_allocator()

    args = [1.2345] * 512
    args = tuple(args)

    stream = cupy.cuda.stream.Stream(non_blocking=True)

    with stream:
        benchmark(launch, kernel, args)
        stream.synchronize()

# Measure launch latency with many parameters using builtin parameter packing
@pytest.mark.skipif(skip_tests, reason="cupy is not installed")
@pytest.mark.benchmark(group="cupy")
def test_launch_latency_small_kernel_512_ints(benchmark):
    module = cupy.RawModule(code=kernel_string)
    kernel = module.get_function('small_kernel_512_ints')
    cupy.cuda.set_allocator()

    args = [123] * 512
    args = tuple(args)

    stream = cupy.cuda.stream.Stream(non_blocking=True)

    with stream:
        benchmark(launch, kernel, args)
        stream.synchronize()

# Measure launch latency with many parameters using builtin parameter packing
@pytest.mark.skipif(skip_tests, reason="cupy is not installed")
@pytest.mark.benchmark(group="cupy")
def test_launch_latency_small_kernel_512_bytes(benchmark):
    module = cupy.RawModule(code=kernel_string)
    kernel = module.get_function('small_kernel_512_chars')
    cupy.cuda.set_allocator()

    args = [127] * 512
    args = tuple(args)

    stream = cupy.cuda.stream.Stream(non_blocking=True)

    with stream:
        benchmark(launch, kernel, args)
        stream.synchronize()

# Measure launch latency with many parameters using builtin parameter packing
@pytest.mark.skipif(skip_tests, reason="cupy is not installed")
@pytest.mark.benchmark(group="cupy")
def test_launch_latency_small_kernel_512_longlongs(benchmark):
    module = cupy.RawModule(code=kernel_string)
    kernel = module.get_function('small_kernel_512_longlongs')
    cupy.cuda.set_allocator()

    args = [9223372036854775806] * 512
    args = tuple(args)

    stream = cupy.cuda.stream.Stream(non_blocking=True)

    with stream:
        benchmark(launch, kernel, args)
        stream.synchronize()

# Measure launch latency with many parameters using builtin parameter packing
@pytest.mark.skipif(skip_tests, reason="cupy is not installed")
@pytest.mark.benchmark(group="cupy")
def test_launch_latency_small_kernel_256_args(benchmark):
    module = cupy.RawModule(code=kernel_string)
    kernel = module.get_function('small_kernel_256_args')
    cupy.cuda.set_allocator()

    args = []
    for _ in range(256):
        args.append(cupy.cuda.alloc(ctypes.sizeof(ctypes.c_int)))
    args = tuple(args)

    stream = cupy.cuda.stream.Stream(non_blocking=True)

    with stream:
        benchmark(launch, kernel, args)
        stream.synchronize()

# Measure launch latency with many parameters using builtin parameter packing
@pytest.mark.skipif(skip_tests, reason="cupy is not installed")
@pytest.mark.benchmark(group="cupy")
def test_launch_latency_small_kernel_16_args(benchmark):
    module = cupy.RawModule(code=kernel_string)
    kernel = module.get_function('small_kernel_16_args')
    cupy.cuda.set_allocator()

    args = []
    for _ in range(16):
        args.append(cupy.cuda.alloc(ctypes.sizeof(ctypes.c_int)))
    args = tuple(args)

    stream = cupy.cuda.stream.Stream(non_blocking=True)

    with stream:
        benchmark(launch, kernel, args)
        stream.synchronize()
