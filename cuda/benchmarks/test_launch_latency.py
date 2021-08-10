# Copyright 2021 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.
import pytest
from cuda import cuda
import ctypes

from .perf_test_utils import ASSERT_DRV, init_cuda, load_module
from .kernels import kernel_string

def launch(kernel, stream, args=(), arg_types=()):
    cuda.cuLaunchKernel(kernel,
                        1, 1, 1,   # grid dim
                        1, 1, 1,   # block dim
                        0, stream, # shared mem and stream
                        (args, arg_types), 0) # arguments

def launch_packed(kernel, stream, params):
    cuda.cuLaunchKernel(kernel,
                        1, 1, 1,   # grid dim
                        1, 1, 1,   # block dim
                        0, stream, # shared mem and stream
                        params, 0) # arguments

# Measure launch latency with no parmaeters
@pytest.mark.benchmark(group="launch-latency")
def test_launch_latency_empty_kernel(benchmark, init_cuda, load_module):
    device, ctx, stream = init_cuda
    module = load_module(kernel_string, device)

    err, func = cuda.cuModuleGetFunction(module, b'empty_kernel')
    ASSERT_DRV(err)

    benchmark(launch, func, stream)

    cuda.cuCtxSynchronize()

# Measure launch latency with a single parameter
@pytest.mark.benchmark(group="launch-latency")
def test_launch_latency_small_kernel(benchmark, init_cuda, load_module):
    device, ctx, stream = init_cuda
    module = load_module(kernel_string, device)

    err, func = cuda.cuModuleGetFunction(module, b'small_kernel')
    ASSERT_DRV(err)

    err, f = cuda.cuMemAlloc(ctypes.sizeof(ctypes.c_float))
    ASSERT_DRV(err)

    benchmark(launch, func, stream, args=(f,), arg_types=(None,))

    cuda.cuCtxSynchronize()

    err, = cuda.cuMemFree(f)
    ASSERT_DRV(err)

# Measure launch latency with many parameters using builtin parameter packing
@pytest.mark.benchmark(group="launch-latency")
def test_launch_latency_small_kernel_512_args(benchmark, init_cuda, load_module):
    device, ctx, stream = init_cuda
    module = load_module(kernel_string, device)

    err, func = cuda.cuModuleGetFunction(module, b'small_kernel_512_args')
    ASSERT_DRV(err)

    args = []
    arg_types = [None] * 512
    for _ in arg_types:
        err, p = cuda.cuMemAlloc(ctypes.sizeof(ctypes.c_int))
        ASSERT_DRV(err)
        args.append(p)

    args = tuple(args)
    arg_types = tuple(arg_types)

    benchmark(launch, func, stream, args=args, arg_types=arg_types)

    cuda.cuCtxSynchronize()

    for p in args:
        err, = cuda.cuMemFree(p)
        ASSERT_DRV(err)

@pytest.mark.benchmark(group="launch-latency")
def test_launch_latency_small_kernel_512_bools(benchmark, init_cuda, load_module):
    device, ctx, stream = init_cuda
    module = load_module(kernel_string, device)

    err, func = cuda.cuModuleGetFunction(module, b'small_kernel_512_bools')
    ASSERT_DRV(err)

    args = [True] * 512
    arg_types = [ctypes.c_bool] * 512

    args = tuple(args)
    arg_types = tuple(arg_types)

    benchmark(launch, func, stream, args=args, arg_types=arg_types)

    cuda.cuCtxSynchronize()

@pytest.mark.benchmark(group="launch-latency")
def test_launch_latency_small_kernel_512_doubles(benchmark, init_cuda, load_module):
    device, ctx, stream = init_cuda
    module = load_module(kernel_string, device)

    err, func = cuda.cuModuleGetFunction(module, b'small_kernel_512_doubles')
    ASSERT_DRV(err)

    args = [1.2345] * 512
    arg_types = [ctypes.c_double] * 512

    args = tuple(args)
    arg_types = tuple(arg_types)

    benchmark(launch, func, stream, args=args, arg_types=arg_types)

    cuda.cuCtxSynchronize()

@pytest.mark.benchmark(group="launch-latency")
def test_launch_latency_small_kernel_512_ints(benchmark, init_cuda, load_module):
    device, ctx, stream = init_cuda
    module = load_module(kernel_string, device)

    err, func = cuda.cuModuleGetFunction(module, b'small_kernel_512_ints')
    ASSERT_DRV(err)

    args = [123] * 512
    arg_types = [ctypes.c_int] * 512

    args = tuple(args)
    arg_types = tuple(arg_types)

    benchmark(launch, func, stream, args=args, arg_types=arg_types)

    cuda.cuCtxSynchronize()

@pytest.mark.benchmark(group="launch-latency")
def test_launch_latency_small_kernel_512_bytes(benchmark, init_cuda, load_module):
    device, ctx, stream = init_cuda
    module = load_module(kernel_string, device)

    err, func = cuda.cuModuleGetFunction(module, b'small_kernel_512_chars')
    ASSERT_DRV(err)

    args = [127] * 512
    arg_types = [ctypes.c_byte] * 512

    args = tuple(args)
    arg_types = tuple(arg_types)

    benchmark(launch, func, stream, args=args, arg_types=arg_types)

    cuda.cuCtxSynchronize()

@pytest.mark.benchmark(group="launch-latency")
def test_launch_latency_small_kernel_512_longlongs(benchmark, init_cuda, load_module):
    device, ctx, stream = init_cuda
    module = load_module(kernel_string, device)

    err, func = cuda.cuModuleGetFunction(module, b'small_kernel_512_longlongs')
    ASSERT_DRV(err)

    args = [9223372036854775806] * 512
    arg_types = [ctypes.c_longlong] * 512

    args = tuple(args)
    arg_types = tuple(arg_types)

    benchmark(launch, func, stream, args=args, arg_types=arg_types)

    cuda.cuCtxSynchronize()

# Measure launch latency with many parameters using builtin parameter packing
@pytest.mark.benchmark(group="launch-latency")
def test_launch_latency_small_kernel_256_args(benchmark, init_cuda, load_module):
    device, ctx, stream = init_cuda
    module = load_module(kernel_string, device)

    err, func = cuda.cuModuleGetFunction(module, b'small_kernel_256_args')
    ASSERT_DRV(err)

    args = []
    arg_types = [None] * 256
    for _ in arg_types:
        err, p = cuda.cuMemAlloc(ctypes.sizeof(ctypes.c_int))
        ASSERT_DRV(err)
        args.append(p)

    args = tuple(args)
    arg_types = tuple(arg_types)

    benchmark(launch, func, stream, args=args, arg_types=arg_types)

    cuda.cuCtxSynchronize()

    for p in args:
        err, = cuda.cuMemFree(p)
        ASSERT_DRV(err)

# Measure launch latency with many parameters using builtin parameter packing
@pytest.mark.benchmark(group="launch-latency")
def test_launch_latency_small_kernel_16_args(benchmark, init_cuda, load_module):
    device, ctx, stream = init_cuda
    module = load_module(kernel_string, device)

    err, func = cuda.cuModuleGetFunction(module, b'small_kernel_16_args')
    ASSERT_DRV(err)

    args = []
    arg_types = [None] * 16
    for _ in arg_types:
        err, p = cuda.cuMemAlloc(ctypes.sizeof(ctypes.c_int))
        ASSERT_DRV(err)
        args.append(p)

    args = tuple(args)
    arg_types = tuple(arg_types)

    benchmark(launch, func, stream, args=args, arg_types=arg_types)

    cuda.cuCtxSynchronize()

    for p in args:
        err, = cuda.cuMemFree(p)
        ASSERT_DRV(err)

# Measure launch latency with many parameters, excluding parameter packing
@pytest.mark.benchmark(group="launch-latency")
def test_launch_latency_small_kernel_512_args_ctypes(benchmark, init_cuda, load_module):
    device, ctx, stream = init_cuda
    module = load_module(kernel_string, device)

    err, func = cuda.cuModuleGetFunction(module, b'small_kernel_512_args')
    ASSERT_DRV(err)

    vals = []
    val_ps = []
    for i in range(512):
        err, p = cuda.cuMemAlloc(ctypes.sizeof(ctypes.c_int))
        ASSERT_DRV(err)
        vals.append(p)
        val_ps.append(ctypes.c_void_p(int(vals[i])))

    packagedParams = (ctypes.c_void_p * 512)()
    for i in range(512):
        packagedParams[i] = ctypes.addressof(val_ps[i])

    benchmark(launch_packed, func, stream, packagedParams)

    cuda.cuCtxSynchronize()

    for p in vals:
        err, = cuda.cuMemFree(p)
        ASSERT_DRV(err)

def pack_and_launch(kernel, stream, params):
    packed_params = (ctypes.c_void_p * len(params))()
    ptrs = [0] * len(params)
    for i in range(len(params)):
        ptrs[i] = ctypes.c_void_p(int(params[i]))
        packed_params[i] = ctypes.addressof(ptrs[i])

    cuda.cuLaunchKernel(kernel,
                        1, 1, 1,
                        1, 1, 1,
                        0, stream,
                        packed_params, 0)

# Measure launch latency plus parameter packing using ctypes
@pytest.mark.benchmark(group="launch-latency")
def test_launch_latency_small_kernel_512_args_ctypes_with_packing(benchmark, init_cuda, load_module):
    device, ctx, stream = init_cuda
    module = load_module(kernel_string, device)

    err, func = cuda.cuModuleGetFunction(module, b'small_kernel_512_args')
    ASSERT_DRV(err)

    vals = []
    for i in range(512):
        err, p = cuda.cuMemAlloc(ctypes.sizeof(ctypes.c_int))
        ASSERT_DRV(err)
        vals.append(p)

    benchmark(pack_and_launch, func, stream, vals)

    cuda.cuCtxSynchronize()

    for p in vals:
        err, = cuda.cuMemFree(p)
        ASSERT_DRV(err)

# Measure launch latency with a single large struct parameter
@pytest.mark.benchmark(group="launch-latency")
def test_launch_latency_small_kernel_2048B(benchmark, init_cuda, load_module):
    device, ctx, stream = init_cuda
    module = load_module(kernel_string, device)

    err, func = cuda.cuModuleGetFunction(module, b'small_kernel_2048B')
    ASSERT_DRV(err)

    class struct_2048B(ctypes.Structure):
        _fields_ = [('values',ctypes.c_uint8 * 2048)]

    benchmark(launch, func, stream, args=(struct_2048B(),), arg_types=(None,))

    cuda.cuCtxSynchronize()
