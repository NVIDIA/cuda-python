# Copyright 2021-2026 NVIDIA Corporation.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ################################################################################
#
# This example demonstrates JIT compilation of CUDA kernels using NVRTC
# and the Driver API (saxpy kernel).
#
# ################################################################################

# /// script
# dependencies = ["cuda_bindings>13.2.1", "numpy"]
# ///

import ctypes

import numpy as np

from cuda.bindings import driver as cuda
from cuda.bindings._v2 import nvrtc


def assert_drv(err):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"Cuda Error: {err}")
    elif isinstance(err, nvrtc.Result):
        if err != nvrtc.Result.SUCCESS:
            raise RuntimeError(f"Nvrtc Error: {err}")
    else:
        raise RuntimeError(f"Unknown error type: {err}")


saxpy = """\
extern "C" __global__
void saxpy(float a, float *x, float *y, float *out, size_t n)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        out[tid] = a * x[tid] + y[tid];
    }
}
"""


def main():
    # Init
    (err,) = cuda.cuInit(0)
    assert_drv(err)

    # Device
    err, cu_device = cuda.cuDeviceGet(0)
    assert_drv(err)

    # Ctx
    err, context = cuda.cuCtxCreate(None, 0, cu_device)
    assert_drv(err)

    # Create program
    prog = nvrtc.create_program(str.encode(saxpy), b"saxpy.cu")

    # Get target architecture
    err, major = cuda.cuDeviceGetAttribute(
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cu_device
    )
    assert_drv(err)
    err, minor = cuda.cuDeviceGetAttribute(
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cu_device
    )
    assert_drv(err)
    nvrtc_major, nvrtc_minor = nvrtc.version()
    use_cubin = nvrtc_minor >= 1
    prefix = "sm" if use_cubin else "compute"
    arch_arg = bytes(f"--gpu-architecture={prefix}_{major}{minor}", "ascii")

    # Compile program
    opts = [b"--fmad=false", arch_arg]
    nvrtc.compile_program(prog, opts)

    # Get log from compilation
    log = nvrtc.get_program_log(prog)
    print(log.decode())

    # Get data from compilation
    if use_cubin:
        data = nvrtc.get_cubin(prog)
    else:
        data = nvrtc.get_ptx(prog)

    # Load data as module data and retrieve function
    data = np.char.array(data)
    err, module = cuda.cuModuleLoadData(data)
    assert_drv(err)
    err, kernel = cuda.cuModuleGetFunction(module, b"saxpy")
    assert_drv(err)

    # Test the kernel
    num_threads = 128
    num_blocks = 32

    a = np.float32(2.0)
    n = np.array(num_threads * num_blocks, dtype=np.uint32)
    buffer_size = n * a.itemsize

    err, d_x = cuda.cuMemAlloc(buffer_size)
    assert_drv(err)
    err, d_y = cuda.cuMemAlloc(buffer_size)
    assert_drv(err)
    err, d_out = cuda.cuMemAlloc(buffer_size)
    assert_drv(err)

    h_x = np.random.rand(n).astype(dtype=np.float32)
    h_y = np.random.rand(n).astype(dtype=np.float32)
    h_out = np.zeros(n).astype(dtype=np.float32)

    err, stream = cuda.cuStreamCreate(0)
    assert_drv(err)

    (err,) = cuda.cuMemcpyHtoDAsync(d_x, h_x, buffer_size, stream)
    assert_drv(err)
    (err,) = cuda.cuMemcpyHtoDAsync(d_y, h_y, buffer_size, stream)
    assert_drv(err)

    (err,) = cuda.cuStreamSynchronize(stream)
    assert_drv(err)

    # Assert values are different before running kernel
    h_z = a * h_x + h_y
    if np.allclose(h_out, h_z):
        raise ValueError("Error inside tolerence for host-device vectors")

    arg_values = (a, d_x, d_y, d_out, n)
    arg_types = (ctypes.c_float, None, None, None, ctypes.c_size_t)
    (err,) = cuda.cuLaunchKernel(
        kernel,
        num_blocks,
        1,
        1,  # grid dim
        num_threads,
        1,
        1,  # block dim
        0,
        stream,  # shared mem and stream
        (arg_values, arg_types),
        0,
    )  # arguments
    assert_drv(err)

    (err,) = cuda.cuMemcpyDtoHAsync(h_out, d_out, buffer_size, stream)
    assert_drv(err)
    (err,) = cuda.cuStreamSynchronize(stream)
    assert_drv(err)

    # Assert values are same after running kernel
    h_z = a * h_x + h_y
    if not np.allclose(h_out, h_z):
        raise ValueError("Error outside tolerence for host-device vectors")

    (err,) = cuda.cuStreamDestroy(stream)
    assert_drv(err)

    (err,) = cuda.cuMemFree(d_x)
    assert_drv(err)
    (err,) = cuda.cuMemFree(d_y)
    assert_drv(err)
    (err,) = cuda.cuMemFree(d_out)
    assert_drv(err)

    (err,) = cuda.cuModuleUnload(module)
    assert_drv(err)
    (err,) = cuda.cuCtxDestroy(context)
    assert_drv(err)


if __name__ == "__main__":
    main()
