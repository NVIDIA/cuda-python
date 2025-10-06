# Copyright 2021-2025 NVIDIA Corporation.  All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import ctypes

import numpy as np
from cuda.bindings import driver as cuda
from cuda.bindings import nvrtc


def ASSERT_DRV(err):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"Cuda Error: {err}")
    elif isinstance(err, nvrtc.nvrtcResult):
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
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
    ASSERT_DRV(err)

    # Device
    err, cuDevice = cuda.cuDeviceGet(0)
    ASSERT_DRV(err)

    # Ctx
    err, context = cuda.cuCtxCreate(None, 0, cuDevice)
    ASSERT_DRV(err)

    # Create program
    err, prog = nvrtc.nvrtcCreateProgram(str.encode(saxpy), b"saxpy.cu", 0, None, None)
    ASSERT_DRV(err)

    # Get target architecture
    err, major = cuda.cuDeviceGetAttribute(
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice
    )
    ASSERT_DRV(err)
    err, minor = cuda.cuDeviceGetAttribute(
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice
    )
    ASSERT_DRV(err)
    err, nvrtc_major, nvrtc_minor = nvrtc.nvrtcVersion()
    ASSERT_DRV(err)
    use_cubin = nvrtc_minor >= 1
    prefix = "sm" if use_cubin else "compute"
    arch_arg = bytes(f"--gpu-architecture={prefix}_{major}{minor}", "ascii")

    # Compile program
    opts = [b"--fmad=false", arch_arg]
    (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    ASSERT_DRV(err)

    # Get log from compilation
    err, logSize = nvrtc.nvrtcGetProgramLogSize(prog)
    ASSERT_DRV(err)
    log = b" " * logSize
    (err,) = nvrtc.nvrtcGetProgramLog(prog, log)
    ASSERT_DRV(err)
    print(log.decode())

    # Get data from compilation
    if use_cubin:
        err, dataSize = nvrtc.nvrtcGetCUBINSize(prog)
        ASSERT_DRV(err)
        data = b" " * dataSize
        (err,) = nvrtc.nvrtcGetCUBIN(prog, data)
        ASSERT_DRV(err)
    else:
        err, dataSize = nvrtc.nvrtcGetPTXSize(prog)
        ASSERT_DRV(err)
        data = b" " * dataSize
        (err,) = nvrtc.nvrtcGetPTX(prog, data)
        ASSERT_DRV(err)

    # Load data as module data and retrieve function
    data = np.char.array(data)
    err, module = cuda.cuModuleLoadData(data)
    ASSERT_DRV(err)
    err, kernel = cuda.cuModuleGetFunction(module, b"saxpy")
    ASSERT_DRV(err)

    # Test the kernel
    NUM_THREADS = 128
    NUM_BLOCKS = 32

    a = np.float32(2.0)
    n = np.array(NUM_THREADS * NUM_BLOCKS, dtype=np.uint32)
    bufferSize = n * a.itemsize

    err, dX = cuda.cuMemAlloc(bufferSize)
    ASSERT_DRV(err)
    err, dY = cuda.cuMemAlloc(bufferSize)
    ASSERT_DRV(err)
    err, dOut = cuda.cuMemAlloc(bufferSize)
    ASSERT_DRV(err)

    hX = np.random.rand(n).astype(dtype=np.float32)
    hY = np.random.rand(n).astype(dtype=np.float32)
    hOut = np.zeros(n).astype(dtype=np.float32)

    err, stream = cuda.cuStreamCreate(0)
    ASSERT_DRV(err)

    (err,) = cuda.cuMemcpyHtoDAsync(dX, hX, bufferSize, stream)
    ASSERT_DRV(err)
    (err,) = cuda.cuMemcpyHtoDAsync(dY, hY, bufferSize, stream)
    ASSERT_DRV(err)

    (err,) = cuda.cuStreamSynchronize(stream)
    ASSERT_DRV(err)

    # Assert values are different before running kernel
    hZ = a * hX + hY
    if np.allclose(hOut, hZ):
        raise ValueError("Error inside tolerence for host-device vectors")

    arg_values = (a, dX, dY, dOut, n)
    arg_types = (ctypes.c_float, None, None, None, ctypes.c_size_t)
    (err,) = cuda.cuLaunchKernel(
        kernel,
        NUM_BLOCKS,
        1,
        1,  # grid dim
        NUM_THREADS,
        1,
        1,  # block dim
        0,
        stream,  # shared mem and stream
        (arg_values, arg_types),
        0,
    )  # arguments
    ASSERT_DRV(err)

    (err,) = cuda.cuMemcpyDtoHAsync(hOut, dOut, bufferSize, stream)
    ASSERT_DRV(err)
    (err,) = cuda.cuStreamSynchronize(stream)
    ASSERT_DRV(err)

    # Assert values are same after running kernel
    hZ = a * hX + hY
    if not np.allclose(hOut, hZ):
        raise ValueError("Error outside tolerence for host-device vectors")

    (err,) = cuda.cuStreamDestroy(stream)
    ASSERT_DRV(err)

    (err,) = cuda.cuMemFree(dX)
    ASSERT_DRV(err)
    (err,) = cuda.cuMemFree(dY)
    ASSERT_DRV(err)
    (err,) = cuda.cuMemFree(dOut)
    ASSERT_DRV(err)

    (err,) = cuda.cuModuleUnload(module)
    ASSERT_DRV(err)
    (err,) = cuda.cuCtxDestroy(context)
    ASSERT_DRV(err)


if __name__ == "__main__":
    main()
