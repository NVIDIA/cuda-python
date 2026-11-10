# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""SAXPY with the CUDA Python low-level bindings (written for cuda-python 11.8)."""

import numpy as np

from cuda import cuda, nvrtc


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

# Initialize CUDA Driver API
(err,) = cuda.cuInit(0)
ASSERT_DRV(err)

# Retrieve handle for device 0
err, cuDevice = cuda.cuDeviceGet(0)
ASSERT_DRV(err)

# Derive target architecture for device 0
err, major = cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice)
ASSERT_DRV(err)
err, minor = cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice)
ASSERT_DRV(err)
arch_arg = bytes(f"--gpu-architecture=compute_{major}{minor}", "ascii")

# Create program
err, prog = nvrtc.nvrtcCreateProgram(str.encode(saxpy), b"saxpy.cu", 0, [], [])
ASSERT_DRV(err)

# Compile program
opts = [b"--fmad=false", arch_arg]
(err,) = nvrtc.nvrtcCompileProgram(prog, 2, opts)
ASSERT_DRV(err)

# Get PTX from compilation
err, ptxSize = nvrtc.nvrtcGetPTXSize(prog)
ASSERT_DRV(err)
ptx = b" " * ptxSize
(err,) = nvrtc.nvrtcGetPTX(prog, ptx)
ASSERT_DRV(err)

# Create context
err, context = cuda.cuCtxCreate(0, cuDevice)
ASSERT_DRV(err)

# Load PTX as module data and retrieve function
ptx = np.char.array(ptx)
err, module = cuda.cuModuleLoadData(ptx.ctypes.data)
ASSERT_DRV(err)
err, kernel = cuda.cuModuleGetFunction(module, b"saxpy")
ASSERT_DRV(err)

NUM_THREADS = 512  # Threads per block
NUM_BLOCKS = 32768  # Blocks per grid

a = np.array([2.0], dtype=np.float32)
n = np.array(NUM_THREADS * NUM_BLOCKS, dtype=np.uint64)
bufferSize = n * a.itemsize

hX = (np.arange(n, dtype=np.float32) % 1000.0).astype(dtype=np.float32)
hY = (np.arange(n, dtype=np.float32) % 7.0).astype(dtype=np.float32)
hOut = np.zeros(n).astype(dtype=np.float32)

err, dXclass = cuda.cuMemAlloc(bufferSize)
ASSERT_DRV(err)
err, dYclass = cuda.cuMemAlloc(bufferSize)
ASSERT_DRV(err)
err, dOutclass = cuda.cuMemAlloc(bufferSize)
ASSERT_DRV(err)

err, stream = cuda.cuStreamCreate(0)
ASSERT_DRV(err)

(err,) = cuda.cuMemcpyHtoDAsync(dXclass, hX.ctypes.data, bufferSize, stream)
ASSERT_DRV(err)
(err,) = cuda.cuMemcpyHtoDAsync(dYclass, hY.ctypes.data, bufferSize, stream)
ASSERT_DRV(err)

# The following code example is not intuitive
# Subject to change in a future release
dX = np.array([int(dXclass)], dtype=np.uint64)
dY = np.array([int(dYclass)], dtype=np.uint64)
dOut = np.array([int(dOutclass)], dtype=np.uint64)

args = [a, dX, dY, dOut, n]
args = np.array([arg.ctypes.data for arg in args], dtype=np.uint64)

(err,) = cuda.cuLaunchKernel(
    kernel,
    NUM_BLOCKS,  # grid x dim
    1,  # grid y dim
    1,  # grid z dim
    NUM_THREADS,  # block x dim
    1,  # block y dim
    1,  # block z dim
    0,  # dynamic shared memory
    stream,  # stream
    args.ctypes.data,  # kernel arguments
    0,  # extra (ignore)
)
ASSERT_DRV(err)

(err,) = cuda.cuMemcpyDtoHAsync(hOut.ctypes.data, dOutclass, bufferSize, stream)
ASSERT_DRV(err)
(err,) = cuda.cuStreamSynchronize(stream)
ASSERT_DRV(err)

# Assert values are same after running kernel
hZ = a * hX + hY
if not np.allclose(hOut, hZ):
    raise ValueError("Error outside tolerance for host-device vectors")

print(f"saxpy ok: n={int(n)} out[:4]={hOut[:4].tolist()} sum={hOut.sum(dtype=np.float64):.1f}")

(err,) = cuda.cuStreamDestroy(stream)
ASSERT_DRV(err)
(err,) = cuda.cuMemFree(dXclass)
ASSERT_DRV(err)
(err,) = cuda.cuMemFree(dYclass)
ASSERT_DRV(err)
(err,) = cuda.cuMemFree(dOutclass)
ASSERT_DRV(err)
(err,) = cuda.cuModuleUnload(module)
ASSERT_DRV(err)
(err,) = cuda.cuCtxDestroy(context)
ASSERT_DRV(err)
