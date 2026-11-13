# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""SAXPY with the CUDA Python low-level bindings (written for cuda-python 11.8)."""

import sys

import numpy as np

from cuda import cuda, nvrtc


def assert_drv(err):
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
assert_drv(err)

# Retrieve handle for device 0
err, cu_device = cuda.cuDeviceGet(0)
assert_drv(err)

# Derive target architecture for device 0
err, major = cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cu_device)
assert_drv(err)
err, minor = cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cu_device)
assert_drv(err)
arch_arg = bytes(f"--gpu-architecture=compute_{major}{minor}", "ascii")

# Create program
err, prog = nvrtc.nvrtcCreateProgram(str.encode(saxpy), b"saxpy.cu", 0, [], [])
assert_drv(err)

# Compile program
opts = [b"--fmad=false", arch_arg]
(err,) = nvrtc.nvrtcCompileProgram(prog, 2, opts)
assert_drv(err)

# Get PTX from compilation
err, ptx_size = nvrtc.nvrtcGetPTXSize(prog)
assert_drv(err)
ptx = b" " * ptx_size
(err,) = nvrtc.nvrtcGetPTX(prog, ptx)
assert_drv(err)

# Create context
err, context = cuda.cuCtxCreate(0, cu_device)
assert_drv(err)

# Load PTX as module data and retrieve function
ptx = np.char.array(ptx)
err, module = cuda.cuModuleLoadData(ptx.ctypes.data)
assert_drv(err)
err, kernel = cuda.cuModuleGetFunction(module, b"saxpy")
assert_drv(err)

NUM_THREADS = 512  # Threads per block
NUM_BLOCKS = 32768  # Blocks per grid

a = np.array([2.0], dtype=np.float32)
n = np.array(NUM_THREADS * NUM_BLOCKS, dtype=np.uint64)
buffer_size = n * a.itemsize

h_x = (np.arange(n, dtype=np.float32) % 1000.0).astype(dtype=np.float32)
h_y = (np.arange(n, dtype=np.float32) % 7.0).astype(dtype=np.float32)
h_out = np.zeros(n).astype(dtype=np.float32)

err, d_x_class = cuda.cuMemAlloc(buffer_size)
assert_drv(err)
err, d_y_class = cuda.cuMemAlloc(buffer_size)
assert_drv(err)
err, d_out_class = cuda.cuMemAlloc(buffer_size)
assert_drv(err)

err, stream = cuda.cuStreamCreate(0)
assert_drv(err)

(err,) = cuda.cuMemcpyHtoDAsync(d_x_class, h_x.ctypes.data, buffer_size, stream)
assert_drv(err)
(err,) = cuda.cuMemcpyHtoDAsync(d_y_class, h_y.ctypes.data, buffer_size, stream)
assert_drv(err)

# The following code example is not intuitive
# Subject to change in a future release
d_x = np.array([int(d_x_class)], dtype=np.uint64)
d_y = np.array([int(d_y_class)], dtype=np.uint64)
d_out = np.array([int(d_out_class)], dtype=np.uint64)

args = [a, d_x, d_y, d_out, n]
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
assert_drv(err)

(err,) = cuda.cuMemcpyDtoHAsync(h_out.ctypes.data, d_out_class, buffer_size, stream)
assert_drv(err)
(err,) = cuda.cuStreamSynchronize(stream)
assert_drv(err)

# Assert values are same after running kernel
h_z = a * h_x + h_y
if not np.allclose(h_out, h_z):
    raise ValueError("Error outside tolerance for host-device vectors")

sys.stdout.write(f"saxpy ok: n={int(n)} out[:4]={h_out[:4].tolist()} sum={h_out.sum(dtype=np.float64):.1f}\n")

(err,) = cuda.cuStreamDestroy(stream)
assert_drv(err)
(err,) = cuda.cuMemFree(d_x_class)
assert_drv(err)
(err,) = cuda.cuMemFree(d_y_class)
assert_drv(err)
(err,) = cuda.cuMemFree(d_out_class)
assert_drv(err)
(err,) = cuda.cuModuleUnload(module)
assert_drv(err)
(err,) = cuda.cuCtxDestroy(context)
assert_drv(err)
