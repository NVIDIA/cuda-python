# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# /// script
# dependencies = ["cuda-python>=13.0.0", "numpy>=1.24"]
# ///

"""
Stream-ordered memory allocation with cudaMallocAsync

Demonstrates the raw ``cudaMallocAsync`` / ``cudaFreeAsync`` API and the
memory-pool release-threshold attribute
(``cudaMemPoolAttrReleaseThreshold``).

Two variants are run back to back on the same 1M-element vector add:

  1. **Basic** -- ``cudaMallocAsync`` / launch / ``cudaFreeAsync`` per
     iteration on the default memory pool. The default release threshold is
     zero, so the pool may release chunks back to the OS between calls.
  2. **Post-sync** -- set ``cudaMemPoolAttrReleaseThreshold`` to a very
     large value so ``cudaFreeAsync`` never releases memory back to the OS
     during the steady-state loop, then time the loop with CUDA events.

This is the low-level counterpart to the high-level
[`samples/cuda_core/memoryResources/`](../../../cuda_core/memoryResources/) sample, whose
``DeviceMemoryResource`` sits on top of the same pool but hides the
attribute knobs.

Waives on Darwin (Metal-only) and on GPUs without memory-pool support.
"""

import ctypes
import math
import platform
import random as rnd
import sys
from pathlib import Path

# Add samples/cuda_bindings/Utilities/ to the import path for shared bindings helpers.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "Utilities"))

try:
    import numpy as np
    from cuda_bindings_utils import (
        KernelHelper,
        check_cmd_line_flag,
        check_cuda_errors,
        find_cuda_device,
        requirement_not_met,
    )

    from cuda.bindings import driver as cuda
    from cuda.bindings import runtime as cudart
except ImportError as e:
    print(f"Error: Required package not found: {e}")
    print("Please install from requirements.txt:")
    print("  pip install -r requirements.txt")
    sys.exit(1)


VECTOR_ADD_KERNEL = """\
/* Add two vectors on the GPU */
extern "C"
__global__ void vectorAddGPU(const float *a, const float *b, float *c, int N)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}
"""


MAX_ITER = 20

# Populated once in main() so the two demo functions can call the same kernel.
_vector_add_gpu = None


def _launch_add(stream, nelem, d_a, d_b, d_c):
    block = cudart.dim3()
    block.x = 256
    block.y = 1
    block.z = 1
    grid = cudart.dim3()
    grid.x = math.ceil(nelem / float(block.x))
    grid.y = 1
    grid.z = 1

    kernel_args = (
        (d_a, d_b, d_c, nelem),
        (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int),
    )
    check_cuda_errors(
        cuda.cuLaunchKernel(
            _vector_add_gpu,
            grid.x,
            grid.y,
            grid.z,
            block.x,
            block.y,
            block.z,
            0,
            stream,
            kernel_args,
            0,
        )
    )


def _l2_norm_error(nelem, a, b, c):
    error_norm = 0.0
    ref_norm = 0.0
    for n in range(nelem):
        ref = a[n] + b[n]
        diff = c[n] - ref
        error_norm += diff * diff
        ref_norm += ref * ref
    error_norm = math.sqrt(error_norm)
    ref_norm = math.sqrt(ref_norm)
    return error_norm / ref_norm if ref_norm > 0 else error_norm


def basic_stream_ordered_allocation(dev, nelem, a, b, c):
    """Basic stream-ordered alloc/free of the three vectors, no pool tuning."""
    num_bytes = nelem * np.dtype(np.float32).itemsize

    print("Starting basicStreamOrderedAllocation()")
    check_cuda_errors(cudart.cudaSetDevice(dev))
    stream = check_cuda_errors(cudart.cudaStreamCreateWithFlags(cudart.cudaStreamNonBlocking))

    d_a = check_cuda_errors(cudart.cudaMallocAsync(num_bytes, stream))
    d_b = check_cuda_errors(cudart.cudaMallocAsync(num_bytes, stream))
    d_c = check_cuda_errors(cudart.cudaMallocAsync(num_bytes, stream))
    check_cuda_errors(cudart.cudaMemcpyAsync(d_a, a, num_bytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream))
    check_cuda_errors(cudart.cudaMemcpyAsync(d_b, b, num_bytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream))

    _launch_add(stream, nelem, d_a, d_b, d_c)

    check_cuda_errors(cudart.cudaFreeAsync(d_a, stream))
    check_cuda_errors(cudart.cudaFreeAsync(d_b, stream))
    check_cuda_errors(cudart.cudaMemcpyAsync(c, d_c, num_bytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream))
    check_cuda_errors(cudart.cudaFreeAsync(d_c, stream))
    check_cuda_errors(cudart.cudaStreamSynchronize(stream))

    print("> Checking the results from vectorAddGPU() ...")
    err = _l2_norm_error(nelem, a, b, c)
    check_cuda_errors(cudart.cudaStreamDestroy(stream))
    return err < 1.0e-6


def stream_ordered_allocation_post_sync(dev, nelem, a, b, c):
    """Same alloc/launch/free loop, but with a huge release threshold so the
    pool retains its backing between iterations and does not thrash the OS."""
    num_bytes = nelem * np.dtype(np.float32).itemsize

    print("Starting streamOrderedAllocationPostSync()")
    check_cuda_errors(cudart.cudaSetDevice(dev))
    stream = check_cuda_errors(cudart.cudaStreamCreateWithFlags(cudart.cudaStreamNonBlocking))
    start = check_cuda_errors(cudart.cudaEventCreate())
    end = check_cuda_errors(cudart.cudaEventCreate())

    mem_pool = check_cuda_errors(cudart.cudaDeviceGetDefaultMemPool(dev))
    threshold_val = cuda.cuuint64_t(ctypes.c_uint64(-1).value)  # ~UINT64_MAX
    check_cuda_errors(
        cudart.cudaMemPoolSetAttribute(
            mem_pool,
            cudart.cudaMemPoolAttr.cudaMemPoolAttrReleaseThreshold,
            threshold_val,
        )
    )

    check_cuda_errors(cudart.cudaEventRecord(start, stream))
    for _ in range(MAX_ITER):
        d_a = check_cuda_errors(cudart.cudaMallocAsync(num_bytes, stream))
        d_b = check_cuda_errors(cudart.cudaMallocAsync(num_bytes, stream))
        d_c = check_cuda_errors(cudart.cudaMallocAsync(num_bytes, stream))
        check_cuda_errors(
            cudart.cudaMemcpyAsync(d_a, a, num_bytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
        )
        check_cuda_errors(
            cudart.cudaMemcpyAsync(d_b, b, num_bytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
        )

        _launch_add(stream, nelem, d_a, d_b, d_c)

        check_cuda_errors(cudart.cudaFreeAsync(d_a, stream))
        check_cuda_errors(cudart.cudaFreeAsync(d_b, stream))
        check_cuda_errors(
            cudart.cudaMemcpyAsync(c, d_c, num_bytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
        )
        check_cuda_errors(cudart.cudaFreeAsync(d_c, stream))
        check_cuda_errors(cudart.cudaStreamSynchronize(stream))
    check_cuda_errors(cudart.cudaEventRecord(end, stream))
    check_cuda_errors(cudart.cudaEventSynchronize(end))

    msec_total = check_cuda_errors(cudart.cudaEventElapsedTime(start, end))
    print(f"Total elapsed time = {msec_total:.3f} ms over {MAX_ITER} iterations")

    print("> Checking the results from vectorAddGPU() ...")
    err = _l2_norm_error(nelem, a, b, c)
    check_cuda_errors(cudart.cudaStreamDestroy(stream))
    return err < 1.0e-6


def main():
    if platform.system() == "Darwin":
        requirement_not_met("streamOrderedAllocation is not supported on Mac OSX")

    cuda.cuInit(0)
    if check_cmd_line_flag("help"):
        print("Usage: streamOrderedAllocation [OPTION]")
        print("Options:")
        print("  device=[device #]  Specify the device to be used")
        return 0

    dev = find_cuda_device()

    version = check_cuda_errors(cudart.cudaDriverGetVersion())
    if version < 11030:
        is_mem_pool_supported = False
    else:
        is_mem_pool_supported = check_cuda_errors(
            cudart.cudaDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED, dev)
        )
    if not is_mem_pool_supported:
        requirement_not_met("Waiving execution: device does not support Memory Pools")

    global _vector_add_gpu
    kernel_helper = KernelHelper(VECTOR_ADD_KERNEL, dev)
    _vector_add_gpu = kernel_helper.get_function(b"vectorAddGPU")

    nelem = 1048576
    a = np.zeros(nelem, dtype="float32")
    b = np.zeros(nelem, dtype="float32")
    c = np.zeros(nelem, dtype="float32")
    for i in range(nelem):
        a[i] = rnd.random()
        b[i] = rnd.random()

    ret1 = basic_stream_ordered_allocation(dev, nelem, a, b, c)
    ret2 = stream_ordered_allocation_post_sync(dev, nelem, a, b, c)

    if not (ret1 and ret2):
        return 1

    print("Both stream-ordered allocation variants verified.")
    print("Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
