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
Kernel timing with the device clock() intrinsic and dynamic shared memory

This sample uses the driver API through ``cuda.bindings`` to compile
(with NVRTC) and launch a reduction kernel that reads the SM cycle counter
via CUDA's device-side ``clock()`` intrinsic. Each block records the cycle
count at kernel entry and exit; the host then reports the average
cycles-per-block spent in the reduction.

Two low-level techniques are demonstrated together:

  * **On-device timing** -- ``clock_t clock()`` is a SM-cycle register you
    can read from a kernel. This gives you the on-device analogue of a host
    CUDA event, useful for measuring kernel-internal work at the block level.
  * **Dynamic shared memory** -- the kernel declares ``extern __shared__
    float shared[]`` and the host passes the byte size at launch time via
    ``cuLaunchKernel``'s ``sharedMemBytes`` argument.

Waives on 32-bit ARM (armv7l), where the sample isn't supported.
"""

import platform
import sys
from pathlib import Path

# Add samples/cuda_bindings/Utilities/ to the import path for shared bindings helpers.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "Utilities"))

try:
    import numpy as np
    from cuda_bindings_utils import KernelHelper, check_cuda_errors, find_cuda_device, requirement_not_met

    from cuda.bindings import driver as cuda
except ImportError as e:
    print(f"Error: Required package not found: {e}")
    print("Please install from requirements.txt:")
    print("  pip install -r requirements.txt")
    sys.exit(1)


CLOCK_NVRTC_KERNEL = """\
extern "C" __global__ void timedReduction(const float *hinput, float *output, clock_t *timer)
{
    // __shared__ float shared[2 * blockDim.x];
    extern __shared__ float shared[];

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    if (tid == 0) timer[bid] = clock();

    // Copy input.
    shared[tid] = hinput[tid];
    shared[tid + blockDim.x] = hinput[tid + blockDim.x];

    // Perform reduction to find minimum.
    for (int d = blockDim.x; d > 0; d /= 2)
    {
        __syncthreads();

        if (tid < d)
        {
            float f0 = shared[tid];
            float f1 = shared[tid + d];

            if (f1 < f0)
            {
                shared[tid] = f1;
            }
        }
    }

    // Write result.
    if (tid == 0) output[bid] = shared[0];

    __syncthreads();

    if (tid == 0) timer[bid + gridDim.x] = clock();
}
"""


NUM_BLOCKS = 64
NUM_THREADS = 256


def _elems_to_bytes(nelems, dtype):
    return nelems * np.dtype(dtype).itemsize


def _check_requirements():
    if platform.machine() == "armv7l":
        requirement_not_met("clockNvrtc is not supported on ARMv7")


def main():
    _check_requirements()

    timer = np.empty(NUM_BLOCKS * 2, dtype="int64")
    hinput = np.empty(NUM_THREADS * 2, dtype="float32")

    for i in range(NUM_THREADS * 2):
        hinput[i] = i

    dev_id = find_cuda_device()
    kernel_helper = KernelHelper(CLOCK_NVRTC_KERNEL, dev_id)
    kernel_addr = kernel_helper.get_function(b"timedReduction")

    dinput = check_cuda_errors(cuda.cuMemAlloc(hinput.nbytes))
    doutput = check_cuda_errors(cuda.cuMemAlloc(_elems_to_bytes(NUM_BLOCKS, np.float32)))
    dtimer = check_cuda_errors(cuda.cuMemAlloc(timer.nbytes))
    check_cuda_errors(cuda.cuMemcpyHtoD(dinput, hinput, hinput.nbytes))

    args = ((dinput, doutput, dtimer), (None, None, None))
    shared_memory_nbytes = _elems_to_bytes(2 * NUM_THREADS, np.float32)

    grid_dims = (NUM_BLOCKS, 1, 1)
    block_dims = (NUM_THREADS, 1, 1)

    # Pass shared_memory_nbytes so the kernel's extern __shared__ array is
    # allocated at launch time. Zero is the default (no dynamic shmem).
    check_cuda_errors(
        cuda.cuLaunchKernel(
            kernel_addr,
            *grid_dims,
            *block_dims,
            shared_memory_nbytes,
            0,  # stream
            args,
            0,
        )
    )

    check_cuda_errors(cuda.cuCtxSynchronize())
    check_cuda_errors(cuda.cuMemcpyDtoH(timer, dtimer, timer.nbytes))
    check_cuda_errors(cuda.cuMemFree(dinput))
    check_cuda_errors(cuda.cuMemFree(doutput))
    check_cuda_errors(cuda.cuMemFree(dtimer))

    avg_elapsed_clocks = 0.0
    for i in range(NUM_BLOCKS):
        avg_elapsed_clocks += timer[i + NUM_BLOCKS] - timer[i]

    avg_elapsed_clocks = avg_elapsed_clocks / NUM_BLOCKS
    print(f"Average clocks/block = {avg_elapsed_clocks}")
    print("Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
