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
# dependencies = ["cuda-python>=13.0.0", "cuda-core>=1.0.0", "cupy-cuda13x>=14.0.0"]
# ///

"""
Independent Multi-GPU launches with cuda.core

Compile and launch different kernels on two GPUs concurrently, each running
against its own CuPy-allocated buffers and its own ``cuda.core.Stream``.

The sample does not use MPI, peer-to-peer, or any inter-GPU
communication. It just shows the plain "same host process, two independent
GPUs" pattern:

  * GPU 0 computes ``c = a + b`` (float32).
  * GPU 1 computes ``z = x - y`` (float32).

Each GPU has its own ``Program``, ``Stream``, and CuPy buffers. The sample
also demonstrates the ``StreamAdaptor`` idiom for bridging a foreign stream
(CuPy's current stream) into ``cuda.core`` so that memory initialized by
CuPy is ordered before the kernel launch on our ``cuda.core`` stream.

Waives when fewer than 2 CUDA-capable devices are available.
"""

import os
import sys

try:
    import cupy as cp

    from cuda.core import Device, LaunchConfig, Program, ProgramOptions, launch, system
except ImportError as e:
    print(f"Error: Required package not found: {e}")
    print("Please install from requirements.txt:")
    print("  pip install -r requirements.txt")
    sys.exit(1)


EXIT_WAIVED = int(os.environ.get("CUDA_PYTHON_SAMPLE_WAIVER_EXIT_CODE", "2"))

DTYPE = cp.float32
SIZE = 50000


# Bridge foreign streams (e.g. CuPy) that do not yet implement
# __cuda_stream__ into cuda.core. CuPy streams expose a ``.ptr`` attribute
# holding the raw CUstream handle.
class StreamAdaptor:
    def __init__(self, obj):
        self.obj = obj

    def __cuda_stream__(self):
        return (0, self.obj.ptr)


ADD_KERNEL = """
extern "C"
__global__ void vector_add(const float* A,
                           const float* B,
                           float* C,
                           size_t N) {
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (size_t i = tid; i < N; i += gridDim.x * blockDim.x) {
        C[i] = A[i] + B[i];
    }
}
"""

SUB_KERNEL = """
extern "C"
__global__ void vector_sub(const float* A,
                           const float* B,
                           float* C,
                           size_t N) {
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (size_t i = tid; i < N; i += gridDim.x * blockDim.x) {
        C[i] = A[i] - B[i];
    }
}
"""


def _compile(device, source):
    prog = Program(
        source,
        code_type="c++",
        options=ProgramOptions(std="c++17", arch=f"sm_{device.arch}"),
    )
    return prog.compile("cubin")


def main():
    num_devices = system.get_num_devices()
    if num_devices < 2:
        print(
            f"This sample requires at least 2 CUDA-capable devices (found {num_devices}). Waiving.",
            file=sys.stderr,
        )
        sys.exit(EXIT_WAIVED)

    dev0 = Device(0)
    dev0.set_current()
    stream0 = dev0.create_stream()
    stream1 = None
    cp_stream0 = None
    cp_stream1 = None

    try:
        # ---- GPU 0: compile the add kernel ----
        add_kernel = _compile(dev0, ADD_KERNEL).get_kernel("vector_add")

        # ---- GPU 1: switch context and compile the subtract kernel ----
        dev1 = Device(1)
        dev1.set_current()
        stream1 = dev1.create_stream()
        sub_kernel = _compile(dev1, SUB_KERNEL).get_kernel("vector_sub")

        # One launch config per GPU; both grids target the same problem size.
        block = 256
        grid = (SIZE + block - 1) // block
        config = LaunchConfig(grid=grid, block=block)

        # ---- Allocate on GPU 0 (uses CuPy's current stream on GPU 0) ----
        dev0.set_current()
        rng = cp.random.default_rng()
        a = rng.random(SIZE, dtype=DTYPE)
        b = rng.random(SIZE, dtype=DTYPE)
        c = cp.empty_like(a)
        # Wrap CuPy's current stream as a cuda.core Stream so we can wait on
        # it: this guarantees the random initialization completes before our
        # kernel touches these buffers on stream0.
        cp_stream0 = dev0.create_stream(StreamAdaptor(cp.cuda.get_current_stream()))
        stream0.wait(cp_stream0)

        launch(stream0, config, add_kernel, a.data.ptr, b.data.ptr, c.data.ptr, cp.uint64(SIZE))

        # ---- Allocate on GPU 1 (uses CuPy's current stream on GPU 1) ----
        dev1.set_current()
        rng = cp.random.default_rng()
        x = rng.random(SIZE, dtype=DTYPE)
        y = rng.random(SIZE, dtype=DTYPE)
        z = cp.empty_like(x)
        cp_stream1 = dev1.create_stream(StreamAdaptor(cp.cuda.get_current_stream()))
        stream1.wait(cp_stream1)

        launch(stream1, config, sub_kernel, x.data.ptr, y.data.ptr, z.data.ptr, cp.uint64(SIZE))

        # ---- Synchronize both GPUs and verify ----
        dev0.set_current()
        stream0.sync()
        assert cp.allclose(c, a + b), "GPU 0 vector_add produced incorrect results"
        dev1.set_current()
        stream1.sync()
        assert cp.allclose(z, x - y), "GPU 1 vector_sub produced incorrect results"

        print(f"GPU 0: vector_add on {SIZE} elements verified")
        print(f"GPU 1: vector_sub on {SIZE} elements verified")
        print("Done")
        return 0
    finally:
        if cp_stream1 is not None:
            cp_stream1.close()
        if cp_stream0 is not None:
            cp_stream0.close()
        if stream1 is not None:
            stream1.close()
        stream0.close()


if __name__ == "__main__":
    sys.exit(main())
