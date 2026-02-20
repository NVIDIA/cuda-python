# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# ################################################################################
#
# This demo illustrates how to use `cuda.core` to compile and launch kernels
# on multiple GPUs.
#
# ################################################################################

import sys

import cupy as cp
from cuda.core import Device, LaunchConfig, Program, launch, system


def main():
    if system.get_num_devices() < 2:
        print("this example requires at least 2 GPUs", file=sys.stderr)
        sys.exit(0)

    dtype = cp.float32
    size = 50000

    # Set GPU 0
    dev0 = Device(0)
    dev0.set_current()
    stream0 = dev0.create_stream()

    # Compile a kernel targeting GPU 0 to compute c = a + b
    code_add = """
    extern "C"
    __global__ void vector_add(const float* A, const float* B, float* C, size_t N) {
        const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
        for (size_t i=tid; i<N; i+=gridDim.x*blockDim.x) {
            C[i] = A[i] + B[i];
        }
    }
    """
    prog_add = Program(code_add, code_type="c++", options={"std": "c++17", "arch": f"sm_{dev0.arch}"})
    mod_add = prog_add.compile("cubin")
    ker_add = mod_add.get_kernel("vector_add")

    # Set GPU 1
    dev1 = Device(1)
    dev1.set_current()
    stream1 = dev1.create_stream()

    # Compile a kernel targeting GPU 1 to compute c = a - b
    code_sub = """
    extern "C"
    __global__ void vector_sub(const float* A, const float* B, float* C, size_t N) {
        const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
        for (size_t i=tid; i<N; i+=gridDim.x*blockDim.x) {
            C[tid] = A[tid] - B[tid];
        }
    }
    """
    prog_sub = Program(code_sub, code_type="c++", options={"std": "c++17", "arch": f"sm_{dev1.arch}"})
    mod_sub = prog_sub.compile("cubin")
    ker_sub = mod_sub.get_kernel("vector_sub")

    # This adaptor ensures that any foreign stream (ex: from CuPy) that have not
    # yet supported the __cuda_stream__ protocol can still be recognized by
    # cuda.core.
    class StreamAdaptor:
        def __init__(self, obj):
            self.obj = obj

        def __cuda_stream__(self):
            # Note: CuPy streams have a .ptr attribute
            return (0, self.obj.ptr)

    # Create launch configs for each kernel that will be executed on the respective
    # CUDA streams.
    block = 256
    grid = (size + block - 1) // block
    config0 = LaunchConfig(grid=grid, block=block)
    config1 = LaunchConfig(grid=grid, block=block)

    # Allocate memory on GPU 0
    # Note: This runs on CuPy's current stream for GPU 0
    dev0.set_current()
    rng = cp.random.default_rng()
    a = rng.random(size, dtype=dtype)
    b = rng.random(size, dtype=dtype)
    c = cp.empty_like(a)
    cp_stream0 = dev0.create_stream(StreamAdaptor(cp.cuda.get_current_stream()))

    # Establish a stream order to ensure that memory has been initialized before
    # accessed by the kernel.
    stream0.wait(cp_stream0)

    # Launch the add kernel on GPU 0 / stream 0
    launch(stream0, config0, ker_add, a.data.ptr, b.data.ptr, c.data.ptr, cp.uint64(size))

    # Allocate memory on GPU 1
    # Note: This runs on CuPy's current stream for GPU 1.
    dev1.set_current()
    rng = cp.random.default_rng()
    x = rng.random(size, dtype=dtype)
    y = rng.random(size, dtype=dtype)
    z = cp.empty_like(a)
    cp_stream1 = dev1.create_stream(StreamAdaptor(cp.cuda.get_current_stream()))

    # Establish a stream order
    stream1.wait(cp_stream1)

    # Launch the subtract kernel on GPU 1 / stream 1
    launch(stream1, config1, ker_sub, x.data.ptr, y.data.ptr, z.data.ptr, cp.uint64(size))

    # Synchronize both GPUs are validate the results
    dev0.set_current()
    stream0.sync()
    assert cp.allclose(c, a + b)
    dev1.set_current()
    stream1.sync()
    assert cp.allclose(z, x - y)

    print("done")


if __name__ == "__main__":
    main()
