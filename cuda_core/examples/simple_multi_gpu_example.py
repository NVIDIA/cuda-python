# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import sys

import cupy as cp

from cuda.core.experimental import Device, LaunchConfig, Program, launch, system

if system.num_devices < 2:
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
__global__ void vector_add(const float* A,
                           const float* B,
                           float* C,
                           size_t N) {
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (size_t i=tid; i<N; i+=gridDim.x*blockDim.x) {
        C[tid] = A[tid] + B[tid];
    }
}
"""
arch0 = "".join(f"{i}" for i in dev0.compute_capability)
prog_add = Program(code_add, code_type="c++")
mod_add = prog_add.compile(
    "cubin",
    options=(
        "-std=c++17",
        "-arch=sm_" + arch0,
    ),
)
ker_add = mod_add.get_kernel("vector_add")

# Set GPU 1
dev1 = Device(1)
dev1.set_current()
stream1 = dev1.create_stream()

# Compile a kernel targeting GPU 1 to compute c = a - b
code_sub = """
extern "C"
__global__ void vector_sub(const float* A,
                           const float* B,
                           float* C,
                           size_t N) {
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (size_t i=tid; i<N; i+=gridDim.x*blockDim.x) {
        C[tid] = A[tid] - B[tid];
    }
}
"""
arch1 = "".join(f"{i}" for i in dev1.compute_capability)
prog_sub = Program(code_sub, code_type="c++")
mod_sub = prog_sub.compile(
    "cubin",
    options=(
        "-std=c++17",
        "-arch=sm_" + arch1,
    ),
)
ker_sub = mod_sub.get_kernel("vector_sub")


# This adaptor ensures that any foreign stream (ex: from CuPy) that have not
# yet supported the __cuda_stream__ protocol can still be recognized by
# cuda.core.
class StreamAdaptor:
    def __init__(self, obj):
        self.obj = obj

    @property
    def __cuda_stream__(self):
        # Note: CuPy streams have a .ptr attribute
        return (0, self.obj.ptr)


# Create launch configs for each kernel that will be executed on the respective
# CUDA streams.
block = 256
grid = (size + block - 1) // block
config0 = LaunchConfig(grid=grid, block=block, stream=stream0)
config1 = LaunchConfig(grid=grid, block=block, stream=stream1)

# Allocate memory on GPU 0
# Note: This runs on CuPy's current stream for GPU 0
dev0.set_current()
a = cp.random.random(size, dtype=dtype)
b = cp.random.random(size, dtype=dtype)
c = cp.empty_like(a)
cp_stream0 = StreamAdaptor(cp.cuda.get_current_stream())

# Establish a stream order to ensure that memory has been initialized before
# accessed by the kernel.
stream0.wait(cp_stream0)

# Launch the add kernel on GPU 0 / stream 0
launch(ker_add, config0, a.data.ptr, b.data.ptr, c.data.ptr, cp.uint64(size))

# Allocate memory on GPU 1
# Note: This runs on CuPy's current stream for GPU 1.
dev1.set_current()
x = cp.random.random(size, dtype=dtype)
y = cp.random.random(size, dtype=dtype)
z = cp.empty_like(a)
cp_stream1 = StreamAdaptor(cp.cuda.get_current_stream())

# Establish a stream order
stream1.wait(cp_stream1)

# Launch the subtract kernel on GPU 1 / stream 1
launch(ker_sub, config1, x.data.ptr, y.data.ptr, z.data.ptr, cp.uint64(size))

# Synchronize both GPUs are validate the results
dev0.set_current()
stream0.sync()
assert cp.allclose(c, a + b)
dev1.set_current()
stream1.sync()
assert cp.allclose(z, x - y)

print("done")
