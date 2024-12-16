#!/usr/bin/env python
# coding: utf-8

from cuda.core.experimental import Device
from cuda import cuda, cudart, nvrtc
from cuda.core.experimental import LaunchConfig, launch
from cuda.core.experimental import Program
from cuda.core.experimental._memory import Buffer

import cupy as cp

dtype = cp.float32
size = 50000

# Set GPU0
dev0 = Device(0)
dev0.set_current()
stream0 = dev0.create_stream()

# allocate memory to GPU0
a = cp.random.random(size, dtype=dtype)
b = cp.random.random(size, dtype=dtype)
c = cp.empty_like(a)

# Set GPU1
dev1 = Device(1)
dev1.set_current()
stream1 = dev1.create_stream()

# allocate memory to GPU1
x = cp.random.random(size, dtype=dtype)
y = cp.random.random(size, dtype=dtype)
z = cp.empty_like(a)

# compute c = a + b
code_add = """
template<typename T>
__global__ void vector_add(const T* A,
                           const T* B,
                           T* C,
                           size_t N) {
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (size_t i=tid; i<N; i+=gridDim.x*blockDim.x) {
        C[tid] = A[tid] + B[tid];
    }
}
"""

# compute c = a - b
code_sub = """
template<typename T>
__global__ void vector_sub(const T* A,
                           const T* B,
                           T* C,
                           size_t N) {
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (size_t i=tid; i<N; i+=gridDim.x*blockDim.x) {
        C[tid] = A[tid] - B[tid];
    }
}
"""

prog_add = Program(code_add, code_type="c++")
mod_add = prog_add.compile(
    "cubin",
    options=("-std=c++17", "-arch=sm_" + "".join(f"{i}" for i in dev0.compute_capability),),
    name_expressions=("vector_add<float>",))

# run in single precision
ker_add = mod_add.get_kernel("vector_add<float>")

prog_sub = Program(code_sub, code_type="c++")
mod_sub = prog_sub.compile(
    "cubin",
    options=("-std=c++17", "-arch=sm_" + "".join(f"{i}" for i in dev1.compute_capability),),
    name_expressions=("vector_sub<float>",))

# run in single precision
ker_sub = mod_sub.get_kernel("vector_sub<float>")


# Synchronize devices to ensure that memory has been created
dev0.sync()
dev1.sync()

block = 256
grid0 = (size + block - 1) // block
grid1 = (size + block - 1) // block

config0 = LaunchConfig(grid=grid0, block=block, stream=stream0)
config1 = LaunchConfig(grid=grid1, block=block, stream=stream1)

# First we update device 0 data with host data
dev0.set_current()
launch(ker_add, config0, a.data.ptr, b.data.ptr, c.data.ptr, cp.uint64(size))
stream0.sync()
# Validate  result
assert cp.allclose(c, a + b)

dev1.set_current()
launch(ker_sub, config1, x.data.ptr, y.data.ptr, z.data.ptr, cp.uint64(size))
stream1.sync()
assert cp.allclose(z, x -y)

print('done')
