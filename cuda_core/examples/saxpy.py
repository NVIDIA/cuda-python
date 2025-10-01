# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# ################################################################################
#
# This demo illustrates how to use `cuda.core` to compile a templated CUDA kernel
# and launch it using `cupy` arrays as inputs. This is a simple example of a
# templated kernel, where the kernel is instantiated for both `float` and `double`
# data types.
#
# ################################################################################

import sys

import cupy as cp
from cuda.core.experimental import Device, LaunchConfig, Program, ProgramOptions, launch

# compute out = a * x + y
code = """
template<typename T>
__global__ void saxpy(const T a,
                      const T* x,
                      const T* y,
                      T* out,
                      size_t N) {
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (size_t i=tid; i<N; i+=gridDim.x*blockDim.x) {
        out[i] = a * x[i] + y[i];
    }
}
"""


dev = Device()
dev.set_current()
s = dev.create_stream()

# prepare program
program_options = ProgramOptions(std="c++11", arch=f"sm_{dev.arch}")
prog = Program(code, code_type="c++", options=program_options)

# Note the use of the `name_expressions` argument to specify the template
# instantiations of the kernel that we will use. For non-templated kernels,
# `name_expressions` will simply contain the name of the kernels.
mod = prog.compile(
    "cubin",
    logs=sys.stdout,
    name_expressions=("saxpy<float>", "saxpy<double>"),
)

# run in single precision
ker = mod.get_kernel("saxpy<float>")
dtype = cp.float32

# prepare input/output
size = cp.uint64(64)
a = dtype(10)
rng = cp.random.default_rng()
x = rng.random(size, dtype=dtype)
y = rng.random(size, dtype=dtype)
out = cp.empty_like(x)
dev.sync()  # cupy runs on a different stream from s, so sync before accessing

# prepare launch
block = 32
grid = int((size + block - 1) // block)
config = LaunchConfig(grid=grid, block=block)
ker_args = (a, x.data.ptr, y.data.ptr, out.data.ptr, size)

# launch kernel on stream s
launch(s, config, ker, *ker_args)
s.sync()

# check result
assert cp.allclose(out, a * x + y)

# let's repeat again, this time allocates our own out buffer instead of cupy's
# run in double precision
ker = mod.get_kernel("saxpy<double>")
dtype = cp.float64

# prepare input
size = cp.uint64(128)
a = dtype(42)
x = rng.random(size, dtype=dtype)
y = rng.random(size, dtype=dtype)
dev.sync()

# prepare output
buf = dev.allocate(
    size * 8,  # = dtype.itemsize
    stream=s,
)

# prepare launch
block = 64
grid = int((size + block - 1) // block)
config = LaunchConfig(grid=grid, block=block)
ker_args = (a, x.data.ptr, y.data.ptr, buf, size)

# launch kernel on stream s
launch(s, config, ker, *ker_args)
s.sync()

# check result
# we wrap output buffer as a cupy array for simplicity
out = cp.ndarray(
    size, dtype=dtype, memptr=cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(int(buf.handle), buf.size, buf), 0)
)
assert cp.allclose(out, a * x + y)

# clean up resources that we allocate
# cupy cleans up automatically the rest
buf.close(s)
s.close()

print("done!")
