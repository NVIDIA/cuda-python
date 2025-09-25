# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# ################################################################################
#
# This demo illustrates how to use `cuda.core` to compile and launch a simple
# vector addition kernel.
#
# ################################################################################

import cupy as cp
from cuda.core.experimental import Device, LaunchConfig, Program, ProgramOptions, launch

# compute c = a + b
code = """
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


dev = Device()
dev.set_current()
s = dev.create_stream()

# prepare program
program_options = ProgramOptions(std="c++17", arch=f"sm_{dev.arch}")
prog = Program(code, code_type="c++", options=program_options)
mod = prog.compile("cubin", name_expressions=("vector_add<float>",))

# run in single precision
ker = mod.get_kernel("vector_add<float>")
dtype = cp.float32

# prepare input/output
size = 50000
rng = cp.random.default_rng()
a = rng.random(size, dtype=dtype)
b = rng.random(size, dtype=dtype)
c = cp.empty_like(a)

# cupy runs on a different stream from s, so sync before accessing
dev.sync()

# prepare launch
block = 256
grid = (size + block - 1) // block
config = LaunchConfig(grid=grid, block=block)

# launch kernel on stream s
launch(s, config, ker, a.data.ptr, b.data.ptr, c.data.ptr, cp.uint64(size))
s.sync()

# check result
assert cp.allclose(c, a + b)
print("done!")
