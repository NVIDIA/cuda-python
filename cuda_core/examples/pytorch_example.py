# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# ################################################################################
#
# This demo illustrates how to use `cuda.core` to compile a CUDA kernel
# and launch it using PyTorch tensors as inputs.
#
# ## Usage: pip install "cuda-core[cu12]"
# ## python pytorch_example.py
#
# ################################################################################

import sys

import torch
from cuda.core.experimental import Device, LaunchConfig, Program, ProgramOptions, launch

# SAXPY kernel - passing a as a pointer to avoid any type issues
code = """
template<typename T>
__global__ void saxpy_kernel(const T* a, const T* x, const T* y, T* out, size_t N) {
 const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
 if (tid < N) {
   // Dereference a to get the scalar value
   out[tid] = (*a) * x[tid] + y[tid];
 }
}
"""

dev = Device()
dev.set_current()

# Get PyTorch's current stream
pt_stream = torch.cuda.current_stream()
print(f"PyTorch stream: {pt_stream}")


# Create a wrapper class that implements __cuda_stream__
class PyTorchStreamWrapper:
    def __init__(self, pt_stream):
        self.pt_stream = pt_stream

    def __cuda_stream__(self):
        stream_id = self.pt_stream.cuda_stream
        return (0, stream_id)  # Return format required by CUDA Python


s = dev.create_stream(PyTorchStreamWrapper(pt_stream))

# prepare program
program_options = ProgramOptions(std="c++11", arch=f"sm_{dev.arch}")
prog = Program(code, code_type="c++", options=program_options)
mod = prog.compile(
    "cubin",
    logs=sys.stdout,
    name_expressions=("saxpy_kernel<float>", "saxpy_kernel<double>"),
)

# Run in single precision
ker = mod.get_kernel("saxpy_kernel<float>")
dtype = torch.float32

# prepare input/output
size = 64
# Use a single element tensor for 'a'
a = torch.tensor([10.0], dtype=dtype, device="cuda")
x = torch.rand(size, dtype=dtype, device="cuda")
y = torch.rand(size, dtype=dtype, device="cuda")
out = torch.empty_like(x)

# prepare launch
block = 32
grid = int((size + block - 1) // block)
config = LaunchConfig(grid=grid, block=block)
ker_args = (a.data_ptr(), x.data_ptr(), y.data_ptr(), out.data_ptr(), size)

# launch kernel on our stream
launch(s, config, ker, *ker_args)

# check result
assert torch.allclose(out, a.item() * x + y)
print("Single precision test passed!")

# let's repeat again with double precision
ker = mod.get_kernel("saxpy_kernel<double>")
dtype = torch.float64

# prepare input
size = 128
# Use a single element tensor for 'a'
a = torch.tensor([42.0], dtype=dtype, device="cuda")
x = torch.rand(size, dtype=dtype, device="cuda")
y = torch.rand(size, dtype=dtype, device="cuda")

# prepare output
out = torch.empty_like(x)

# prepare launch
block = 64
grid = int((size + block - 1) // block)
config = LaunchConfig(grid=grid, block=block)
ker_args = (a.data_ptr(), x.data_ptr(), y.data_ptr(), out.data_ptr(), size)

# launch kernel on PyTorch's stream
launch(s, config, ker, *ker_args)

# check result
assert torch.allclose(out, a * x + y)
print("Double precision test passed!")
print("All tests passed successfully!")
