# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# ################################################################################
#
# This demo illustrates:
#
#   1. How to use different memory resources to allocate and manage memory
#   2. How to copy data between different memory types
#   3. How to use DLPack to interoperate with other libraries
#
# ################################################################################

import sys

import cupy as cp
import numpy as np

from cuda.core.experimental import (
    Device,
    LaunchConfig,
    LegacyPinnedMemoryResource,
    Program,
    ProgramOptions,
    launch,
)

if np.__version__ < "2.1.0":
    print("This example requires NumPy 2.1.0 or later", file=sys.stderr)
    sys.exit(0)

# Kernel for memory operations
code = """
extern "C"
__global__ void memory_ops(float* device_data,
                          float* pinned_data,
                          size_t N) {
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        // Access device memory
        device_data[tid] = device_data[tid] + 1.0f;

        // Access pinned memory (zero-copy from GPU)
        pinned_data[tid] = pinned_data[tid] * 3.0f;
    }
}
"""

dev = Device()
dev.set_current()
stream = dev.create_stream()
# tell CuPy to use our stream as the current stream:
cp.cuda.ExternalStream(int(stream.handle)).use()

# Compile kernel
program_options = ProgramOptions(std="c++17", arch=f"sm_{dev.arch}")
prog = Program(code, code_type="c++", options=program_options)
mod = prog.compile("cubin")
kernel = mod.get_kernel("memory_ops")

# Create different memory resources
device_mr = dev.memory_resource
pinned_mr = LegacyPinnedMemoryResource()

# Allocate different types of memory
size = 1024
dtype = cp.float32
element_size = dtype().itemsize
total_size = size * element_size

# 1. Device Memory (GPU-only)
device_buffer = device_mr.allocate(total_size, stream=stream)
device_array = cp.from_dlpack(device_buffer).view(dtype=dtype)

# 2. Pinned Memory (CPU memory, GPU accessible)
pinned_buffer = pinned_mr.allocate(total_size, stream=stream)
pinned_array = np.from_dlpack(pinned_buffer).view(dtype=dtype)

# Initialize data
rng = cp.random.default_rng()
device_array[:] = rng.random(size, dtype=dtype)
pinned_array[:] = rng.random(size, dtype=dtype).get()

# Store original values for verification
device_original = device_array.copy()
pinned_original = pinned_array.copy()

# Sync before kernel launch
stream.sync()

# Launch kernel
block = 256
grid = (size + block - 1) // block
config = LaunchConfig(grid=grid, block=block)

launch(stream, config, kernel, device_buffer, pinned_buffer, cp.uint64(size))
stream.sync()

# Verify kernel operations
assert cp.allclose(device_array, device_original + 1.0), "Device memory operation failed"
assert cp.allclose(pinned_array, pinned_original * 3.0), "Pinned memory operation failed"

# Copy data between different memory types
print("\nCopying data between memory types...")

# Copy from device to pinned memory
device_buffer.copy_to(pinned_buffer, stream=stream)
stream.sync()

# Verify the copy operation
assert cp.allclose(pinned_array, device_array), "Device to pinned copy failed"

# Create a new device buffer and copy from pinned
new_device_buffer = device_mr.allocate(total_size, stream=stream)
new_device_array = cp.from_dlpack(new_device_buffer).view(dtype=dtype)

pinned_buffer.copy_to(new_device_buffer, stream=stream)
stream.sync()

# Verify the copy operation
assert cp.allclose(new_device_array, pinned_array), "Pinned to device copy failed"

# Clean up
device_buffer.close(stream)
pinned_buffer.close(stream)
new_device_buffer.close(stream)
stream.close()
cp.cuda.Stream.null.use()  # reset CuPy's current stream to the null stream

# Verify buffers are properly closed
assert device_buffer.handle is None, "Device buffer should be closed"
assert pinned_buffer.handle is None, "Pinned buffer should be closed"
assert new_device_buffer.handle is None, "New device buffer should be closed"

print("Memory management example completed!")
