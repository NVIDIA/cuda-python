# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import cupy as cp
import numpy as np

from cuda.core.experimental import (
    Device,
    DeviceMemoryResource,
    LaunchConfig,
    LegacyPinnedMemoryResource,
    Program,
    ProgramOptions,
    launch,
)
from cuda.core.experimental._dlpack import DLDeviceType

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

# Compile kernel
arch = "".join(f"{i}" for i in dev.compute_capability)
program_options = ProgramOptions(std="c++17", arch=f"sm_{arch}")
prog = Program(code, code_type="c++", options=program_options)
mod = prog.compile("cubin")
kernel = mod.get_kernel("memory_ops")

# Create different memory resources
device_mr = DeviceMemoryResource(dev.device_id)
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

# Demonstrate DLPack integration
print("\nDLPack device information:")
print(f"Device buffer DLPack device: {device_buffer.__dlpack_device__()}")
print(f"Pinned buffer DLPack device: {pinned_buffer.__dlpack_device__()}")

# Assert DLPack device types
device_dlpack = device_buffer.__dlpack_device__()
pinned_dlpack = pinned_buffer.__dlpack_device__()

assert device_dlpack[0] == DLDeviceType.kDLCUDA, "Device buffer should have CUDA device type"
assert pinned_dlpack[0] == DLDeviceType.kDLCUDAHost, "Pinned buffer should have CUDA host device type"

# Clean up
device_buffer.close(stream)
pinned_buffer.close(stream)
new_device_buffer.close(stream)
stream.close()

# Verify buffers are properly closed
assert device_buffer.handle == 0, "Device buffer should be closed"
assert pinned_buffer.handle == 0, "Pinned buffer should be closed"
assert new_device_buffer.handle == 0, "New device buffer should be closed"

print("Memory management example completed!")
