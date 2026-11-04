# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
# dependencies = ["cuda-core[cu13]>=1.2.2", "cupy-cuda13x>=14.0.0", "numpy>=2.2.5"]
# ///

"""Run CuPy directly on host memory on Grace Hopper.

On Grace Hopper, NVLink-C2C makes CPU memory coherently accessible to the GPU,
and the GPU translates addresses through the host page tables. A kernel can
therefore read and write pinned, managed and even ordinary pageable host memory
in place, and the CPU sees the results without a copy.

The sample allocates one buffer from each cuda.core memory resource (device,
pinned on the GPU's nearest host NUMA node, and managed) plus a plain NumPy
array. It wraps each one as a CuPy array, runs the same CuPy operations on all
of them and checks the results on the CPU.
"""

import os
import sys

try:
    import cupy as cp
    import numpy as np

    from cuda.core import (
        Device,
        DeviceMemoryResource,
        ManagedMemoryResource,
        PinnedMemoryResource,
        PinnedMemoryResourceOptions,
    )
except ImportError as e:
    print(f"Error: Required package not found: {e}")
    print("Install with: pip install -r requirements.txt")
    sys.exit(1)

EXIT_WAIVED = int(os.environ.get("CUDA_PYTHON_SAMPLE_WAIVER_EXIT_CODE", "2"))
N = 1 << 24  # 64 MiB of float32 values per buffer


def host_as_cupy(ptr, nbytes, owner, device_id):
    # cp.from_dlpack() does not import host memory (kDLCUDAHost or kDLCPU), so
    # wrap the pointer directly. owner keeps the memory alive.
    memory = cp.cuda.UnownedMemory(ptr, nbytes, owner, device_id=device_id)
    return cp.ndarray((nbytes // 4,), cp.float32, cp.cuda.MemoryPointer(memory, 0))


def run(label, gpu_array, cpu_array):
    """Fill on the CPU, compute on the GPU, check on the CPU."""
    cpu_array[:] = np.arange(N, dtype=np.float32) % 1024
    gpu_array *= 2
    total = int(gpu_array.sum(dtype=cp.float64))
    cp.cuda.get_current_stream().synchronize()

    expected = 2 * (np.arange(N, dtype=np.float32) % 1024)
    if total != int(expected.sum(dtype=np.float64)) or not np.array_equal(cpu_array, expected):
        raise RuntimeError(f"{label}: result mismatch")
    print(f"  {label:<38} sum={total} verified on the CPU without a copy")


def main():
    dev = Device(0)
    dev.set_current()
    props = dev.properties
    cc = dev.compute_capability
    print(f"Device: {dev.name}, compute capability {cc.major}.{cc.minor}")
    print(f"  host_numa_id: {props.host_numa_id}")
    print(f"  pageable_memory_access_uses_host_page_tables: {props.pageable_memory_access_uses_host_page_tables}")
    print(f"  concurrent_managed_access: {props.concurrent_managed_access}")
    print(f"  host_numa_memory_pools_supported: {props.host_numa_memory_pools_supported}")

    if not (
        props.pageable_memory_access_uses_host_page_tables
        and props.concurrent_managed_access
        and props.host_numa_memory_pools_supported
        and props.host_numa_id >= 0
    ):
        print("This sample requires a Grace Hopper class system; waiving.")
        return EXIT_WAIVED

    stream = dev.create_stream()
    nbytes = N * 4
    device_buffer = DeviceMemoryResource(dev).allocate(nbytes, stream=stream)
    pinned_buffer = PinnedMemoryResource(PinnedMemoryResourceOptions(numa_id=props.host_numa_id)).allocate(
        nbytes, stream=stream
    )
    managed_buffer = ManagedMemoryResource().allocate(nbytes, stream=stream)
    pageable = np.empty(N, dtype=np.float32)
    # Allocations are ordered on stream; CuPy uses its own stream.
    stream.sync()

    try:
        print("GPU work on host-resident memory:")
        run(
            f"pinned, host NUMA node {props.host_numa_id}",
            host_as_cupy(pinned_buffer.handle, nbytes, pinned_buffer, dev.device_id),
            np.from_dlpack(pinned_buffer).view(np.float32),
        )
        run(
            "managed",
            cp.from_dlpack(managed_buffer).view(cp.float32),
            np.from_dlpack(managed_buffer).view(np.float32),
        )
        run(
            "pageable NumPy array",
            host_as_cupy(pageable.ctypes.data, nbytes, pageable, dev.device_id),
            pageable,
        )

        print("For comparison, device memory needs explicit copies:")
        device_array = cp.from_dlpack(device_buffer).view(cp.float32)
        device_array.set(np.arange(N, dtype=np.float32) % 1024)
        device_array *= 2
        if not np.array_equal(device_array.get(), 2 * (np.arange(N, dtype=np.float32) % 1024)):
            raise RuntimeError("device: result mismatch")
        print(f"  {'device':<38} verified after a device-to-host copy")
    finally:
        cp.cuda.get_current_stream().synchronize()
        for buffer in (managed_buffer, pinned_buffer, device_buffer):
            buffer.close(stream)
        stream.sync()

    print("Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
