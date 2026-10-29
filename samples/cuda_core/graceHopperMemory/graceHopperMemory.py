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
# dependencies = ["cuda-python>=13.0.0", "cuda-core>=1.3.0", "numpy>=2.2.5"]
# ///

"""Compare GPU reads from host, device, and managed memory on Grace Hopper."""

import os
import sys
import time

import numpy as np

from cuda.core import (
    Device,
    DeviceMemoryResource,
    Host,
    LaunchConfig,
    ManagedMemoryResource,
    PinnedMemoryResource,
    PinnedMemoryResourceOptions,
    Program,
    ProgramOptions,
    launch,
)

EXIT_WAIVED = int(os.environ.get("CUDA_PYTHON_SAMPLE_WAIVER_EXIT_CODE", "2"))
N = 1 << 28  # 1 GiB of float32 values
REPS = 10

SOURCE = r"""
extern "C" __global__ void read_sum(const float *x, unsigned long long n, double *out) {
    __shared__ float part[512];
    float acc = 0.f;
    for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; i < n;
         i += (unsigned long long)gridDim.x * blockDim.x)
        acc += x[i];
    part[threadIdx.x] = acc;
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) part[threadIdx.x] += part[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(out, (double)part[0]);
}
extern "C" __global__ void scale(float *x, unsigned long long n, float a) {
    for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; i < n;
         i += (unsigned long long)gridDim.x * blockDim.x)
        x[i] *= a;
}
extern "C" __global__ void fill(float *x, unsigned long long n, float v) {
    for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; i < n;
         i += (unsigned long long)gridDim.x * blockDim.x)
        x[i] = v;
}
extern "C" __global__ void zero(double *out) { *out = 0.0; }
"""


def main():
    dev = Device(0)
    dev.set_current()
    props = dev.properties
    print(f"Device: {dev.name}, compute capability {dev.compute_capability}")
    for name in (
        "numa_config",
        "numa_id",
        "host_numa_id",
        "pageable_memory_access",
        "pageable_memory_access_uses_host_page_tables",
        "concurrent_managed_access",
    ):
        print(f"  {name}: {getattr(props, name)}")
    print(f"  system.Device.is_c2c_enabled: {dev.to_system_device().is_c2c_enabled}")

    if not props.pageable_memory_access_uses_host_page_tables:
        print("Host page tables are unavailable for pageable GPU access; waiving sample.")
        return EXIT_WAIVED
    if (
        not props.concurrent_managed_access
        or not props.host_memory_pools_supported
        or not props.host_numa_memory_pools_supported
        or props.host_numa_id < 0
    ):
        print("Managed access, host memory pools, or host NUMA support is unavailable; waiving sample.")
        return EXIT_WAIVED

    stream = dev.create_stream()
    arch = f"sm_{dev.compute_capability.major}{dev.compute_capability.minor}"
    module = Program(SOURCE, code_type="c++", options=ProgramOptions(arch=arch)).compile("cubin")
    read_sum, scale, fill, zero = (module.get_kernel(name) for name in ("read_sum", "scale", "fill", "zero"))
    config = LaunchConfig(grid=props.multiprocessor_count * 8, block=512)
    one = LaunchConfig(grid=1, block=1)
    allocations = []
    resources = []

    def allocate(resource, size):
        resources.append(resource)
        buffer = resource.allocate(size, stream=stream)
        allocations.append(buffer)
        return buffer

    try:
        result = allocate(DeviceMemoryResource(dev), 8)
        host_result = allocate(PinnedMemoryResource(), 8)

        def run_once(ptr, location=None):
            if location is not None:
                managed.prefetch(location, stream=stream)
                stream.sync()
            launch(stream, one, zero, result)
            stream.sync()
            start = time.perf_counter()
            launch(stream, config, read_sum, np.uint64(ptr), np.uint64(N), result)
            stream.sync()
            elapsed = time.perf_counter() - start
            result.copy_to(host_result, stream=stream)
            stream.sync()
            actual = float(np.from_dlpack(host_result).view(np.float64)[0])
            if actual != N:
                raise RuntimeError(f"read_sum returned {actual}, expected {N}")
            return elapsed

        def time_read(ptr, label, location=None):
            if location is None:
                run_once(ptr)
            durations = sorted(run_once(ptr, location) for _ in range(REPS))
            elapsed = (durations[REPS // 2 - 1] + durations[REPS // 2]) / 2
            print(f"  {label:<52} {N * 4 / elapsed / 1e9:8.1f} GB/s ({elapsed * 1e3:.2f} ms)")

        print("[1] In-place kernel on pageable NumPy memory")
        array = np.arange(1 << 20, dtype=np.float32)
        expected = array * 3.0
        launch(stream, config, scale, np.uint64(array.ctypes.data), np.uint64(array.size), np.float32(3.0))
        stream.sync()
        if not np.array_equal(array, expected):
            raise RuntimeError("Pageable array result does not match")
        print("  Result verified")

        print("[2] Read bandwidth, 1 GiB")
        pinned = allocate(PinnedMemoryResource(PinnedMemoryResourceOptions(numa_id=props.host_numa_id)), N * 4)
        stream.sync()
        np.from_dlpack(pinned).view(np.float32)[:] = 1.0
        time_read(pinned.handle, f"pinned, host NUMA node {props.host_numa_id}")
        pageable = np.ones(N, dtype=np.float32)
        time_read(pageable.ctypes.data, "pageable NumPy memory")
        device_buffer = allocate(DeviceMemoryResource(dev), N * 4)
        launch(stream, config, fill, device_buffer, np.uint64(N), np.float32(1.0))
        time_read(device_buffer.handle, "device memory")

        print("[3] Managed memory, first kernel after each prefetch")
        managed = allocate(ManagedMemoryResource(), N * 4)
        stream.sync()
        np.from_dlpack(managed).view(np.float32)[:] = 1.0
        time_read(managed.handle, f"managed, host NUMA node {props.host_numa_id}", Host(numa_id=props.host_numa_id))
        time_read(managed.handle, "managed, GPU", dev)
    finally:
        for buffer in reversed(allocations):
            buffer.close(stream)
        stream.sync()
    print("Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
