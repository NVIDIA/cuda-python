# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# ################################################################################
#
# This example demonstrates how to use TMA (Tensor Memory Accelerator)
# descriptors with cuda.core on Hopper+ GPUs (compute capability >= 9.0).
#
# TMA enables efficient bulk data movement between global and shared memory
# using hardware-managed tensor map descriptors. This example shows:
#
#   1. Creating a TMA tiled descriptor from a CuPy device array
#   2. Passing the descriptor to a kernel via launch()
#   3. Using libcudacxx TMA/barrier wrappers instead of raw PTX
#   4. Reusing the same descriptor with replace_address()
#
# Requirements:
#   - Hopper or later GPU (compute capability >= 9.0)
#   - CuPy
#   - CUDA toolkit headers (CUDA_PATH or CUDA_HOME set)
#
# ################################################################################

import os
import sys

import cupy as cp
import numpy as np

from cuda.core import (
    Device,
    LaunchConfig,
    Program,
    ProgramOptions,
    StridedMemoryView,
    launch,
)
from cuda.pathfinder import get_cuda_path_or_home

# ---------------------------------------------------------------------------
# CUDA kernel that uses TMA to load a 1-D tile into shared memory, then
# copies the tile to an output buffer so we can verify correctness.
#
# The CUtensorMap struct (128 bytes) is defined inline so the kernel can be
# compiled with NVRTC without pulling in the full driver-API header. The
# kernel uses libcudacxx's `cuda::barrier` and TMA wrapper helpers rather
# than embedding raw PTX strings.
#
# Key points:
#   - The tensor map is passed by value with __grid_constant__ so the TMA
#     hardware can read it from grid-constant memory.
#   - Thread 0 in each block issues the TMA load and waits on the barrier.
#   - All threads synchronize before copying from shared to global memory.
# ---------------------------------------------------------------------------
TILE_SIZE = 128  # elements per tile (must match the kernel constant)

code = r"""
#include <cuda/barrier>

// Minimal definition of the 128-byte opaque tensor map struct.
struct __align__(64) TensorMap { unsigned long long opaque[16]; };

static constexpr int TILE_SIZE = 128;
using TmaBarrier = cuda::barrier<cuda::thread_scope_block>;

extern "C"
__global__ void tma_copy(
    const __grid_constant__ TensorMap tensor_map,
    float* output,
    int N)
{
    __shared__ __align__(128) float smem[TILE_SIZE];
    __shared__ TmaBarrier bar;

    const int tid        = threadIdx.x;
    const int tile_start = blockIdx.x * TILE_SIZE;

    if (tid == 0)
    {
        init(&bar, 1);
    }
    __syncthreads();

    if (tid == 0)
    {
        cuda::device::experimental::cp_async_bulk_tensor_1d_global_to_shared(
            smem,
            reinterpret_cast<const CUtensorMap*>(&tensor_map),
            tile_start,
            bar);
        bar.wait(cuda::device::barrier_arrive_tx(bar, 1, TILE_SIZE * sizeof(float)));
    }
    __syncthreads();

    if (tid < TILE_SIZE)
    {
        const int idx = tile_start + tid;
        if (idx < N)
            output[idx] = smem[tid];
    }
}
"""


def _get_cccl_include_paths():
    cuda_path = get_cuda_path_or_home()
    if cuda_path is None:
        print("This example requires CUDA_PATH or CUDA_HOME to point to a CUDA toolkit.", file=sys.stderr)
        sys.exit(1)

    cuda_include = os.path.join(cuda_path, "include")
    if not os.path.isdir(cuda_include):
        print(f"CUDA include directory not found: {cuda_include}", file=sys.stderr)
        sys.exit(1)

    include_path = [cuda_include]
    cccl_include = os.path.join(cuda_include, "cccl")
    if os.path.isdir(cccl_include):
        include_path.insert(0, cccl_include)
    return include_path


def main():
    # -----------------------------------------------------------------------
    # Check for Hopper+ GPU
    # -----------------------------------------------------------------------
    dev = Device()
    arch = dev.compute_capability
    if arch < (9, 0):
        print(
            "TMA requires compute capability >= 9.0 (Hopper or later)",
            file=sys.stderr,
        )
        sys.exit(0)
    dev.set_current()
    include_path = _get_cccl_include_paths()

    # -----------------------------------------------------------------------
    # Compile the kernel
    # -----------------------------------------------------------------------
    prog = Program(
        code,
        code_type="c++",
        options=ProgramOptions(std="c++17", arch=f"sm_{dev.arch}", include_path=include_path),
    )
    mod = prog.compile("cubin")
    ker = mod.get_kernel("tma_copy")

    # -----------------------------------------------------------------------
    # 1) Prepare input data and verify the initial TMA copy
    # -----------------------------------------------------------------------
    n = 1024
    src = cp.arange(n, dtype=cp.float32)
    output = cp.zeros(n, dtype=cp.float32)
    dev.sync()  # CuPy uses its own stream

    tensor_map = StridedMemoryView.from_any_interface(src, stream_ptr=-1).as_tensor_map(box_dim=(TILE_SIZE,))

    n_tiles = n // TILE_SIZE
    config = LaunchConfig(grid=n_tiles, block=TILE_SIZE)
    launch(dev.default_stream, config, ker, tensor_map, output.data.ptr, np.int32(n))
    dev.sync()

    assert cp.array_equal(output, src), "TMA copy produced incorrect results"
    print(f"TMA copy verified: {n} elements across {n_tiles} tiles")

    # -----------------------------------------------------------------------
    # 2) Demonstrate replace_address() without rebuilding the descriptor
    # -----------------------------------------------------------------------
    replacement = cp.full(n, fill_value=42.0, dtype=cp.float32)
    dev.sync()

    tensor_map.replace_address(replacement)

    output2 = cp.zeros(n, dtype=cp.float32)
    launch(dev.default_stream, config, ker, tensor_map, output2.data.ptr, np.int32(n))
    dev.sync()

    assert cp.array_equal(output2, replacement), "replace_address produced incorrect results"
    print("replace_address verified: descriptor reused with new source tensor")


if __name__ == "__main__":
    main()
