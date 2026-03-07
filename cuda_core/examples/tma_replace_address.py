# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# ################################################################################
#
# This example demonstrates how to use replace_address() to repoint a TMA
# (Tensor Memory Accelerator) descriptor at a different tensor without
# rebuilding the descriptor from scratch.
#
# The workflow is:
#
#   1. Create a TMA tiled descriptor and launch a kernel to verify it works
#   2. Allocate a second tensor with different content
#   3. Call replace_address() to repoint the same descriptor at the new tensor
#   4. Re-launch the kernel and verify it reads from the new tensor
#
# This is useful when the tensor layout (shape, dtype, tile size) stays the
# same but the underlying data buffer changes, e.g. double-buffering or
# iterating over a sequence of same-shaped tensors.
#
# Requirements:
#   - Hopper or later GPU (compute capability >= 9.0)
#   - CuPy
#   - CUDA toolkit headers (CUDA_PATH or CUDA_HOME set)
#
# ################################################################################

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

# ---------------------------------------------------------------------------
# Check for Hopper+ GPU
# ---------------------------------------------------------------------------
dev = Device()
arch = dev.compute_capability
if arch < (9, 0):
    print(
        "TMA requires compute capability >= 9.0 (Hopper or later)",
        file=sys.stderr,
    )
    sys.exit(0)
dev.set_current()

arch_str = "".join(f"{i}" for i in arch)

# ---------------------------------------------------------------------------
# CUDA kernel that uses TMA to load a 1-D tile into shared memory, then
# copies the tile to an output buffer so we can verify correctness.
#
# The CUtensorMap struct (128 bytes) is defined inline so the kernel can be
# compiled with NVRTC without pulling in the full driver-API header.
#
# Key points:
#   - The tensor map is passed by value with __grid_constant__ so the TMA
#     hardware can read it from grid-constant memory.
#   - Thread 0 in each block issues the TMA load and manages the mbarrier.
#   - All threads wait on the mbarrier, then copy from shared to global.
# ---------------------------------------------------------------------------
TILE_SIZE = 128  # elements per tile (must match the kernel constant)

code = r"""
// Minimal definition of the 128-byte opaque tensor map struct.
struct __align__(64) TensorMap { unsigned long long opaque[16]; };

static constexpr int TILE_SIZE = 128;

extern "C"
__global__ void tma_copy(
    const __grid_constant__ TensorMap tensor_map,
    float* output,
    int N)
{
    __shared__ __align__(128) float smem[TILE_SIZE];
    __shared__ __align__(8) unsigned long long mbar;

    const int tid        = threadIdx.x;
    const int tile_start = blockIdx.x * TILE_SIZE;

    // ---- Thread 0: set up mbarrier and issue the TMA load ----
    if (tid == 0)
    {
        // Initialise a single-phase mbarrier (1 arriving thread).
        asm volatile(
            "mbarrier.init.shared.b64 [%0], 1;"
            :: "r"((unsigned)__cvta_generic_to_shared(&mbar)));

        // Ask TMA to copy TILE_SIZE floats starting at element 'tile_start'
        // from the tensor described by 'tensor_map' into shared memory.
        asm volatile(
            "cp.async.bulk.tensor.1d.shared::cluster.global.tile"
            ".mbarrier::complete_tx::bytes"
            " [%0], [%1, {%2}], [%3];"
            :: "r"((unsigned)__cvta_generic_to_shared(smem)),
               "l"(&tensor_map),
               "r"(tile_start),
               "r"((unsigned)__cvta_generic_to_shared(&mbar)));

        // Tell the mbarrier how many bytes the TMA will deliver.
        asm volatile(
            "mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
            :: "r"((unsigned)__cvta_generic_to_shared(&mbar)),
               "r"((unsigned)(TILE_SIZE * sizeof(float))));
    }

    __syncthreads();

    // ---- Wait for the TMA load to complete ----
    if (tid == 0)
    {
        asm volatile(
            "{ .reg .pred P;                                           \n"
            "WAIT:                                                     \n"
            "  mbarrier.try_wait.parity.shared.b64 P, [%0], 0;         \n"
            "  @!P bra WAIT;                                           \n"
            "}                                                         \n"
            :: "r"((unsigned)__cvta_generic_to_shared(&mbar)));
    }

    __syncthreads();

    // ---- Copy the tile from shared memory to the output buffer ----
    if (tid < TILE_SIZE)
    {
        const int idx = tile_start + tid;
        if (idx < N)
            output[idx] = smem[tid];
    }
}
"""

# ---------------------------------------------------------------------------
# Compile the kernel
# ---------------------------------------------------------------------------
prog = Program(
    code,
    code_type="c++",
    options=ProgramOptions(std="c++17", arch=f"sm_{arch_str}"),
)
mod = prog.compile("cubin")
ker = mod.get_kernel("tma_copy")

# ---------------------------------------------------------------------------
# 1) Prepare input data and verify the initial TMA copy
# ---------------------------------------------------------------------------
N = 1024
a = cp.arange(N, dtype=cp.float32)  # [0, 1, 2, ..., N-1]
output = cp.zeros(N, dtype=cp.float32)
dev.sync()  # cupy uses its own stream

tensor_map = StridedMemoryView.from_any_interface(a, stream_ptr=-1).as_tensor_map(box_dim=(TILE_SIZE,))

n_tiles = N // TILE_SIZE
config = LaunchConfig(grid=n_tiles, block=TILE_SIZE)
launch(dev.default_stream, config, ker, tensor_map, output.data.ptr, np.int32(N))
dev.sync()

assert cp.array_equal(output, a), "TMA copy produced incorrect results"
print(f"TMA copy verified: {N} elements across {n_tiles} tiles")

# ---------------------------------------------------------------------------
# 2) Demonstrate replace_address()
#    Create a second tensor with different content, point the *same*
#    descriptor at it, and re-launch without rebuilding the descriptor.
# ---------------------------------------------------------------------------
b = cp.full(N, fill_value=42.0, dtype=cp.float32)
dev.sync()

tensor_map.replace_address(b)

output2 = cp.zeros(N, dtype=cp.float32)
dev.sync()

launch(dev.default_stream, config, ker, tensor_map, output2.data.ptr, np.int32(N))
dev.sync()

assert cp.array_equal(output2, b), "replace_address produced incorrect results"
print("replace_address verified: descriptor reused with new source tensor")
