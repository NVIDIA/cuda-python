# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# ################################################################################
#
# This example demonstrates how to use TMA (Tensor Memory Accelerator) descriptors
# with cuda.core on Hopper+ GPUs (compute capability >= 9.0).
#
# TMA enables efficient bulk data movement between global and shared memory using
# hardware-managed tensor map descriptors. This example shows:
#
#   1. Creating a TMA tiled descriptor from a CuPy device array
#   2. Passing the descriptor to a kernel via launch()
#   3. Using TMA to load tiles into shared memory (via inline PTX)
#   4. Updating the descriptor's source address with replace_address()
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
    TensorMapDescriptor,
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
# 1) Prepare input data on the device
# ---------------------------------------------------------------------------
N = 1024
a = cp.arange(N, dtype=cp.float32)  # [0, 1, 2, ..., N-1]
output = cp.zeros(N, dtype=cp.float32)
dev.sync()  # cupy uses its own stream

# ---------------------------------------------------------------------------
# 2) Create a TMA tiled descriptor
#    from_tiled() accepts any DLPack / __cuda_array_interface__ object.
#    The dtype (float32) is inferred automatically from the CuPy array.
# ---------------------------------------------------------------------------
tensor_map = TensorMapDescriptor.from_tiled(a, box_dim=(TILE_SIZE,))

# ---------------------------------------------------------------------------
# 3) Launch the kernel
#    The TensorMapDescriptor is passed directly as a kernel argument — the
#    128-byte struct is copied into kernel parameter space automatically.
# ---------------------------------------------------------------------------
n_tiles = N // TILE_SIZE
config = LaunchConfig(grid=n_tiles, block=TILE_SIZE)
launch(dev.default_stream, config, ker, tensor_map, output.data.ptr, np.int32(N))
dev.sync()

assert cp.array_equal(output, a), "TMA copy produced incorrect results"
print(f"TMA copy verified: {N} elements across {n_tiles} tiles")

# ---------------------------------------------------------------------------
# 4) Demonstrate replace_address()
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
