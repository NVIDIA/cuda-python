# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# ################################################################################
#
# This demo illustrates the use of thread block clusters in the CUDA launch
# configuration and verifies that the correct grid size is passed to the kernel.
#
# ################################################################################

import os
import sys

import numpy as np
from cuda.core import (
    Device,
    LaunchConfig,
    LegacyPinnedMemoryResource,
    Program,
    ProgramOptions,
    launch,
)

if np.lib.NumpyVersion(np.__version__) < "2.2.5":
    print("This example requires NumPy 2.2.5 or later", file=sys.stderr)
    sys.exit(0)

# prepare include
cuda_path = os.environ.get("CUDA_PATH", os.environ.get("CUDA_HOME"))
if cuda_path is None:
    print("this demo requires a valid CUDA_PATH environment variable set", file=sys.stderr)
    sys.exit(0)
cuda_include = os.path.join(cuda_path, "include")
assert os.path.isdir(cuda_include)
include_path = [cuda_include]
cccl_include = os.path.join(cuda_include, "cccl")
if os.path.isdir(cccl_include):
    include_path.insert(0, cccl_include)

# print cluster info using a kernel and store results in pinned memory
code = r"""
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

extern "C"
__global__ void check_cluster_info(unsigned int* grid_dims, unsigned int* cluster_dims, unsigned int* block_dims) {
    auto g = cg::this_grid();
    auto b = cg::this_thread_block();

    if (g.cluster_rank() == 0 && g.block_rank() == 0 && g.thread_rank() == 0) {
        // Store grid dimensions (in blocks)
        grid_dims[0] = g.dim_blocks().x;
        grid_dims[1] = g.dim_blocks().y;
        grid_dims[2] = g.dim_blocks().z;

        // Store cluster dimensions
        cluster_dims[0] = g.dim_clusters().x;
        cluster_dims[1] = g.dim_clusters().y;
        cluster_dims[2] = g.dim_clusters().z;

        // Store block dimensions (in threads)
        block_dims[0] = b.dim_threads().x;
        block_dims[1] = b.dim_threads().y;
        block_dims[2] = b.dim_threads().z;

        // Also print to console
        printf("grid dim: (%u, %u, %u)\n", g.dim_blocks().x, g.dim_blocks().y, g.dim_blocks().z);
        printf("cluster dim: (%u, %u, %u)\n", g.dim_clusters().x, g.dim_clusters().y, g.dim_clusters().z);
        printf("block dim: (%u, %u, %u)\n", b.dim_threads().x, b.dim_threads().y, b.dim_threads().z);
    }
}
"""

dev = Device()
arch = dev.compute_capability
if arch < (9, 0):
    print(
        "this demo requires compute capability >= 9.0 (since thread block cluster is a hardware feature)",
        file=sys.stderr,
    )
    sys.exit(0)
arch = "".join(f"{i}" for i in arch)

# prepare program & compile kernel
dev.set_current()
prog = Program(
    code,
    code_type="c++",
    options=ProgramOptions(arch=f"sm_{arch}", std="c++17", include_path=include_path),
)
mod = prog.compile(target_type="cubin")
ker = mod.get_kernel("check_cluster_info")

# prepare launch config
grid = 4
cluster = 2
block = 32
config = LaunchConfig(grid=grid, cluster=cluster, block=block)

# allocate pinned memory to store kernel results
pinned_mr = LegacyPinnedMemoryResource()
element_size = np.dtype(np.uint32).itemsize

# allocate 3 uint32 values each for grid, cluster, and block dimensions
grid_buffer = pinned_mr.allocate(3 * element_size)
cluster_buffer = pinned_mr.allocate(3 * element_size)
block_buffer = pinned_mr.allocate(3 * element_size)

# create NumPy arrays from the pinned memory
grid_dims = np.from_dlpack(grid_buffer).view(dtype=np.uint32)
cluster_dims = np.from_dlpack(cluster_buffer).view(dtype=np.uint32)
block_dims = np.from_dlpack(block_buffer).view(dtype=np.uint32)

# initialize arrays to zero
grid_dims[:] = 0
cluster_dims[:] = 0
block_dims[:] = 0

# launch kernel on the default stream
launch(dev.default_stream, config, ker, grid_buffer, cluster_buffer, block_buffer)
dev.sync()

# verify results
print("\nResults stored in pinned memory:")
print(f"Grid dimensions (blocks): {tuple(grid_dims)}")
print(f"Cluster dimensions: {tuple(cluster_dims)}")
print(f"Block dimensions (threads): {tuple(block_dims)}")

# verify that grid conversion worked correctly:
# LaunchConfig(grid=4, cluster=2) should result in 8 total blocks (4 clusters * 2 blocks/cluster)
expected_grid_blocks = grid * cluster  # 4 * 2 = 8
actual_grid_blocks = grid_dims[0]

print("\nVerification:")
print(f"LaunchConfig specified: grid={grid} clusters, cluster={cluster} blocks/cluster")
print(f"Expected total blocks: {expected_grid_blocks}")
print(f"Actual total blocks: {actual_grid_blocks}")

if actual_grid_blocks == expected_grid_blocks:
    print("✓ Grid conversion is correct!")
else:
    print("✗ Grid conversion failed!")
    sys.exit(1)

print("done!")
