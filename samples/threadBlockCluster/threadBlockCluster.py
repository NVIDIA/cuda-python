# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
# dependencies = ["cuda-python>=13.0.0", "cuda-core>=1.0.0", "numpy>=2.3.2"]
# ///

"""
Thread Block Clusters with cuda.core

Thread Block Clusters are a Hopper-class hardware feature (Compute
Capability 9.0+): a group of thread blocks that can share distributed
shared memory and coordinate through cooperative_groups.

This sample:

  * Compiles a kernel that queries ``cg::this_grid().dim_blocks()``,
    ``dim_clusters()``, and ``cg::this_thread_block().dim_threads()``.
  * Launches it with ``LaunchConfig(grid=..., cluster=..., block=...)``.
  * Reads the reported dimensions back through host-visible pinned memory.
  * Verifies that ``LaunchConfig(grid=G, cluster=C, block=B)`` produces
    ``G * C`` total blocks arranged as ``G`` clusters of ``C`` blocks each.

Waives with exit code 2 when:

  * the current device's compute capability is below 9.0, or
  * ``CUDA_PATH`` / ``CUDA_HOME`` is not set (needed to locate
    ``cooperative_groups.h``).
"""

import os
import sys

try:
    import numpy as np

    from cuda.core import (
        Device,
        LaunchConfig,
        LegacyPinnedMemoryResource,
        Program,
        ProgramOptions,
        launch,
    )
    from cuda.pathfinder import get_cuda_path_or_home
except ImportError as e:
    print(f"Error: Required package not found: {e}")
    print("Please install from requirements.txt:")
    print("  pip install -r requirements.txt")
    sys.exit(1)


EXIT_WAIVED = 2


CLUSTER_INFO_KERNEL = r"""
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

extern "C"
__global__ void check_cluster_info(unsigned int* grid_dims,
                                   unsigned int* cluster_dims,
                                   unsigned int* block_dims) {
    auto g = cg::this_grid();
    auto b = cg::this_thread_block();

    // Only one thread of the launch writes the results.
    if (g.cluster_rank() == 0 && g.block_rank() == 0 && g.thread_rank() == 0) {
        grid_dims[0] = g.dim_blocks().x;
        grid_dims[1] = g.dim_blocks().y;
        grid_dims[2] = g.dim_blocks().z;

        cluster_dims[0] = g.dim_clusters().x;
        cluster_dims[1] = g.dim_clusters().y;
        cluster_dims[2] = g.dim_clusters().z;

        block_dims[0] = b.dim_threads().x;
        block_dims[1] = b.dim_threads().y;
        block_dims[2] = b.dim_threads().z;

        printf("grid dim: (%u, %u, %u)\n", g.dim_blocks().x, g.dim_blocks().y, g.dim_blocks().z);
        printf("cluster dim: (%u, %u, %u)\n", g.dim_clusters().x, g.dim_clusters().y, g.dim_clusters().z);
        printf("block dim: (%u, %u, %u)\n", b.dim_threads().x, b.dim_threads().y, b.dim_threads().z);
    }
}
"""


def _find_include_paths():
    """Resolve include paths for cooperative_groups.h (and optional CCCL)."""
    cuda_path = get_cuda_path_or_home()
    if cuda_path is None:
        return None
    cuda_include = os.path.join(cuda_path, "include")
    if not os.path.isdir(cuda_include):
        return None
    paths = [cuda_include]
    cccl_include = os.path.join(cuda_include, "cccl")
    if os.path.isdir(cccl_include):
        paths.insert(0, cccl_include)
    return paths


def main():
    if np.lib.NumpyVersion(np.__version__) < "2.2.5":
        print("This sample requires NumPy 2.2.5 or later.", file=sys.stderr)
        sys.exit(EXIT_WAIVED)

    include_path = _find_include_paths()
    if include_path is None:
        print(
            "Could not locate a CUDA include directory. "
            "Set CUDA_PATH or CUDA_HOME to a CUDA toolkit installation. Waiving.",
            file=sys.stderr,
        )
        sys.exit(EXIT_WAIVED)

    dev = Device()
    arch = dev.compute_capability
    if arch < (9, 0):
        print(
            f"This sample requires compute capability >= 9.0 "
            f"(found sm_{arch[0]}{arch[1]}). Thread Block Clusters are Hopper+. Waiving.",
            file=sys.stderr,
        )
        sys.exit(EXIT_WAIVED)

    dev.set_current()
    arch_str = f"{arch[0]}{arch[1]}"

    prog = Program(
        CLUSTER_INFO_KERNEL,
        code_type="c++",
        options=ProgramOptions(arch=f"sm_{arch_str}", std="c++17", include_path=include_path),
    )
    mod = prog.compile(target_type="cubin")
    kernel = mod.get_kernel("check_cluster_info")

    # LaunchConfig(grid=G, cluster=C, block=B) → G clusters of C blocks
    # each = G*C total blocks; each block launches B threads.
    grid = 4
    cluster = 2
    block = 32
    config = LaunchConfig(grid=grid, cluster=cluster, block=block)

    # Pinned host-visible memory for the kernel's output.
    pinned_mr = LegacyPinnedMemoryResource()
    element_size = np.dtype(np.uint32).itemsize
    grid_buffer = cluster_buffer = block_buffer = None
    try:
        grid_buffer = pinned_mr.allocate(3 * element_size)
        cluster_buffer = pinned_mr.allocate(3 * element_size)
        block_buffer = pinned_mr.allocate(3 * element_size)

        grid_dims = np.from_dlpack(grid_buffer).view(dtype=np.uint32)
        cluster_dims = np.from_dlpack(cluster_buffer).view(dtype=np.uint32)
        block_dims = np.from_dlpack(block_buffer).view(dtype=np.uint32)

        grid_dims[:] = 0
        cluster_dims[:] = 0
        block_dims[:] = 0

        launch(dev.default_stream, config, kernel, grid_buffer, cluster_buffer, block_buffer)
        dev.sync()

        print("\nResults stored in pinned memory:")
        print(f"  Grid dimensions (blocks):   {tuple(grid_dims)}")
        print(f"  Cluster dimensions:         {tuple(cluster_dims)}")
        print(f"  Block dimensions (threads): {tuple(block_dims)}")

        expected_grid_blocks = grid * cluster  # 4 * 2 = 8
        actual_grid_blocks = int(grid_dims[0])
        assert actual_grid_blocks == expected_grid_blocks, (
            f"Grid conversion failed: expected {expected_grid_blocks} total blocks, got {actual_grid_blocks}"
        )

        print(f"\nLaunchConfig(grid={grid}, cluster={cluster}) produced {actual_grid_blocks} total blocks as expected.")
        print("Done")
        return 0
    finally:
        if block_buffer is not None:
            block_buffer.close()
        if cluster_buffer is not None:
            cluster_buffer.close()
        if grid_buffer is not None:
            grid_buffer.close()


if __name__ == "__main__":
    sys.exit(main())
