# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import os
import sys

from cuda.core.experimental import Device, LaunchConfig, Program, launch

# prepare include
cuda_path = os.environ.get("CUDA_PATH", os.environ.get("CUDA_HOME"))
if cuda_path is None:
    print("this demo requires a valid CUDA_PATH environment variable set", file=sys.stderr)
    sys.exit(0)
cuda_include_path = os.path.join(cuda_path, "include")

# print cluster info using a kernel
code = r"""
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

extern "C"
__global__ void check_cluster_info() {
    auto g = cg::this_grid();
    auto b = cg::this_thread_block();
    if (g.cluster_rank() == 0 && g.block_rank() == 0 && g.thread_rank() == 0) {
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
prog = Program(code, code_type="c++")
mod = prog.compile(
    target_type="cubin",
    # TODO: update this after NVIDIA/cuda-python#237 is merged
    options=(f"-arch=sm_{arch}", "-std=c++17", f"-I{cuda_include_path}"),
)
ker = mod.get_kernel("check_cluster_info")

# prepare launch config
grid = 4
cluster = 2
block = 32
config = LaunchConfig(grid=grid, cluster=cluster, block=block, stream=dev.default_stream)

# launch kernel on the default stream
launch(ker, config)
dev.sync()

print("done!")
