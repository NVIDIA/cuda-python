# Sample: Thread Block Clusters (Python)

## Description

Thread Block Clusters are a hardware feature of Hopper-class (and newer)
GPUs — a Compute Capability 9.0+ concept that groups several thread blocks
into a *cluster* so they can share distributed shared memory and coordinate
through `cooperative_groups`.

`cuda.core.LaunchConfig` exposes this via a `cluster=` argument. This
sample:

1. Compiles a kernel that queries the cluster/grid/block dimensions from
   inside `cooperative_groups`.
2. Launches with `LaunchConfig(grid=4, cluster=2, block=32)`.
3. Reads the reported dimensions back through host-visible pinned memory
   (via `LegacyPinnedMemoryResource`).
4. Verifies that `LaunchConfig(grid=G, cluster=C, block=B)` produces
   `G * C` total blocks — `G` clusters of `C` blocks each.

The sample **waives itself** when:

- the current device has Compute Capability < 9.0, or
- neither `CUDA_PATH` nor `CUDA_HOME` points to a CUDA toolkit whose
  `include/` directory holds `cooperative_groups.h`.

## What You'll Learn

- Using `LaunchConfig(cluster=...)` for thread block cluster launches
- Reading `cg::this_grid().dim_clusters()` / `dim_blocks()` /
  `cg::this_thread_block().dim_threads()` inside a kernel
- Passing an `include_path` to `ProgramOptions` so NVRTC can find
  `cooperative_groups.h`
- Allocating pinned memory with `LegacyPinnedMemoryResource` and viewing it
  as a NumPy array via DLPack

## Key Libraries

- [`cuda.core`](https://nvidia.github.io/cuda-python/cuda-core/latest/) - `Device`, `Program`, `LaunchConfig`, `LegacyPinnedMemoryResource`, `launch`
- [`cuda.pathfinder`](https://nvidia.github.io/cuda-python/cuda-pathfinder/latest/) - `get_cuda_path_or_home` to locate the CUDA toolkit
- `numpy` (>=2.2.5) - viewing pinned buffers as arrays via DLPack

## Key APIs

### From `cuda.core`

- `LaunchConfig(grid=..., cluster=..., block=...)` - configure a
  cluster-aware launch
- `ProgramOptions(arch=..., std=..., include_path=...)` - hand NVRTC the
  include search paths for CG headers
- `LegacyPinnedMemoryResource` / `.allocate(nbytes)` - allocate host-visible
  pinned memory
- `launch`, `Device.default_stream`, `Device.sync()` - launch and wait

### Kernel-side (CUDA)

- `cg::this_grid().dim_blocks()`, `dim_clusters()`, `cluster_rank()`,
  `block_rank()`, `thread_rank()`
- `cg::this_thread_block().dim_threads()`

## Requirements

### Hardware

- **NVIDIA Hopper or newer GPU** (Compute Capability 9.0+). Ada / Ampere /
  Turing GPUs do not support thread block clusters and will waive the
  sample.

### Software

- CUDA Toolkit 13.0 or newer with `cooperative_groups.h` in its `include/`
  directory. Set `CUDA_PATH` or `CUDA_HOME`.
- Python 3.10 or newer
- `cuda-python` (>=13.0.0)
- `cuda-core` (>=1.0.0)
- `numpy` (>=2.2.5)

## Installation

```bash
pip install -r requirements.txt
```

## How to Run

```bash
export CUDA_HOME=/usr/local/cuda        # or wherever your CTK is installed
python threadBlockCluster.py
```

## Expected Output

On a Hopper-class GPU:

```
grid dim: (8, 1, 1)
cluster dim: (4, 1, 1)
block dim: (32, 1, 1)

Results stored in pinned memory:
  Grid dimensions (blocks):   (8, 1, 1)
  Cluster dimensions:         (4, 1, 1)
  Block dimensions (threads): (32, 1, 1)

LaunchConfig(grid=4, cluster=2) produced 8 total blocks as expected.
Done
```

On a pre-Hopper GPU (CC < 9.0) the sample self-waives:

```
This sample requires compute capability >= 9.0 (found sm_89). Thread Block Clusters are Hopper+. Waiving.
```

## Files

- `threadBlockCluster.py` - Python implementation
- `README.md` - This file
- `requirements.txt` - Sample dependencies

## See Also

- [CUDA Python Documentation](https://nvidia.github.io/cuda-python/)
- [`cuda.core` LaunchConfig API](https://nvidia.github.io/cuda-python/cuda-core/latest/api.html#cuda.core.LaunchConfig)
- [CUDA C++ Programming Guide — Thread Block Clusters](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-block-clusters)
- [Cooperative Groups](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cooperative-groups)
