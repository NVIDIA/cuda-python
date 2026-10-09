# Sample: Multi-GPU Isotropic FD Wave Propagation (Python)

## Description

3D acoustic finite-difference wave propagator, striped across every visible
CUDA device. The X-Y volume is split evenly and each device owns a subset
of Z slices; neighboring subvolumes exchange halo regions via
``cuMemcpyPeerAsync`` between timesteps.

Compute and halo exchange run on separate CUDA streams per device
(``streamCenter`` and ``streamHalo``) so halo copies overlap with the
interior update. Peer-to-peer access is enabled with
``cuCtxEnablePeerAccess`` between every pair of participating devices.

The kernels compute a 2-nd-order-in-time / 8-th-order-in-space stencil
using vectorized ``float2`` loads plus shared memory tiling for the
horizontal neighborhood, and register accumulation for the vertical
neighborhood — a fairly realistic HPC kernel.

This is the only sample in ``/samples/cuda_bindings`` that teaches the multi-GPU HPC
pattern: per-device contexts, halo exchange via ``cuMemcpyPeerAsync``,
and compute/comm overlap on two streams per device.

Waives with exit code 2 unless there are 2+ CUDA devices with
peer-to-peer access enabled between them. The sample displays the final
wavefield with ``matplotlib`` by default; pass ``--no-display`` for a
non-interactive run.

## What You'll Learn

- Managing one CUDA context per device with `cuCtxCreate`, `cuCtxSetCurrent`,
  and `cuCtxEnablePeerAccess`
- Splitting a volume across GPUs and exchanging halos with
  `cuMemcpyPeerAsync`
- Overlapping compute (`streamCenter`) with halo exchange (`streamHalo`) on
  two streams per device using `cuStreamWaitEvent`
- A realistic FD wave kernel using shared-memory tiling, register queues,
  and vectorized `float2` loads
- Feature-detecting P2P support with `cuDeviceCanAccessPeer` before
  attempting a multi-GPU run

## Key Libraries

- [`cuda.bindings`](https://nvidia.github.io/cuda-python/cuda-bindings/latest/) - driver + runtime bindings
- `numpy` - array plumbing on the host
- `matplotlib` - final-wavefield display

## Key APIs

### From `cuda.bindings.driver`

- `cuCtxCreate`, `cuCtxSetCurrent`, `cuCtxGetDevice`, `cuCtxDestroy`
- `cuCtxEnablePeerAccess`, `cuDeviceCanAccessPeer`
- `cuStreamCreate` / `cuStreamDestroy` / `cuStreamSynchronize`
- `cuMemAlloc` / `cuMemFree` / `cuMemsetD32`
- `cuMemcpyPeerAsync`, `cuMemcpyDtoHAsync`, `cuMemcpyHtoDAsync`
- `cuLaunchKernel`

## Requirements

### Hardware

- **2 or more CUDA-capable GPUs with peer-to-peer access enabled between
  them.** On many consumer GPUs (e.g. GeForce RTX 4090) P2P is disabled;
  the sample waives with exit code 2 in that case.
- Data-center / NVLink-connected GPUs (A100, H100, etc.) are the intended
  target hardware.

### Software

- CUDA Toolkit 13.0 or newer
- Python 3.10 or newer
- `cuda-python` (>=13.4.0)
- `numpy`
- `matplotlib` (optional)

## Installation

```bash
pip install -r requirements.txt
```

## How to Run

```bash
python isoFdModelling.py               # Display the final wavefield
python isoFdModelling.py --no-display  # Run without opening a plot
```

## Expected Output

On a P2P-capable multi-GPU system the sample propagates a wavefield for
several hundred timesteps and prints per-iteration stats. On systems
without peer access it waives:

```
Two or more GPUs with Peer-to-Peer access capability are required
```

## Files

- `isoFdModelling.py` - Python implementation using `cuda.bindings.driver`
- `README.md` - This file
- `requirements.txt` - Sample dependencies
- `../../Utilities/cuda_bindings_utils.py` - Shared bindings helpers (imported by this sample)

## See Also

- [CUDA Python Documentation](https://nvidia.github.io/cuda-python/)
- [`cuda.bindings` driver API](https://nvidia.github.io/cuda-python/cuda-bindings/latest/module/driver.html)
- [`samples/cuda_core/simpleP2P/`](../../../cuda_core/simpleP2P/) - basic peer-to-peer access
- [CUDA C++ Programming Guide — Peer-to-Peer Memory Access](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#peer-to-peer-memory-access)
