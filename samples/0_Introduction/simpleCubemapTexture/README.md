# Sample: Cubemap Textures (Python)

## Description

Creates a cubemap ``cudaArray`` (six 2D faces glued together), wraps it as
a bindless ``cudaTextureObject_t``, and samples it from a kernel using the
``texCubemap<float>(tex, cx, cy, cz)`` intrinsic. Also demonstrates 3D
memory transfers with ``cudaMemcpy3DParms`` and ``cudaMemcpy3D``.

This is currently the only sample in ``/samples`` that teaches CUDA
texture objects, cubemap arrays, or ``cudaMemcpy3D``.

The kernel walks all 6 cubemap faces, converts each pixel to a 3D
direction vector for that face, samples the cubemap, negates the result,
and writes it to global memory. Verification compares against a NumPy
reference on the host.

## What You'll Learn

- Allocating a cubemap ``cudaArray`` via ``cudaMalloc3DArray`` with the
  ``cudaArrayCubemap`` flag
- Populating a 3D destination with ``cudaMemcpy3DParms`` + ``cudaMemcpy3D``
- Building a bindless texture object with ``cudaResourceDesc`` +
  ``cudaTextureDesc`` + ``cudaCreateTextureObject``
- Setting texture parameters: normalized coordinates, linear filtering,
  wrap addressing (per axis)
- Sampling a cubemap face from a kernel with
  ``texCubemap<float>(tex, cx, cy, cz)``
- Tearing down texture / array / device memory resources cleanly

## Key Libraries

- [`cuda.bindings`](https://nvidia.github.io/cuda-python/cuda-bindings/latest/) - runtime + driver bindings
- `numpy` - input generation and host-side reference

## Key APIs

### From `cuda.bindings.runtime`

- `cudaMalloc` / `cudaFree`
- `cudaMalloc3DArray(..., cudaArrayCubemap)` / `cudaFreeArray`
- `cudaCreateChannelDesc`
- `cudaMemcpy3DParms`, `cudaMemcpy3D`
- `make_cudaPos`, `make_cudaExtent`, `make_cudaPitchedPtr`
- `cudaResourceDesc`, `cudaTextureDesc`
- `cudaCreateTextureObject` / `cudaDestroyTextureObject`
- `cudaMemcpy`, `cudaDeviceSynchronize`, `cudaGetDeviceProperties`

### From `cuda.bindings.driver`

- `cuLaunchKernel`

### Kernel-side

- `texCubemap<float>(cudaTextureObject_t, float, float, float)`

## Requirements

### Hardware

- NVIDIA GPU with SM 2.0 or higher (all Kepler-and-later cards)

### Software

- CUDA Toolkit 13.0 or newer
- Python 3.10 or newer
- `cuda-python` (>=13.0.0)
- `numpy`

## Installation

```bash
pip install -r requirements.txt
```

## How to Run

```bash
python simpleCubemapTexture.py
python simpleCubemapTexture.py --device=1     # use a specific GPU
```

## Expected Output

Throughput depends on GPU.

```
CUDA device [NVIDIA GeForce RTX 4090] has 128 Multi-Processors SM 8.9
Covering Cubemap data array of 64^3 x 1: Grid size is 8 x 8, each block has 8 x 8 threads
Processing time: 0.041 msec
595.35 Mtexlookups/sec
Verification PASSED (max error 0.000e+00)
Done
```

## Files

- `simpleCubemapTexture.py` - Python implementation using `cuda.bindings`
- `README.md` - This file
- `requirements.txt` - Sample dependencies
- `../../Utilities/cuda_bindings_utils.py` - Shared bindings helpers (imported by this sample)

## See Also

- [CUDA Python Documentation](https://nvidia.github.io/cuda-python/)
- [`cuda.bindings` runtime API](https://nvidia.github.io/cuda-python/cuda-bindings/latest/module/runtime.html)
- [CUDA C++ Programming Guide — Texture Object API](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-object-api)
- [CUDA C++ Programming Guide — Cubemap Textures](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cubemap-textures)
