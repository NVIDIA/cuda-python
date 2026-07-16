# Sample: Texture Sampling (Python)

## Description

Minimal ``cuda.core.texture`` demo: build a 2D CUDA ``OpaqueArray``, bind
it as a bindless ``TextureObject``, and sample it from a kernel using both
POINT-exact and LINEAR-interpolated coordinates.

Texture coordinate convention (non-normalized): each texel ``(i, j)`` is
centered at ``(i + 0.5, j + 0.5)``. So ``tex2D(tex, 0.5, 0.5)`` returns
texel ``(0, 0)`` exactly, while ``tex2D(tex, 1.0, 0.5)`` returns the linear
blend of texels ``(0, 0)`` and ``(1, 0)``.

## What You'll Learn

- Allocating a 2D `OpaqueArray` (`cudaArray_t`) via `cuda.core.texture`
- Copying host data into the array
- Configuring a `TextureObject` for POINT vs LINEAR filtering with
  non-normalized coordinates
- Sampling the texture from a kernel with `tex2D<T>` and verifying both
  filter modes against a NumPy reference

## Requirements

### Hardware

- NVIDIA GPU with Compute Capability 5.0 or higher

### Software

- CUDA Toolkit 13.0 or newer
- Python 3.10 or newer
- `cuda-python` (>=13.0.0)
- `cuda-core` (>=1.0.0)
- `numpy` (>=1.24)

## Installation

```bash
pip install -r requirements.txt
```

## How to Run

```bash
python textureSample.py
```

## Files

- `textureSample.py` — Python implementation
- `README.md` — This file
- `requirements.txt` — Sample dependencies

## See Also

- [`cuda.core.texture` API](https://nvidia.github.io/cuda-python/cuda-core/latest/api.html#module-cuda.core.texture)
- [`samples/glInteropFluid/`](../glInteropFluid/) — larger demo using TextureObject + SurfaceObject
- [`samples/glInteropMipmapLod/`](../glInteropMipmapLod/) — mipmap / LOD demo
