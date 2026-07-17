# Sample: Mipmapped Textures with LOD (Python)

## Description

Demonstrates the new ``cuda.core.texture`` stack — ``MipmappedArray``,
``SurfaceObject``, and a ``TextureObject`` that does **trilinear** (LINEAR
mipmap + LINEAR filter) sampling with user-controlled LOD bias — rendered
live to an OpenGL window with the same interop pattern as
[`samples/cuda_core/glInteropPlasma/`](../glInteropPlasma/).

The sample:

1. Allocates a mipmap pyramid as a single ``MipmappedArray``.
2. Populates each level from a CUDA kernel bound to that level as a
   ``SurfaceObject``.
3. Samples the whole pyramid from a ``TextureObject`` with a manual LOD
   bias controlled from the keyboard.

## What You'll Learn

- Allocating a mipmap pyramid via `MipmappedArray`
- Binding a specific mip level as a `SurfaceObject` to write to it from a
  kernel
- Configuring a `TextureObject` for `LINEAR` mipmap + `LINEAR` filter
  (trilinear) sampling
- Passing a user-controlled `lod` bias through to `tex2DLod<float4>`
- OpenGL PBO interop through `GraphicsResource.from_gl_buffer`

## Requirements

### Hardware

- NVIDIA GPU with Compute Capability 7.0 or higher
- A display (X11 on Linux, native window server on macOS / Windows).
  Headless environments waive the sample.

### Software

- CUDA Toolkit 13.0 or newer
- Python 3.10 or newer
- `cuda-python` (>=13.0.0)
- `cuda-core` (>=1.0.0)
- `numpy` (>=1.24)
- `pyglet` (>=2.0)

## Installation

```bash
pip install -r requirements.txt
```

## How to Run

```bash
python glInteropMipmapLod.py
```

## Files

- `glInteropMipmapLod.py` — Python implementation
- `README.md` — This file
- `requirements.txt` — Sample dependencies

## See Also

- [`cuda.core.texture` API](https://nvidia.github.io/cuda-python/cuda-core/latest/api.html#module-cuda.core.texture)
- [`samples/cuda_core/glInteropFluid/`](../glInteropFluid/) — Stable Fluids demo (also uses TextureObject / SurfaceObject)
- [`samples/cuda_core/textureSample/`](../textureSample/) — minimal texture sampling
