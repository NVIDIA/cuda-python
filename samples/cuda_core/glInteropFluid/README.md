# Sample: OpenGL Interop — Stable Fluids (Python)

## Description

A real-time Stable Fluids (Jos Stam) smoke/ink solver rendered live to an
OpenGL window using ``cuda.core``'s texture / surface / graphics-resource
stack. Velocity, pressure, and dye fields live in ping-ponged ``cudaArray``s,
are read through ``TextureObject``s with hardware bilinear filtering (the
heart of semi-Lagrangian advection), and written back through
``SurfaceObject``s. The final dye field is colorized straight into an OpenGL
PBO. Drag the mouse to inject swirling ink.

This is the flagship demo for the ``cuda.core.texture`` package
(``OpaqueArray``, ``TextureObject``, ``SurfaceObject``, ``MipmappedArray``)
introduced in cuda-core 1.1.

## What You'll Learn

- Building a `cudaArray` (`OpaqueArray`) and binding it as both a
  `TextureObject` (cached, hardware-filtered reads) and a `SurfaceObject`
  (raw writes)
- Semi-Lagrangian advection via a single hardware bilinear `tex2D<float2>`
  fetch at a fractional back-traced coordinate
- Ping-ponged read/write between two arrays each frame
- Registering an OpenGL PBO with CUDA via `GraphicsResource.from_gl_buffer`
  and writing pixels directly into it (no CPU round-trip)

## Key Libraries

- [`cuda.core`](https://nvidia.github.io/cuda-python/cuda-core/latest/) — Device, Program, `texture` submodule (OpaqueArray / TextureObject / SurfaceObject), GraphicsResource
- `pyglet` (>=2.0) — windowing and OpenGL bindings
- `numpy` — kernel arguments and array shape metadata

## Requirements

### Hardware

- NVIDIA GPU with Compute Capability 7.0 or higher
- A display (X11 on Linux, native window server on macOS / Windows).
  Headless Linux environments (no `$DISPLAY`) waive the sample.

### Software

- CUDA Toolkit 13.0 or newer (matches ``cuda-python`` 13.x)
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
python glInteropFluid.py             # Run until the window is closed
python glInteropFluid.py --frames 3  # Render three frames, then exit
```

## Files

- `glInteropFluid.py` — Python implementation
- `README.md` — This file
- `requirements.txt` — Sample dependencies

## See Also

- [CUDA Python Documentation](https://nvidia.github.io/cuda-python/)
- [`cuda.core.texture` API](https://nvidia.github.io/cuda-python/cuda-core/latest/api.html#module-cuda.core.texture)
- [`samples/cuda_core/glInteropPlasma/`](../glInteropPlasma/) — simpler OpenGL interop example
- [`samples/cuda_core/glInteropMipmapLod/`](../glInteropMipmapLod/) — mipmap / LOD texture demo
- [`samples/cuda_core/textureSample/`](../textureSample/) — minimal texture sampling
- [Jos Stam, *Stable Fluids* (SIGGRAPH '99)](https://www.dgp.toronto.edu/people/stam/reality/Research/pdf/ns.pdf)
