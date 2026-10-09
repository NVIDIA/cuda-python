# Sample: Stable Fluids on numba-cuda-mlir (Python)

## Description

A [numba-cuda-mlir](https://github.com/NVIDIA/numba-cuda) port of the
[`samples/cuda_core/glInteropFluid/`](../glInteropFluid/) Stable Fluids demo. Same
physics; the memory model changes.

The `cuda.core` version binds each field as a `cudaArray` and reads it
through a `TextureObject` (cached, hardware-filtered) and writes it back
through a `SurfaceObject`. `numba-cuda-mlir` has no texture / surface
support, so this port uses ordinary linear device arrays and implements
the hardware bilinear filter by hand. Useful as a side-by-side reference
for the two programming models.

Drag the mouse to inject swirling ink.

## What You'll Learn

- What Stable Fluids looks like without textures — semi-Lagrangian
  advection with a hand-rolled bilinear interpolator
- How `numba-cuda-mlir` handles arrays, kernel launches, and device
  functions
- Registering an OpenGL PBO with CUDA and writing pixels into it

## Requirements

- `numba-cuda-mlir` (from NVIDIA)
- `pyglet` (>=2.0)
- `numpy`

## Installation

```bash
pip install -r requirements.txt
```

## How to Run

```bash
python glInteropFluidNumbaCudaMlir.py
```

## Files

- `glInteropFluidNumbaCudaMlir.py` — Python implementation
- `README.md` — This file
- `requirements.txt` — Sample dependencies

## See Also

- [`samples/cuda_core/glInteropFluid/`](../glInteropFluid/) — `cuda.core` version using textures / surfaces
- [numba-cuda](https://github.com/NVIDIA/numba-cuda)
