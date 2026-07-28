# Sample: OpenGL Interop Plasma (Python)

## Description

A CUDA kernel writes pixel colors directly into an OpenGL Pixel Buffer
Object (PBO) with zero copies through the CPU. The PBO is then blitted
into a texture and drawn on a fullscreen quad, producing an animated
"plasma" effect (layered sine waves).

Without CUDA / OpenGL interop, moving GPU-computed pixels onto the screen
would require:

```
CUDA -> CPU memory -> OpenGL     (two slow copies across the PCIe bus)
```

Using `cuda.core.GraphicsResource.from_gl_buffer()` eliminates the
CPU round-trip: the PBO stays on the GPU the entire time, and CUDA and
OpenGL take turns accessing the same buffer.

Each frame:

1. `resource.map(stream=...)` gives CUDA a device pointer into the PBO.
2. A CUDA kernel writes RGBA pixels into that pointer.
3. The context manager `unmap()`s the resource on exit; OpenGL now owns
   the PBO again.
4. `glTexSubImage2D` copies the PBO into a texture (GPU-to-GPU, fast).
5. OpenGL draws the texture on a fullscreen quad.

## What You'll Learn

- Registering an OpenGL PBO with CUDA via `GraphicsResource.from_gl_buffer()`
- Mapping/unmapping a graphics resource with a context manager
- Launching a CUDA kernel that writes directly into GPU-side OpenGL memory
- Blitting a PBO into a GL texture without a CPU round-trip
- Running an interactive real-time visualization from Python

## Key Libraries

- [`cuda.core`](https://nvidia.github.io/cuda-python/cuda-core/latest/) - Pythonic access to CUDA runtime, programs, and graphics interop
- `pyglet` (>=2.0) - windowing and OpenGL bindings
- `numpy` - kernel argument construction

## Key APIs

### From `cuda.core`

- `GraphicsResource.from_gl_buffer(gl_buffer_id, flags="write_discard")` - register an OpenGL buffer with CUDA
- `GraphicsResource.map(stream=...)` - context manager yielding a `Buffer` that CUDA can write to
- `Program`, `ProgramOptions`, `LaunchConfig`, `launch` - standard cuda.core compile/launch flow

### From pyglet

- `pyglet.window.Window` - open a GL context
- `pyglet.graphics.shader.Shader` / `ShaderProgram` - build the passthrough shader used to draw the texture

## Requirements

### Hardware

- NVIDIA GPU with Compute Capability 7.0 or higher
- A display (X11 on Linux, native window server on macOS/Windows). Headless
  Linux environments (no `$DISPLAY`) will waive the sample.

### Software

- CUDA Toolkit 13.0 or newer (matches `cuda-python` 13.x)
- Python 3.10 or newer
- `cuda-python` (>=13.0.0)
- `cuda-core` (>=1.0.0)
- `numpy` (>=2.3.2)
- `pyglet` (>=2.0)

## Installation

Install the required packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

## How to Run

### Default (bounded run, exits after N frames):

```bash
python glInteropPlasma.py             # 60 frames, then exits
python glInteropPlasma.py --frames 300 # 300 frames
```

### Interactive (window stays open until closed):

```bash
python glInteropPlasma.py --interactive
```

### Choose a specific GPU:

```bash
python glInteropPlasma.py --device 1
```

## Expected Output

A window titled *GraphicsResource Example - CUDA Plasma* opens and shows
smoothly animated, colorful swirling patterns. The window title updates
every second with the current FPS. In non-interactive mode the window
closes automatically after `--frames` frames:

```
Rendered 60 frames via CUDA/OpenGL interop. Done
```

On a headless Linux runner the sample self-waives with:

```
No DISPLAY available; waiving GL interop sample.
```

## Files

- `glInteropPlasma.py` - Python implementation using `cuda.core.GraphicsResource`
- `README.md` - This file
- `requirements.txt` - Sample dependencies

## See Also

- [CUDA Python Documentation](https://nvidia.github.io/cuda-python/)
- [`cuda.core` GraphicsResource API](https://nvidia.github.io/cuda-python/cuda-core/latest/api.html#module-cuda.core)
- [pyglet Documentation](https://pyglet.readthedocs.io/en/latest/)
