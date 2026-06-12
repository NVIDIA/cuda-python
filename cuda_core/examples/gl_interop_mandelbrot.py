# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# ################################################################################
#
# This example demonstrates cuda.core.CUDAArray and TextureObject used as a *color
# lookup table* (palette LUT) for a real-time Mandelbrot deep-zoom explorer.
# A CUDA kernel computes smooth iteration counts and uses tex1D<float4> with
# LINEAR + CLAMP + NORMALIZED_FLOAT sampling to read a 256-entry RGBA palette,
# writing the final RGBA bytes straight into an OpenGL PBO via GraphicsResource.
# Requires pyglet.
#
# ################################################################################

# What this example teaches
# =========================
# - How to use a 1D cuda.core.CUDAArray as a palette and bind it via a
#   TextureObject for hardware-filtered color lookups inside a kernel.
# - How LINEAR + AddressMode.CLAMP + ReadMode.NORMALIZED_FLOAT + normalized
#   coordinates give you a free `texture(palette, t)` style sampler that
#   returns a float4 in [0, 1] regardless of the underlying storage format.
# - How to drive a real-time interactive viewer: mouse pan, scroll-wheel zoom
#   anchored at the cursor, and key-driven iteration cap.
#
# How it works
# ============
# The Mandelbrot set is defined by iterating z -> z^2 + c starting from
# z = 0; pixels are colored by how quickly z escapes the disk of radius 2.
#
#     +---------+   ResourceDescriptor.from_array
#     |  CUDAArray  | --------------------------------+
#     | float4  |                                 v
#     | size 256|                       +-------------------+
#     +---------+                       |   TextureObject   |
#       ^  copy_from(host)              |  (palette LUT)    |
#       |                               +---------+---------+
#     host palette                                |
#     (numpy float32x4, 256 stops)                |
#                                                 v
#                                  tex1D<float4>(palette, t)
#                                                 |
#                                                 v
#                                     +-----------------------+
#                                     |  mandelbrot kernel    |
#                                     |  (one thread / pixel) |
#                                     +-----------+-----------+
#                                                 |
#                                                 v   GraphicsResource.map
#                                     +-----------------------+
#                                     |   OpenGL PBO (RGBA8)  |
#                                     +-----------------------+
#
# Smooth iteration count
# ----------------------
# A plain integer escape count produces ugly banded colors. With a bailout
# radius R = 2 (escape when |z|^2 > 4), we use the standard smooth formula:
#
#     mu = iter + 1 - log(log(|z|)) / log(2)
#
# At the escape step |z| > 2, so log(|z|) > log(2) > 0 and log(log(|z|)) is
# finite. We compute this in double and cast to float for the palette lookup.
#
# Cursor-anchored zoom
# --------------------
# On scroll, we want the world point under the mouse cursor to remain under
# the cursor after the zoom. We capture (wx, wy) under the cursor with the
# old scale, multiply the scale by 0.9 (zoom in) or 1.1 (zoom out), then
# back-solve cx, cy so the same screen pixel still maps to (wx, wy):
#
#     cx_new = wx - (mouse_x - W/2) * scale_new
#     cy_new = wy - (mouse_y - H/2) * scale_new
#
# Why double precision for cx, cy, scale?
# ---------------------------------------
# Float32 runs out of mantissa bits around 1e6x zoom; double gets you to
# roughly 1e13x before the pixel grid coarsens visibly. The kernel takes
# cx, cy, scale as doubles and only narrows to float for the color lookup.
#
# Address mode note
# -----------------
# We use AddressMode.CLAMP (per the example brief). Combined with the
# `fmodf(mu * 0.02f, 1.0f)` cycling formula, the palette index is already
# guaranteed to be in [0, 1), so CLAMP and WRAP both produce identical
# results in practice -- there is no visible seam.
#
# What you should see
# ===================
# A window showing the Mandelbrot set. Drag with the left mouse button to
# pan, scroll the wheel to zoom in/out at the cursor, press R to reset the
# view, and `[`/`]` to lower/raise the iteration cap. The window title shows
# the current zoom level, center, max_iter, and FPS. Close the window or
# press Escape to exit.
#

# /// script
# dependencies = ["cuda_bindings", "cuda_core>0.6.0", "pyglet"]
# ///

import ctypes
import sys
import time

import numpy as np

from cuda.core import (
    AddressMode,
    ArrayFormat,
    CUDAArray,
    Device,
    FilterMode,
    GraphicsResource,
    LaunchConfig,
    Program,
    ProgramOptions,
    ReadMode,
    ResourceDescriptor,
    TextureDescriptor,
    TextureObject,
    launch,
)

# ---------------------------------------------------------------------------
# Window and viewer parameters (feel free to change these)
# ---------------------------------------------------------------------------
WIDTH = 1024
HEIGHT = 768
PALETTE_SIZE = 256

# Default view: classic Mandelbrot framing centered slightly left of origin.
DEFAULT_CX = -0.5
DEFAULT_CY = 0.0
DEFAULT_SCALE = 4.0 / HEIGHT  # world-units per pixel (4-unit-tall view)
DEFAULT_MAX_ITER = 512

# Bounds for [/] iteration adjust.
MIN_MAX_ITER = 64
MAX_MAX_ITER = 8192
ITER_STEP = 64


# ============================= Helper functions =============================
#
# The functions below set up CUDA and OpenGL. If you're here to learn about
# CUDAArray/TextureObject as a palette LUT, skip ahead to main() -- the interesting
# part is there. These helpers exist so that main() reads like a short story
# instead of a wall of boilerplate.
# ============================================================================


def setup_cuda():
    """Compile the CUDA kernel and return (device, stream, kernel, config)."""
    dev = Device(0)
    dev.set_current()

    # Bindless texture objects (cuTexObjectCreate) require SM 3.0+.
    cc = dev.compute_capability
    if cc.major < 3:
        print(
            "This example requires a GPU with compute capability >= 3.0 for "
            f"bindless texture objects. Found sm_{cc.major}{cc.minor}.",
            file=sys.stderr,
        )
        sys.exit(1)

    stream = dev.create_stream()

    # Compile as C++ so the templated tex1D<float4> overload resolves.
    program_options = ProgramOptions(std="c++17", arch=f"sm_{dev.arch}")
    prog = Program(KERNEL_SOURCE, code_type="c++", options=program_options)
    mod = prog.compile("cubin", name_expressions=("mandelbrot",))

    kernel = mod.get_kernel("mandelbrot")

    block = (16, 16, 1)
    grid = (
        (WIDTH + block[0] - 1) // block[0],
        (HEIGHT + block[1] - 1) // block[1],
        1,
    )
    config = LaunchConfig(grid=grid, block=block)

    return dev, stream, kernel, config


def create_window():
    """Open a pyglet window and return (window, gl_module, pyglet)."""
    try:
        import pyglet
        from pyglet.gl import gl as _gl
    except ImportError:
        print(
            "This example requires pyglet >= 2.0.\nInstall it with:  pip install pyglet",
            file=sys.stderr,
        )
        sys.exit(1)

    window = pyglet.window.Window(
        WIDTH,
        HEIGHT,
        caption="cuda.core CUDAArray/Texture - Mandelbrot Deep Zoom",
        vsync=False,
    )
    return window, _gl, pyglet


def create_display_resources(gl, width, height):
    """Create the GL objects needed to show a texture on screen.

    This sets up a shader program, a fullscreen quad, and an empty texture.
    None of this is CUDA-specific -- it's standard OpenGL boilerplate for
    rendering a textured quad.

    Returns (shader_program, vertex_array_id, texture_id). The shader_program
    is a pyglet ShaderProgram object (must be kept alive).
    """
    from pyglet.graphics.shader import Shader, ShaderProgram

    # Shader program -- just passes texture coordinates through
    vert = Shader(VERTEX_SHADER_SOURCE, "vertex")
    frag = Shader(FRAGMENT_SHADER_SOURCE, "fragment")
    shader_prog = ShaderProgram(vert, frag)

    # Fullscreen quad (two triangles covering the entire window)
    quad_verts = np.array(
        [
            # x,  y,    s, t      (position + texture coordinate)
            -1,
            -1,
            0,
            0,
            1,
            -1,
            1,
            0,
            1,
            1,
            1,
            1,
            -1,
            -1,
            0,
            0,
            1,
            1,
            1,
            1,
            -1,
            1,
            0,
            1,
        ],
        dtype=np.float32,
    )

    vao = ctypes.c_uint(0)
    gl.glGenVertexArrays(1, ctypes.byref(vao))
    gl.glBindVertexArray(vao.value)

    vbo = ctypes.c_uint(0)
    gl.glGenBuffers(1, ctypes.byref(vbo))
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo.value)
    gl.glBufferData(
        gl.GL_ARRAY_BUFFER,
        quad_verts.nbytes,
        quad_verts.ctypes.data_as(ctypes.c_void_p),
        gl.GL_STATIC_DRAW,
    )

    stride = 4 * 4  # 4 floats * 4 bytes each = 16 bytes per vertex
    pos_loc = gl.glGetAttribLocation(shader_prog.id, b"position")
    gl.glEnableVertexAttribArray(pos_loc)
    gl.glVertexAttribPointer(pos_loc, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(0))

    tc_loc = gl.glGetAttribLocation(shader_prog.id, b"texcoord")
    gl.glEnableVertexAttribArray(tc_loc)
    gl.glVertexAttribPointer(tc_loc, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(8))

    gl.glBindVertexArray(0)

    # Empty texture (will be filled each frame from the PBO)
    tex = ctypes.c_uint(0)
    gl.glGenTextures(1, ctypes.byref(tex))
    gl.glBindTexture(gl.GL_TEXTURE_2D, tex.value)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
    gl.glTexImage2D(
        gl.GL_TEXTURE_2D,
        0,
        gl.GL_RGBA8,
        width,
        height,
        0,
        gl.GL_RGBA,
        gl.GL_UNSIGNED_BYTE,
        None,
    )

    return shader_prog, vao.value, tex.value


def create_pixel_buffer(gl, width, height):
    """Create a Pixel Buffer Object (PBO) -- the bridge between CUDA and OpenGL.

    A PBO is a GPU-side buffer that OpenGL can read from when uploading pixels
    to a texture. By registering this same buffer with CUDA, the CUDA kernel
    can write directly into it.

    Returns (pbo_gl_name, size_in_bytes).
    """
    pbo = ctypes.c_uint(0)
    gl.glGenBuffers(1, ctypes.byref(pbo))
    gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, pbo.value)
    nbytes = width * height * 4  # RGBA, 1 byte per channel
    gl.glBufferData(gl.GL_PIXEL_UNPACK_BUFFER, nbytes, None, gl.GL_DYNAMIC_DRAW)
    gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)
    return pbo.value, nbytes


def copy_pbo_to_texture(gl, pbo_id, tex_id, width, height):
    """Copy pixel data from the PBO into the GL texture (GPU-to-GPU)."""
    gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, pbo_id)
    gl.glBindTexture(gl.GL_TEXTURE_2D, tex_id)
    gl.glTexSubImage2D(
        gl.GL_TEXTURE_2D,
        0,
        0,
        0,
        width,
        height,
        gl.GL_RGBA,
        gl.GL_UNSIGNED_BYTE,
        None,  # None = read from the currently bound PBO, not from CPU
    )
    gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)


def draw_fullscreen_quad(gl, shader_prog, vao_id, tex_id):
    """Draw the texture to the screen using the fullscreen quad."""
    gl.glUseProgram(shader_prog.id)
    gl.glBindTexture(gl.GL_TEXTURE_2D, tex_id)
    gl.glBindVertexArray(vao_id)
    gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)
    gl.glBindVertexArray(0)
    gl.glUseProgram(0)


def build_palette():
    """Build a 256-entry RGBA float32 palette by lerping through color stops.

    Returns a flat numpy array of shape (PALETTE_SIZE * 4,) dtype=float32
    suitable for CUDAArray.copy_from(). Each color channel is in [0, 1].
    """
    # Hand-picked stops: deep blue -> cyan -> yellow -> orange -> red ->
    # magenta -> black (the final stop is used by points that hit max_iter
    # and don't escape).
    stops = np.array(
        [
            [0.00, 0.02, 0.05, 0.30, 1.0],  # position, R, G, B, A
            [0.16, 0.10, 0.50, 0.90, 1.0],  # cyan
            [0.42, 1.00, 0.95, 0.20, 1.0],  # yellow
            [0.58, 1.00, 0.55, 0.10, 1.0],  # orange
            [0.74, 0.95, 0.10, 0.10, 1.0],  # red
            [0.90, 0.65, 0.10, 0.85, 1.0],  # magenta
            [1.00, 0.00, 0.00, 0.00, 1.0],  # black
        ],
        dtype=np.float32,
    )

    pal = np.empty((PALETTE_SIZE, 4), dtype=np.float32)
    positions = stops[:, 0]
    colors = stops[:, 1:]
    for i in range(PALETTE_SIZE):
        t = i / (PALETTE_SIZE - 1)
        # Find the bracketing segment.
        j = int(np.searchsorted(positions, t, side="right")) - 1
        j = max(0, min(j, len(positions) - 2))
        t0 = positions[j]
        t1 = positions[j + 1]
        seg = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
        pal[i] = colors[j] + seg * (colors[j + 1] - colors[j])

    # Flatten to (PALETTE_SIZE * 4,) so the byte layout matches a
    # float4 x PALETTE_SIZE 1D CUDAArray.
    return np.ascontiguousarray(pal.reshape(-1), dtype=np.float32)


def make_palette_texture(arr):
    """Bind `arr` as a TextureObject configured for LINEAR + CLAMP + normalized."""
    res_desc = ResourceDescriptor.from_array(arr)
    tex_desc = TextureDescriptor(
        address_mode=AddressMode.CLAMP,
        filter_mode=FilterMode.LINEAR,
        # NORMALIZED_FLOAT is a no-op for FLOAT32 storage (the data is already
        # in [0, 1]); we set it because the spec calls for it and to document
        # the intent for readers building palettes from UINT8 storage.
        read_mode=ReadMode.NORMALIZED_FLOAT,
        normalized_coords=True,
    )
    return TextureObject.from_descriptor(resource=res_desc, texture_descriptor=tex_desc)


# ================================== main() ==================================


def main():
    # --- Step 1: Set up CUDA (compile kernel, create stream) ---
    dev, stream, kernel, config = setup_cuda()

    # --- Step 2: Open a window ---
    window, gl, pyglet = create_window()

    # --- Step 3: Create GL resources for drawing a texture to screen ---
    #     (Standard OpenGL boilerplate -- not CUDA-specific.)
    shader_prog, quad_vao, tex_id = create_display_resources(gl, WIDTH, HEIGHT)

    # --- Step 4: Create the Pixel Buffer Object (PBO) ---
    #     The PBO is GPU memory owned by OpenGL. It's the bridge between the
    #     two worlds: CUDA writes into it, OpenGL reads from it.
    pbo_id, _ = create_pixel_buffer(gl, WIDTH, HEIGHT)

    # --- Step 5: Register the PBO with CUDA ---
    resource = GraphicsResource.from_gl_buffer(pbo_id, flags="write_discard")

    # --- Step 6: Build and upload the palette LUT ---
    #     One 1D CUDAArray, 256 entries of float4 RGBA. The host-side palette is
    #     a flat numpy float32 array; copy_from() does an async H2D copy, so
    #     we sync the stream once afterwards to make sure the data has landed
    #     before we start sampling from it in the render loop.
    host_palette = build_palette()
    palette_arr = CUDAArray.from_descriptor(
        shape=(PALETTE_SIZE,),
        format=ArrayFormat.FLOAT32,
        num_channels=4,
    )
    palette_arr.copy_from(host_palette, stream=stream)
    stream.sync()

    # --- Step 7: Bind the palette CUDAArray as a TextureObject (LUT) ---
    palette_tex = make_palette_texture(palette_arr)

    # --- Step 8: Render loop ---
    start_time = time.monotonic()
    frame_count = 0
    fps_time = start_time

    # View state. cx, cy, scale are kept in Python floats (double precision)
    # and converted to np.float64 on each kernel launch.
    view = {
        "cx": float(DEFAULT_CX),
        "cy": float(DEFAULT_CY),
        "scale": float(DEFAULT_SCALE),
        "max_iter": int(DEFAULT_MAX_ITER),
        # Pan-drag state (left mouse button).
        "dragging": False,
    }

    def screen_to_world(mouse_x, mouse_y):
        """Map a pyglet mouse coordinate to the world point currently under it.

        Pyglet's window origin is bottom-left and the rendered texture's
        origin is also bottom-left, so no y-flip is needed.
        """
        wx = view["cx"] + (mouse_x - WIDTH / 2.0) * view["scale"]
        wy = view["cy"] + (mouse_y - HEIGHT / 2.0) * view["scale"]
        return wx, wy

    @window.event
    def on_key_press(symbol, _modifiers):
        key = pyglet.window.key
        if symbol == key.ESCAPE:
            window.close()
            return
        if symbol == key.R:
            view["cx"] = float(DEFAULT_CX)
            view["cy"] = float(DEFAULT_CY)
            view["scale"] = float(DEFAULT_SCALE)
            view["max_iter"] = int(DEFAULT_MAX_ITER)
            return
        if symbol == key.BRACKETLEFT:
            view["max_iter"] = max(MIN_MAX_ITER, view["max_iter"] - ITER_STEP)
            return
        if symbol == key.BRACKETRIGHT:
            view["max_iter"] = min(MAX_MAX_ITER, view["max_iter"] + ITER_STEP)
            return

    @window.event
    def on_mouse_press(_x, _y, button, _modifiers):
        if button == pyglet.window.mouse.LEFT:
            view["dragging"] = True

    @window.event
    def on_mouse_release(_x, _y, button, _modifiers):
        if button == pyglet.window.mouse.LEFT:
            view["dragging"] = False

    @window.event
    def on_mouse_drag(_x, _y, dx, dy, buttons, _modifiers):
        if buttons & pyglet.window.mouse.LEFT:
            # Pan: move the center opposite to the cursor drag (so the scene
            # follows the cursor). dy is positive when moving up in pyglet's
            # bottom-left origin space, matching the texture orientation.
            view["cx"] -= dx * view["scale"]
            view["cy"] += dy * view["scale"]

    @window.event
    def on_mouse_scroll(x, y, _scroll_x, scroll_y):
        # Cursor-anchored zoom: keep the world point under the cursor pinned.
        wx, wy = screen_to_world(x, y)
        factor = 0.9 if scroll_y > 0 else 1.1
        view["scale"] *= factor
        # Back-solve cx, cy so screen pixel (x, y) still maps to (wx, wy).
        view["cx"] = wx - (x - WIDTH / 2.0) * view["scale"]
        view["cy"] = wy - (y - HEIGHT / 2.0) * view["scale"]

    @window.event
    def on_draw():
        nonlocal frame_count, fps_time

        window.clear()

        # (a) Map the PBO so CUDA can write to it. This gives us a Buffer
        #     whose .handle is a device pointer pointing into the GL PBO.
        with resource.map(stream=stream) as buf:
            launch(
                stream,
                config,
                kernel,
                np.uint64(palette_tex.handle),  # bindless texture handle
                buf.handle,  # output PBO (RGBA8)
                np.int32(WIDTH),
                np.int32(HEIGHT),
                np.float64(view["cx"]),
                np.float64(view["cy"]),
                np.float64(view["scale"]),
                np.int32(view["max_iter"]),
            )
        # Unmap happens automatically when the `with` block exits.

        # (b) Tell OpenGL to copy the PBO contents into our texture.
        copy_pbo_to_texture(gl, pbo_id, tex_id, WIDTH, HEIGHT)

        # (c) Draw the texture to the screen.
        draw_fullscreen_quad(gl, shader_prog, quad_vao, tex_id)

        # FPS counter (shown in window title)
        frame_count += 1
        now = time.monotonic()
        if now - fps_time >= 1.0:
            fps = frame_count / (now - fps_time)
            zoom = 1.0 / view["scale"] if view["scale"] > 0 else 0.0
            window.set_caption(
                "cuda.core CUDAArray/Texture - Mandelbrot"
                f" | zoom {zoom:.3e}x"
                f" | center ({view['cx']:.6f}, {view['cy']:.6f})"
                f" | iter {view['max_iter']}"
                f" | {fps:.0f} FPS"
            )
            frame_count = 0
            fps_time = now

    @window.event
    def on_close():
        # Release everything we opened, in reverse order. Each of these is a
        # context manager too, but pyglet owns the event loop here so we
        # release explicitly.
        resource.close()
        palette_tex.close()
        palette_arr.close()
        stream.close()

    pyglet.app.run(interval=0)


# ======================== GPU code (CUDA + GLSL) ============================
#
# These source strings are kept at the bottom of the file so they don't
# distract from the Python logic above. The important things to know:
#
#   - KERNEL_SOURCE is a single CUDA C++ kernel `mandelbrot` that computes a
#     smooth iteration count per pixel and looks up the color via
#     tex1D<float4>(palette, t). Coordinates and the scale factor are doubles
#     to support deep zooms; only the color lookup runs in single precision.
#
#   - VERTEX_SHADER_SOURCE / FRAGMENT_SHADER_SOURCE are GLSL. They draw a
#     texture onto a rectangle covering the entire window. Nothing interesting.
#
# ============================================================================

KERNEL_SOURCE = r"""
// Mandelbrot deep-zoom kernel with a TextureObject palette LUT.
//
// Each thread computes one pixel. Coordinates and scale are doubles so the
// zoom doesn't quantize at modest depth. Once we have the smooth iteration
// count we narrow to float and use tex1D<float4> to read the palette.

extern "C"
__global__
void mandelbrot(cudaTextureObject_t palette,
                unsigned char* output,
                int width, int height,
                double cx, double cy, double scale,
                int max_iter) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // Map pixel -> complex plane (doubles).
    double c_re = cx + ((double)x - 0.5 * (double)width)  * scale;
    double c_im = cy + ((double)y - 0.5 * (double)height) * scale;

    // Standard escape iteration with bailout radius 2 (compare squared norm
    // against 4 to skip the sqrt in the inner loop).
    double zr = 0.0;
    double zi = 0.0;
    double zr2 = 0.0;
    double zi2 = 0.0;
    int iter = 0;
    while (iter < max_iter && (zr2 + zi2) <= 4.0) {
        zi = 2.0 * zr * zi + c_im;
        zr = zr2 - zi2 + c_re;
        zr2 = zr * zr;
        zi2 = zi * zi;
        ++iter;
    }

    unsigned char r, g, b;
    if (iter >= max_iter) {
        // Inside the set (or close enough): solid black.
        r = 0;
        g = 0;
        b = 0;
    } else {
        // Smooth iteration count:
        //   mu = iter + 1 - log(log(|z|)) / log(2)
        //      = iter + 1 - log(0.5 * log(|z|^2)) / log(2)
        // At escape, |z|^2 > 4, so 0.5 * log(|z|^2) > log(2) > 0 -- the
        // outer log is well-defined. Compute in double, narrow to float
        // for the palette lookup.
        double log_zn = 0.5 * log(zr2 + zi2);
        double nu = log(log_zn) / log(2.0);
        float mu = (float)((double)(iter + 1) - nu);

        // Cycle through the palette: 0.02 controls how quickly we wrap
        // through the gradient as the iteration count climbs.
        float t = fmodf(mu * 0.02f, 1.0f);
        if (t < 0.0f) t += 1.0f;  // fmodf can return negative for negative mu

        float4 rgba = tex1D<float4>(palette, t);

        // Clamp before narrowing to bytes.
        float fr = rgba.x; if (fr < 0.0f) fr = 0.0f; if (fr > 1.0f) fr = 1.0f;
        float fg = rgba.y; if (fg < 0.0f) fg = 0.0f; if (fg > 1.0f) fg = 1.0f;
        float fb = rgba.z; if (fb < 0.0f) fb = 0.0f; if (fb > 1.0f) fb = 1.0f;
        r = (unsigned char)(fr * 255.0f);
        g = (unsigned char)(fg * 255.0f);
        b = (unsigned char)(fb * 255.0f);
    }

    int idx = (y * width + x) * 4;
    output[idx + 0] = r;
    output[idx + 1] = g;
    output[idx + 2] = b;
    output[idx + 3] = 255;
}
"""

# GLSL shaders -- these just display a texture on a fullscreen rectangle.
# Nothing CUDA-specific here.

VERTEX_SHADER_SOURCE = """#version 330 core
in vec2 position;
in vec2 texcoord;
out vec2 v_texcoord;
void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    v_texcoord = texcoord;
}
"""

FRAGMENT_SHADER_SOURCE = """#version 330 core
in vec2 v_texcoord;
out vec4 fragColor;
uniform sampler2D tex;
void main() {
    fragColor = texture(tex, v_texcoord);
}
"""


if __name__ == "__main__":
    main()
