# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# ################################################################################
#
# Real-time Plasma Effect -- CUDA/OpenGL Interop with cuda.core.GraphicsResource
#
# ################################################################################
#
# What this example teaches
# =========================
# How to use cuda.core.GraphicsResource to let a CUDA kernel write pixels
# directly into an OpenGL buffer with zero copies through the CPU.
#
# How it works
# ============
# Normally, getting CUDA results onto the screen would require:
#   CUDA -> CPU memory -> OpenGL  (two slow copies across the PCIe bus)
#
# GraphicsResource eliminates the CPU round-trip.  The pixel data stays
# on the GPU the entire time:
#
#   1. OpenGL allocates a PBO (Pixel Buffer Object) -- a raw GPU buffer.
#   2. GraphicsResource.from_gl_buffer() registers that PBO with CUDA.
#      Now both CUDA and OpenGL have access to the *same* GPU memory.
#
#   +----------------------+       +---------------------+
#   |    OpenGL PBO        |       |  GraphicsResource   |
#   | (pixel buffer on GPU)| <---> |  (CUDA handle to    |
#   +----------------------+       |  the same memory)   |
#                                  +---------------------+
#
#   EACH FRAME (all on GPU -- nothing touches the CPU or PCIe bus)
#   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. map()           -- CUDA gets a device pointer into the PBO
#   2. launch kernel   -- CUDA writes pixel colors into that memory
#   3. unmap()         -- ownership returns to OpenGL
#   4. glTexSubImage2D -- OpenGL copies PBO into a texture (GPU-to-GPU)
#   5. draw            -- OpenGL renders the texture to the window
#
#   Why is there a copy in step 4?  OpenGL can only render from a
#   "texture" object, not from a raw buffer. The glTexSubImage2D step
#   copies the PBO bytes into a texture, but this happens entirely on
#   the GPU and it is very fast. The big win from GraphicsResource is
#   that we never copy pixels from the CPU to the GPU and then and back.
#
# What you should see
# ===================
# A window showing smoothly animated, colorful swirling patterns (a "plasma"
# effect popular in the demoscene).  The window title shows the current FPS.
# Close the window or press Escape to exit.
#
# Requirements
# ============
#   pip install pyglet
#
# ################################################################################

import ctypes
import sys
import time

import numpy as np
from cuda.core import (
    Device,
    GraphicsResource,
    LaunchConfig,
    Program,
    ProgramOptions,
    launch,
)

# ---------------------------------------------------------------------------
# Window dimensions (feel free to change these)
# ---------------------------------------------------------------------------
WIDTH = 800
HEIGHT = 600


# ============================= Helper functions =============================
#
# The functions below set up CUDA and OpenGL.  If you're here to learn about
# GraphicsResource, you can skip straight to main() -- the interesting part
# is there.  These helpers exist so that main() reads like a short story
# instead of a wall of boilerplate.
# ============================================================================


def setup_cuda(kernel_source):
    """Compile the CUDA kernel and return (device, stream, kernel, launch_config)."""
    dev = Device(0)
    dev.set_current()
    stream = dev.create_stream()

    opts = ProgramOptions(std="c++11", arch=f"sm_{dev.arch}")
    prog = Program(kernel_source, code_type="c++", options=opts)
    mod = prog.compile("cubin")
    kernel = mod.get_kernel("plasma")

    block = (16, 16, 1)
    grid = (
        (WIDTH + block[0] - 1) // block[0],
        (HEIGHT + block[1] - 1) // block[1],
        1,
    )
    config = LaunchConfig(grid=grid, block=block)

    return dev, stream, kernel, config


def create_window():
    """Open a pyglet window and return (window, gl_module)."""
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
        caption="GraphicsResource Example - CUDA Plasma",
        vsync=False,
    )
    return window, _gl, pyglet


def create_display_resources(gl, width, height):
    """Create the GL objects needed to show a texture on screen.

    This sets up a shader program, a fullscreen quad, and an
    empty texture. None of this is CUDA-specific as it's standard
    OpenGL boilerplate for rendering a textured quad to the
    screen.

    Returns (shader_program, vertex_array_id, texture_id).
    The shader_program is a pyglet ShaderProgram object (must be kept alive).
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
    to a texture. By registering this same buffer with CUDA, the CUDA kernel can
    write directly into it.

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


# ================================== main() ==================================


def main():
    # --- Step 1: Set up CUDA (compile kernel, create stream) ---
    dev, stream, kernel, config = setup_cuda(PLASMA_KERNEL_SOURCE)

    # --- Step 2: Open a window ---
    window, gl, pyglet = create_window()

    # --- Step 3: Create GL resources for drawing a texture to screen ---
    #     (This is standard OpenGL boilerplate -- not CUDA-specific.)
    shader_prog, quad_vao, tex_id = create_display_resources(gl, WIDTH, HEIGHT)

    # --- Step 4: Create the Pixel Buffer Object (PBO) ---
    #     A PBO is GPU memory owned by OpenGL.  It's the bridge between the
    #     two worlds: CUDA will write into it, and OpenGL will read from it.
    pbo_id, nbytes = create_pixel_buffer(gl, WIDTH, HEIGHT)

    # --- Step 5: Register the PBO with CUDA ---
    #     THIS IS THE KEY LINE.  GraphicsResource.from_gl_buffer() tells the
    #     CUDA driver "I want to access this OpenGL buffer from CUDA kernels."
    #     WRITE_DISCARD means CUDA will overwrite the entire buffer each frame.
    resource = GraphicsResource.from_gl_buffer(pbo_id, flags="write_discard")

    # --- Step 6: Render loop ---
    start_time = time.monotonic()
    frame_count = 0
    fps_time = start_time

    @window.event
    def on_draw():
        nonlocal frame_count, fps_time

        window.clear()
        t = time.monotonic() - start_time

        # (a) Map the PBO so CUDA can write to it.
        #     This gives us a Buffer whose .handle is a CUDA device pointer
        #     pointing directly into the OpenGL PBO's GPU memory.
        with resource.map(stream=stream) as buf:
            # (b) Launch the plasma kernel -- it writes RGBA pixels into buf.
            launch(
                stream,
                config,
                kernel,
                buf.handle,  # pointer to PBO memory (on GPU)
                np.int32(WIDTH),
                np.int32(HEIGHT),
                np.float32(t),  # animation time
            )
        # (c) Unmap happens automatically when the `with` block exits.
        #     The PBO now belongs to OpenGL again.  No stream.sync() is
        #     needed here -- cuGraphicsUnmapResources guarantees that all
        #     CUDA work on the stream completes before OpenGL can use the
        #     buffer.

        # (d) Tell OpenGL to copy the PBO contents into our texture (GPU-to-GPU).
        copy_pbo_to_texture(gl, pbo_id, tex_id, WIDTH, HEIGHT)

        # (e) Draw the texture to the screen.
        draw_fullscreen_quad(gl, shader_prog, quad_vao, tex_id)

        # FPS counter (shown in window title)
        frame_count += 1
        now = time.monotonic()
        if now - fps_time >= 1.0:
            fps = frame_count / (now - fps_time)
            frame_us = 1_000_000.0 / fps if fps > 0 else 0
            window.set_caption(
                f"GraphicsResource Example - CUDA Plasma"
                f" ({WIDTH}x{HEIGHT}, {fps:.0f} FPS, {frame_us:.0f} \u00b5s frame)"
            )
            frame_count = 0
            fps_time = now

    @window.event
    def on_close():
        resource.close()

    pyglet.app.run(interval=0)
    print("done!")


# ======================== GPU code (CUDA + GLSL) ============================
#
# These source strings are kept at the bottom of the file so they don't
# distract from the Python logic above.  The important thing to know:
#
#   - PLASMA_KERNEL_SOURCE is CUDA C++ that runs on the GPU.  It computes a
#     color for each pixel based on layered sine waves (the "plasma" effect)
#     and writes RGBA bytes into the output buffer.
#
#   - VERTEX_SHADER_SOURCE / FRAGMENT_SHADER_SOURCE are GLSL (OpenGL's shader
#     language).  They simply draw a texture onto a rectangle that covers the
#     entire window.  Nothing interesting happens here.
#
# ============================================================================

PLASMA_KERNEL_SOURCE = r"""
extern "C"
__global__
void plasma(unsigned char* output, int width, int height, float t) {
    // Each CUDA thread computes one pixel.
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // Normalize pixel coordinates to [0, 1]
    float u = (float)x / (float)width;
    float v = (float)y / (float)height;

    // --- Plasma: sum of overlapping sine waves ---
    float val = 0.0f;
    val += sinf(u * 10.0f + t);                        // horizontal ripple
    val += sinf(v * 10.0f + t * 0.7f);                 // vertical ripple
    val += sinf((u + v) * 10.0f + t * 0.5f);           // diagonal ripple

    // Circular wave radiating from the center of the screen
    float cx = u - 0.5f;
    float cy = v - 0.5f;
    val += sinf(sqrtf(cx*cx + cy*cy) * 20.0f - t * 2.0f);

    // Second circular wave whose center slowly drifts over time
    float cx2 = u - 0.5f + 0.3f * sinf(t * 0.3f);
    float cy2 = v - 0.5f + 0.3f * cosf(t * 0.4f);
    val += sinf(sqrtf(cx2*cx2 + cy2*cy2) * 15.0f + t * 1.5f);

    // val now ranges roughly from -5 to +5.  Normalize to [0, 1].
    val = (val + 5.0f) / 10.0f;

    // --- Convert to RGB ---
    // Three sine waves offset by 120 degrees (2*PI/3) give smooth color cycling.
    //   2*PI       = 6.2831853
    //   2*PI / 3   = 2.094   (120 degree offset for green)
    //   4*PI / 3   = 4.189   (240 degree offset for blue)
    float r = sinf(val * 6.2831853f)          * 0.5f + 0.5f;
    float g = sinf(val * 6.2831853f + 2.094f) * 0.5f + 0.5f;
    float b = sinf(val * 6.2831853f + 4.189f) * 0.5f + 0.5f;

    // Write RGBA pixel (4 bytes per pixel)
    int idx = (y * width + x) * 4;
    output[idx + 0] = (unsigned char)(r * 255.0f);   // R
    output[idx + 1] = (unsigned char)(g * 255.0f);   // G
    output[idx + 2] = (unsigned char)(b * 255.0f);   // B
    output[idx + 3] = 255;                            // A (fully opaque)
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
