# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# /// script
# dependencies = ["cuda-python>=13.0.0", "cuda-core>=1.0.0", "numpy>=2.3.2", "pyglet>=2.0"]
# ///

"""
CUDA / OpenGL interop with cuda.core.GraphicsResource

A CUDA kernel writes pixel colors directly into an OpenGL Pixel Buffer Object
(PBO) with zero copies through the CPU. The PBO is then blitted into a
texture and drawn to the window.

The classic way to display CUDA output would be:
    CUDA -> CPU memory -> OpenGL       (two slow copies across the PCIe bus)

``GraphicsResource.from_gl_buffer()`` eliminates the CPU round-trip. The
pixel data stays on the GPU the entire time:

  1. OpenGL allocates a PBO -- a raw GPU buffer.
  2. ``GraphicsResource.from_gl_buffer()`` registers that PBO with CUDA;
     now both CUDA and OpenGL have access to the same GPU memory.
  3. Each frame we ``map()`` the resource (CUDA gets a device pointer into
     the PBO), launch the kernel, and let the context manager ``unmap()``
     the resource so OpenGL can render it.
  4. ``glTexSubImage2D`` copies the PBO into a texture (a fast, GPU-to-GPU
     step) and OpenGL draws the texture on a fullscreen quad.

The animation is a "plasma" effect: layered sine waves that produce
swirling colors. Close the window (or press Escape) to exit.

By default this sample runs for a bounded number of frames so it is
CI-friendly; pass ``--interactive`` to run until the window is closed.
Headless environments (no ``DISPLAY``) waive the sample.
"""

import argparse
import ctypes
import os
import platform
import sys
import time

try:
    import numpy as np

    from cuda.core import (
        Device,
        GraphicsResource,
        LaunchConfig,
        Program,
        ProgramOptions,
        launch,
    )
except ImportError as e:
    print(f"Error: Required package not found: {e}")
    print("Please install from requirements.txt:")
    print("  pip install -r requirements.txt")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Window dimensions (feel free to change these)
# ---------------------------------------------------------------------------
WIDTH = 800
HEIGHT = 600
EXIT_WAIVED = int(os.environ.get("CUDA_PYTHON_SAMPLE_WAIVER_EXIT_CODE", "2"))


# ============================= Helper functions =============================
# The functions below set up CUDA and OpenGL. If you're here to learn about
# GraphicsResource, you can skip straight to main() -- the interesting part
# is there.
# ============================================================================


def setup_cuda(kernel_source, device_id=0):
    """Compile the CUDA kernel and return (device, stream, kernel, launch_config)."""
    dev = Device(device_id)
    dev.set_current()
    stream = dev.create_stream()

    program_options = ProgramOptions(std="c++11", arch=f"sm_{dev.arch}")
    prog = Program(kernel_source, code_type="c++", options=program_options)
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
        caption="GraphicsResource Example - CUDA Plasma",
        vsync=False,
    )
    return window, _gl, pyglet


def create_display_resources(gl, width, height):
    """Create the GL objects needed to show a texture on screen.

    Standard OpenGL boilerplate: a passthrough shader program, a fullscreen
    quad, and an empty texture. Nothing CUDA-specific.

    Returns (shader_program, vao_id, texture_id).
    """
    from pyglet.graphics.shader import Shader, ShaderProgram

    vert = Shader(VERTEX_SHADER_SOURCE, "vertex")
    frag = Shader(FRAGMENT_SHADER_SOURCE, "fragment")
    shader_prog = ShaderProgram(vert, frag)

    # Fullscreen quad (two triangles covering the entire window).
    quad_verts = np.array(
        [
            # x,  y,    s, t   (position + texture coordinate)
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

    A PBO is GPU-side memory that OpenGL can read from when uploading pixels
    to a texture. Registering this same buffer with CUDA lets the kernel
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
    parser = argparse.ArgumentParser(description="CUDA/OpenGL interop plasma demo")
    parser.add_argument(
        "--frames",
        type=int,
        default=60,
        help="Number of frames to render before exiting (default: 60). Ignored in --interactive mode.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run until the window is closed instead of stopping after --frames",
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device id")
    args = parser.parse_args()

    # Waive the sample when there is no display available. On Linux we look
    # at $DISPLAY; on other platforms we just try and let pyglet fail below
    # if it cannot open a window.
    if platform.system() == "Linux" and not os.environ.get("DISPLAY"):
        print("No DISPLAY available; waiving GL interop sample.", file=sys.stderr)
        sys.exit(EXIT_WAIVED)

    # --- Step 1: Set up CUDA (compile kernel, create stream) ---
    _dev, stream, kernel, config = setup_cuda(PLASMA_KERNEL_SOURCE, device_id=args.device)

    # --- Step 2: Open a window ---
    try:
        window, gl, pyglet = create_window()
    except Exception as e:
        print(f"Could not open a pyglet window ({e}); waiving GL interop sample.", file=sys.stderr)
        sys.exit(EXIT_WAIVED)

    # --- Step 3: Create GL resources for drawing a texture to screen ---
    shader_prog, quad_vao, tex_id = create_display_resources(gl, WIDTH, HEIGHT)

    # --- Step 4: Create the Pixel Buffer Object (PBO) ---
    #     A PBO is GPU memory owned by OpenGL. It bridges the two worlds:
    #     CUDA will write into it, and OpenGL will read from it.
    pbo_id, _ = create_pixel_buffer(gl, WIDTH, HEIGHT)

    # --- Step 5: Register the PBO with CUDA ---
    #     KEY LINE. GraphicsResource.from_gl_buffer() tells the CUDA driver
    #     "I want to access this OpenGL buffer from CUDA kernels." WRITE_DISCARD
    #     means CUDA will overwrite the entire buffer each frame.
    resource = GraphicsResource.from_gl_buffer(pbo_id, flags="write_discard")

    # --- Step 6: Render loop ---
    start_time = time.monotonic()
    frame_count = 0
    fps_time = start_time
    frames_rendered = 0
    resources_closed = False

    def close_resources():
        nonlocal resources_closed
        if resources_closed:
            return
        resources_closed = True
        resource.close()
        stream.close()

    @window.event
    def on_draw():
        nonlocal frame_count, fps_time, frames_rendered

        window.clear()
        t = time.monotonic() - start_time

        # (a) Map the PBO so CUDA can write to it. This gives us a Buffer
        #     whose .handle is a CUDA device pointer pointing directly into
        #     the OpenGL PBO's GPU memory.
        with resource.map(stream=stream) as buf:
            # (b) Launch the plasma kernel -- it writes RGBA pixels into buf.
            launch(
                stream,
                config,
                kernel,
                buf.handle,
                np.int32(WIDTH),
                np.int32(HEIGHT),
                np.float32(t),
            )
        # (c) Unmap happens automatically when the `with` block exits.
        #     cuGraphicsUnmapResources guarantees CUDA work on the stream
        #     completes before OpenGL can use the buffer.

        # (d) Copy PBO -> texture (GPU-to-GPU).
        copy_pbo_to_texture(gl, pbo_id, tex_id, WIDTH, HEIGHT)

        # (e) Draw the texture to the screen.
        draw_fullscreen_quad(gl, shader_prog, quad_vao, tex_id)

        frame_count += 1
        frames_rendered += 1
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

        # Terminate after --frames iterations when not interactive.
        if not args.interactive and frames_rendered >= args.frames:
            # Let pyglet finish the current refresh/flip before tearing down
            # the GL context and registered CUDA graphics resource.
            pyglet.app.exit()

    @window.event
    def on_close():
        close_resources()

    try:
        pyglet.app.run(interval=0)
    finally:
        close_resources()
        window.close()
    print(f"\nRendered {frames_rendered} frames via CUDA/OpenGL interop. Done")
    return 0


# ======================== GPU code (CUDA + GLSL) ============================
# PLASMA_KERNEL_SOURCE is CUDA C++ that runs on the GPU. It computes a color
# for each pixel based on layered sine waves (the "plasma" effect) and writes
# RGBA bytes into the output buffer.
#
# VERTEX_SHADER_SOURCE / FRAGMENT_SHADER_SOURCE are GLSL. They draw a texture
# onto a rectangle that covers the entire window.
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

    // val now ranges roughly from -5 to +5. Normalize to [0, 1].
    val = (val + 5.0f) / 10.0f;

    // --- Convert to RGB ---
    // Three sine waves offset by 120 degrees (2*PI/3) give smooth color cycling.
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
    sys.exit(main())
