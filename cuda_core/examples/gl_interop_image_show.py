# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# ################################################################################
#
# Minimal "Hello World" for the cuda.core texture/surface stack.
#
# Allocates a small `CUDAArray`, fills it with a procedural image once, binds it
# as a `TextureObject`, and uses a single CUDA kernel to sample that texture
# at every screen pixel (with a scale + rotation transform) and write the
# result into an OpenGL PBO for display.
#
# Nothing else: no `SurfaceObject`, no ping-pong, no simulation, no mipmaps.
# If you have never touched the new APIs before, open this file first.
#
# ################################################################################
#
# What this example teaches
# =========================
# - Allocate an `CUDAArray` and upload data into it with `CUDAArray.copy_from`.
# - Build a `TextureObject` from a `ResourceDescriptor` + `TextureDescriptor`.
# - The visual difference between `FilterMode.POINT` and `FilterMode.LINEAR`
#   (press F to toggle live).
# - That filter mode is baked into the `TextureDescriptor` at creation time,
#   so changing it requires destroying and rebuilding the `TextureObject`.
#
# How it works
# ============
#   Startup (once):
#     +-------------------+   copy_from   +----------+
#     | host numpy image  | ------------> |  CUDAArray   |  (UINT8 RGBA, 64x64)
#     +-------------------+               +----+-----+
#                                              |
#                                              v
#                                       +-------------+
#                                       | TextureObj  |  (filter mode = POINT)
#                                       +-------------+
#
#   Each frame:
#     - kernel `sample_image` reads from the TextureObject at a transformed
#       (u, v) per screen pixel and writes RGBA bytes to the GL PBO.
#     - OpenGL copies the PBO into a screen texture and draws it.
#
# What you should see
# ===================
# A 64x64 procedural test pattern (checkerboard + colored gradient stripes +
# diagonal lines) magnified to fill the window. Press F to switch between
# POINT (blocky) and LINEAR (smooth) sampling; the difference is immediately
# visible. Press R to start/stop a slow rotation. Esc to quit.
#

# /// script
# dependencies = ["cuda_bindings", "cuda_core>0.6.0", "pyglet"]
# ///

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
from cuda.core.textures import (
    AddressMode,
    ArrayFormat,
    CUDAArray,
    FilterMode,
    ReadMode,
    ResourceDescriptor,
    TextureDescriptor,
    TextureObject,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WIDTH = 640
HEIGHT = 480
IMAGE_SIZE = 64  # the source CUDAArray is IMAGE_SIZE x IMAGE_SIZE RGBA8


# ============================= Helper functions =============================


def make_test_image(size):
    """Build a (size, size, 4) uint8 RGBA test pattern.

    Designed so the filter-mode difference is obvious: hard-edged checkerboard
    (POINT preserves the edges; LINEAR smooths them) plus a vertical color
    gradient stripe (LINEAR blends smoothly between palette stops) plus two
    diagonal hairlines (POINT preserves them; LINEAR softens them).
    """
    img = np.zeros((size, size, 4), dtype=np.uint8)
    # 8x8 black/white checkerboard
    cells = size // 8
    for y in range(size):
        for x in range(size):
            if ((x // cells) + (y // cells)) & 1:
                img[y, x, :3] = 255
    # vertical RGB gradient strip down the left third
    strip = size // 3
    img[:, :strip, 0] = np.linspace(255, 0, size, dtype=np.uint8)[:, None].repeat(strip, axis=1)
    img[:, :strip, 1] = np.linspace(0, 255, size, dtype=np.uint8)[:, None].repeat(strip, axis=1)
    img[:, :strip, 2] = 128
    # two diagonal red hairlines
    for d in range(size):
        img[d, d, :] = [255, 0, 0, 255]
        if d < size - 4:
            img[d, d + 4, :] = [255, 0, 0, 255]
    img[:, :, 3] = 255  # opaque
    return img


def setup_cuda():
    """Compile the kernel and return (device, stream, kernel, launch_config)."""
    dev = Device(0)
    dev.set_current()
    stream = dev.create_stream()

    program_options = ProgramOptions(std="c++17", arch=f"sm_{dev.arch}")
    prog = Program(KERNEL_SOURCE, code_type="c++", options=program_options)
    mod = prog.compile("cubin", name_expressions=("sample_image",))
    kernel = mod.get_kernel("sample_image")

    block = (16, 16, 1)
    grid = (
        (WIDTH + block[0] - 1) // block[0],
        (HEIGHT + block[1] - 1) // block[1],
        1,
    )
    config = LaunchConfig(grid=grid, block=block)
    return dev, stream, kernel, config


def create_window():
    """Open a pyglet window. Returns (window, gl_module, pyglet_module)."""
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
        caption="cuda.core CUDAArray + TextureObject - Image Show",
        vsync=False,
    )
    return window, _gl, pyglet


def create_display_resources(gl, width, height):
    """Standard pyglet boilerplate: shader, fullscreen quad, screen texture."""
    from pyglet.graphics.shader import Shader, ShaderProgram

    vert = Shader(VERTEX_SHADER_SOURCE, "vertex")
    frag = Shader(FRAGMENT_SHADER_SOURCE, "fragment")
    shader_prog = ShaderProgram(vert, frag)

    quad_verts = np.array(
        [
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

    stride = 4 * 4
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
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
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
    """Create the GL PBO that CUDA writes RGBA pixels into each frame."""
    pbo = ctypes.c_uint(0)
    gl.glGenBuffers(1, ctypes.byref(pbo))
    gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, pbo.value)
    nbytes = width * height * 4
    gl.glBufferData(gl.GL_PIXEL_UNPACK_BUFFER, nbytes, None, gl.GL_DYNAMIC_DRAW)
    gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)
    return pbo.value


def copy_pbo_to_texture(gl, pbo_id, tex_id, width, height):
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
        None,
    )
    gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)


def draw_fullscreen_quad(gl, shader_prog, vao_id, tex_id):
    gl.glUseProgram(shader_prog.id)
    gl.glBindTexture(gl.GL_TEXTURE_2D, tex_id)
    gl.glBindVertexArray(vao_id)
    gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)
    gl.glBindVertexArray(0)
    gl.glUseProgram(0)


def make_texture(arr, filter_mode):
    """Build a `TextureObject` for `arr` with the given FilterMode.

    Filter mode is baked into the descriptor at creation; to switch modes
    we close this object and call this helper again.
    """
    res_desc = ResourceDescriptor.from_array(arr)
    tex_desc = TextureDescriptor(
        address_mode=AddressMode.CLAMP,
        filter_mode=filter_mode,
        # UINT8 source + NORMALIZED_FLOAT means tex2D<float4> returns each
        # channel as a float in [0, 1] -- handy for the colorize math below.
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

    # --- Step 3: Create GL resources (shader, fullscreen quad, screen tex) ---
    shader_prog, quad_vao, screen_tex = create_display_resources(gl, WIDTH, HEIGHT)

    # --- Step 4: Create the PBO that CUDA will write into ---
    pbo_id = create_pixel_buffer(gl, WIDTH, HEIGHT)
    resource = GraphicsResource.from_gl_buffer(pbo_id, flags="write_discard")

    # --- Step 5: Allocate the source `CUDAArray` and upload the test pattern ---
    arr = CUDAArray.from_descriptor(
        shape=(IMAGE_SIZE, IMAGE_SIZE),
        format=ArrayFormat.UINT8,
        num_channels=4,
    )
    host_image = make_test_image(IMAGE_SIZE)
    arr.copy_from(np.ascontiguousarray(host_image), stream=stream)
    stream.sync()

    # --- Step 6: Bind the CUDAArray as a TextureObject (initially POINT) ---
    state = {"filter": FilterMode.POINT, "rotate": False, "angle": 0.0}
    tex = make_texture(arr, state["filter"])

    @window.event
    def on_key_press(symbol, _modifiers):
        key = pyglet.window.key
        nonlocal tex
        if symbol == key.ESCAPE:
            window.close()
        elif symbol == key.F:
            # Filter mode is baked at TextureObject creation time. Swapping
            # it means closing the old one and building a new one.
            state["filter"] = FilterMode.LINEAR if state["filter"] == FilterMode.POINT else FilterMode.POINT
            tex.close()
            tex = make_texture(arr, state["filter"])
        elif symbol == key.R:
            state["rotate"] = not state["rotate"]

    # --- Step 7: Render loop ---
    start = time.monotonic()
    last_t = start
    frame_count = 0
    fps_time = start

    @window.event
    def on_draw():
        nonlocal frame_count, fps_time, last_t
        now = time.monotonic()
        if state["rotate"]:
            state["angle"] += (now - last_t) * 0.5  # rad/sec
        last_t = now

        window.clear()
        with resource.map(stream=stream) as buf:
            launch(
                stream,
                config,
                kernel,
                np.uint64(tex.handle),
                buf.handle,
                np.int32(WIDTH),
                np.int32(HEIGHT),
                np.float32(state["angle"]),
            )
        copy_pbo_to_texture(gl, pbo_id, screen_tex, WIDTH, HEIGHT)
        draw_fullscreen_quad(gl, shader_prog, quad_vao, screen_tex)

        frame_count += 1
        if now - fps_time >= 1.0:
            fps = frame_count / (now - fps_time)
            window.set_caption(
                f"cuda.core CUDAArray + TextureObject - Image Show "
                f"(filter={state['filter'].name}, "
                f"rotate={'on' if state['rotate'] else 'off'}, "
                f"{fps:.0f} FPS)"
            )
            frame_count = 0
            fps_time = now

    @window.event
    def on_close():
        tex.close()
        arr.close()
        resource.close()
        stream.close()

    pyglet.app.run(interval=0)


# ============================== GPU code (kernel) ============================

KERNEL_SOURCE = r"""
extern "C"
__global__
void sample_image(cudaTextureObject_t tex,
                  unsigned char* output,
                  int width, int height,
                  float angle) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // Center the screen pixel around (0, 0) in [-aspect, aspect] x [-1, 1].
    float aspect = (float)width / (float)height;
    float sx = ((float)x / (float)width  - 0.5f) * 2.0f * aspect;
    float sy = ((float)y / (float)height - 0.5f) * 2.0f;

    // Inverse-rotate the screen point: rotating the image by +angle means
    // each output pixel reads from the source rotated by -angle.
    float c = cosf(-angle), s = sinf(-angle);
    float rx = c * sx - s * sy;
    float ry = s * sx + c * sy;

    // Map rotated screen point to the [0, 1] x [0, 1] texture domain so the
    // image (drawn centered, fitting ~75% of the window height) lands on it.
    const float scale = 0.75f;
    float u = (rx / (2.0f * scale)) + 0.5f;
    float v = (ry / (2.0f * scale)) + 0.5f;

    // AddressMode.CLAMP means out-of-range u/v sample the edge texel.
    float4 col = tex2D<float4>(tex, u, v);

    int idx = (y * width + x) * 4;
    output[idx + 0] = (unsigned char)(col.x * 255.0f);
    output[idx + 1] = (unsigned char)(col.y * 255.0f);
    output[idx + 2] = (unsigned char)(col.z * 255.0f);
    output[idx + 3] = 255;
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
    main()
