# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# ################################################################################
#
# This example demonstrates cuda.core.TextureObject hardware filtering by
# comparing FilterMode.POINT and FilterMode.LINEAR side by side on the same
# source CUDA Array. Requires pyglet.
#
# ################################################################################

# What this example teaches
# =========================
# How to back two TextureObjects with the SAME CUDA Array and observe the
# difference between POINT (nearest-texel) and LINEAR (bilinear) filtering
# under user-controlled zoom and pan.  Also shows how the address mode
# (WRAP / CLAMP / MIRROR / BORDER) is baked into the texture descriptor at
# creation time, so changing it at runtime means rebuilding the textures.
#
# How it works
# ============
# A single 256x256 RGBA8 Array holds a procedurally-generated test pattern
# (high-contrast checkerboard, diagonals, gradient stripe).  Two
# TextureObjects are built on top of that Array:
#
#       Array (256x256 RGBA UINT8)
#       /                       \
#   tex_point                  tex_linear
#   FilterMode.POINT           FilterMode.LINEAR
#   AddressMode.WRAP           AddressMode.WRAP
#   ReadMode.NORMALIZED_FLOAT  ReadMode.NORMALIZED_FLOAT
#
# Each frame, a single CUDA kernel runs over a 1024x512 OpenGL PBO:
#
#   - Left half of the screen samples tex_point.
#   - Right half samples tex_linear.
#   - Both halves use the same (zoom, pan) -> texture-space mapping, so the
#     two views show the same content with different filtering.
#   - A 2-pixel vertical white line marks the divider.
#
# Because ReadMode.NORMALIZED_FLOAT is used, tex2D<float4>() returns each
# channel as a float in [0, 1]; the kernel multiplies by 255 and writes
# unsigned bytes back into the PBO.
#
# The PBO is then copied to a GL texture and drawn on a fullscreen quad,
# identical to the plasma example.
#
# What you should see
# ===================
# A 1024x512 window split down the middle.  The left half (POINT) shows
# blocky / pixelated magnification; the right half (LINEAR) shows smooth
# bilinear interpolation.  Drag with the left mouse button to pan,
# scroll to zoom, press M to cycle the texture address mode, press R to
# reset, Escape or close the window to exit.  The current address mode
# and FPS are shown in the window title.
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
    Array,
    ArrayFormat,
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
# Window and source-image dimensions (feel free to change these)
# ---------------------------------------------------------------------------
WIDTH = 1024
HEIGHT = 512
SRC_W = 256
SRC_H = 256

# Address modes cycled by pressing the M key.
ADDRESS_MODES = (
    AddressMode.WRAP,
    AddressMode.CLAMP,
    AddressMode.MIRROR,
    AddressMode.BORDER,
)


# ============================= Helper functions =============================
#
# The functions below set up CUDA and OpenGL.  If you're here to learn about
# TextureObject filtering, the most interesting parts are in main() and in
# make_pattern() / make_textures(); everything else is the same kind of
# CUDA-GL interop boilerplate used by gl_interop_plasma.py.
# ============================================================================


def make_pattern(width, height):
    """Build an RGBA8 test pattern that makes POINT vs LINEAR obvious.

    Layout (height, width, 4) of dtype uint8.  Channels are R, G, B, A.
    The pattern contains:
      - 8x8 black/white checkerboard (high-frequency)
      - Two diagonal red lines (1px wide)
      - Horizontal blue->green gradient strip near y = height/4
      - A pair of thin horizontal rectangles ("text-like" blocks)
    """
    img = np.zeros((height, width, 4), dtype=np.uint8)

    # Checkerboard (black / white) at 8x8 cells.
    ys = np.arange(height)[:, None]
    xs = np.arange(width)[None, :]
    cell = ((xs // 8) + (ys // 8)) & 1
    white = np.broadcast_to(cell[..., None].astype(np.uint8) * 255, (height, width, 3))
    img[..., :3] = white
    img[..., 3] = 255

    # Two diagonal red lines.
    diag1 = (xs == ys)
    diag2 = (xs == (width - 1 - ys))
    red_mask = diag1 | diag2
    img[red_mask] = (255, 0, 0, 255)

    # Horizontal gradient strip (blue -> green) ~ 8 rows tall at y ~ height/4.
    g_y = height // 4
    g_h = max(4, height // 32)
    grad = np.linspace(0, 255, width, dtype=np.uint8)
    for row in range(g_y, min(g_y + g_h, height)):
        img[row, :, 0] = 0
        img[row, :, 1] = grad             # G ramps up
        img[row, :, 2] = 255 - grad       # B ramps down
        img[row, :, 3] = 255

    # Two "text-like" thin rectangles, alternating bright/dim.
    def fill_rect(y0, y1, x0, x1, rgba):
        img[y0:y1, x0:x1] = rgba

    bar_y = (3 * height) // 4
    fill_rect(bar_y, bar_y + 4, width // 8, (width * 3) // 8, (255, 255, 0, 255))
    fill_rect(bar_y + 8, bar_y + 12, (width * 5) // 8, (width * 7) // 8,
              (0, 255, 255, 255))

    return np.ascontiguousarray(img)


def make_textures(array, address_mode):
    """Build (tex_point, tex_linear) on the given Array with the given mode.

    The address mode is baked into the descriptor at cuTexObjectCreate time, so
    we recreate both textures whenever the user cycles the mode.  Caller owns
    the returned objects and must close() them.
    """
    res_desc = ResourceDescriptor.from_array(array)

    point_desc = TextureDescriptor(
        address_mode=address_mode,
        filter_mode=FilterMode.POINT,
        read_mode=ReadMode.NORMALIZED_FLOAT,
        normalized_coords=False,
    )
    linear_desc = TextureDescriptor(
        address_mode=address_mode,
        filter_mode=FilterMode.LINEAR,
        read_mode=ReadMode.NORMALIZED_FLOAT,
        normalized_coords=False,
    )
    tex_point = TextureObject.from_descriptor(
        resource=res_desc, texture_descriptor=point_desc
    )
    tex_linear = TextureObject.from_descriptor(
        resource=res_desc, texture_descriptor=linear_desc
    )
    return tex_point, tex_linear


def setup_cuda(kernel_source):
    """Compile the CUDA kernel and return (device, stream, kernel, launch_config)."""
    dev = Device(0)
    dev.set_current()
    stream = dev.create_stream()

    # C++ compile so the templated tex2D<float4> overload resolves.
    program_options = ProgramOptions(std="c++17", arch=f"sm_{dev.arch}")
    prog = Program(kernel_source, code_type="c++", options=program_options)
    mod = prog.compile("cubin", name_expressions=("split_screen_sample",))
    kernel = mod.get_kernel("split_screen_sample")

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
        caption="TextureObject Filter Comparison - POINT vs LINEAR",
        vsync=False,
    )
    return window, _gl, pyglet


def create_display_resources(gl, width, height):
    """Create the GL objects needed to show a texture on screen.

    Standard OpenGL boilerplate for a textured fullscreen quad, identical in
    structure to the plasma example.  Returns (shader_program, vao_id, tex_id).
    """
    from pyglet.graphics.shader import Shader, ShaderProgram

    vert = Shader(VERTEX_SHADER_SOURCE, "vertex")
    frag = Shader(FRAGMENT_SHADER_SOURCE, "fragment")
    shader_prog = ShaderProgram(vert, frag)

    # Fullscreen quad (two triangles).  Each vertex: x, y, s, t.
    quad_verts = np.array(
        [
            -1, -1, 0, 0,
             1, -1, 1, 0,
             1,  1, 1, 1,
            -1, -1, 0, 0,
             1,  1, 1, 1,
            -1,  1, 0, 1,
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

    # Empty GL texture; filled each frame from the PBO.
    tex = ctypes.c_uint(0)
    gl.glGenTextures(1, ctypes.byref(tex))
    gl.glBindTexture(gl.GL_TEXTURE_2D, tex.value)
    # Use nearest filtering on the display texture so the example's own
    # POINT/LINEAR comparison is not muddied by GL's sampler.
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
    """Create a Pixel Buffer Object (PBO) sized for one RGBA8 frame."""
    pbo = ctypes.c_uint(0)
    gl.glGenBuffers(1, ctypes.byref(pbo))
    gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, pbo.value)
    nbytes = width * height * 4
    gl.glBufferData(gl.GL_PIXEL_UNPACK_BUFFER, nbytes, None, gl.GL_DYNAMIC_DRAW)
    gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)
    return pbo.value, nbytes


def copy_pbo_to_texture(gl, pbo_id, tex_id, width, height):
    """Copy pixel data from the PBO into the GL texture (GPU-to-GPU)."""
    gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, pbo_id)
    gl.glBindTexture(gl.GL_TEXTURE_2D, tex_id)
    gl.glTexSubImage2D(
        gl.GL_TEXTURE_2D, 0, 0, 0, width, height,
        gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None,
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
    dev, stream, kernel, config = setup_cuda(KERNEL_SOURCE)

    # The hardware-texture path needs at least compute capability 3.x
    # (it's available essentially everywhere modern, but check anyway so the
    # failure is friendly).
    if dev.compute_capability.major < 3:
        print(
            f"This example requires compute capability >= 3.0, "
            f"got {dev.compute_capability.major}.{dev.compute_capability.minor}.",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Step 2: Open a window ---
    window, gl, pyglet = create_window()

    # --- Step 3: Create GL resources (shader, quad, display texture) ---
    shader_prog, quad_vao, tex_id = create_display_resources(gl, WIDTH, HEIGHT)

    # --- Step 4: Create the Pixel Buffer Object (PBO) ---
    pbo_id, _nbytes = create_pixel_buffer(gl, WIDTH, HEIGHT)

    # --- Step 5: Register the PBO with CUDA ---
    resource = GraphicsResource.from_gl_buffer(pbo_id, flags="write_discard")

    # --- Step 6: Allocate the source Array and upload the test pattern ---
    #     The Array lives for the entire program, so we use a `with` block.
    #     Inside it we create / re-create two TextureObjects whenever the
    #     user cycles the address mode.
    with Array.from_descriptor(
        shape=(SRC_W, SRC_H),
        format=ArrayFormat.UINT8,
        num_channels=4,
    ) as arr:
        pattern = make_pattern(SRC_W, SRC_H)
        # Sanity: 256 * 256 * 4 bytes = 262144.
        assert pattern.nbytes == arr.size_bytes, (
            f"pattern bytes ({pattern.nbytes}) != array bytes ({arr.size_bytes})"
        )
        arr.copy_from(pattern, stream=stream)
        stream.sync()  # upload must finish before kernel reads

        # --- Step 7: Build initial POINT + LINEAR textures (WRAP mode). ---
        # We can't use a `with` block here because the address mode is baked
        # into the descriptor at creation time: cycling modes means closing
        # and recreating these objects.  We instead hold them in mutable
        # closure state and release them in on_close().
        tex_state = {
            "mode_idx": 0,
            "tex_point": None,
            "tex_linear": None,
        }

        def rebuild_textures():
            # Close previous textures (if any) before creating new ones so we
            # don't leak handles when cycling the address mode.
            if tex_state["tex_point"] is not None:
                tex_state["tex_point"].close()
            if tex_state["tex_linear"] is not None:
                tex_state["tex_linear"].close()
            mode = ADDRESS_MODES[tex_state["mode_idx"]]
            tp, tl = make_textures(arr, mode)
            tex_state["tex_point"] = tp
            tex_state["tex_linear"] = tl

        rebuild_textures()

        # --- Step 8: View state (zoom + pan), tight initial framing. ---
        # zoom = pixels_per_texel.  zoom=3 -> roughly 3x magnification, which
        # makes POINT vs LINEAR obvious without any user input.
        view = {
            "zoom": 3.0,
            "pan_x": SRC_W * 0.5,
            "pan_y": SRC_H * 0.5,
            "drag": False,
        }

        def reset_view():
            view["zoom"] = 3.0
            view["pan_x"] = SRC_W * 0.5
            view["pan_y"] = SRC_H * 0.5

        # --- Step 9: Render loop ---
        start_time = time.monotonic()
        frame_count = 0
        fps_time = start_time

        def current_mode_name():
            return ADDRESS_MODES[tex_state["mode_idx"]].name

        @window.event
        def on_draw():
            nonlocal frame_count, fps_time
            window.clear()

            # (a) Map the PBO so CUDA can write to it.
            with resource.map(stream=stream) as buf:
                # (b) Launch the split-screen sampling kernel.
                launch(
                    stream,
                    config,
                    kernel,
                    np.uint64(tex_state["tex_point"].handle),
                    np.uint64(tex_state["tex_linear"].handle),
                    buf.handle,
                    np.int32(WIDTH),
                    np.int32(HEIGHT),
                    np.float32(view["zoom"]),
                    np.float32(view["pan_x"]),
                    np.float32(view["pan_y"]),
                    np.int32(SRC_W),
                    np.int32(SRC_H),
                )
            # (c) Unmap happens automatically when the `with` block exits.

            # (d) PBO -> GL texture (GPU-to-GPU).
            copy_pbo_to_texture(gl, pbo_id, tex_id, WIDTH, HEIGHT)

            # (e) Draw the texture to the screen.
            draw_fullscreen_quad(gl, shader_prog, quad_vao, tex_id)

            frame_count += 1
            now = time.monotonic()
            if now - fps_time >= 1.0:
                fps = frame_count / (now - fps_time)
                window.set_caption(
                    f"TextureObject Filter - POINT | LINEAR  "
                    f"[address={current_mode_name()}, zoom={view['zoom']:.2f}x, "
                    f"{fps:.0f} FPS]"
                )
                frame_count = 0
                fps_time = now

        # --- Mouse: drag to pan, scroll to zoom ------------------------------
        @window.event
        def on_mouse_press(x, y, button, modifiers):
            if button == pyglet.window.mouse.LEFT:
                view["drag"] = True

        @window.event
        def on_mouse_release(x, y, button, modifiers):
            if button == pyglet.window.mouse.LEFT:
                view["drag"] = False

        @window.event
        def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
            if not (buttons & pyglet.window.mouse.LEFT):
                return
            # Pyglet dy is screen-up-positive; texture y is texel-down-positive.
            # One screen pixel = 1/zoom texels in source space.
            view["pan_x"] -= dx / view["zoom"]
            view["pan_y"] += dy / view["zoom"]

        @window.event
        def on_mouse_scroll(x, y, scroll_x, scroll_y):
            # Geometric zoom; clamp to a sensible range.
            factor = 1.1 ** scroll_y
            new_zoom = view["zoom"] * factor
            view["zoom"] = max(0.1, min(32.0, new_zoom))

        # --- Keyboard: M cycles address mode, R resets view ------------------
        @window.event
        def on_key_press(symbol, modifiers):
            key = pyglet.window.key
            if symbol == key.M:
                tex_state["mode_idx"] = (tex_state["mode_idx"] + 1) % len(ADDRESS_MODES)
                rebuild_textures()
            elif symbol == key.R:
                reset_view()
            elif symbol == key.ESCAPE:
                window.close()

        @window.event
        def on_close():
            # Release CUDA resources in reverse order of creation.
            if tex_state["tex_linear"] is not None:
                tex_state["tex_linear"].close()
                tex_state["tex_linear"] = None
            if tex_state["tex_point"] is not None:
                tex_state["tex_point"].close()
                tex_state["tex_point"] = None
            resource.close()

        pyglet.app.run(interval=0)


# ======================== GPU code (CUDA + GLSL) ============================
#
# KERNEL_SOURCE samples the same source Array through two TextureObjects
# (POINT vs LINEAR) and writes RGBA8 pixels into the PBO.  ReadMode.
# NORMALIZED_FLOAT means tex2D<float4>() returns each channel in [0, 1];
# the kernel scales by 255 and writes unsigned bytes back out.
#
# VERTEX_SHADER_SOURCE / FRAGMENT_SHADER_SOURCE are plain GLSL that draws
# a texture on a fullscreen quad -- nothing CUDA-specific.
# ============================================================================

KERNEL_SOURCE = r"""
extern "C" __global__
void split_screen_sample(cudaTextureObject_t point_tex,
                         cudaTextureObject_t linear_tex,
                         unsigned char* out,
                         int w, int h,
                         float zoom,
                         float pan_x, float pan_y,
                         int src_w, int src_h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int half_w = w / 2;

    // 2-pixel-wide white separator down the middle.
    if (x == half_w || x == half_w - 1) {
        int idx = (y * w + x) * 4;
        out[idx + 0] = 255;
        out[idx + 1] = 255;
        out[idx + 2] = 255;
        out[idx + 3] = 255;
        return;
    }

    // Each half of the screen samples the same (src_x, src_y) so the two
    // sides line up visually for an apples-to-apples filter comparison.
    float local_x = (x < half_w) ? (float)x : (float)(x - half_w);

    // (src_x, src_y) in source-texture pixel coordinates.  Non-normalized
    // coords are used, so coordinate (i + 0.5, j + 0.5) selects texel (i, j).
    float src_x = pan_x + (local_x - (float)half_w * 0.5f) / zoom;
    float src_y = pan_y + ((float)y     - (float)h      * 0.5f) / zoom;

    float4 sample;
    if (x < half_w) {
        sample = tex2D<float4>(point_tex,  src_x, src_y);
    } else {
        sample = tex2D<float4>(linear_tex, src_x, src_y);
    }

    int idx = (y * w + x) * 4;
    out[idx + 0] = (unsigned char)(sample.x * 255.0f);
    out[idx + 1] = (unsigned char)(sample.y * 255.0f);
    out[idx + 2] = (unsigned char)(sample.z * 255.0f);
    out[idx + 3] = (unsigned char)(sample.w * 255.0f);
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
