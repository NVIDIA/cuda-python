# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# ################################################################################
#
# This example demonstrates cuda.core.CUDAArray, TextureObject, and SurfaceObject
# in combination with GraphicsResource for CUDA/OpenGL interop: a classic
# "Doom-style" procedural fire effect. A scalar heat field lives on a
# ping-ponged float CUDA CUDAArray; each frame the field is advected upward with a
# horizontal jitter and a small decay, then colorized through a 1D fire-palette
# TextureObject straight into an OpenGL PBO. Requires pyglet.
#
# ################################################################################

# What this example teaches
# =========================
# - How to combine a 2D float CUDAArray (the heat field) and a 1D RGBA8 CUDAArray (the
#   color palette) under the same texture/surface API.
# - How to ping-pong a scalar field via CUDAArray + SurfaceObject writes and
#   TextureObject reads, similar to the reaction-diffusion example but with a
#   single channel.
# - How to use TextureObject(NORMALIZED_FLOAT) on a UINT8 palette so a
#   tex1D<float4> lookup returns RGBA in [0, 1] -- no manual unpacking needed.
# - How to wire mouse / keyboard events into a CUDA simulation without
#   blocking the event loop.
#
# How it works
# ============
# The heat field is a WIDTH x HEIGHT scalar in [0, 1]. Each frame we:
#
#   1. step kernel: for every pixel,
#        - if y is near the bottom AND ambient injection is on, write random
#          high heat ("the embers");
#        - if the mouse button is held, paint a hot disc near the cursor;
#        - otherwise read a horizontally-jittered sample from the row "below"
#          (i.e. one texel toward the bottom of the screen) and subtract a
#          small decay. This is what creates the upward-flickering motion.
#   2. colorize kernel: per pixel, sample the heat, look it up in a 1D RGBA8
#      fire palette via tex1D<float4>, and write RGBA bytes into the PBO.
#
#   PING-PONG (two single-channel float Arrays)
#   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   +-------------+   tex2D<float>    +-------------+
#   |   heat_a    | ----------------> |             |
#   | (FLOAT32 x1)|                   |  step_fire  |
#   +-------------+                   |   kernel    |
#                                     |             |
#   +-------------+   surf2Dwrite     |             |
#   |   heat_b    | <---------------- |             |
#   | (FLOAT32 x1)|                   +-------------+
#   +-------------+
#       (swap)
#
# Orientation
# -----------
# OpenGL displays texel row 0 at the bottom of the window. The fullscreen quad
# in create_display_resources() flips t so that kernel y=0 lands at the TOP of
# the screen -- this lets the kernel keep the intuitive "inject at y = h-1,
# advect from y+1 -> y" convention while the visible flames rise upward.
# Mouse coordinates from pyglet (y=0 at window bottom) are flipped to the
# kernel's y-down convention on entry.
#
# surf2Dwrite x-in-bytes
# ----------------------
# `surf2Dwrite` takes the x coordinate in BYTES, not in elements. For a
# float surface that means `x * sizeof(float)` = `x * 4`. Getting this wrong
# silently corrupts every other column.
#
# What you should see
# ===================
# A flickering wall of doom-style fire rising from the bottom of the window.
# Hold the mouse button and drag to paint a torch of heat at the cursor.
# Press SPACE to toggle the ambient embers along the bottom row (the fire
# will die out when ambient is OFF). Press R to clear the heat field.
# Press Escape or close the window to exit. The window title shows FPS and
# whether ambient injection is currently on.
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
    CUDAArray,
    ArrayFormat,
    Device,
    FilterMode,
    GraphicsResource,
    LaunchConfig,
    Program,
    ProgramOptions,
    ReadMode,
    ResourceDescriptor,
    SurfaceObject,
    TextureDescriptor,
    TextureObject,
    launch,
)

# ---------------------------------------------------------------------------
# Simulation parameters (feel free to change these)
# ---------------------------------------------------------------------------
# Window dimensions (what the user sees).
WINDOW_WIDTH = 640
WINDOW_HEIGHT = 480

# Simulation dimensions (the heat-field grid). Doom's actual screen was
# 320x200; we use 320x100 so the canonical decay rate of ~1 intensity unit
# per row (random {0, 1, 2}, average 1) produces flames that reach ~36% of
# the screen height -- the recognizable "tall licking flames" look.
# NEAREST-filtered upscale to the 640x480 window stretches vertically 4.8x,
# giving the chunky retro pixel-doubled appearance.
WIDTH = 320
HEIGHT = 100

# Canonical Doom fire palette: 37 hand-tuned colors (intensity 0..36 -> RGB).
# Source: https://github.com/tiagomenegaz/doom-fire (and Fabien Sanglard's
# analysis of the original PSX Doom fire effect).
PALETTE_SIZE = 37
MAX_INTENSITY = 36
TORCH_RADIUS = 12  # pixel radius of the mouse-painted hot disc (sim space)


# ============================= Helper functions =============================
#
# The functions below set up CUDA and OpenGL. If you're here to learn about
# CUDAArray/TextureObject/SurfaceObject, skip ahead to main() -- the interesting
# part is there. These helpers exist so that main() reads like a short story
# instead of a wall of boilerplate.
# ============================================================================


def setup_cuda():
    """Compile the CUDA kernels and return (device, stream, kernels, configs)."""
    dev = Device(0)
    dev.set_current()

    # SurfaceObject requires surface load/store, which has existed since SM 2.0,
    # but bindless surface objects (cuSurfObjectCreate) require SM 3.0+.
    cc = dev.compute_capability
    if cc.major < 3:
        print(
            "This example requires a GPU with compute capability >= 3.0 for "
            f"bindless surface objects. Found sm_{cc.major}{cc.minor}.",
            file=sys.stderr,
        )
        sys.exit(1)

    stream = dev.create_stream()

    # Compile as C++ so the templated tex1D<float4> / tex2D<float> overloads
    # resolve.
    program_options = ProgramOptions(std="c++17", arch=f"sm_{dev.arch}")
    prog = Program(KERNEL_SOURCE, code_type="c++", options=program_options)
    mod = prog.compile(
        "cubin",
        name_expressions=("step_fire", "colorize_fire"),
    )

    kernels = {
        "step": mod.get_kernel("step_fire"),
        "colorize": mod.get_kernel("colorize_fire"),
    }

    block = (16, 16, 1)
    grid = (
        (WIDTH + block[0] - 1) // block[0],
        (HEIGHT + block[1] - 1) // block[1],
        1,
    )
    config = LaunchConfig(grid=grid, block=block)
    # Both kernels are pixel-parallel over a WIDTH x HEIGHT grid.
    configs = {"step": config, "colorize": config}

    return dev, stream, kernels, configs


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
        WINDOW_WIDTH,
        WINDOW_HEIGHT,
        caption="cuda.core CUDAArray/Texture/Surface - Doom Fire",
        vsync=False,
    )
    return window, _gl, pyglet


def create_display_resources(gl, width, height):
    """Create the GL objects needed to show a texture on screen.

    Standard OpenGL boilerplate for a textured fullscreen quad. The texcoord
    `t` is flipped versus the plasma example so that kernel y=0 lands at the
    TOP of the screen. That lets the fire kernel keep the intuitive
    "inject at the largest y, advect upward" convention while the visible
    flames rise toward the top.

    Returns (shader_program, vertex_array_id, texture_id).
    """
    from pyglet.graphics.shader import Shader, ShaderProgram

    vert = Shader(VERTEX_SHADER_SOURCE, "vertex")
    frag = Shader(FRAGMENT_SHADER_SOURCE, "fragment")
    shader_prog = ShaderProgram(vert, frag)

    # Fullscreen quad (two triangles covering the entire window). Note the
    # flipped t coordinates compared to gl_interop_plasma: (-1, -1) gets t=1
    # so screen-bottom samples the kernel's largest-y row.
    quad_verts = np.array(
        [
            # x,  y,    s, t      (position + texture coordinate)
            -1, -1, 0, 1,
             1, -1, 1, 1,
             1,  1, 1, 0,
            -1, -1, 0, 1,
             1,  1, 1, 0,
            -1,  1, 0, 0,
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

    # Empty texture (filled each frame from the PBO).
    tex = ctypes.c_uint(0)
    gl.glGenTextures(1, ctypes.byref(tex))
    gl.glBindTexture(gl.GL_TEXTURE_2D, tex.value)
    # NEAREST upscale: makes the low-res simulation render with crisp,
    # blocky pixels instead of bilinear-blended mush. Critical to the
    # Doom-fire look.
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
    """Create a Pixel Buffer Object (PBO) -- the bridge between CUDA and OpenGL.

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


def make_heat_arrays():
    """Allocate two single-channel UINT8 ping-pong Arrays for the heat field.

    Intensity is an integer in [0, 36] indexing the canonical Doom palette.
    UINT8 is exactly one byte per texel -- surf2Dwrite x-coord = x * 1.
    """
    arr_a = CUDAArray.from_descriptor(
        shape=(WIDTH, HEIGHT),
        format=ArrayFormat.UINT8,
        num_channels=1,
        surface_load_store=True,
    )
    arr_b = CUDAArray.from_descriptor(
        shape=(WIDTH, HEIGHT),
        format=ArrayFormat.UINT8,
        num_channels=1,
        surface_load_store=True,
    )
    return arr_a, arr_b


def make_heat_texture(arr):
    """Bind `arr` as a TextureObject configured for POINT + CLAMP reads.

    POINT filtering is what gives Doom fire its chunky retro look. LINEAR
    smooths the per-frame horizontal jitter into a uniform glow that
    doesn't read as fire.
    """
    res_desc = ResourceDescriptor.from_array(arr)
    tex_desc = TextureDescriptor(
        address_mode=AddressMode.CLAMP,
        filter_mode=FilterMode.POINT,
        read_mode=ReadMode.ELEMENT_TYPE,
        # Non-normalized: the step kernel addresses texels in pixel space.
        normalized_coords=False,
    )
    return TextureObject.from_descriptor(resource=res_desc, texture_descriptor=tex_desc)


def build_fire_palette():
    """Return the canonical Doom fire palette as a (37, 4) uint8 array.

    The 37 entries map intensity 0 (black) -> 36 (white). Each entry is
    indexed by the integer intensity in the heat field.

    Source: Fabien Sanglard's PSX Doom analysis, reproduced in
    https://github.com/tiagomenegaz/doom-fire.
    """
    rgb = [
        (  7,   7,   7), ( 31,   7,   7), ( 47,  15,   7), ( 71,  15,   7),
        ( 87,  23,   7), (103,  31,   7), (119,  31,   7), (143,  39,   7),
        (159,  47,   7), (175,  63,   7), (191,  71,   7), (199,  71,   7),
        (223,  79,   7), (223,  87,   7), (223,  87,   7), (215,  95,   7),
        (215,  95,   7), (215, 103,  15), (207, 111,  15), (207, 119,  15),
        (207, 127,  15), (207, 135,  23), (199, 135,  23), (199, 143,  23),
        (199, 151,  31), (191, 159,  31), (191, 159,  31), (191, 167,  39),
        (191, 167,  39), (191, 175,  47), (183, 175,  47), (183, 183,  47),
        (183, 183,  55), (207, 207, 111), (223, 223, 159), (239, 239, 199),
        (255, 255, 255),
    ]
    # Index 0 (the "no fire" color) is rendered as pure black so dead pixels
    # don't glow. The canonical (7, 7, 7) reads as a dim background which is
    # less dramatic against the dark window.
    rgb[0] = (0, 0, 0)
    assert len(rgb) == PALETTE_SIZE
    rgba = np.empty((PALETTE_SIZE, 4), dtype=np.uint8)
    rgba[:, :3] = np.array(rgb, dtype=np.uint8)
    rgba[:, 3] = 255
    return rgba


def make_palette_array_and_texture(stream):
    """Allocate the 1D RGBA8 palette CUDAArray, upload, and bind as a texture.

    Returns (palette_array, palette_texture). Both must be closed by the
    caller (or used inside `with` blocks).
    """
    palette = build_fire_palette()  # shape (PALETTE_SIZE, 4), uint8
    arr = CUDAArray.from_descriptor(
        shape=(PALETTE_SIZE,),
        format=ArrayFormat.UINT8,
        num_channels=4,
    )
    # 1D CUDAArray bytes match a flat (PALETTE_SIZE * 4) uint8 buffer.
    arr.copy_from(np.ascontiguousarray(palette), stream=stream)

    res_desc = ResourceDescriptor.from_array(arr)
    tex_desc = TextureDescriptor(
        address_mode=AddressMode.CLAMP,
        # POINT keeps the palette stops as discrete color bands -- the
        # classic Doom fire palette is indexed, not gradient-blended.
        filter_mode=FilterMode.POINT,
        # NORMALIZED_FLOAT: tex1D<float4> returns each UINT8 channel as a
        # float in [0, 1], so the colorize kernel can multiply by 255 and
        # store directly without manual unpacking.
        read_mode=ReadMode.NORMALIZED_FLOAT,
        # Normalized: the kernel feeds a heat value in [0, 1] as the LUT
        # coordinate. With normalized_coords=True the LINEAR filter blends
        # adjacent palette entries smoothly.
        normalized_coords=True,
    )
    tex = TextureObject.from_descriptor(resource=res_desc, texture_descriptor=tex_desc)
    return arr, tex


# ================================== main() ==================================


def main():
    # --- Step 1: Set up CUDA (compile kernels, create stream) ---
    dev, stream, kernels, configs = setup_cuda()

    # --- Step 2: Open a window ---
    window, gl, pyglet = create_window()

    # --- Step 3: Create GL resources for drawing a texture to screen ---
    #     (Standard OpenGL boilerplate -- not CUDA-specific.)
    shader_prog, quad_vao, tex_id = create_display_resources(gl, WIDTH, HEIGHT)

    # --- Step 4: Create the Pixel Buffer Object (PBO) ---
    pbo_id, _ = create_pixel_buffer(gl, WIDTH, HEIGHT)

    # --- Step 5: Register the PBO with CUDA ---
    resource = GraphicsResource.from_gl_buffer(pbo_id, flags="write_discard")

    # --- Step 6: Allocate heat-field Arrays, palette CUDAArray, and the four
    #             bindless handles (textures + surfaces). We hold them open
    #             for the lifetime of the window and release in on_close(),
    #             matching the reaction-diffusion example. (Using `with`
    #             blocks here would close everything before the pyglet event
    #             loop has a chance to use them.)
    arr_a, arr_b = make_heat_arrays()
    palette_arr, palette_tex = make_palette_array_and_texture(stream)
    tex_a = make_heat_texture(arr_a)
    tex_b = make_heat_texture(arr_b)
    surf_a = SurfaceObject.from_array(arr_a)
    surf_b = SurfaceObject.from_array(arr_b)

    # The heat field is born zeroed by CUDAArray.from_descriptor. No seed pass.
    state = {
        "current": "a",            # which array holds the latest heat field
        "frame_index": 0,           # passed into the step kernel as `t`
        "ambient": True,            # SPACE toggles bottom-row injection
        "mouse_down": False,
        "mouse_x": 0,
        "mouse_y": 0,
    }

    def current_read_write():
        if state["current"] == "a":
            return tex_a, surf_b, "b"  # read a, write b, next current = b
        return tex_b, surf_a, "a"

    def clear_field():
        """Zero both heat arrays and seed the bottom row at full intensity.

        CUDAArray.copy_from is the simplest reset path -- a dedicated clear
        kernel would be faster but is unnecessary for an interactive demo.
        The bottom row is set to MAX_INTENSITY so the very first frame
        already has a fire source to advect from.
        """
        seed = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        seed[HEIGHT - 1, :] = MAX_INTENSITY  # canonical Doom fire source
        arr_a.copy_from(np.ascontiguousarray(seed), stream=stream)
        arr_b.copy_from(np.ascontiguousarray(seed), stream=stream)
        state["current"] = "a"

    # Seed at startup so frame 1 already has a source row.
    clear_field()
    stream.sync()

    # --- Step 7: Render loop ---
    start_time = time.monotonic()
    frame_count = 0
    fps_time = start_time

    @window.event
    def on_key_press(symbol, _modifiers):
        key = pyglet.window.key
        if symbol == key.ESCAPE:
            window.close()
            return
        if symbol == key.SPACE:
            state["ambient"] = not state["ambient"]
            return
        if symbol == key.R:
            clear_field()
            return

    # Map window coords (WINDOW_WIDTH x WINDOW_HEIGHT, y=0 at bottom) to
    # simulation coords (WIDTH x HEIGHT, y=0 at top).
    def _window_to_sim(x, y):
        sx = int(x * WIDTH / WINDOW_WIDTH)
        sy = int((WINDOW_HEIGHT - 1 - y) * HEIGHT / WINDOW_HEIGHT)
        return sx, sy

    @window.event
    def on_mouse_press(x, y, _button, _modifiers):
        state["mouse_down"] = True
        state["mouse_x"], state["mouse_y"] = _window_to_sim(x, y)

    @window.event
    def on_mouse_release(_x, _y, _button, _modifiers):
        state["mouse_down"] = False

    @window.event
    def on_mouse_drag(x, y, _dx, _dy, _buttons, _modifiers):
        state["mouse_down"] = True
        state["mouse_x"], state["mouse_y"] = _window_to_sim(x, y)

    @window.event
    def on_draw():
        nonlocal frame_count, fps_time

        window.clear()

        # (a) Advance the heat field by one step.
        tex_read, surf_write, next_current = current_read_write()
        launch(
            stream,
            configs["step"],
            kernels["step"],
            np.uint64(tex_read.handle),
            np.uint64(surf_write.handle),
            np.int32(WIDTH),
            np.int32(HEIGHT),
            np.uint32(state["frame_index"]),
            np.int32(state["mouse_x"]),
            np.int32(state["mouse_y"]),
            np.int32(1 if state["mouse_down"] else 0),
            np.int32(1 if state["ambient"] else 0),
        )
        state["current"] = next_current
        state["frame_index"] += 1

        # (b) Colorize the latest state into the OpenGL PBO.
        tex_heat = tex_a if state["current"] == "a" else tex_b
        with resource.map(stream=stream) as buf:
            launch(
                stream,
                configs["colorize"],
                kernels["colorize"],
                np.uint64(tex_heat.handle),
                np.uint64(palette_tex.handle),
                buf.handle,
                np.int32(WIDTH),
                np.int32(HEIGHT),
            )
        # Unmap happens automatically when the `with` block exits.

        # (c) Tell OpenGL to copy the PBO contents into our texture.
        copy_pbo_to_texture(gl, pbo_id, tex_id, WIDTH, HEIGHT)

        # (d) Draw the texture to the screen.
        draw_fullscreen_quad(gl, shader_prog, quad_vao, tex_id)

        # FPS counter (shown in window title)
        frame_count += 1
        now = time.monotonic()
        if now - fps_time >= 1.0:
            fps = frame_count / (now - fps_time)
            ambient_label = "on" if state["ambient"] else "off"
            window.set_caption(
                "cuda.core CUDAArray/Texture/Surface - Doom Fire"
                f" ({WIDTH}x{HEIGHT}, {fps:.0f} FPS,"
                f" ambient {ambient_label})"
            )
            frame_count = 0
            fps_time = now

    @window.event
    def on_close():
        # Release everything we opened, in reverse order. Each of these is a
        # context manager too, but pyglet owns the event loop here so we
        # release explicitly to be deterministic about ordering.
        resource.close()
        tex_a.close()
        tex_b.close()
        surf_a.close()
        surf_b.close()
        palette_tex.close()
        palette_arr.close()
        arr_a.close()
        arr_b.close()
        stream.close()

    pyglet.app.run(interval=0)


# ======================== GPU code (CUDA + GLSL) ============================
#
# These source strings are kept at the bottom of the file so they don't
# distract from the Python logic above. The important things to know:
#
#   - KERNEL_SOURCE contains two CUDA C++ kernels:
#       * step_fire     -- advances the heat field. Reads previous state via a
#                          TextureObject (LINEAR + CLAMP, non-normalized) and
#                          writes the next state via a SurfaceObject. Bakes
#                          the bottom-row injection, mouse torch, and upward
#                          jittered advection into a single pass.
#       * colorize_fire -- per pixel: read heat from the heat TextureObject,
#                          look up the fire palette via tex1D<float4>, write
#                          RGBA bytes to the OpenGL PBO.
#
#   - VERTEX_SHADER_SOURCE / FRAGMENT_SHADER_SOURCE are GLSL. They draw a
#     texture onto a rectangle covering the entire window. The quad's t
#     coordinate is flipped versus the plasma example so that y=0 maps to the
#     top of the screen (see create_display_resources for why).
#
# ============================================================================

KERNEL_SOURCE = r"""
// Small, deterministic, GPU-friendly hash. Returns a value in [0, 1).
// Used both for bottom-row ember intensity and for the per-pixel jitter that
// gives the fire its characteristic horizontal flicker.
__device__ __forceinline__ float hash3(unsigned int x, unsigned int y,
                                       unsigned int t) {
    unsigned int h = x * 374761393u + y * 668265263u + t * 2246822519u;
    h = (h ^ (h >> 13)) * 1274126177u;
    h ^= (h >> 16);
    return (float)(h & 0x00ffffffu) / (float)0x01000000u;
}

// Canonical Doom-fire step (gather form of the original scatter algorithm).
//
// Reference scatter (one cell per JS source row):
//     decay = random in {0, 1, 2}
//     below = state[x, y+1]
//     new = max(0, below - decay)
//     state[x - decay, y] = new        // writes LEFT of source -> leftward lean
//
// Equivalent gather (one CUDA thread per destination cell):
//     decay = hash(x, y, t) in {0, 1, 2}
//     below = state[x + decay, y+1]    // reads from the right-shifted source
//     new = max(0, below - decay)
//     state[x, y] = new
//
// The right-shifted gather reads the same data the leftward-shifted scatter
// would have produced.

extern "C"
__global__
void step_fire(cudaTextureObject_t tex_read,
               cudaSurfaceObject_t surf_write,
               int width, int height,
               unsigned int t,
               int mouse_x, int mouse_y, int mouse_active,
               int ambient_on) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const int MAX_I = 36;

    // 1) Mouse torch: a hot disc painted at the cursor (overrides everything).
    if (mouse_active) {
        int dx = x - mouse_x;
        int dy = y - mouse_y;
        if (dx * dx + dy * dy <= 12 * 12) {  // matches host TORCH_RADIUS
            surf2Dwrite((unsigned char)MAX_I, surf_write, x, y);
            return;
        }
    }

    // 2) Bottom row is the steady fire source. Hardcoded to MAX_I when the
    //    ambient ember bed is on; zero otherwise (lets the fire die down).
    if (y == height - 1) {
        surf2Dwrite((unsigned char)(ambient_on ? MAX_I : 0),
                    surf_write, x, y);
        return;
    }

    // 3) Gather from the row below with random {0, 1, 2} horizontal shift
    //    and matching intensity decay -- the canonical Doom-fire update.
    float jitter_h = hash3((unsigned int)x, (unsigned int)y, t);
    int decay = (int)(jitter_h * 3.0f);             // 0, 1, or 2
    int src_x = x + decay;
    if (src_x >= width) src_x = width - 1;
    unsigned char below = tex2D<unsigned char>(tex_read,
                                               (float)src_x + 0.5f,
                                               (float)y + 1.5f);
    int new_i = (int)below - decay;
    if (new_i < 0) new_i = 0;

    // UINT8 is 1 byte, so surf2Dwrite's x argument is already the byte offset.
    surf2Dwrite((unsigned char)new_i, surf_write, x, y);
}

extern "C"
__global__
void colorize_fire(cudaTextureObject_t tex_heat,
                   cudaTextureObject_t palette_tex,
                   unsigned char* output,
                   int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // Heat texture is UINT8 + ELEMENT_TYPE: tex2D<unsigned char> returns the
    // raw intensity byte (0..36).
    unsigned char h = tex2D<unsigned char>(tex_heat,
                                           (float)x + 0.5f,
                                           (float)y + 0.5f);

    // Palette texture is 1D normalized RGBA8 with POINT filtering and 37
    // entries. Index i lands at coord (i + 0.5) / 37 -- the texel center,
    // which POINT samples exactly.
    const float palette_size = 37.0f;
    float u = ((float)h + 0.5f) / palette_size;
    float4 c = tex1D<float4>(palette_tex, u);

    int idx = (y * width + x) * 4;
    output[idx + 0] = (unsigned char)(c.x * 255.0f);
    output[idx + 1] = (unsigned char)(c.y * 255.0f);
    output[idx + 2] = (unsigned char)(c.z * 255.0f);
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
