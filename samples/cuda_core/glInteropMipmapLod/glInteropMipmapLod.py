# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# ################################################################################
#
# This example demonstrates the new cuda.core texture/surface stack:
# MipmappedArray, SurfaceObject, and a TextureObject that does trilinear
# (LINEAR mipmap + LINEAR filter) sampling with user-controlled LOD bias.
# Requires pyglet.
#
# ################################################################################

# What this example teaches
# =========================
# How to allocate a mipmap pyramid as a single MipmappedArray, populate each
# level from a CUDA kernel by binding it as a SurfaceObject, and then sample
# the whole pyramid from a TextureObject with manual LOD bias.
#
# How it works
# ============
# A mipmap pyramid is a stack of progressively-halved images of the same
# texture. The base level (level 0) holds the highest-resolution version; each
# subsequent level is a 2x2 box-filtered downsample of the level below it:
#
#     level 0: 512 x 512   <- highest detail
#     level 1: 256 x 256
#     level 2: 128 x 128
#     ...
#     level 9:   1 x 1     <- a single average color
#
# At sample time, the GPU picks the mip level that best matches the on-screen
# size of the texel, optionally blending between adjacent levels (trilinear).
# Selecting a coarser level than the "right" one is called a positive LOD bias
# and produces a softer/blurrier image; a negative bias selects finer levels
# (sharper but more aliased when undersampled).
#
#   +----------------------+       +-----------------------+
#   |   MipmappedArray     |       |   TextureObject       |
#   | (single allocation,  | <---  | (samples the whole    |
#   |  10 mip levels)      |       |  pyramid w/ trilinear |
#   +----------------------+       |  filtering)           |
#         ^      ^                 +-----------------------+
#         |      |
#         |      +---- one SurfaceObject per level, used at BUILD time only
#         |            to let a kernel write pixels into that level.
#         |
#         +----------- get_level(L) returns a NON-OWNING OpaqueArray view of level L;
#                      the storage belongs to the parent MipmappedArray.
#
#   STARTUP -- one-time mipmap build
#   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. Allocate MipmappedArray (10 levels, float4 RGBA, is_surface_load_store=True).
#   2. Level 0: launch `seed_base` kernel -> SurfaceObject -> high-frequency
#      procedural pattern.
#   3. For L = 1..num_levels-1: launch `downsample` kernel:
#        - reads level L-1 through a TextureObject (POINT-filtered)
#        - writes level L   through a SurfaceObject
#        - 4-sample box average of the parent's 2x2 footprint.
#
#   PER FRAME (render loop)
#   ~~~~~~~~~~~~~~~~~~~~~~~
#   The display TextureObject samples the whole pyramid with `tex2DLod`,
#   where the LOD is computed per-pixel as `log2(zoom) + lod_bias`. The result
#   is written to a GL PBO via GraphicsResource, then drawn as a textured quad.
#
# What you should see
# ===================
# A 512x512 procedural pattern (concentric rings + diagonal grid) shown
# stretched across the window. Use the mouse wheel to zoom in/out (this
# implicitly changes the LOD), and use the bracket keys `[` / `]` to add a
# manual LOD bias on top of that. Press `R` to reset.
#
#   Mouse wheel       zoom in / out
#   [                 LOD bias -= 0.25  (sharper, more aliased)
#   ]                 LOD bias += 0.25  (blurrier, samples a coarser level)
#   R                 reset zoom + bias
#   Escape / close    quit
#
# The window title shows the current zoom, manual bias, and effective LOD.
# Close the window or press Escape to exit.
#

# /// script
# dependencies = ["cuda_bindings", "cuda_core>0.6.0", "pyglet"]
# ///

import ctypes
import math
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
from cuda.core.texture import (
    MipmappedArrayOptions,
    ResourceDescriptor,
    TextureObjectOptions,
)
from cuda.core.typing import (
    AddressModeType,
    ArrayFormatType,
    FilterModeType,
    ReadModeType,
)

# ---------------------------------------------------------------------------
# Configuration (feel free to change these)
# ---------------------------------------------------------------------------
WIDTH = 800
HEIGHT = 600
BASE_SIZE = 512  # Texture base-level edge length (must be a power of two).
LOD_BIAS_STEP = 0.25


# ============================= Helper functions =============================
#
# The functions below set up CUDA, OpenGL, and the mipmap pyramid. If you're
# here to learn about MipmappedArray / SurfaceObject / mipmapped TextureObject,
# you can skip straight to main() -- the interesting part is there. These
# helpers exist so that main() reads like a short story.
# ============================================================================


def _check_compute_capability(dev):
    """Surface load/store + mipmapped arrays require sm_30+."""
    cc = dev.compute_capability
    if cc.major < 3:
        print(
            f"This example requires compute capability >= 3.0, got sm_{cc.major}{cc.minor}.",
            file=sys.stderr,
        )
        sys.exit(1)


def setup_cuda():
    """Compile the three kernels and return everything we need to drive them.

    Returns
    -------
    (dev, stream, kernels, arch_str)
        kernels is a dict with keys "seed_base", "downsample", "display".
    """
    dev = Device(0)
    dev.set_current()
    _check_compute_capability(dev)
    stream = dev.create_stream()

    program_options = ProgramOptions(std="c++17", arch=f"sm_{dev.arch}")
    prog = Program(KERNEL_SOURCE, code_type="c++", options=program_options)
    mod = prog.compile(
        "cubin",
        name_expressions=("seed_base", "downsample", "display"),
    )
    kernels = {
        "seed_base": mod.get_kernel("seed_base"),
        "downsample": mod.get_kernel("downsample"),
        "display": mod.get_kernel("display"),
    }
    return dev, stream, kernels, f"sm_{dev.arch}"


def build_mipmap_pyramid(mip, num_levels, stream, kernels):
    """Populate every level of `mip` using SurfaceObject writes.

    Strategy
    --------
    * Level 0 is filled directly by `seed_base`, which writes a procedural
      pattern through a SurfaceObject bound to level 0.
    * Each subsequent level L is filled by `downsample`, which reads level L-1
      through a POINT-filtered TextureObject and box-averages a 2x2 footprint
      into level L through a SurfaceObject.
    * All operations are issued on a single stream, so they serialize
      implicitly -- no per-level sync is needed.
    """
    # ---- Level 0: seed the base image -------------------------------------
    base_arr = mip.get_level(0)  # non-owning view; do NOT use a `with` block
    with Device().create_surface_object(resource=ResourceDescriptor.from_opaque_array(base_arr)) as base_surf:
        block = (16, 16, 1)
        grid = (
            (BASE_SIZE + block[0] - 1) // block[0],
            (BASE_SIZE + block[1] - 1) // block[1],
            1,
        )
        launch(
            stream,
            LaunchConfig(grid=grid, block=block),
            kernels["seed_base"],
            np.uint64(base_surf.handle),
            np.int32(BASE_SIZE),
            np.int32(BASE_SIZE),
        )
    # base_arr (non-owning) is allowed to fall out of scope here; the parent
    # MipmappedArray keeps the underlying storage alive.

    # ---- Levels 1..N-1: box-filter downsample ------------------------------
    # Each iteration reads level (L-1) through a temporary TextureObject and
    # writes level L through a temporary SurfaceObject. Both close cleanly
    # at the end of their `with` blocks.
    src_tex_desc = TextureObjectOptions(
        address_mode=AddressModeType.CLAMP,
        filter_mode=FilterModeType.POINT,  # explicit per-texel reads
        read_mode=ReadModeType.ELEMENT_TYPE,
        normalized_coords=False,  # integer pixel coordinates
    )
    for level in range(1, num_levels):
        parent_size = BASE_SIZE >> (level - 1)
        level_size = BASE_SIZE >> level
        if level_size < 1:
            break

        src_arr = mip.get_level(level - 1)
        dst_arr = mip.get_level(level)
        src_res = ResourceDescriptor.from_opaque_array(src_arr)
        with (
            Device().create_texture_object(resource=src_res, options=src_tex_desc) as src_tex,
            Device().create_surface_object(resource=ResourceDescriptor.from_opaque_array(dst_arr)) as dst_surf,
        ):
            block = (16, 16, 1)
            grid = (
                (level_size + block[0] - 1) // block[0],
                (level_size + block[1] - 1) // block[1],
                1,
            )
            launch(
                stream,
                LaunchConfig(grid=grid, block=block),
                kernels["downsample"],
                np.uint64(src_tex.handle),
                np.uint64(dst_surf.handle),
                np.int32(parent_size),
                np.int32(level_size),
            )
        # src_arr, dst_arr (non-owning) fall out of scope; storage stays
        # alive via the parent MipmappedArray.

    # One sync at the end is enough -- the whole build chain ran on this
    # stream and serialized naturally.
    stream.sync()


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
        caption="MipmappedArray Example - Mipmap LOD viewer",
        vsync=False,
    )
    return window, _gl, pyglet


def create_display_resources(gl, width, height):
    """Standard GL boilerplate: a shader program, a fullscreen quad, and an
    empty texture that we'll repeatedly fill from a PBO. Not CUDA-specific.

    Returns (shader_program, vertex_array_id, texture_id).
    """
    from pyglet.graphics.shader import Shader, ShaderProgram

    vert = Shader(VERTEX_SHADER_SOURCE, "vertex")
    frag = Shader(FRAGMENT_SHADER_SOURCE, "fragment")
    shader_prog = ShaderProgram(vert, frag)

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

    stride = 4 * 4  # 4 floats * 4 bytes each
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
    """Create a Pixel Buffer Object (PBO) -- the CUDA/GL bridge.

    Returns (pbo_gl_name, size_in_bytes).
    """
    pbo = ctypes.c_uint(0)
    gl.glGenBuffers(1, ctypes.byref(pbo))
    gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, pbo.value)
    nbytes = width * height * 4  # RGBA8
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
        None,
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
    # Waive when no display is available (headless CI, Wayland-only, etc.).
    import os
    import platform
    import sys

    if platform.system() == "Linux" and not os.environ.get("DISPLAY"):
        print("No DISPLAY available; waiving glInteropMipmapLod.", file=sys.stderr)
        sys.exit(2)

    # --- Step 1: Set up CUDA (compile kernels, create stream) ---
    dev, stream, kernels, _arch = setup_cuda()

    # --- Step 2: Allocate the mipmap pyramid and build every level ---
    #     is_surface_load_store=True is required for kernel-side writes.
    num_levels = int(math.log2(BASE_SIZE)) + 1
    mip = Device().create_mipmapped_array(
        MipmappedArrayOptions(
            shape=(BASE_SIZE, BASE_SIZE),
            format=ArrayFormatType.FLOAT32,
            num_channels=4,
            num_levels=num_levels,
            is_surface_load_store=True,
        )
    )
    build_mipmap_pyramid(mip, num_levels, stream, kernels)

    # --- Step 3: Bind the WHOLE pyramid as a trilinear-filtered texture ---
    #     Normalized coordinates (0..1) make zoom-by-uv simple. The texture
    #     descriptor's mipmap_level_bias stays 0.0; the display kernel
    #     receives the user-controlled bias as a kernel argument and folds
    #     it into the tex2DLod call (avoids rebuilding the TextureObject
    #     whenever the user changes the bias).
    display_tex_desc = TextureObjectOptions(
        address_mode=AddressModeType.WRAP,
        filter_mode=FilterModeType.LINEAR,
        read_mode=ReadModeType.ELEMENT_TYPE,
        normalized_coords=True,
        mipmap_filter_mode=FilterModeType.LINEAR,  # trilinear
        mipmap_level_bias=0.0,
        min_mipmap_level_clamp=0.0,
        max_mipmap_level_clamp=float(num_levels - 1),
    )
    display_tex = Device().create_texture_object(
        resource=ResourceDescriptor.from_mipmapped_array(mip),
        options=display_tex_desc,
    )

    # --- Step 4: Open a window and set up the GL/CUDA bridge ---
    window, gl, pyglet = create_window()
    shader_prog, quad_vao, tex_id = create_display_resources(gl, WIDTH, HEIGHT)
    pbo_id, _ = create_pixel_buffer(gl, WIDTH, HEIGHT)
    resource = GraphicsResource.from_gl_buffer(pbo_id, flags="write_discard")

    # --- Step 5: Render loop state ---
    # `zoom` controls how big a texel is on screen: zoom > 1 stretches the
    # texture and selects coarser mip levels (positive LOD); zoom < 1 shrinks
    # the texture and selects finer levels. `lod_bias` is a manual offset
    # added on top.
    state = {"zoom": 1.0, "lod_bias": 0.0}
    start_time = time.monotonic()
    frame_count = [0]
    fps_time = [start_time]

    block = (16, 16, 1)
    grid = (
        (WIDTH + block[0] - 1) // block[0],
        (HEIGHT + block[1] - 1) // block[1],
        1,
    )
    config = LaunchConfig(grid=grid, block=block)

    def effective_lod():
        # Same formula the display kernel uses, clamped to the legal range so
        # the window title matches what the GPU actually sees.
        raw = math.log2(max(state["zoom"], 1e-6)) + state["lod_bias"]
        return max(0.0, min(float(num_levels - 1), raw))

    @window.event
    def on_draw():
        window.clear()

        # (a) Map the PBO so CUDA can write into it.
        with resource.map(stream=stream) as buf:
            # (b) Launch the display kernel -- samples the mipmap and writes RGBA.
            launch(
                stream,
                config,
                kernels["display"],
                buf.handle,
                np.int32(WIDTH),
                np.int32(HEIGHT),
                np.uint64(display_tex.handle),
                np.float32(state["zoom"]),
                np.float32(state["lod_bias"]),
                np.float32(float(num_levels - 1)),
            )
        # (c) Unmap happens automatically; cuGraphicsUnmapResources serializes
        #     the CUDA work against subsequent OpenGL use.

        copy_pbo_to_texture(gl, pbo_id, tex_id, WIDTH, HEIGHT)
        draw_fullscreen_quad(gl, shader_prog, quad_vao, tex_id)

        frame_count[0] += 1
        now = time.monotonic()
        if now - fps_time[0] >= 1.0:
            fps = frame_count[0] / (now - fps_time[0])
            window.set_caption(
                f"MipmappedArray LOD viewer "
                f"({WIDTH}x{HEIGHT}, {fps:.0f} FPS) -- "
                f"zoom={state['zoom']:.2f}, "
                f"bias={state['lod_bias']:+.2f}, "
                f"LOD={effective_lod():.2f}"
            )
            frame_count[0] = 0
            fps_time[0] = now

    @window.event
    def on_mouse_scroll(_x, _y, _scroll_x, scroll_y):
        # One wheel step changes zoom by ~12.5%. Clamped to keep LOD in range.
        if scroll_y == 0:
            return
        factor = 1.125**scroll_y
        state["zoom"] = max(1.0 / 64.0, min(64.0, state["zoom"] * factor))

    @window.event
    def on_key_press(symbol, _modifiers):
        key = pyglet.window.key
        if symbol == key.BRACKETLEFT:
            state["lod_bias"] = max(-float(num_levels), state["lod_bias"] - LOD_BIAS_STEP)
        elif symbol == key.BRACKETRIGHT:
            state["lod_bias"] = min(float(num_levels), state["lod_bias"] + LOD_BIAS_STEP)
        elif symbol == key.R:
            state["zoom"] = 1.0
            state["lod_bias"] = 0.0

    @window.event
    def on_close():
        # Release CUDA-side resources in reverse construction order. GL
        # objects clean up via pyglet on window close.
        resource.close()
        display_tex.close()
        mip.close()
        stream.close()

    pyglet.app.run(interval=0)


# ======================== GPU code (CUDA + GLSL) ============================
#
# Three CUDA kernels are concatenated into one program string so they share a
# single NVRTC compile. All three operate on float4 RGBA pixels.
#
#   seed_base   -- writes a high-frequency procedural pattern to level 0 via a
#                  SurfaceObject. NOTE: surf2Dwrite's x-coordinate is in BYTES,
#                  not in elements, so we multiply by sizeof(float4) every time.
#
#   downsample  -- reads level L-1 through a POINT-filtered TextureObject and
#                  writes the 2x2 box average to level L through a SurfaceObject.
#                  tex2D with non-normalized coords needs the +0.5 half-texel
#                  offset to hit exact texel centers.
#
#   display     -- samples the WHOLE mipmap pyramid with tex2DLod, where the
#                  per-thread LOD is `clamp(log2(zoom) + lod_bias, 0, maxLod)`.
#                  Writes 8-bit RGBA into the PBO.
#
# GLSL shaders at the very bottom just draw a textured quad. Nothing CUDA-
# specific there.
#
# ============================================================================

KERNEL_SOURCE = r"""
// --------------------------------------------------------------------------
// Helper: clamp a float to [a, b].
// --------------------------------------------------------------------------
__device__ __forceinline__ float clampf(float v, float a, float b) {
    return fminf(fmaxf(v, a), b);
}

// CUDA does not ship a builtin "fract" so we provide one (used by seed_base).
__device__ __forceinline__ float fracf(float v) {
    return v - floorf(v);
}

// --------------------------------------------------------------------------
// seed_base: write a procedural high-frequency pattern to level 0.
//
// surf is a SurfaceObject bound to the level-0 OpaqueArray (float4 RGBA). The
// pattern is a colorful blend of concentric rings, a diagonal grid, and a
// radial sweep, designed to have plenty of fine detail so the difference
// between mip levels is visually obvious.
// --------------------------------------------------------------------------
extern "C" __global__
void seed_base(cudaSurfaceObject_t surf, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float u = ((float)x + 0.5f) / (float)width;
    float v = ((float)y + 0.5f) / (float)height;

    // Concentric rings centered on the image.
    float cx = u - 0.5f;
    float cy = v - 0.5f;
    float r = sqrtf(cx * cx + cy * cy);
    float rings = 0.5f + 0.5f * sinf(r * 80.0f);

    // Diagonal grid -- thin lines about every 1/16 of the image.
    float gx = fabsf(fracf(u * 16.0f) - 0.5f);
    float gy = fabsf(fracf(v * 16.0f) - 0.5f);
    float grid = (gx < 0.05f || gy < 0.05f) ? 1.0f : 0.0f;

    // Angular sweep gives the rings some color variation.
    float theta = atan2f(cy, cx);
    float sweep = 0.5f + 0.5f * sinf(theta * 6.0f);

    // Combine into an RGBA color. Keep values in [0, 1].
    float red   = clampf(rings * (0.4f + 0.6f * sweep) + 0.3f * grid, 0.0f, 1.0f);
    float green = clampf(rings * (0.6f - 0.4f * sweep) + 0.3f * grid, 0.0f, 1.0f);
    float blue  = clampf(0.4f + 0.4f * sweep + 0.5f * grid,            0.0f, 1.0f);
    float alpha = 1.0f;

    float4 px = make_float4(red, green, blue, alpha);

    // Surface writes index x in BYTES (this is the classic gotcha).
    surf2Dwrite<float4>(px, surf, x * (int)sizeof(float4), y);
}

// --------------------------------------------------------------------------
// downsample: box-filter a 2x2 footprint of the parent level into one texel.
//
// src is a POINT-filtered TextureObject bound to level (L-1).
// dst is a SurfaceObject bound to level L.
// (dst_w, dst_h) is the size of level L.
// (src_w = 2 * dst_w, src_h = 2 * dst_h is implicit and unused; we pass it
// only for the bounds check.)
//
// Texture coordinates: tex2D with non-normalized coords returns texel (i, j)
// when sampled at (i + 0.5, j + 0.5). So for output texel (x, y) the four
// parent texels live at parent-coords (2x + 0.5, 2y + 0.5), (2x + 1.5, ...).
// --------------------------------------------------------------------------
extern "C" __global__
void downsample(cudaTextureObject_t src,
                cudaSurfaceObject_t dst,
                int src_size,
                int dst_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_size || y >= dst_size) return;

    float fx = 2.0f * (float)x;
    float fy = 2.0f * (float)y;

    float4 a = tex2D<float4>(src, fx + 0.5f, fy + 0.5f);
    float4 b = tex2D<float4>(src, fx + 1.5f, fy + 0.5f);
    float4 c = tex2D<float4>(src, fx + 0.5f, fy + 1.5f);
    float4 d = tex2D<float4>(src, fx + 1.5f, fy + 1.5f);

    float4 px;
    px.x = 0.25f * (a.x + b.x + c.x + d.x);
    px.y = 0.25f * (a.y + b.y + c.y + d.y);
    px.z = 0.25f * (a.z + b.z + c.z + d.z);
    px.w = 0.25f * (a.w + b.w + c.w + d.w);

    // Silence unused-variable warning for the convenience parameter.
    (void)src_size;

    surf2Dwrite<float4>(px, dst, x * (int)sizeof(float4), y);
}

// --------------------------------------------------------------------------
// display: per-pixel mipmap sample with manual LOD bias.
//
// tex is a TextureObject built from the whole MipmappedArray (LINEAR +
// LINEAR mipmap filter, normalized coords). For each output pixel we compute
// a single per-thread LOD from `zoom` and `lod_bias`, then sample with
// tex2DLod. Output is written as RGBA8 into a linear byte buffer.
// --------------------------------------------------------------------------
extern "C" __global__
void display(unsigned char *output,
             int width,
             int height,
             cudaTextureObject_t tex,
             float zoom,
             float lod_bias,
             float max_lod) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // Normalized window coords in [0, 1].
    float u = ((float)x + 0.5f) / (float)width;
    float v = ((float)y + 0.5f) / (float)height;

    // Zoom around the window center so the user sees the effect symmetrically.
    u = (u - 0.5f) * zoom + 0.5f;
    v = (v - 0.5f) * zoom + 0.5f;

    // LOD: zoom > 1 means the texture is being stretched (each texel covers
    // more screen area), which intuitively corresponds to selecting a coarser
    // (higher) mip level. log2(zoom) yields exactly that. lod_bias is added
    // on top, and the final value is clamped to the legal range.
    float lod = log2f(fmaxf(zoom, 1e-6f)) + lod_bias;
    lod = clampf(lod, 0.0f, max_lod);

    float4 c = tex2DLod<float4>(tex, u, v, lod);

    int idx = (y * width + x) * 4;
    output[idx + 0] = (unsigned char)(clampf(c.x, 0.0f, 1.0f) * 255.0f);
    output[idx + 1] = (unsigned char)(clampf(c.y, 0.0f, 1.0f) * 255.0f);
    output[idx + 2] = (unsigned char)(clampf(c.z, 0.0f, 1.0f) * 255.0f);
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
