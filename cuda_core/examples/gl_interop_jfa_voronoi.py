# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# ################################################################################
#
# This example demonstrates cuda.core.CUDAArray, TextureObject, and SurfaceObject
# in combination with GraphicsResource for CUDA/OpenGL interop. A Voronoi diagram
# is computed every frame with the Jump Flood Algorithm (JFA): a float2 "nearest
# seed" map is ping-ponged between two CUDA arrays across log2(N) passes. Each
# pass reads the previous map through a POINT-filtered TextureObject (exact texel
# reads -- no interpolation) and writes the refined map through a SurfaceObject.
# The final nearest-seed map is colorized straight into an OpenGL PBO as neon
# Voronoi cells or glowing metaballs. Seeds drift continuously so it animates.
# Requires pyglet.
#
# ################################################################################

# What this example teaches
# =========================
# - How to allocate a CUDA CUDAArray with `is_surface_load_store=True` so the same
#   memory can be bound as both a TextureObject (for sampled reads) and a
#   SurfaceObject (for typed writes).
# - How to use FilterMode.POINT + AddressMode.BORDER + border_color +
#   non-normalized coordinates to get EXACT texel reads with a clean
#   "off-grid = no seed" sentinel. JFA fundamentally requires reading the
#   precise value stored at an integer neighbor offset -- bilinear interpolation
#   between two different seed coordinates would be meaningless. This is the
#   deliberate inverse of the reaction-diffusion example's LINEAR/WRAP/normalized
#   choice.
#   API MAP: FilterMode.POINT -> exact texel reads (JFA needs no interpolation);
#   AddressMode.BORDER + border_color -> off-grid neighbor fetches return a
#   "no seed" sentinel instead of CLAMP-replicating an edge seed.
# - How varying the read offset (the JFA "step") each pass, combined with
#   ping-pong surface writes, propagates seed information across the whole image
#   in O(log N) passes instead of O(N).
# - How to compose CUDAArray/TextureObject/SurfaceObject with GraphicsResource so
#   the entire pipeline never leaves the GPU.
#
# How it works
# ============
# The Jump Flood Algorithm computes, for every pixel, the coordinate of its
# nearest seed. We store that coordinate in a `float2` map (channel 0 = seed x,
# channel 1 = seed y), using the sentinel (-1, -1) for "no seed known yet".
#
#   1. seed_clear   -- fill the whole map with the sentinel.
#   2. seed_splat   -- for each seed, write its own (x, y) into the cell it
#                      occupies. One tiny 1-thread launch per seed (seeds live
#                      in a host numpy array and are passed as scalar params;
#                      see "Why splat seeds as scalars" below).
#   3. jfa_step     -- the heart of the algorithm. With the current step size s
#                      (s = K, K/2, ..., 1), every pixel examines itself and its
#                      8 neighbors at offset +/- s. Among all non-sentinel seed
#                      coordinates found, it keeps the one closest to this pixel
#                      and writes it out. Run once per step size, ping-ponging
#                      the two arrays each pass.
#   4. colorize     -- read the final nearest-seed map and write RGBA bytes
#                      into the OpenGL PBO.
#
#   PING-PONG over JFA passes (two arrays, swap each pass)
#   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   +--------------+  tex2D<float2>   +--------------+
#   |   arr_read   | ---------------> |              |
#   | nearest-seed |  (POINT, exact   |  jfa_step    |
#   |     map      |   texel reads at |   (step s)   |
#   +--------------+   +/- step)      |              |
#                                     |              |
#   +--------------+  surf2Dwrite     |              |
#   |   arr_write  | <--------------- |              |
#   | nearest-seed |                  +--------------+
#   |     map      |
#   +--------------+
#       (swap, halve step)
#
# The step schedule starts at K = next power of two >= max(W, H) / 2 and halves
# down to 1, giving floor(log2(K)) + 1 passes. Because we ping-pong every pass,
# the final result lands in whichever array was written last; we track that
# explicitly (see the loop in on_draw) rather than assuming it is a fixed array.
# The full JFA is re-run from scratch every frame because the seeds move.
#
# Why POINT + BORDER + border_color + non-normalized coords?
# -----------------------------------------------------------
# JFA reads the exact seed coordinate stored at a specific integer neighbor.
# LINEAR filtering would blend two stored coordinates into a meaningless
# average, so we use FilterMode.POINT. For the addressing mode we use BORDER
# with an explicit border_color equal to the map's "no seed" sentinel
# (-1, -1). The earlier version used CLAMP, but CLAMP makes an off-edge
# neighbor lookup silently return the *edge* texel's real seed coordinate; that
# can make a border pixel pick a seed that is not actually its nearest one.
# BORDER instead returns the sentinel for any out-of-range fetch, which the
# kernel ignores -- the correct "there is no neighbor here" answer. (WRAP and
# MIRROR are the only address modes that require normalized coordinates; BORDER
# and CLAMP work with non-normalized coords, so we keep the integer-style
# sampling.) With non-normalized coordinates a texel at integer (nx, ny) is read
# at `tex2D<float2>(tex, nx + 0.5f, ny + 0.5f)` -- the +0.5 lands on the texel
# center. This is intentionally the opposite of the LINEAR/WRAP/normalized
# choice used by the reaction-diffusion example.
#
# Why splat seeds as scalars (no device buffer)?
# ----------------------------------------------
# Seeds live in a host numpy array and drift via sin/cos on the CPU each frame.
# Rather than allocating a device buffer, we pass each seed's position to a tiny
# 1-thread `seed_splat` kernel as float scalars. With only tens of seeds this is
# a handful of trivial launches per frame. Note the seed *list* is only needed
# for splatting: colorize and the cell-border test read seed coordinates back
# out of the JFA map, never from the host list.
#
# Channel byte width in surf2Dwrite
# ---------------------------------
# `surf2Dwrite` takes the x coordinate in BYTES, not in elements. For a
# `float2` surface that means `x * sizeof(float2)` = `x * 8`. Getting this
# wrong silently corrupts every other column.
#
# What you should see
# ===================
# A window of animated, drifting Voronoi cells (smooth vivid per-cell neon
# colors with glowing seams) or shimmering metaball-style blobs. Press M to
# toggle the two modes,
# +/- to change the seed count, R to reseed, and Escape to exit. The window
# title shows the mode, seed count, and FPS.
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
from cuda.core.textures import (
    AddressMode,
    ArrayFormat,
    CUDAArray,
    FilterMode,
    ReadMode,
    ResourceDescriptor,
    SurfaceObject,
    TextureDescriptor,
    TextureObject,
)

# ---------------------------------------------------------------------------
# Parameters (feel free to change these)
# ---------------------------------------------------------------------------
WIDTH = 512
HEIGHT = 512
MAX_SEEDS = 64  # upper bound on the seed count (host array is sized for this)
DEFAULT_SEEDS = 16
MIN_SEEDS = 2

# Visual modes for the colorize kernel. The integer value is passed to the
# kernel; the label is shown in the window caption.
MODE_VORONOI = 0
MODE_METABALL = 1
MODE_LABELS = {MODE_VORONOI: "voronoi", MODE_METABALL: "metaball"}


def jfa_steps(width, height):
    """Return the JFA step schedule: K, K/2, ..., 1.

    K is the next power of two >= max(width, height) / 2. The number of passes
    is floor(log2(K)) + 1.
    """
    longest = max(width, height)
    step = 1
    while step < longest // 2:
        step *= 2
    steps = []
    while step >= 1:
        steps.append(step)
        step //= 2
    return steps


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

    # Compile as C++ so the templated tex2D<float2> overload resolves.
    program_options = ProgramOptions(std="c++17", arch=f"sm_{dev.arch}")
    prog = Program(KERNEL_SOURCE, code_type="c++", options=program_options)
    mod = prog.compile(
        "cubin",
        name_expressions=("seed_clear", "seed_splat", "jfa_step", "colorize"),
    )

    kernels = {
        "seed_clear": mod.get_kernel("seed_clear"),
        "seed_splat": mod.get_kernel("seed_splat"),
        "jfa_step": mod.get_kernel("jfa_step"),
        "colorize": mod.get_kernel("colorize"),
    }

    block = (16, 16, 1)
    grid = (
        (WIDTH + block[0] - 1) // block[0],
        (HEIGHT + block[1] - 1) // block[1],
        1,
    )
    grid_config = LaunchConfig(grid=grid, block=block)
    # seed_clear, jfa_step, and colorize are pixel-parallel over a WIDTH x HEIGHT
    # grid and can share this config. seed_splat is a single 1-thread launch.
    point_config = LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1))
    configs = {
        "seed_clear": grid_config,
        "jfa_step": grid_config,
        "colorize": grid_config,
        "seed_splat": point_config,
    }

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
        WIDTH,
        HEIGHT,
        caption="cuda.core CUDAArray/Texture/Surface - JFA Voronoi",
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


def make_state_arrays():
    """Allocate the two `float2` ping-pong arrays that hold the nearest-seed map."""
    arr_a = CUDAArray.from_descriptor(
        shape=(WIDTH, HEIGHT),
        format=ArrayFormat.FLOAT32,
        num_channels=2,
        is_surface_load_store=True,
    )
    arr_b = CUDAArray.from_descriptor(
        shape=(WIDTH, HEIGHT),
        format=ArrayFormat.FLOAT32,
        num_channels=2,
        is_surface_load_store=True,
    )
    return arr_a, arr_b


def make_texture(arr):
    """Bind `arr` as a TextureObject configured for POINT + BORDER + non-normalized.

    API MAP:
      FilterMode.POINT            -> exact texel reads (JFA needs no interpolation)
      AddressMode.BORDER          -> off-grid neighbor fetches return border_color
      border_color (sentinel)     -> a "no seed" value the kernel ignores, instead
                                     of CLAMP-replicating a real edge seed

    JFA needs exact texel reads at integer neighbor offsets, so we use POINT
    filtering (no interpolation). We address with BORDER + an explicit
    border_color set to the same "no seed" sentinel as the map's empty cells
    (x = -1). When a JFA neighbor lookup lands off the grid, the texture unit
    returns that sentinel and the kernel ignores it. This is strictly more
    correct than CLAMP: with CLAMP an off-edge fetch silently replicates the
    edge texel's seed, which can pull a border pixel toward a seed that is not
    actually its nearest one. BORDER turns those out-of-range fetches into a
    clean "no candidate".

    Note on coordinates: BORDER addressing is valid with non-normalized
    coordinates (only WRAP/MIRROR require normalized coords), so we keep the
    integer-style `(nx + 0.5)` sampling used throughout the JFA. The border
    sentinel is a 4-tuple because the descriptor always carries four channels;
    a float2 read consumes channels 0-1, so (-1, -1) lands in (.x, .y) and the
    trailing (0, 0) is unused.
    """
    res_desc = ResourceDescriptor.from_array(arr)
    tex_desc = TextureDescriptor(
        address_mode=AddressMode.BORDER,
        filter_mode=FilterMode.POINT,
        read_mode=ReadMode.ELEMENT_TYPE,
        normalized_coords=False,
        border_color=(-1.0, -1.0, 0.0, 0.0),
    )
    return TextureObject.from_descriptor(resource=res_desc, texture_descriptor=tex_desc)


def make_seeds(count):
    """Create `count` drifting seeds.

    Each seed has a base position, an angular speed, and a radius. The instant
    position is recomputed every frame from these via sin/cos. Returns a dict of
    numpy arrays sized for MAX_SEEDS (only the first `count` are used).
    """
    rng = np.random.default_rng()
    return {
        "base_x": rng.uniform(0.2, 0.8, MAX_SEEDS).astype(np.float32) * WIDTH,
        "base_y": rng.uniform(0.2, 0.8, MAX_SEEDS).astype(np.float32) * HEIGHT,
        "radius": rng.uniform(0.05, 0.25, MAX_SEEDS).astype(np.float32) * min(WIDTH, HEIGHT),
        "phase": rng.uniform(0.0, 2.0 * math.pi, MAX_SEEDS).astype(np.float32),
        "speed": rng.uniform(0.3, 1.2, MAX_SEEDS).astype(np.float32),
        "count": count,
    }


def seed_positions(seeds, t):
    """Return (xs, ys) instant positions for the active seeds at time `t`.

    Seeds drift along small circles via sin/cos so the Voronoi diagram animates
    smoothly. Positions are clamped to the interior of the image.
    """
    n = seeds["count"]
    ang = seeds["phase"][:n] + seeds["speed"][:n] * t
    xs = seeds["base_x"][:n] + seeds["radius"][:n] * np.cos(ang)
    ys = seeds["base_y"][:n] + seeds["radius"][:n] * np.sin(ang)
    xs = np.clip(xs, 0.0, WIDTH - 1.0).astype(np.float32)
    ys = np.clip(ys, 0.0, HEIGHT - 1.0).astype(np.float32)
    return xs, ys


def run_jfa(stream, kernels, configs, seeds, t, tex_a, tex_b, surf_a, surf_b):
    """Run a full JFA pass for the current seed positions.

    Clears arr_a (via surf_a) to the sentinel, splats each seed into arr_a, then
    ping-pongs the step loop between (tex_a/surf_a) and (tex_b/surf_b).

    Returns the TextureObject bound to the array that was written last, which
    holds the final nearest-seed map for colorize.
    """
    # 1. Clear arr_a to the sentinel (-1, -1).
    launch(
        stream,
        configs["seed_clear"],
        kernels["seed_clear"],
        np.uint64(surf_a.handle),
        np.int32(WIDTH),
        np.int32(HEIGHT),
    )

    # 2. Splat each seed's own coordinate into arr_a (one 1-thread launch each).
    xs, ys = seed_positions(seeds, t)
    for i in range(seeds["count"]):
        launch(
            stream,
            configs["seed_splat"],
            kernels["seed_splat"],
            np.uint64(surf_a.handle),
            np.int32(WIDTH),
            np.int32(HEIGHT),
            np.float32(xs[i]),
            np.float32(ys[i]),
        )

    # 3. Ping-pong the JFA step loop. Start reading arr_a / writing arr_b.
    read_tex, write_surf = tex_a, surf_b
    other_tex, other_surf = tex_b, surf_a
    final_tex = tex_a  # if the loop body never runs, arr_a holds the result
    for step in jfa_steps(WIDTH, HEIGHT):
        launch(
            stream,
            configs["jfa_step"],
            kernels["jfa_step"],
            np.uint64(read_tex.handle),
            np.uint64(write_surf.handle),
            np.int32(WIDTH),
            np.int32(HEIGHT),
            np.int32(step),
        )
        # The array we just wrote is now the current map; swap for next pass.
        final_tex = tex_b if write_surf is surf_b else tex_a
        read_tex, other_tex = other_tex, read_tex
        write_surf, other_surf = other_surf, write_surf
    return final_tex


# ================================== main() ==================================


def main():
    # --- Step 1: Set up CUDA (compile kernels, create stream) ---
    dev, stream, kernels, configs = setup_cuda()

    # --- Step 2: Open a window ---
    window, gl, pyglet = create_window()

    # --- Step 3: Create GL resources for drawing a texture to screen ---
    shader_prog, quad_vao, tex_id = create_display_resources(gl, WIDTH, HEIGHT)

    # --- Step 4: Create the Pixel Buffer Object (PBO) ---
    pbo_id, _ = create_pixel_buffer(gl, WIDTH, HEIGHT)

    # --- Step 5: Register the PBO with CUDA ---
    resource = GraphicsResource.from_gl_buffer(pbo_id, flags="write_discard")

    # --- Step 6: Allocate the two ping-pong nearest-seed map Arrays ---
    #     Both are `float2` (channel 0 = seed x, channel 1 = seed y) with
    #     is_surface_load_store=True so they can be bound as SurfaceObjects.
    arr_a, arr_b = make_state_arrays()

    # --- Step 7: Pre-create the four bindless handles (once, kept alive) ---
    tex_a = make_texture(arr_a)
    tex_b = make_texture(arr_b)
    surf_a = SurfaceObject.from_array(arr_a)
    surf_b = SurfaceObject.from_array(arr_b)

    # --- Step 8: Initialize seeds and view state ---
    state = {"mode": MODE_VORONOI, "seeds": make_seeds(DEFAULT_SEEDS)}

    # --- Step 9: Render loop ---
    start_time = time.monotonic()
    frame_count = 0
    fps_time = start_time

    @window.event
    def on_key_press(symbol, _modifiers):
        key = pyglet.window.key
        if symbol == key.ESCAPE:
            window.close()
            return
        if symbol == key.M:
            state["mode"] = MODE_METABALL if state["mode"] == MODE_VORONOI else MODE_VORONOI
            return
        if symbol == key.R:
            state["seeds"] = make_seeds(state["seeds"]["count"])
            return
        if symbol in (key.PLUS, key.EQUAL, key.NUM_ADD):
            new_count = min(MAX_SEEDS, state["seeds"]["count"] + 1)
            if new_count != state["seeds"]["count"]:
                state["seeds"] = make_seeds(new_count)
            return
        if symbol in (key.MINUS, key.NUM_SUBTRACT):
            new_count = max(MIN_SEEDS, state["seeds"]["count"] - 1)
            if new_count != state["seeds"]["count"]:
                state["seeds"] = make_seeds(new_count)
            return

    @window.event
    def on_draw():
        nonlocal frame_count, fps_time

        window.clear()
        t = time.monotonic() - start_time

        # (a) Run the full Jump Flood Algorithm for the current seed positions.
        #     final_tex is the TextureObject over the array written last.
        final_tex = run_jfa(stream, kernels, configs, state["seeds"], t, tex_a, tex_b, surf_a, surf_b)

        # (b) Colorize the nearest-seed map into the OpenGL PBO.
        with resource.map(stream=stream) as buf:
            launch(
                stream,
                configs["colorize"],
                kernels["colorize"],
                np.uint64(final_tex.handle),
                buf.handle,
                np.int32(WIDTH),
                np.int32(HEIGHT),
                np.int32(state["mode"]),
                np.float32(t),
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
            label = MODE_LABELS[state["mode"]]
            window.set_caption(
                "cuda.core JFA Voronoi"
                " | TextureObject[POINT|BORDER|border_color] float2 + SurfaceObject"
                f" | mode={label} | {state['seeds']['count']} seeds"
                f" | {WIDTH}x{HEIGHT} | {fps:.0f} FPS"
            )
            frame_count = 0
            fps_time = now

    @window.event
    def on_close():
        # Release everything we opened, in reverse order.
        resource.close()
        tex_a.close()
        tex_b.close()
        surf_a.close()
        surf_b.close()
        arr_a.close()
        arr_b.close()
        stream.close()

    pyglet.app.run(interval=0)


# ======================== GPU code (CUDA + GLSL) ============================
#
# These source strings are kept at the bottom of the file so they don't
# distract from the Python logic above. KERNEL_SOURCE contains four CUDA C++
# kernels:
#
#   * seed_clear -- fills the map with the sentinel (-1, -1) via surface writes.
#   * seed_splat -- writes one seed's own coordinate into the cell it occupies.
#   * jfa_step   -- reads the previous map via a POINT-filtered, BORDER-addressed
#                   TextureObject at +/- step offsets and writes the refined
#                   nearest-seed map via a SurfaceObject. Off-grid fetches return
#                   the sentinel border_color. Coordinates are non-normalized.
#   * colorize   -- reads the final nearest-seed map and writes RGBA bytes into
#                   the OpenGL PBO, either as smooth neon Voronoi cells with
#                   glowing borders (mode 0) or glowing metaballs (mode 1).
#
# VERTEX_SHADER_SOURCE / FRAGMENT_SHADER_SOURCE are GLSL that draw a texture on
# a fullscreen rectangle. Nothing interesting.
# ============================================================================

KERNEL_SOURCE = r"""
// The nearest-seed map is a float2 per texel: (.x, .y) = coordinate of the
// nearest known seed, or the sentinel (-1, -1) for "none yet". With POINT
// filtering + non-normalized coords, texel (ix, iy) is read at
// tex2D<float2>(tex, ix + 0.5f, iy + 0.5f). The texture is BORDER-addressed
// with border_color == the sentinel, so a fetch with an out-of-range coord
// also returns (-1, -1) and is rejected by is_seed() -- the same path as an
// empty interior cell.

#define SENTINEL_X (-1.0f)

__device__ __forceinline__ bool is_seed(float2 s) {
    // Any non-negative x marks a valid stored seed coordinate.
    return s.x >= 0.0f;
}

// Fully-saturated HSV->RGB, hue/value driven by hash, returns vivid neon RGB.
__device__ __forceinline__ void hsv_to_rgb(float hue, float sat, float val,
                                           float* r, float* g, float* b) {
    hue -= floorf(hue);            // wrap hue into [0, 1)
    float h6 = hue * 6.0f;
    float c = val * sat;
    float x = c * (1.0f - fabsf(fmodf(h6, 2.0f) - 1.0f));
    float m = val - c;
    float rr, gg, bb;
    if (h6 < 1.0f)      { rr = c; gg = x; bb = 0.0f; }
    else if (h6 < 2.0f) { rr = x; gg = c; bb = 0.0f; }
    else if (h6 < 3.0f) { rr = 0.0f; gg = c; bb = x; }
    else if (h6 < 4.0f) { rr = 0.0f; gg = x; bb = c; }
    else if (h6 < 5.0f) { rr = x; gg = 0.0f; bb = c; }
    else                { rr = c; gg = 0.0f; bb = x; }
    *r = rr + m; *g = gg + m; *b = bb + m;
}

// Hash a seed coordinate into a smooth, vivid per-cell neon color. The hash
// drives a hue around the full color wheel; saturation/value stay high so
// neighboring cells read as distinct saturated hues rather than muddy bytes.
__device__ __forceinline__ void seed_color(float sx, float sy,
                                           float* r, float* g, float* b) {
    unsigned int h = (unsigned int)(sx + 0.5f) * 374761393u +
                     (unsigned int)(sy + 0.5f) * 668265263u;
    h = (h ^ (h >> 13)) * 1274126177u;
    h = h ^ (h >> 16);
    float hue = (h & 0xffffu) / 65535.0f;
    // A little value jitter from the high bits keeps equal-hue cells separable.
    float val = 0.85f + 0.15f * (((h >> 16) & 0xffu) / 255.0f);
    hsv_to_rgb(hue, 0.92f, val, r, g, b);
}

extern "C"
__global__
void seed_clear(cudaSurfaceObject_t surf, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    // float2 is 8 bytes; surf2Dwrite takes the x offset in BYTES.
    surf2Dwrite(make_float2(SENTINEL_X, SENTINEL_X), surf,
                x * (int)sizeof(float2), y);
}

extern "C"
__global__
void seed_splat(cudaSurfaceObject_t surf, int width, int height,
                float sx, float sy) {
    // Single-thread launch: write this seed's own coordinate into its cell.
    int ix = (int)(sx + 0.5f);
    int iy = (int)(sy + 0.5f);
    if (ix < 0) ix = 0;
    if (ix >= width) ix = width - 1;
    if (iy < 0) iy = 0;
    if (iy >= height) iy = height - 1;
    surf2Dwrite(make_float2(sx, sy), surf, ix * (int)sizeof(float2), iy);
}

extern "C"
__global__
void jfa_step(cudaTextureObject_t tex, cudaSurfaceObject_t surf,
              int width, int height, int step) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float px = (float)x;
    float py = (float)y;

    float best_x = SENTINEL_X;
    float best_y = SENTINEL_X;
    float best_d2 = 3.0e38f;  // ~FLT_MAX

    // Examine self (dx=dy=0) and the 8 neighbors at +/- step. We deliberately
    // do NOT clamp the neighbor coordinate: off-grid lookups are left out of
    // range so the BORDER-addressed texture returns the sentinel border_color
    // (-1, -1). is_seed() then rejects it, exactly as it would reject an empty
    // interior cell. Under the old CLAMP scheme an off-edge fetch returned the
    // edge texel's real seed, which could win the nearest-seed test for a
    // border pixel even though that seed is not actually its nearest.
    #pragma unroll
    for (int dy = -1; dy <= 1; ++dy) {
        #pragma unroll
        for (int dx = -1; dx <= 1; ++dx) {
            int nx = x + dx * step;
            int ny = y + dy * step;

            float2 s = tex2D<float2>(tex, (float)nx + 0.5f, (float)ny + 0.5f);
            if (is_seed(s)) {
                float ddx = s.x - px;
                float ddy = s.y - py;
                float d2 = ddx * ddx + ddy * ddy;
                if (d2 < best_d2) {
                    best_d2 = d2;
                    best_x = s.x;
                    best_y = s.y;
                }
            }
        }
    }

    surf2Dwrite(make_float2(best_x, best_y), surf, x * (int)sizeof(float2), y);
}

extern "C"
__global__
void colorize(cudaTextureObject_t tex, unsigned char* output,
              int width, int height, int mode, float t) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float2 c = tex2D<float2>(tex, (float)x + 0.5f, (float)y + 0.5f);

    float r = 0.0f, g = 0.0f, b = 0.0f;

    if (is_seed(c)) {
        float dx = c.x - (float)x;
        float dy = c.y - (float)y;
        float dist = sqrtf(dx * dx + dy * dy);

        if (mode == 0) {
            // --- Voronoi cells: smooth neon color + glowing cell borders. ---
            seed_color(c.x, c.y, &r, &g, &b);

            // Border proximity: count how many 8-neighbors belong to a different
            // cell. A pixel deep inside a cell sees 0; a pixel right on the edge
            // sees several. We use this as a smooth edge factor rather than a
            // hard on/off so borders read as a luminous seam, not a jagged line.
            int diff = 0;
            const int ox[8] = {-1, 1, 0, 0, -1, -1, 1, 1};
            const int oy[8] = {0, 0, -1, 1, -1, 1, -1, 1};
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                int nx = x + ox[i];
                int ny = y + oy[i];
                if (nx < 0) nx = 0;
                if (nx >= width) nx = width - 1;
                if (ny < 0) ny = 0;
                if (ny >= height) ny = height - 1;
                float2 n = tex2D<float2>(tex, (float)nx + 0.5f, (float)ny + 0.5f);
                if (is_seed(n) && (n.x != c.x || n.y != c.y)) {
                    ++diff;
                }
            }

            // Smooth interior shading: gentle radial falloff from the cell seed
            // for a soft volumetric look, slowly breathing in time.
            float shade = 1.0f / (1.0f + 0.0006f * dist * dist);
            float pulse = 0.92f + 0.08f * sinf(1.5f * t + 0.02f * dist);
            shade = (0.55f + 0.45f * shade) * pulse;
            r *= shade; g *= shade; b *= shade;

            if (diff > 0) {
                // edge in [0,1]: stronger the more neighbors disagree.
                float edge = (float)diff / 8.0f;
                edge = edge * edge;  // bias toward the true seam
                // Darken the base color toward the seam, then add a bright neon
                // rim on top so cell boundaries glow instead of just going dark.
                float dark = 1.0f - 0.85f * edge;
                r *= dark; g *= dark; b *= dark;
                float rim = edge * (0.65f + 0.35f * sinf(2.5f * t));
                r += rim; g += rim * 0.9f; b += rim;
            }
        } else {
            // --- Metaballs: glowing neon falloff from the nearest seed. ---
            // Brightness peaks at the seed and decays smoothly with distance.
            float glow = 1.0f / (1.0f + 0.0018f * dist * dist);
            // A couple of animated isoline ripples add a layered plasma pulse.
            float ripple = 0.5f + 0.5f * sinf(0.13f * dist - 3.0f * t);
            float ripple2 = 0.5f + 0.5f * sinf(0.05f * dist + 1.7f * t);
            float intensity = glow * (0.55f + 0.30f * ripple + 0.15f * ripple2);
            // A soft core bloom keeps seed centers reading as hot points.
            float core = 1.0f / (1.0f + 0.02f * dist * dist);
            intensity += 0.5f * core;

            // Hue sweeps with distance + time so blobs shimmer through the neon
            // spectrum; value tracks intensity so falloff still fades to black.
            float hue = 0.6f + 0.0015f * dist + 0.05f * t;
            float val = intensity;
            if (val > 1.0f) val = 1.0f;
            hsv_to_rgb(hue, 0.85f, val, &r, &g, &b);
            // Lift toward white at the very brightest cores for a hot-tip look.
            float hot = intensity - 1.0f;
            if (hot > 0.0f) {
                if (hot > 1.0f) hot = 1.0f;
                r += hot * (1.0f - r);
                g += hot * (1.0f - g);
                b += hot * (1.0f - b);
            }
        }
    }

    // Clamp to [0, 1] before writing bytes.
    if (r < 0.0f) r = 0.0f; if (r > 1.0f) r = 1.0f;
    if (g < 0.0f) g = 0.0f; if (g > 1.0f) g = 1.0f;
    if (b < 0.0f) b = 0.0f; if (b > 1.0f) b = 1.0f;

    int idx = (y * width + x) * 4;
    output[idx + 0] = (unsigned char)(r * 255.0f);
    output[idx + 1] = (unsigned char)(g * 255.0f);
    output[idx + 2] = (unsigned char)(b * 255.0f);
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
