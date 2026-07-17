# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# ################################################################################
#
# This example demonstrates cuda.core.OpaqueArray, TextureObject, and SurfaceObject
# in combination with GraphicsResource for CUDA/OpenGL interop. It runs a
# real-time Stable Fluids (Jos Stam) smoke/ink solver entirely on the GPU:
# velocity, pressure, and dye fields live in ping-ponged CUDA arrays, are read
# through TextureObjects with free hardware bilinear filtering (the heart of
# semi-Lagrangian advection), and written back through SurfaceObjects. The dye
# is colorized straight into an OpenGL PBO. Drag the mouse to inject swirling
# ink. Requires pyglet.
#
# ################################################################################

# What this example teaches
# =========================
# - How semi-Lagrangian advection uses tex2D LINEAR sampling: trace each cell
#   backward along the velocity field and read the old quantity with free
#   hardware bilinear interpolation (no manual lerp, no neighbor gather).
# - How to drive several distinct kernels (advect, divergence, Jacobi pressure
#   solve, gradient subtraction, dye advect, colorize) over a shared set of
#   pre-created TextureObject/SurfaceObject handles, ping-ponging multiple
#   fields without recreating handles per frame.
# - How to fold live mouse input into a GPU simulation: capture the mouse delta
#   and splat velocity + dye into the field via a SurfaceObject (in-place
#   read-modify-write, one thread per cell -> no race).
#
# How it works
# ============
# Stam's "Stable Fluids" solves the incompressible Navier-Stokes equations on a
# regular grid by splitting each step into stages that are each individually
# stable:
#
#   1. ADVECT VELOCITY  - move the velocity field along itself. For each cell we
#      back-trace its center one timestep against the local velocity and read
#      the old velocity there with tex2D<float2> LINEAR (bilinear). This is the
#      unconditionally-stable semi-Lagrangian scheme.
#   2. SPLAT (input)    - add the mouse-drag velocity and a dab of dye in a soft
#      radial brush around the cursor (in-place on the velocity/dye surfaces).
#   3. DIVERGENCE       - compute div(velocity), the amount each cell is a
#      source/sink. An incompressible fluid must have zero divergence.
#   4. PRESSURE SOLVE   - Jacobi-iterate the Poisson equation lap(p) = div,
#      ping-ponging two pressure buffers for ~30 iterations.
#   5. SUBTRACT GRADIENT- v <- v - grad(p). This projects the velocity onto its
#      divergence-free part, enforcing incompressibility.
#   6. ADVECT DYE       - move the ink along the (now divergence-free) velocity,
#      again with tex2D LINEAR back-tracing.
#   7. COLORIZE         - map dye density through a vivid gradient into the PBO.
#
#   PING-PONG (read one array, write the other, then swap)
#   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   +-----------+   tex2D<float2> LINEAR   +-------------+   surf2Dwrite   +-----------+
#   |  vel_a    | -----------------------> |  advect /   | --------------> |  vel_b    |
#   | (vx, vy)  |                          |  jacobi /   |                 | (vx, vy)  |
#   +-----------+                          |  advect_dye |                 +-----------+
#        ^                                 +-------------+                       |
#        +-------------------------------- (swap) ------------------------------+
#
# Why LINEAR + CLAMP + normalized coords?
# ---------------------------------------
# Semi-Lagrangian advection traces a cell's center back to an arbitrary
# fractional position and needs the interpolated field value there. LINEAR
# filtering gives that bilinear interpolation for free in hardware. We use a
# bounded box (CLAMP) rather than a torus so ink piles up against the walls
# instead of wrapping. CLAMP, like all addressing modes, behaves cleanly with
# normalized coordinates, and we sample at texel centers `(i + 0.5) / N` so a
# zero-velocity cell reads back exactly its own value.
#
# Channel byte width in surf2Dwrite
# ---------------------------------
# `surf2Dwrite` takes the x coordinate in BYTES, not in elements. Velocity is a
# `float2` (8 bytes) so its x offset is `x * sizeof(float2)`; pressure and
# divergence are `float` (4 bytes, `x * sizeof(float)`); the dye is a `float4`
# RGBA color (16 bytes, `x * sizeof(float4)`). Getting this wrong silently
# corrupts every other column.
#
# What you should see
# ===================
# Big blobs of saturated color are dropped into the fluid every fraction of a
# second and immediately billow, swirl, and mix into turbulent ribbons that
# fill the window -- "ink dropped in water." Drag the mouse to paint your own
# rainbow ink. Press R to clear, Escape to exit. The window title shows the
# current FPS, pressure-iteration count, and live texture/surface config.
#

# /// script
# dependencies = ["cuda_bindings", "cuda_core>0.6.0", "pyglet"]
# ///

import argparse
import colorsys
import ctypes
import math
import random
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
    OpaqueArrayOptions,
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
# Simulation parameters (feel free to change these)
# ---------------------------------------------------------------------------
WIDTH = 512
HEIGHT = 512
DT = 1.0  # simulation timestep
PRESSURE_ITERS = 30  # Jacobi iterations for the pressure solve per frame
VELOCITY_DISSIPATION = 0.999  # per-step velocity decay (1.0 = no decay)
DYE_DISSIPATION = 0.994  # per-step dye decay; ink lingers and builds, then fades
SPLAT_RADIUS = 24.0  # brush radius in cells for mouse injection
SPLAT_FORCE = 6.0  # how strongly a mouse delta becomes velocity
SPLAT_DYE = 1.0  # mouse ink intensity (color * this is deposited)
CURL_SEED = 2.5  # strength of the ambient curl seeded on reset
# Vorticity confinement pushes velocity back toward regions of high |curl|,
# sharpening the swirls that numerical diffusion would otherwise smear out.
# This is the single extra kernel that turns soft blobs into crisp curling
# plumes. Tunable: ~0.1-0.3 reads well at DT=1.0; higher gets turbulent.
VORTICITY = 0.28  # confinement strength (0.0 disables it)

# Auto-bursts keep the simulation alive and colorful without any input: when
# the mouse is idle we periodically drop a big blob of a random bright color
# with a random velocity impulse at a random spot -- the classic "ink dropped
# in water" look that quickly fills the frame with billowing, swirling color.
# Grab the cursor and drag to paint your own ink.
AUTO_EMIT = True
BURST_INTERVAL = 0.45  # seconds between automatic colored bursts
BURSTS_PER_EVENT = 2  # blobs dropped each burst event
BURST_RADIUS = 42.0  # blob radius in cells (big, soft)
BURST_FORCE = 18.0  # velocity impulse magnitude per blob
BURST_DYE = 1.2  # ink intensity per blob (random color * this)

# This solver advances one step per displayed frame, so its per-step rates
# (dissipation, advection distance) would otherwise depend on the frame rate --
# on a fast GPU the dye would dissipate away almost instantly between bursts.
# We make it frame-rate INDEPENDENT instead: every frame, the real elapsed time
# is expressed in units of a REF_FPS reference step and the dissipation and
# advection distance are scaled by it, so the ink evolves at the same wall-clock
# rate (and looks the same) whether the loop runs at 60 or 2000 FPS. Running
# faster just means more, smaller, smoother substeps.
REF_FPS = 60.0


# ============================= Helper functions =============================
#
# The functions below set up CUDA and OpenGL. If you're here to learn about
# OpaqueArray/TextureObject/SurfaceObject, skip ahead to main() -- the interesting
# part is there. These helpers exist so that main() reads like a short story
# instead of a wall of boilerplate.
# ============================================================================


def setup_cuda():
    """Compile the CUDA kernels and return (device, stream, kernels, configs).

    Returns a dict of kernels keyed by name and a shared LaunchConfig (every
    kernel is pixel-parallel over the same WIDTH x HEIGHT grid).
    """
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
        name_expressions=(
            "seed_field",
            "splat",
            "advect_velocity",
            "vorticity_confinement",
            "divergence",
            "pressure_jacobi",
            "subtract_gradient",
            "advect_dye",
            "colorize",
        ),
    )

    kernels = {
        "seed": mod.get_kernel("seed_field"),
        "splat": mod.get_kernel("splat"),
        "advect_vel": mod.get_kernel("advect_velocity"),
        "vorticity": mod.get_kernel("vorticity_confinement"),
        "divergence": mod.get_kernel("divergence"),
        "jacobi": mod.get_kernel("pressure_jacobi"),
        "subtract": mod.get_kernel("subtract_gradient"),
        "advect_dye": mod.get_kernel("advect_dye"),
        "colorize": mod.get_kernel("colorize"),
    }

    block = (16, 16, 1)
    grid = (
        (WIDTH + block[0] - 1) // block[0],
        (HEIGHT + block[1] - 1) // block[1],
        1,
    )
    config = LaunchConfig(grid=grid, block=block)

    return dev, stream, kernels, config


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
        caption="cuda.core OpaqueArray/Texture/Surface - Stable Fluids",
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


# ============================ API MAP (cuda.core) ===========================
#
# The three helpers below are where every OpaqueArray / ResourceDescriptor /
# TextureObjectOptions / TextureObject / SurfaceObject knob in this example is set.
# Each visible setting maps to a concrete piece of cuda.core / CUDA behavior:
#
#   Device.create_opaque_array(...)  -> allocates a CUDA *array* (opaque, tiled
#                                       layout optimized for 2D texture fetches),
#                                       not linear device memory.
#   ArrayFormatType.FLOAT32          -> each channel is a 32-bit float texel.
#   num_channels=2 / num_channels=1  -> float2 (vx, vy) vs scalar (pressure /
#                                       divergence / dye); also fixes the
#                                       surf2Dwrite byte offset per element.
#   is_surface_load_store=True       -> the SAME array can be bound both as a
#                                       TextureObject (cached, filtered READS)
#                                       and as a SurfaceObject (raw WRITES). This
#                                       is what lets each field be sampled and
#                                       then written back in the ping-pong.
#
#   ResourceDescriptor.from_opaque_array -> wraps the OpaqueArray as the resource a
#                                           TextureObject reads from.
#   FilterModeType.LINEAR            -> free HARDWARE bilinear interpolation;
#                                       this is what makes semi-Lagrangian
#                                       advection a single tex2D fetch at a
#                                       fractional back-traced position (no
#                                       manual lerp, no neighbor gather).
#   AddressModeType.CLAMP            -> bounded box boundary: out-of-range traces
#                                       read the edge texel (ink piles up at the
#                                       walls instead of wrapping like a torus).
#   ReadModeType.ELEMENT_TYPE        -> return the stored float value as-is (no
#                                       integer->[0,1] normalization of texels).
#   normalized_coords=True           -> sample in [0, 1) so CLAMP is well-defined
#                                       and texel centers are (i + 0.5) / N.
#
#   make_surface(arr)                -> binds the array for surf2Dread/surf2Dwrite.
#                                       The x coordinate is in BYTES, so it is
#                                       x * sizeof(elem): sizeof(float2)=8 for
#                                       velocity, sizeof(float)=4 for the scalars.
# ============================================================================


def make_velocity_array():
    """Allocate a `float2` velocity CUDA array (channel 0 = vx, channel 1 = vy)."""
    return Device().create_opaque_array(
        OpaqueArrayOptions(
            shape=(WIDTH, HEIGHT),
            format=ArrayFormatType.FLOAT32,
            num_channels=2,
            is_surface_load_store=True,
        )
    )


def make_scalar_array():
    """Allocate a single-channel `float` CUDA array (pressure / divergence / dye)."""
    return Device().create_opaque_array(
        OpaqueArrayOptions(
            shape=(WIDTH, HEIGHT),
            format=ArrayFormatType.FLOAT32,
            num_channels=1,
            is_surface_load_store=True,
        )
    )


def make_color_array():
    """Allocate a `float4` RGBA dye CUDA array.

    The dye carries a full color per cell (not just a density), so different
    bursts inject different hues that advect and mix. Same LINEAR sampling and
    surface-write machinery as the scalar fields -- only the channel count
    (and the surf2Dwrite byte stride, sizeof(float4) = 16) differ.
    """
    return Device().create_opaque_array(
        OpaqueArrayOptions(
            shape=(WIDTH, HEIGHT),
            format=ArrayFormatType.FLOAT32,
            num_channels=4,
            is_surface_load_store=True,
        )
    )


def make_texture(arr):
    """Bind `arr` as a TextureObject configured for LINEAR + CLAMP + normalized.

    One descriptor serves every read in this example: semi-Lagrangian advection
    needs the bilinear interpolation, and the stencil reads (divergence, Jacobi,
    gradient) sample exactly at texel centers so LINEAR returns the exact value.
    """
    res_desc = ResourceDescriptor.from_opaque_array(arr)
    tex_desc = TextureObjectOptions(
        address_mode=AddressModeType.CLAMP,
        filter_mode=FilterModeType.LINEAR,
        read_mode=ReadModeType.ELEMENT_TYPE,
        # Normalized coordinates keep CLAMP addressing well-defined and let us
        # sample at texel centers as (i + 0.5) / N.
        normalized_coords=True,
    )
    return Device().create_texture_object(resource=res_desc, options=tex_desc)


def make_surface(arr):
    """Bind `arr` as a SurfaceObject for raw surf2Dwrite writes."""
    res_desc = ResourceDescriptor.from_opaque_array(arr)
    return Device().create_surface_object(resource=res_desc)


def seed_field(stream, kernels, config, vel_surf, dye_surf, prs_surf, seed_value):
    """Reset the field: gentle ambient curl in velocity, zero pressure/dye.

    Takes long-lived SurfaceObjects (not freshly created ones): `launch` is
    async, so a SurfaceObject created inside a `with` block that closes right
    after `launch` returns would destroy the handle before the kernel runs.
    """
    launch(
        stream,
        config,
        kernels["seed"],
        np.uint64(vel_surf.handle),
        np.uint64(dye_surf.handle),
        np.uint64(prs_surf.handle),
        np.int32(WIDTH),
        np.int32(HEIGHT),
        np.float32(CURL_SEED),
        np.uint32(seed_value),
    )


# ================================== main() ==================================


def main(argv=None):
    parser = argparse.ArgumentParser(description="CUDA/OpenGL interop Stable Fluids demo")
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Number of frames to render before exiting (default: run until the window is closed)",
    )
    args = parser.parse_args(argv)

    # Waive when no display is available (headless CI, Wayland-only, etc.).
    import os
    import platform
    import sys

    if platform.system() == "Linux" and not os.environ.get("DISPLAY"):
        print("No DISPLAY available; waiving glInteropFluid.", file=sys.stderr)
        sys.exit(2)

    # --- Step 1: Set up CUDA (compile kernels, create stream) ---
    dev, stream, kernels, config = setup_cuda()

    # --- Step 2: Open a window ---
    try:
        window, gl, pyglet = create_window()
    except Exception as exc:
        print(f"Could not open a pyglet window ({exc}); waiving glInteropFluid.", file=sys.stderr)
        sys.exit(2)

    # --- Step 3: Create GL resources for drawing a texture to screen ---
    #     (Standard OpenGL boilerplate -- not CUDA-specific.)
    shader_prog, quad_vao, tex_id = create_display_resources(gl, WIDTH, HEIGHT)

    # --- Step 4: Create the Pixel Buffer Object (PBO) ---
    #     The PBO is GPU memory owned by OpenGL. It's the bridge between the
    #     two worlds: CUDA writes into it, OpenGL reads from it.
    pbo_id, _ = create_pixel_buffer(gl, WIDTH, HEIGHT)

    # --- Step 5: Register the PBO with CUDA ---
    resource = GraphicsResource.from_gl_buffer(pbo_id, flags="write_discard")

    # --- Step 6: Allocate the simulation fields ---
    #     velocity (float2) and dye (float) ping-pong; pressure (float)
    #     ping-pongs across Jacobi iterations; divergence (float) is a single
    #     scratch target written once per frame.
    vel_a = make_velocity_array()
    vel_b = make_velocity_array()
    prs_a = make_scalar_array()
    prs_b = make_scalar_array()
    div = make_scalar_array()
    dye_a = make_color_array()
    dye_b = make_color_array()

    # --- Step 7: Pre-create every bindless handle ONCE ---
    #     Creating texture/surface objects is comparatively expensive, and they
    #     must outlive the async launches that reference them, so we build them
    #     up front and keep them alive for the whole run.
    #     API MAP: make_texture binds an array as a read-only TextureObject
    #     (LINEAR + CLAMP + normalized; see the API MAP block above), while
    #     make_surface binds the SAME array for raw surf2Dwrite
    #     writes -- the read/write halves of one ping-pong buffer.
    vel_tex_a = make_texture(vel_a)
    vel_tex_b = make_texture(vel_b)
    vel_surf_a = make_surface(vel_a)
    vel_surf_b = make_surface(vel_b)

    prs_tex_a = make_texture(prs_a)
    prs_tex_b = make_texture(prs_b)
    prs_surf_a = make_surface(prs_a)
    prs_surf_b = make_surface(prs_b)

    div_tex = make_texture(div)
    div_surf = make_surface(div)

    dye_tex_a = make_texture(dye_a)
    dye_tex_b = make_texture(dye_b)
    dye_surf_a = make_surface(dye_a)
    dye_surf_b = make_surface(dye_b)

    # --- Step 8: Seed the initial field (curl into vel_a, zero pressure/dye) ---
    seed_field(stream, kernels, config, vel_surf_a, dye_surf_a, prs_surf_a, seed_value=0)
    stream.sync()

    # `vel` / `dye` track which ping-pong array currently holds the live state.
    state = {"vel": "a", "dye": "a", "seed": 0, "next_burst": 0.0}

    # Mouse state shared with the event handlers. Coordinates are in SIMULATION
    # space (y = 0 at top); the framebuffer has y = 0 at the bottom, so we flip.
    mouse = {"down": False, "x": 0.0, "y": 0.0, "dx": 0.0, "dy": 0.0}

    def vel_pair():
        # Read live velocity, write the other buffer; returns (read_tex, write_surf, next).
        if state["vel"] == "a":
            return vel_tex_a, vel_surf_b, "b"
        return vel_tex_b, vel_surf_a, "a"

    def vel_live_tex():
        return vel_tex_a if state["vel"] == "a" else vel_tex_b

    def vel_live_surf():
        return vel_surf_a if state["vel"] == "a" else vel_surf_b

    def dye_pair():
        if state["dye"] == "a":
            return dye_tex_a, dye_surf_b, "b"
        return dye_tex_b, dye_surf_a, "a"

    def dye_live_tex():
        return dye_tex_a if state["dye"] == "a" else dye_tex_b

    def dye_live_surf():
        return dye_surf_a if state["dye"] == "a" else dye_surf_b

    # --- Step 9: Render loop ---
    start_time = time.monotonic()
    frame_count = 0
    frames_rendered = 0
    fps_time = start_time
    clock = {"last": start_time}  # wall-clock time of the previous frame
    resources_closed = False

    def close_resources():
        nonlocal resources_closed
        if resources_closed:
            return
        resources_closed = True

        # Release everything we opened, in reverse order. Each of these is a
        # context manager too, but pyglet owns the event loop here so we
        # release explicitly to be deterministic about ordering.
        resource.close()
        dye_tex_a.close()
        dye_tex_b.close()
        dye_surf_a.close()
        dye_surf_b.close()
        div_tex.close()
        div_surf.close()
        prs_tex_a.close()
        prs_tex_b.close()
        prs_surf_a.close()
        prs_surf_b.close()
        vel_tex_a.close()
        vel_tex_b.close()
        vel_surf_a.close()
        vel_surf_b.close()
        dye_a.close()
        dye_b.close()
        div.close()
        prs_a.close()
        prs_b.close()
        vel_a.close()
        vel_b.close()
        stream.close()

    def _window_to_sim(x, y):
        # Window: y = 0 at bottom. Simulation: y = 0 at top. Flip vertically.
        sx = float(x)
        sy = float(HEIGHT - 1 - y)
        return sx, sy

    @window.event
    def on_key_press(symbol, _modifiers):
        key = pyglet.window.key
        if symbol == key.ESCAPE:
            pyglet.app.exit()
            return pyglet.event.EVENT_HANDLED
        if symbol == key.R:
            state["seed"] += 1
            seed_field(
                stream,
                kernels,
                config,
                vel_surf_a,
                dye_surf_a,
                prs_surf_a,
                seed_value=state["seed"],
            )
            state["vel"] = "a"
            state["dye"] = "a"
            return

    @window.event
    def on_mouse_press(x, y, _button, _modifiers):
        mouse["down"] = True
        mouse["x"], mouse["y"] = _window_to_sim(x, y)
        mouse["dx"] = 0.0
        mouse["dy"] = 0.0

    @window.event
    def on_mouse_release(_x, _y, _button, _modifiers):
        mouse["down"] = False
        mouse["dx"] = 0.0
        mouse["dy"] = 0.0

    @window.event
    def on_mouse_drag(x, y, dx, dy, _buttons, _modifiers):
        # The mouse delta IS the injected velocity. Framebuffer dy is up-positive
        # while simulation y is down-positive, so the sim-space delta is -dy.
        mouse["down"] = True
        mouse["x"], mouse["y"] = _window_to_sim(x, y)
        mouse["dx"] = float(dx)
        mouse["dy"] = float(-dy)

    @window.event
    def on_draw():
        nonlocal frame_count, frames_rendered, fps_time

        window.clear()
        now_t = time.monotonic()
        elapsed = now_t - start_time

        # Frame-rate independence: express this frame's real duration in units of
        # a REF_FPS reference step. `step` scales the advection distance, and the
        # per-step dissipations are raised to `step` so their per-SECOND rate is
        # constant no matter how fast the loop runs. Clamp to absorb the first
        # frame and any hitch without launching a giant (unstable-looking) step.
        dt_real = now_t - clock["last"]
        clock["last"] = now_t
        step = min(max(dt_real * REF_FPS, 0.0), 3.0)
        dt_adv = DT * step
        vel_diss = VELOCITY_DISSIPATION**step
        dye_diss = DYE_DISSIPATION**step

        # (a) Advect velocity along itself (semi-Lagrangian, tex2D LINEAR).
        vel_read, vel_write, vel_next = vel_pair()
        launch(
            stream,
            config,
            kernels["advect_vel"],
            np.uint64(vel_read.handle),
            np.uint64(vel_write.handle),
            np.int32(WIDTH),
            np.int32(HEIGHT),
            np.float32(dt_adv),
            np.float32(vel_diss),
        )
        state["vel"] = vel_next

        # (b) Splat mouse-drag velocity and colored dye into the live fields.
        #     The injected color cycles through hues over time so dragging
        #     paints a rainbow ribbon of ink.
        inject = 1 if mouse["down"] else 0
        mr, mg, mb = colorsys.hsv_to_rgb((elapsed * 0.15) % 1.0, 0.85, 1.0)
        launch(
            stream,
            config,
            kernels["splat"],
            np.uint64(vel_live_surf().handle),
            np.uint64(dye_live_surf().handle),
            np.int32(WIDTH),
            np.int32(HEIGHT),
            np.float32(mouse["x"]),
            np.float32(mouse["y"]),
            np.float32(mouse["dx"] * SPLAT_FORCE),
            np.float32(mouse["dy"] * SPLAT_FORCE),
            np.float32(SPLAT_RADIUS),
            np.float32(mr * SPLAT_DYE),
            np.float32(mg * SPLAT_DYE),
            np.float32(mb * SPLAT_DYE),
            np.int32(inject),
        )

        # (b2) When the user is not dragging, periodically drop big blobs of a
        #      random bright color with a random velocity impulse at random
        #      spots -- the classic "ink in water" look. Reuses the same `splat`
        #      kernel as the mouse, just with a color argument.
        if AUTO_EMIT and not mouse["down"] and elapsed >= state["next_burst"]:
            state["next_burst"] = elapsed + BURST_INTERVAL
            for _ in range(BURSTS_PER_EVENT):
                bx = random.uniform(0.12, 0.88) * WIDTH
                by = random.uniform(0.12, 0.88) * HEIGHT
                ang = random.uniform(0.0, 2.0 * math.pi)
                bfx = math.cos(ang) * BURST_FORCE
                bfy = math.sin(ang) * BURST_FORCE
                br, bg, bb = colorsys.hsv_to_rgb(random.random(), 0.9, 1.0)
                launch(
                    stream,
                    config,
                    kernels["splat"],
                    np.uint64(vel_live_surf().handle),
                    np.uint64(dye_live_surf().handle),
                    np.int32(WIDTH),
                    np.int32(HEIGHT),
                    np.float32(bx),
                    np.float32(by),
                    np.float32(bfx),
                    np.float32(bfy),
                    np.float32(BURST_RADIUS),
                    np.float32(br * BURST_DYE),
                    np.float32(bg * BURST_DYE),
                    np.float32(bb * BURST_DYE),
                    np.int32(1),
                )

        # (b3) Vorticity confinement: read the live velocity through its
        #      TextureObject, compute curl + grad|curl|, and add a force that
        #      pushes velocity back toward high-vorticity regions -- this is the
        #      one extra kernel that sharpens the curling plumes. Like
        #      advect_velocity, it reads neighbor velocities, so it MUST
        #      ping-pong (read old buffer, write the other) -- aliasing a
        #      texture read with a surface write of the same array in one launch
        #      is undefined.
        if VORTICITY > 0.0:
            vort_read, vort_write, vort_next = vel_pair()
            launch(
                stream,
                config,
                kernels["vorticity"],
                np.uint64(vort_read.handle),
                np.uint64(vort_write.handle),
                np.int32(WIDTH),
                np.int32(HEIGHT),
                np.float32(dt_adv),
                np.float32(VORTICITY),
            )
            state["vel"] = vort_next

        # (c) Compute divergence of the live velocity field.
        launch(
            stream,
            config,
            kernels["divergence"],
            np.uint64(vel_live_tex().handle),
            np.uint64(div_surf.handle),
            np.int32(WIDTH),
            np.int32(HEIGHT),
        )

        # (d) Pressure solve: Jacobi-iterate lap(p) = div, ping-ponging pressure.
        #     Start from a cleared pressure field (prs_a) each frame.
        launch(
            stream,
            config,
            kernels["jacobi"],
            np.uint64(prs_tex_a.handle),  # ignored on the first pass via clear flag
            np.uint64(div_tex.handle),
            np.uint64(prs_surf_b.handle),
            np.int32(WIDTH),
            np.int32(HEIGHT),
            np.int32(1),  # clear: treat the previous pressure as zero
        )
        # After the clearing pass the result lives in prs_b. Continue iterating.
        prs_cur = "b"
        for _ in range(PRESSURE_ITERS - 1):
            if prs_cur == "b":
                read_tex, write_surf, prs_cur = prs_tex_b, prs_surf_a, "a"
            else:
                read_tex, write_surf, prs_cur = prs_tex_a, prs_surf_b, "b"
            launch(
                stream,
                config,
                kernels["jacobi"],
                np.uint64(read_tex.handle),
                np.uint64(div_tex.handle),
                np.uint64(write_surf.handle),
                np.int32(WIDTH),
                np.int32(HEIGHT),
                np.int32(0),  # do not clear: read the previous pressure
            )
        # `prs_cur` now names the buffer holding the converged pressure.
        prs_final_tex = prs_tex_a if prs_cur == "a" else prs_tex_b

        # (e) Subtract pressure gradient from the live velocity (in-place).
        launch(
            stream,
            config,
            kernels["subtract"],
            np.uint64(prs_final_tex.handle),
            np.uint64(vel_live_surf().handle),
            np.int32(WIDTH),
            np.int32(HEIGHT),
        )

        # (f) Advect the dye along the (now divergence-free) velocity field.
        dye_read, dye_write, dye_next = dye_pair()
        launch(
            stream,
            config,
            kernels["advect_dye"],
            np.uint64(dye_read.handle),
            np.uint64(vel_live_tex().handle),
            np.uint64(dye_write.handle),
            np.int32(WIDTH),
            np.int32(HEIGHT),
            np.float32(dt_adv),
            np.float32(dye_diss),
        )
        state["dye"] = dye_next

        # (g) Colorize the latest dye into the OpenGL PBO.
        with resource.map(stream=stream) as buf:
            launch(
                stream,
                config,
                kernels["colorize"],
                np.uint64(dye_live_tex().handle),
                buf.handle,
                np.int32(WIDTH),
                np.int32(HEIGHT),
            )
        # Unmap happens automatically when the `with` block exits.

        # (h) Tell OpenGL to copy the PBO contents into our texture.
        copy_pbo_to_texture(gl, pbo_id, tex_id, WIDTH, HEIGHT)

        # (i) Draw the texture to the screen.
        draw_fullscreen_quad(gl, shader_prog, quad_vao, tex_id)

        # Reset the per-frame mouse delta so a held-still cursor stops pushing.
        mouse["dx"] = 0.0
        mouse["dy"] = 0.0

        # FPS counter (shown in window title)
        frame_count += 1
        frames_rendered += 1
        now = time.monotonic()
        if now - fps_time >= 1.0:
            fps = frame_count / (now - fps_time)
            window.set_caption(
                "cuda.core OpaqueArray/Texture/Surface - Stable Fluids"
                f" ({WIDTH}x{HEIGHT}, {fps:.0f} FPS,"
                f" {PRESSURE_ITERS} pressure iters)"
                " | TextureObject[LINEAR|CLAMP|norm|float2]"
                " + SurfaceObject writes + GraphicsResource(PBO)"
            )
            frame_count = 0
            fps_time = now

        if args.frames is not None and frames_rendered >= args.frames:
            # Let pyglet finish the current refresh/flip before tearing down
            # the GL context and registered CUDA graphics resource.
            pyglet.app.exit()

    @window.event
    def on_close():
        close_resources()

    # Render as fast as the GPU allows; the per-step rates are scaled by real
    # elapsed time (see REF_FPS) so the look is frame-rate independent.
    try:
        pyglet.app.run(interval=0)
    finally:
        close_resources()
        window.close()
    print(f"\nRendered {frames_rendered} fluid simulation frames. Done")


# ======================== GPU code (CUDA + GLSL) ============================
#
# These source strings are kept at the bottom of the file so they don't
# distract from the Python logic above. The important things to know:
#
#   - KERNEL_SOURCE contains the eight CUDA C++ kernels of the Stable Fluids
#     pipeline. Reads go through cudaTextureObject_t (LINEAR + CLAMP +
#     normalized coords); writes go through cudaSurfaceObject_t with the x
#     offset in BYTES. A small helper converts pixel coords to normalized
#     texel-center coords.
#
#   - VERTEX_SHADER_SOURCE / FRAGMENT_SHADER_SOURCE are GLSL. They draw a
#     texture onto a rectangle covering the entire window. Nothing interesting.
#
# ============================================================================

KERNEL_SOURCE = r"""
// Sample a float2 (velocity) field at pixel center (px, py) with bilinear
// filtering. CLAMP addressing keeps out-of-range traces at the border.
__device__ __forceinline__
float2 sample_vec(cudaTextureObject_t tex, float px, float py,
                  int width, int height) {
    float u = (px + 0.5f) / (float)width;
    float v = (py + 0.5f) / (float)height;
    return tex2D<float2>(tex, u, v);
}

// Sample a scalar (float) field at pixel center (px, py) with bilinear filtering.
__device__ __forceinline__
float sample_scalar(cudaTextureObject_t tex, float px, float py,
                    int width, int height) {
    float u = (px + 0.5f) / (float)width;
    float v = (py + 0.5f) / (float)height;
    return tex2D<float>(tex, u, v);
}

// Sample a float4 (RGBA dye) field at pixel center with bilinear filtering.
__device__ __forceinline__
float4 sample_color(cudaTextureObject_t tex, float px, float py,
                    int width, int height) {
    float u = (px + 0.5f) / (float)width;
    float v = (py + 0.5f) / (float)height;
    return tex2D<float4>(tex, u, v);
}

extern "C"
__global__
void seed_field(cudaSurfaceObject_t vel_surf,
                cudaSurfaceObject_t dye_surf,
                cudaSurfaceObject_t prs_surf,
                int width, int height,
                float curl, unsigned int seed) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // Seed a gentle global rotation: velocity perpendicular to the radius from
    // the center gives a curl, so even with no mouse input there is motion.
    float cx = width * 0.5f;
    float cy = height * 0.5f;
    float rx = (x - cx) / cx;   // ~[-1, 1]
    float ry = (y - cy) / cy;
    float2 vel = make_float2(-ry * curl, rx * curl);

    // A touch of deterministic noise so successive resets look a little
    // different and to break perfect symmetry.
    unsigned int h = (unsigned int)x * 374761393u +
                     (unsigned int)y * 668265263u + seed * 2246822519u;
    h = (h ^ (h >> 13)) * 1274126177u;
    h = h ^ (h >> 16);
    float noise = ((h & 0xffffu) / 65535.0f) - 0.5f;   // [-0.5, 0.5]
    vel.x += noise * 0.2f;
    vel.y += noise * 0.2f;

    // Dye starts black; the colored bursts (or the mouse) paint the ink, so
    // there is nothing to seed here beyond clearing to zero.
    surf2Dwrite(vel, vel_surf, x * (int)sizeof(float2), y);
    surf2Dwrite(make_float4(0.0f, 0.0f, 0.0f, 0.0f), dye_surf,
                x * (int)sizeof(float4), y);
    surf2Dwrite(0.0f, prs_surf, x * (int)sizeof(float), y);
}

// Inject mouse-drag velocity and dye into a soft radial brush around the
// cursor. In-place read-modify-write: each thread owns its own cell, no race.
extern "C"
__global__
void splat(cudaSurfaceObject_t vel_surf,
           cudaSurfaceObject_t dye_surf,
           int width, int height,
           float mx, float my,
           float fx, float fy,
           float radius, float dr, float dg, float db,
           int inject) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    if (!inject) return;

    float dx = (float)x - mx;
    float dy = (float)y - my;
    float d2 = dx * dx + dy * dy;
    float falloff = expf(-d2 / (radius * radius));
    if (falloff < 1e-3f) return;

    float2 vel;
    surf2Dread(&vel, vel_surf, x * (int)sizeof(float2), y);
    vel.x += fx * falloff;
    vel.y += fy * falloff;
    surf2Dwrite(vel, vel_surf, x * (int)sizeof(float2), y);

    // Additive colored ink. float4 surface element is 16 bytes.
    float4 dye;
    surf2Dread(&dye, dye_surf, x * (int)sizeof(float4), y);
    dye.x += dr * falloff;
    dye.y += dg * falloff;
    dye.z += db * falloff;
    dye.w = 1.0f;
    surf2Dwrite(dye, dye_surf, x * (int)sizeof(float4), y);
}

// Semi-Lagrangian advection of the velocity field along itself.
extern "C"
__global__
void advect_velocity(cudaTextureObject_t vel_tex,
                     cudaSurfaceObject_t vel_out,
                     int width, int height,
                     float dt, float dissipation) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float2 v = sample_vec(vel_tex, (float)x, (float)y, width, height);
    // Trace this cell's center backward along the velocity field.
    float px = (float)x - dt * v.x;
    float py = (float)y - dt * v.y;
    float2 advected = sample_vec(vel_tex, px, py, width, height);
    advected.x *= dissipation;
    advected.y *= dissipation;
    surf2Dwrite(advected, vel_out, x * (int)sizeof(float2), y);
}

// Vorticity confinement. Curl of a 2D velocity field is the scalar
// w = dVy/dx - dVx/dy. Where |w| has a gradient we add a force that pushes
// velocity along the swirl, reinjecting the small-scale rotation that
// numerical diffusion smears away -- the result is crisper, longer-lived
// curls. Reads neighbor velocities through the TextureObject and writes the
// updated velocity to a SEPARATE ping-pong buffer (no read/write aliasing).
__device__ __forceinline__
float curl_at(cudaTextureObject_t vel_tex, float px, float py,
              int width, int height) {
    float2 l = sample_vec(vel_tex, px - 1.0f, py, width, height);
    float2 r = sample_vec(vel_tex, px + 1.0f, py, width, height);
    float2 d = sample_vec(vel_tex, px, py - 1.0f, width, height);
    float2 u = sample_vec(vel_tex, px, py + 1.0f, width, height);
    return 0.5f * ((r.y - l.y) - (u.x - d.x));
}

extern "C"
__global__
void vorticity_confinement(cudaTextureObject_t vel_tex,
                           cudaSurfaceObject_t vel_out,
                           int width, int height,
                           float dt, float eps) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float fx = (float)x;
    float fy = (float)y;

    // Curl at this cell and at the 4 neighbors (for grad|curl|).
    float w = curl_at(vel_tex, fx, fy, width, height);
    float wl = curl_at(vel_tex, fx - 1.0f, fy, width, height);
    float wr = curl_at(vel_tex, fx + 1.0f, fy, width, height);
    float wd = curl_at(vel_tex, fx, fy - 1.0f, width, height);
    float wu = curl_at(vel_tex, fx, fy + 1.0f, width, height);

    // Gradient of |curl|, normalized to a unit direction N.
    float gx = 0.5f * (fabsf(wr) - fabsf(wl));
    float gy = 0.5f * (fabsf(wu) - fabsf(wd));
    float len = sqrtf(gx * gx + gy * gy) + 1e-5f;
    float nx = gx / len;
    float ny = gy / len;

    // Confinement force = eps * (N x w_hat). In 2D: (N_y * w, -N_x * w).
    float2 v = sample_vec(vel_tex, fx, fy, width, height);
    v.x += eps * dt * (ny * w);
    v.y += eps * dt * (-nx * w);
    surf2Dwrite(v, vel_out, x * (int)sizeof(float2), y);
}

// Divergence of the velocity field (central differences), written as a scalar.
extern "C"
__global__
void divergence(cudaTextureObject_t vel_tex,
                cudaSurfaceObject_t div_out,
                int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float2 l = sample_vec(vel_tex, (float)x - 1.0f, (float)y, width, height);
    float2 r = sample_vec(vel_tex, (float)x + 1.0f, (float)y, width, height);
    float2 d = sample_vec(vel_tex, (float)x, (float)y - 1.0f, width, height);
    float2 u = sample_vec(vel_tex, (float)x, (float)y + 1.0f, width, height);

    float div = 0.5f * ((r.x - l.x) + (u.y - d.y));
    surf2Dwrite(div, div_out, x * (int)sizeof(float), y);
}

// One Jacobi iteration of lap(p) = div. With unit grid spacing the update is
// p = (p_left + p_right + p_down + p_up - div) / 4. When `clear` is set the
// previous pressure is treated as zero so the first pass starts clean.
extern "C"
__global__
void pressure_jacobi(cudaTextureObject_t prs_tex,
                     cudaTextureObject_t div_tex,
                     cudaSurfaceObject_t prs_out,
                     int width, int height,
                     int clear) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float pl = 0.0f, pr = 0.0f, pd = 0.0f, pu = 0.0f;
    if (!clear) {
        pl = sample_scalar(prs_tex, (float)x - 1.0f, (float)y, width, height);
        pr = sample_scalar(prs_tex, (float)x + 1.0f, (float)y, width, height);
        pd = sample_scalar(prs_tex, (float)x, (float)y - 1.0f, width, height);
        pu = sample_scalar(prs_tex, (float)x, (float)y + 1.0f, width, height);
    }
    float div = sample_scalar(div_tex, (float)x, (float)y, width, height);
    float p = (pl + pr + pd + pu - div) * 0.25f;
    surf2Dwrite(p, prs_out, x * (int)sizeof(float), y);
}

// v <- v - grad(p): project the velocity onto its divergence-free part.
extern "C"
__global__
void subtract_gradient(cudaTextureObject_t prs_tex,
                       cudaSurfaceObject_t vel_surf,
                       int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float pl = sample_scalar(prs_tex, (float)x - 1.0f, (float)y, width, height);
    float pr = sample_scalar(prs_tex, (float)x + 1.0f, (float)y, width, height);
    float pd = sample_scalar(prs_tex, (float)x, (float)y - 1.0f, width, height);
    float pu = sample_scalar(prs_tex, (float)x, (float)y + 1.0f, width, height);

    float2 v;
    surf2Dread(&v, vel_surf, x * (int)sizeof(float2), y);
    v.x -= 0.5f * (pr - pl);
    v.y -= 0.5f * (pu - pd);
    surf2Dwrite(v, vel_surf, x * (int)sizeof(float2), y);
}

// Semi-Lagrangian advection of the dye along the velocity field.
extern "C"
__global__
void advect_dye(cudaTextureObject_t dye_tex,
                cudaTextureObject_t vel_tex,
                cudaSurfaceObject_t dye_out,
                int width, int height,
                float dt, float dissipation) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float2 v = sample_vec(vel_tex, (float)x, (float)y, width, height);
    float px = (float)x - dt * v.x;
    float py = (float)y - dt * v.y;
    float4 d = sample_color(dye_tex, px, py, width, height);
    d.x *= dissipation;
    d.y *= dissipation;
    d.z *= dissipation;
    d.w *= dissipation;
    surf2Dwrite(d, dye_out, x * (int)sizeof(float4), y);
}

// Tonemap the accumulated float4 dye color into the PBO. The ink color is
// whatever the bursts/mouse injected and advection mixed; we apply a filmic
// 1 - exp(-c) curve so dense ink stays vivid without harshly clipping.
extern "C"
__global__
void colorize(cudaTextureObject_t dye_tex,
              unsigned char* output,
              int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float4 c = sample_color(dye_tex, (float)x, (float)y, width, height);
    const float gain = 1.3f;
    float r = 1.0f - expf(-fmaxf(c.x, 0.0f) * gain);
    float g = 1.0f - expf(-fmaxf(c.y, 0.0f) * gain);
    float b = 1.0f - expf(-fmaxf(c.z, 0.0f) * gain);

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
