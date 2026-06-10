# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# ################################################################################
#
# This example demonstrates cuda.core.CUDAArray, TextureObject, and SurfaceObject
# in combination with GraphicsResource for CUDA/OpenGL interop. A Lenia
# continuous cellular automaton is ping-ponged between two CUDA arrays each
# frame: a TextureObject provides smooth (LINEAR + WRAP) sampled reads through
# a large bell-shaped neighborhood kernel, and a SurfaceObject provides typed
# writes. The final state is colorized straight into an OpenGL PBO. Requires
# pyglet.
#
# ################################################################################

# What this example teaches
# =========================
# - How to drive a wide-radius convolution from a TextureObject configured for
#   LINEAR + WRAP + normalized coordinates. The same CUDAArray is then bound as a
#   SurfaceObject for the typed write back, requiring `is_surface_load_store=True`
#   at allocation time.
# - How a single-channel `float` CUDAArray differs from the multi-channel layout
#   used in the Gray-Scott example: `num_channels=1`, `tex2D<float>` reads, and
#   a 4-byte x-stride in `surf2Dwrite`.
# - How to host-precompute a normalization constant for a stencil with a
#   variable-shape support (the bell-curve neighborhood), then pass it as a
#   plain float kernel argument.
#
# How it works
# ============
# Lenia (Bert Wang-Chak Chan, 2018) generalizes Conway's Game of Life to
# continuous space, time, and state. Each cell holds a real value in [0, 1].
# Per step, every cell:
#
#   1. Integrates a smooth bell-shaped neighborhood kernel K against the
#      current state to produce a "potential" U:
#
#          U(x) = sum over offsets (dx, dy) inside a disk of radius R of
#                  K(|(dx, dy)|) * state(x + (dx, dy))
#                 divided by  sum of K  (host-precomputed).
#
#      K(r) = exp(-((r / R) - mu_K)^2 / (2 * sigma_K^2)) for r <= R.
#
#   2. Applies the growth function G and updates the state:
#
#          state_new = clamp(state_old + dt * (2 * exp(-(U - mu)^2 /
#                            (2 * sigma^2)) - 1),  0,  1).
#
# Two single-channel `float` arrays are ping-ponged each frame: a
# TextureObject reads one (sampled with LINEAR + WRAP so the disk wraps
# toroidally) and a SurfaceObject writes the other.
#
#   PING-PONG (two arrays, swap each step)
#   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   +--------------+   tex2D<float>    +------------------+
#   |   arr_a      | ----------------> |                  |
#   |    state     |                   |  convolve_lenia  |
#   +--------------+                   |     kernel       |
#                                      |  (+ growth fn)   |
#   +--------------+   surf2Dwrite     |                  |
#   |   arr_b      | <---------------- |                  |
#   |    state     |                   +------------------+
#   +--------------+
#       (swap)
#
# After the step we run a separate `colorize_lenia` kernel that samples the
# new state and writes RGBA bytes straight into the OpenGL PBO via
# GraphicsResource. No data ever travels across the PCIe bus during the frame.
#
# Why LINEAR + WRAP + normalized coords?
# --------------------------------------
# Lenia's neighborhood radius (R = 13) is wide enough that boundary handling
# really matters. AddressMode.WRAP gives a toroidal world for free, and it is
# only supported in normalized coordinate mode (see the CUDA Programming
# Guide). LINEAR filtering is essentially free on the hardware -- here it
# softens the integer-offset reads a hair, which keeps the dynamics smooth.
# Sample coordinates are `(x + dx + 0.5) / W`; values < 0 or > 1 are fine,
# WRAP handles them.
#
# Channel byte width in surf2Dwrite
# ---------------------------------
# `surf2Dwrite` takes the x coordinate in BYTES, not in elements. For a
# single-channel `float` surface that means `x * sizeof(float)` = `x * 4`.
# (The Gray-Scott example uses 8 because it stores `float2`.)
#
# One step per frame
# ------------------
# Each step convolves a (2R+1)^2 = 729-tap neighborhood for every pixel, which
# is much heavier than a Gray-Scott 5-point Laplacian. With dt = 0.1 the
# dynamics are slow enough that one step per displayed frame is plenty. There
# is no `N_STEPS` loop.
#
# What you should see
# ===================
# A window showing soft, glider-like blobs drifting across the field on a
# teal-on-black palette. Press R to reseed with a new Gaussian blob, 1 to
# clear the field, and Escape to exit. The window title shows the current
# FPS.
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
    SurfaceObject,
    TextureDescriptor,
    TextureObject,
    launch,
)

# ---------------------------------------------------------------------------
# Simulation parameters (feel free to change these)
# ---------------------------------------------------------------------------
WIDTH = 256
HEIGHT = 256

# Neighborhood / kernel shape
R = 13  # convolution radius in pixels (texture-space)
MU_K = 0.5  # bell center for the neighborhood weight K(r/R)
SIGMA_K = 0.15  # bell width for K

# Growth function shape
MU = 0.15  # bell center for the growth function G(U)
SIGMA = 0.015  # bell width for G

DT = 0.1  # time step

# Initial blob radius and peak for the Gaussian seed.
# The radius must be large relative to the neighborhood radius R=13 so the
# kernel-integrated potential U lands near the growth bell's center mu=0.15.
# With SEED_RADIUS=36, U at the blob's centre starts near mu and the field
# survives the first step; smaller seeds collapse to zero within one frame
# because U is far outside the narrow (sigma=0.015) growth bell.
SEED_RADIUS = 36.0
SEED_PEAK = 0.5

# Seed modes (kept in sync with the seed_blob kernel)
SEED_MODE_CLEAR = 0
SEED_MODE_BLOB = 1


# ============================= Helper functions =============================
#
# The functions below set up CUDA and OpenGL. If you're here to learn about
# CUDAArray/TextureObject/SurfaceObject, skip ahead to main() -- the interesting
# part is there. These helpers exist so that main() reads like a short story
# instead of a wall of boilerplate.
# ============================================================================


def compute_kernel_norm(radius, mu_k, sigma_k):
    """Precompute 1 / (sum of K(r)) for the bell-shaped neighborhood weight.

    Mirrors exactly what the device kernel does so the convolution is energy-
    preserving: walks the (2R+1)x(2R+1) box, accumulates
    `exp(-(r/R - mu_k)^2 / (2*sigma_k^2))` for `r <= R`, and returns the
    reciprocal sum as a float32.
    """
    inv_two_sigma2 = 1.0 / (2.0 * sigma_k * sigma_k)
    inv_r = 1.0 / float(radius)
    total = 0.0
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            r = math.sqrt(dx * dx + dy * dy)
            if r > radius:
                continue
            rn = r * inv_r - mu_k
            total += math.exp(-(rn * rn) * inv_two_sigma2)
    if total <= 0.0:
        raise RuntimeError("kernel normalization sum collapsed to zero")
    return np.float32(1.0 / total)


def setup_cuda():
    """Compile the CUDA kernels and return (device, stream, kernels, configs).

    Returns a dict of kernels keyed by name and matching LaunchConfigs.
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

    # Compile as C++ so the templated tex2D<float> overload resolves.
    program_options = ProgramOptions(std="c++17", arch=f"sm_{dev.arch}")
    prog = Program(KERNEL_SOURCE, code_type="c++", options=program_options)
    mod = prog.compile(
        "cubin",
        name_expressions=("convolve_lenia", "colorize_lenia", "seed_blob"),
    )

    kernels = {
        "step": mod.get_kernel("convolve_lenia"),
        "colorize": mod.get_kernel("colorize_lenia"),
        "seed": mod.get_kernel("seed_blob"),
    }

    block = (16, 16, 1)
    grid = (
        (WIDTH + block[0] - 1) // block[0],
        (HEIGHT + block[1] - 1) // block[1],
        1,
    )
    config = LaunchConfig(grid=grid, block=block)
    # All three kernels are pixel-parallel over a WIDTH x HEIGHT grid, so they
    # can share a launch config.
    configs = {"step": config, "colorize": config, "seed": config}

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
        caption="cuda.core CUDAArray/Texture/Surface - Lenia",
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
    """Allocate the two single-channel `float` ping-pong arrays.

    `is_surface_load_store=True` is what lets the same CUDAArray be bound as both a
    TextureObject (sampled reads) and a SurfaceObject (typed writes).
    """
    arr_a = CUDAArray.from_descriptor(
        shape=(WIDTH, HEIGHT),
        format=ArrayFormat.FLOAT32,
        num_channels=1,
        is_surface_load_store=True,
    )
    arr_b = CUDAArray.from_descriptor(
        shape=(WIDTH, HEIGHT),
        format=ArrayFormat.FLOAT32,
        num_channels=1,
        is_surface_load_store=True,
    )
    return arr_a, arr_b


def make_texture(arr):
    """Bind `arr` as a TextureObject configured for LINEAR + WRAP + normalized."""
    res_desc = ResourceDescriptor.from_array(arr)
    tex_desc = TextureDescriptor(
        address_mode=AddressMode.WRAP,
        filter_mode=FilterMode.LINEAR,
        read_mode=ReadMode.ELEMENT_TYPE,
        # WRAP/MIRROR addressing modes require normalized coordinates.
        normalized_coords=True,
    )
    return TextureObject.from_descriptor(resource=res_desc, texture_descriptor=tex_desc)


def seed_state(stream, kernels, configs, write_surf, mode, seed_value):
    """Re-initialize the array behind `write_surf` with a Gaussian blob or zeros.

    `mode = SEED_MODE_CLEAR` zeroes the field; `mode = SEED_MODE_BLOB` places a
    Gaussian blob with peak ~SEED_PEAK at the center, jittered by `seed_value`
    so successive reseeds give different patterns.

    Takes a long-lived SurfaceObject (not a fresh one): `launch` is async, so
    creating a SurfaceObject inside a `with` block that closes immediately
    after `launch` returns would destroy the surface handle before the kernel
    actually runs against it.
    """
    launch(
        stream,
        configs["seed"],
        kernels["seed"],
        np.uint64(write_surf.handle),
        np.int32(WIDTH),
        np.int32(HEIGHT),
        np.int32(mode),
        np.uint32(seed_value),
        np.float32(SEED_RADIUS),
        np.float32(SEED_PEAK),
    )


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
    #     The PBO is GPU memory owned by OpenGL. It's the bridge between the
    #     two worlds: CUDA writes into it, OpenGL reads from it.
    pbo_id, _ = create_pixel_buffer(gl, WIDTH, HEIGHT)

    # --- Step 5: Register the PBO with CUDA ---
    resource = GraphicsResource.from_gl_buffer(pbo_id, flags="write_discard")

    # --- Step 6: Allocate the two ping-pong state Arrays ---
    #     Both are single-channel `float` with `is_surface_load_store=True` so
    #     they can be bound as SurfaceObjects.
    arr_a, arr_b = make_state_arrays()

    # --- Step 7: Pre-create the four bindless handles ---
    #     Creating these once is much cheaper than rebuilding them every
    #     step. The simulation loop just picks which read/write pair to use.
    tex_a = make_texture(arr_a)
    tex_b = make_texture(arr_b)
    surf_a = SurfaceObject.from_array(arr_a)
    surf_b = SurfaceObject.from_array(arr_b)

    # --- Step 8: Precompute the bell-curve normalization constant ---
    #     The neighborhood weight K(r) is unnormalized in the kernel; we
    #     divide by sum(K) so the convolution is a weighted mean rather than
    #     an unbounded integral. Doing this on the host once at startup is
    #     much cheaper than redoing it on the device every step.
    inv_weight_sum = compute_kernel_norm(R, MU_K, SIGMA_K)

    # --- Step 9: Seed an initial Gaussian blob into arr_a (writes via surf_a) ---
    seed_state(stream, kernels, configs, surf_a, SEED_MODE_BLOB, seed_value=0)
    # After seeding, `arr_a` is the "current" state.
    state = {"current": "a", "seed": 0}

    # --- Step 10: Render loop ---
    start_time = time.monotonic()
    frame_count = 0
    fps_time = start_time

    def current_read_write():
        if state["current"] == "a":
            return tex_a, surf_b, "b"  # read a, write b, next current = b
        return tex_b, surf_a, "a"

    @window.event
    def on_key_press(symbol, _modifiers):
        key = pyglet.window.key
        if symbol == key.ESCAPE:
            window.close()
            return
        if symbol == key.R:
            # Reseed with a new Gaussian blob; bump the seed so the jitter
            # pattern changes each time.
            state["seed"] += 1
            seed_state(stream, kernels, configs, surf_a, SEED_MODE_BLOB, state["seed"])
            state["current"] = "a"
            return
        if symbol == key._1:
            # Clear the field. Useful to confirm the simulation is quiet when
            # the state is zero.
            seed_state(stream, kernels, configs, surf_a, SEED_MODE_CLEAR, 0)
            state["current"] = "a"
            return

    @window.event
    def on_draw():
        nonlocal frame_count, fps_time

        window.clear()

        # (a) Run one Lenia step. The convolution kernel reads the current
        #     state via a TextureObject (LINEAR + WRAP gives toroidal
        #     wrapping at the border), evaluates the growth function, and
        #     writes the new state via a SurfaceObject. One step per frame
        #     is intentional: dt = 0.1 is small, and the (2R+1)^2 = 729-tap
        #     stencil is heavy enough that going faster would not help.
        tex_read, surf_write, next_current = current_read_write()
        launch(
            stream,
            configs["step"],
            kernels["step"],
            np.uint64(tex_read.handle),
            np.uint64(surf_write.handle),
            np.int32(WIDTH),
            np.int32(HEIGHT),
            np.int32(R),
            np.float32(MU_K),
            np.float32(SIGMA_K),
            np.float32(MU),
            np.float32(SIGMA),
            np.float32(DT),
            inv_weight_sum,
        )
        state["current"] = next_current

        # (b) Colorize the latest state into the OpenGL PBO.
        tex_read = tex_a if state["current"] == "a" else tex_b
        with resource.map(stream=stream) as buf:
            launch(
                stream,
                configs["colorize"],
                kernels["colorize"],
                np.uint64(tex_read.handle),
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
            window.set_caption(f"cuda.core CUDAArray/Texture/Surface - Lenia ({WIDTH}x{HEIGHT}, R={R}, {fps:.0f} FPS)")
            frame_count = 0
            fps_time = now

    @window.event
    def on_close():
        # Release everything we opened, in reverse order. Each of these is a
        # context manager too, but pyglet owns the event loop here so we
        # release explicitly.
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
# distract from the Python logic above. The important things to know:
#
#   - KERNEL_SOURCE contains three CUDA C++ kernels:
#       * seed_blob       -- sets the initial state via SurfaceObject writes.
#                            Either clears the field (mode = 0) or paints a
#                            Gaussian blob centered in the field (mode = 1).
#       * convolve_lenia  -- reads previous state via TextureObject (with
#                            LINEAR + WRAP bilinear filtering), integrates a
#                            bell-shaped neighborhood K(r/R) to produce the
#                            potential U, applies the growth function G(U),
#                            and writes the next state via SurfaceObject.
#       * colorize_lenia  -- reads the new state via TextureObject and writes
#                            RGBA bytes into the OpenGL PBO using a simple
#                            teal-on-black gradient.
#
#   - VERTEX_SHADER_SOURCE / FRAGMENT_SHADER_SOURCE are GLSL. They draw a
#     texture onto a rectangle covering the entire window. Nothing interesting.
#
# ============================================================================

KERNEL_SOURCE = r"""
// All kernels run one thread per output pixel and bounds-check at the top.
// `surf2Dwrite` takes the x offset in BYTES; for a single-channel float
// surface that means `x * sizeof(float)` = `x * 4`.

extern "C"
__global__
void seed_blob(cudaSurfaceObject_t surf,
               int width, int height,
               int mode,
               unsigned int seed,
               float radius,
               float peak) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float value = 0.0f;
    if (mode == 1) {
        // Gaussian blob centered in the field with a small deterministic
        // jitter that breaks symmetry differently on each reseed.
        float cx = (float)(width  / 2);
        float cy = (float)(height / 2);
        float dx = (float)x - cx;
        float dy = (float)y - cy;
        float r2 = dx * dx + dy * dy;
        float inv = 1.0f / (radius * radius);
        value = peak * expf(-r2 * inv);

        unsigned int h = (unsigned int)x * 374761393u +
                         (unsigned int)y * 668265263u + seed * 2246822519u;
        h = (h ^ (h >> 13)) * 1274126177u;
        h = h ^ (h >> 16);
        float noise = (h & 0xffffu) / 65535.0f;  // in [0, 1]
        value += 0.02f * (noise - 0.5f);
        if (value < 0.0f) value = 0.0f;
        if (value > 1.0f) value = 1.0f;
    }

    // float is 4 bytes; surf2Dwrite takes the x offset in BYTES.
    surf2Dwrite(value, surf, x * (int)sizeof(float), y);
}

extern "C"
__global__
void convolve_lenia(cudaTextureObject_t tex,
                    cudaSurfaceObject_t surf,
                    int width, int height,
                    int R,
                    float mu_k, float sigma_k,
                    float mu, float sigma,
                    float dt,
                    float inv_weight_sum) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // Normalized texture coordinates: WRAP addressing requires them. The
    // (x + dx + 0.5) / W idiom places the sample at the texel center; values
    // outside [0, 1] are fine because WRAP wraps them toroidally.
    float inv_w = 1.0f / (float)width;
    float inv_h = 1.0f / (float)height;
    float inv_R = 1.0f / (float)R;
    float inv_two_sigma_k2 = 1.0f / (2.0f * sigma_k * sigma_k);
    float inv_two_sigma2   = 1.0f / (2.0f * sigma     * sigma);

    // Integrate the bell-shaped weight K(r/R) against the current state.
    float U = 0.0f;
    for (int dy = -R; dy <= R; ++dy) {
        for (int dx = -R; dx <= R; ++dx) {
            float fdx = (float)dx;
            float fdy = (float)dy;
            float r2 = fdx * fdx + fdy * fdy;
            float r  = sqrtf(r2);
            if (r > (float)R) continue;   // restrict to the disk
            float rn = r * inv_R - mu_k;
            float w  = expf(-(rn * rn) * inv_two_sigma_k2);

            float sx = ((float)x + fdx + 0.5f) * inv_w;
            float sy = ((float)y + fdy + 0.5f) * inv_h;
            float s  = tex2D<float>(tex, sx, sy);
            U += w * s;
        }
    }
    U *= inv_weight_sum;   // host-precomputed 1 / sum(K)

    // Read the current cell value (point sample at the texel center).
    float sx0 = ((float)x + 0.5f) * inv_w;
    float sy0 = ((float)y + 0.5f) * inv_h;
    float state = tex2D<float>(tex, sx0, sy0);

    // Growth function G(U) = 2 * exp(-(U - mu)^2 / (2 * sigma^2)) - 1,
    // mapping U near mu to +1 (grow) and U far from mu to -1 (shrink).
    float du = U - mu;
    float G  = 2.0f * expf(-(du * du) * inv_two_sigma2) - 1.0f;

    float new_state = state + dt * G;
    if (new_state < 0.0f) new_state = 0.0f;
    if (new_state > 1.0f) new_state = 1.0f;

    surf2Dwrite(new_state, surf, x * (int)sizeof(float), y);
}

extern "C"
__global__
void colorize_lenia(cudaTextureObject_t tex,
                    unsigned char* output,
                    int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float inv_w = 1.0f / (float)width;
    float inv_h = 1.0f / (float)height;
    float cx = ((float)x + 0.5f) * inv_w;
    float cy = ((float)y + 0.5f) * inv_h;

    float v = tex2D<float>(tex, cx, cy);
    if (v < 0.0f) v = 0.0f;
    if (v > 1.0f) v = 1.0f;

    // Linear interpolation from a deep teal at v = 0 to a bright teal at
    // v = 1. Two stops -- simple, easy to read, no LUT required.
    //   (0, 15, 30, 255)  ->  (50, 200, 180, 255)
    float r = (  0.0f + v * ( 50.0f -   0.0f));
    float g = ( 15.0f + v * (200.0f -  15.0f));
    float b = ( 30.0f + v * (180.0f -  30.0f));

    int idx = (y * width + x) * 4;
    output[idx + 0] = (unsigned char)r;
    output[idx + 1] = (unsigned char)g;
    output[idx + 2] = (unsigned char)b;
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
