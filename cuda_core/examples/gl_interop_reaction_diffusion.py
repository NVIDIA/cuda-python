# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# ################################################################################
#
# This example demonstrates cuda.core.CUDAArray, TextureObject, and SurfaceObject
# in combination with GraphicsResource for CUDA/OpenGL interop. A Gray-Scott
# reaction-diffusion simulation is ping-ponged between two CUDA arrays each
# frame: a TextureObject provides smooth (LINEAR + WRAP) sampled reads, and a
# SurfaceObject provides typed writes. The final state is colorized straight
# into an OpenGL PBO. Requires pyglet.
#
# ################################################################################

# What this example teaches
# =========================
# - How to allocate a CUDA CUDAArray with `surface_load_store=True` so the same
#   memory can be bound as both a TextureObject (for sampled reads) and a
#   SurfaceObject (for typed writes).
# - How to use FilterMode.LINEAR + AddressMode.WRAP + normalized coordinates
#   to get free hardware bilinear interpolation on a toroidal world.
# - How to compose CUDAArray/TextureObject/SurfaceObject with GraphicsResource so
#   the entire simulation never leaves the GPU.
#
# How it works
# ============
# Gray-Scott is a two-species (U, V) reaction-diffusion system. At each cell
# the rule is roughly:
#
#     du/dt = Du * laplacian(u) - u*v*v + F*(1 - u)
#     dv/dt = Dv * laplacian(v) + u*v*v - (F + k)*v
#
# Different choices of F and k yield strikingly different patterns: coral,
# mitosis, spots, and many more. We pack (U, V) into the two channels of a
# `float2` CUDAArray.
#
#   PING-PONG (two arrays, swap each step)
#   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   +--------------+   tex2D<float2>   +--------------+
#   |   arr_a      | ----------------> |              |
#   | (U, V) state |                   |  gray_scott  |
#   +--------------+                   |    kernel    |
#                                      |              |
#   +--------------+   surf2Dwrite     |              |
#   |   arr_b      | <---------------- |              |
#   | (U, V) state |                   +--------------+
#   +--------------+
#       (swap)
#
# Each frame we do N_STEPS iterations of the kernel above, then run a separate
# `colorize` kernel that samples V from the final state and writes RGBA bytes
# straight into the OpenGL PBO via GraphicsResource. No data ever travels
# across the PCIe bus during the frame.
#
# Why LINEAR + WRAP + normalized coords?
# --------------------------------------
# Addressing modes WRAP and MIRROR are only supported with normalized
# coordinates (see the CUDA Programming Guide and the SDK's
# simplePitchLinearTexture sample). We use WRAP so that neighbor lookups at
# the image edge automatically wrap around -- i.e. a torus. LINEAR filtering
# is essentially free on the hardware and gives smoother diffusion than POINT
# sampling would. We sample at the texel center `(x + 0.5) / W` so the
# neighbor offsets line up exactly on integer texel positions.
#
# Channel byte width in surf2Dwrite
# ---------------------------------
# `surf2Dwrite` takes the x coordinate in BYTES, not in elements. For a
# `float2` surface that means `x * sizeof(float2)` = `x * 8`. Getting this
# wrong silently corrupts every other column.
#
# What you should see
# ===================
# A window showing animated, organic-looking patterns growing and dividing
# (think coral, spots, or mitosing cells). Press 1/2/3 to switch presets,
# R to reseed, and Escape to exit. The window title shows the current FPS
# and active preset.
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
    SurfaceObject,
    TextureDescriptor,
    TextureObject,
    launch,
)

# ---------------------------------------------------------------------------
# Simulation parameters (feel free to change these)
# ---------------------------------------------------------------------------
WIDTH = 512
HEIGHT = 512
N_STEPS = 8  # Gray-Scott iterations per displayed frame
DU = 0.16  # diffusion rate for U
DV = 0.08  # diffusion rate for V
DT = 1.0  # time step (Gray-Scott is stable at 1.0 with these D's)

# Named presets: (F, k, label) tuples. F is the feed rate, k is the kill rate.
# These are classic Gray-Scott regimes documented all over the literature.
PRESETS = {
    "1": (0.0545, 0.062, "coral"),
    "2": (0.0367, 0.0649, "mitosis"),
    "3": (0.030, 0.062, "spots"),
}
DEFAULT_PRESET = "1"


# ============================= Helper functions =============================
#
# The functions below set up CUDA and OpenGL. If you're here to learn about
# CUDAArray/TextureObject/SurfaceObject, skip ahead to main() -- the interesting
# part is there. These helpers exist so that main() reads like a short story
# instead of a wall of boilerplate.
# ============================================================================


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

    # Compile as C++ so the templated tex2D<float2> overload resolves.
    program_options = ProgramOptions(std="c++17", arch=f"sm_{dev.arch}")
    prog = Program(KERNEL_SOURCE, code_type="c++", options=program_options)
    mod = prog.compile(
        "cubin",
        name_expressions=("gray_scott_step", "colorize", "seed_initial"),
    )

    kernels = {
        "step": mod.get_kernel("gray_scott_step"),
        "colorize": mod.get_kernel("colorize"),
        "seed": mod.get_kernel("seed_initial"),
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
        caption="cuda.core CUDAArray/Texture/Surface - Gray-Scott Reaction Diffusion",
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
    """Allocate the two `float2` ping-pong arrays that hold the (U, V) state."""
    arr_a = CUDAArray.from_descriptor(
        shape=(WIDTH, HEIGHT),
        format=ArrayFormat.FLOAT32,
        num_channels=2,
        surface_load_store=True,
    )
    arr_b = CUDAArray.from_descriptor(
        shape=(WIDTH, HEIGHT),
        format=ArrayFormat.FLOAT32,
        num_channels=2,
        surface_load_store=True,
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


def seed_state(stream, kernels, configs, write_surf, seed_value):
    """Re-initialize the array behind `write_surf` with the Gray-Scott starting state.

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
        np.uint32(seed_value),
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
    #     Both are `float2` (channel 0 = U, channel 1 = V) with
    #     surface_load_store=True so they can be bound as SurfaceObjects.
    arr_a, arr_b = make_state_arrays()

    # --- Step 7: Pre-create the four bindless handles ---
    #     Per advisor: doing this once is much cheaper than recreating them
    #     every step. We keep both texture and surface handles for each
    #     array; the simulation loop just picks which pair to use.
    tex_a = make_texture(arr_a)
    tex_b = make_texture(arr_b)
    surf_a = SurfaceObject.from_array(arr_a)
    surf_b = SurfaceObject.from_array(arr_b)

    # --- Step 8: Seed the initial state into arr_a (writes via surf_a) ---
    seed_state(stream, kernels, configs, surf_a, seed_value=0)
    # After seeding, `arr_a` is the "current" state.
    state = {"current": "a", "preset": DEFAULT_PRESET, "seed": 0}

    # --- Step 9: Render loop ---
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
            state["seed"] += 1
            seed_state(stream, kernels, configs, surf_a, seed_value=state["seed"])
            state["current"] = "a"
            return
        for digit_key, name in (
            (key._1, "1"),
            (key._2, "2"),
            (key._3, "3"),
        ):
            if symbol == digit_key:
                state["preset"] = name
                return

    @window.event
    def on_draw():
        nonlocal frame_count, fps_time

        window.clear()
        f, k, _label = PRESETS[state["preset"]]

        # (a) Run N_STEPS Gray-Scott iterations. Each step reads from one
        #     array via a TextureObject (LINEAR + WRAP gives wrapping +
        #     bilinear sampling) and writes to the other via a SurfaceObject.
        for _ in range(N_STEPS):
            tex_read, surf_write, next_current = current_read_write()
            launch(
                stream,
                configs["step"],
                kernels["step"],
                np.uint64(tex_read.handle),
                np.uint64(surf_write.handle),
                np.int32(WIDTH),
                np.int32(HEIGHT),
                np.float32(DU),
                np.float32(DV),
                np.float32(f),
                np.float32(k),
                np.float32(DT),
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
            label = PRESETS[state["preset"]][2]
            window.set_caption(
                "cuda.core CUDAArray/Texture/Surface - Gray-Scott"
                f" [{label}] ({WIDTH}x{HEIGHT}, {fps:.0f} FPS,"
                f" {N_STEPS} steps/frame)"
            )
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
#       * seed_initial   -- sets initial (U, V) state via SurfaceObject writes
#       * gray_scott_step -- reads previous state via TextureObject (with
#                            LINEAR + WRAP bilinear filtering) and writes the
#                            next state via SurfaceObject. Coordinates are
#                            normalized to [0, 1] because WRAP requires it.
#       * colorize       -- reads the V channel via TextureObject and writes
#                            RGBA bytes into the OpenGL PBO using a simple
#                            three-stop "magma-ish" gradient.
#
#   - VERTEX_SHADER_SOURCE / FRAGMENT_SHADER_SOURCE are GLSL. They draw a
#     texture onto a rectangle covering the entire window. Nothing interesting.
#
# ============================================================================

KERNEL_SOURCE = r"""
// Inverse texture dimensions are precomputed by the host and passed as
// floats so the kernel can convert integer pixel coordinates to normalized
// texture coordinates with a single multiply.

extern "C"
__global__
void seed_initial(cudaSurfaceObject_t surf,
                  int width, int height,
                  unsigned int seed) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // U = 1 everywhere; V = 1 inside a ~40x40 centered square plus a small
    // deterministic perturbation that breaks symmetry differently each reseed.
    float u = 1.0f;
    float v = 0.0f;

    int half_w = width / 2;
    int half_h = height / 2;
    if (x >= half_w - 20 && x < half_w + 20 &&
        y >= half_h - 20 && y < half_h + 20) {
        v = 1.0f;
        // Knock U down a bit inside the seed square so V can grow.
        u = 0.5f;
    }

    // Cheap deterministic pseudo-random noise (xorshift on packed coords).
    unsigned int h = (unsigned int)x * 374761393u +
                     (unsigned int)y * 668265263u + seed * 2246822519u;
    h = (h ^ (h >> 13)) * 1274126177u;
    h = h ^ (h >> 16);
    float noise = (h & 0xffffu) / 65535.0f;   // in [0, 1]
    v += 0.02f * (noise - 0.5f);              // small +/- jitter
    if (v < 0.0f) v = 0.0f;
    if (v > 1.0f) v = 1.0f;

    // float2 is 8 bytes; surf2Dwrite takes the x offset in BYTES.
    surf2Dwrite(make_float2(u, v), surf, x * (int)sizeof(float2), y);
}

extern "C"
__global__
void gray_scott_step(cudaTextureObject_t tex,
                     cudaSurfaceObject_t surf,
                     int width, int height,
                     float Du, float Dv,
                     float F, float k_kill,
                     float dt) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // Normalized coordinates: WRAP addressing only works in normalized mode.
    // Each texel center sits at ((i + 0.5) / W, (j + 0.5) / H).
    float inv_w = 1.0f / (float)width;
    float inv_h = 1.0f / (float)height;
    float cx = (x + 0.5f) * inv_w;
    float cy = (y + 0.5f) * inv_h;

    // 5-point Laplacian stencil. LINEAR filtering does nothing extra here
    // because the offsets land exactly on neighboring texel centers, but the
    // toroidal WRAP at the boundary is essential for a periodic world.
    float2 c = tex2D<float2>(tex, cx, cy);
    float2 l = tex2D<float2>(tex, cx - inv_w, cy);
    float2 r = tex2D<float2>(tex, cx + inv_w, cy);
    float2 u_n = tex2D<float2>(tex, cx, cy - inv_h);
    float2 d_n = tex2D<float2>(tex, cx, cy + inv_h);

    float lap_u = (l.x + r.x + u_n.x + d_n.x) - 4.0f * c.x;
    float lap_v = (l.y + r.y + u_n.y + d_n.y) - 4.0f * c.y;

    float u = c.x;
    float v = c.y;
    float uvv = u * v * v;

    float du = Du * lap_u - uvv + F * (1.0f - u);
    float dv = Dv * lap_v + uvv - (F + k_kill) * v;

    float new_u = u + dt * du;
    float new_v = v + dt * dv;

    // Clamp to keep things numerically sane after long runs.
    if (new_u < 0.0f) new_u = 0.0f;
    if (new_u > 1.0f) new_u = 1.0f;
    if (new_v < 0.0f) new_v = 0.0f;
    if (new_v > 1.0f) new_v = 1.0f;

    surf2Dwrite(make_float2(new_u, new_v), surf,
                x * (int)sizeof(float2), y);
}

extern "C"
__global__
void colorize(cudaTextureObject_t tex,
              unsigned char* output,
              int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float inv_w = 1.0f / (float)width;
    float inv_h = 1.0f / (float)height;
    float cx = (x + 0.5f) * inv_w;
    float cy = (y + 0.5f) * inv_h;

    float2 c = tex2D<float2>(tex, cx, cy);
    float v = c.y;
    if (v < 0.0f) v = 0.0f;
    if (v > 1.0f) v = 1.0f;

    // Three-stop "magma-ish" gradient: dark purple -> orange -> pale yellow.
    // Implemented as two linear interpolations stitched together at v = 0.5
    // so the result is reasonably perceptually smooth without a lookup table.
    float r, g, b;
    if (v < 0.5f) {
        float t = v * 2.0f;                  // [0, 1] over the low half
        r = 0.05f + t * (0.85f - 0.05f);
        g = 0.02f + t * (0.30f - 0.02f);
        b = 0.20f + t * (0.10f - 0.20f);
    } else {
        float t = (v - 0.5f) * 2.0f;         // [0, 1] over the high half
        r = 0.85f + t * (1.00f - 0.85f);
        g = 0.30f + t * (0.95f - 0.30f);
        b = 0.10f + t * (0.70f - 0.10f);
    }

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
