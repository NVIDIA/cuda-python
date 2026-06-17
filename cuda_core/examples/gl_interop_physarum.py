# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# ################################################################################
#
# This example demonstrates cuda.core.CUDAArray, TextureObject, and SurfaceObject
# together with a plain device Buffer and GraphicsResource for CUDA/OpenGL
# interop. A large population of "slime mold" (Physarum) agents crawls over a
# single-channel float trail map: each agent senses the trail ahead via a
# TextureObject (LINEAR + WRAP sampling), steers toward the strongest scent,
# steps forward, and deposits pheromone through a SurfaceObject. A separate
# diffuse/decay pass blurs and fades the trail (ping-ponged between two CUDA
# arrays), and a colorize pass writes a neon palette straight into an OpenGL
# PBO. The result is emergent, self-organizing vein/network patterns. Requires
# pyglet.
#
# ################################################################################

# What this example teaches
# =========================
# - How to combine a plain device Buffer (per-agent state) with CUDAArray-backed
#   TextureObject/SurfaceObject pairs in a single simulation, all on the GPU.
# - How to allocate a single-channel float CUDAArray with
#   `is_surface_load_store=True` so the same memory can be read as a
#   TextureObject (LINEAR + WRAP + normalized) and written as a SurfaceObject.
# - How to initialize a device Buffer from host data without a third-party array
#   library: stage through a host-accessible pinned Buffer, fill it via NumPy,
#   then `copy_from` into the device Buffer.
#
# How it works
# ============
# Physarum is an agent-based transport-network model. Every agent stores
# (x, y, heading) and, once per frame:
#
#   1. Samples the trail at three sensors (left / center / right of its heading,
#      a fixed sensor distance ahead) using tex2D<float> LINEAR sampling.
#   2. Rotates toward whichever sensor reads strongest (with a little random
#      jitter from a per-agent xorshift RNG seeded by index + frame).
#   3. Steps forward by a fixed speed and wraps around the toroidal edges.
#   4. Deposits a constant amount of pheromone into the trail via surf2Dwrite.
#      Concurrent agents may race on the same texel -- that is acceptable and
#      even characteristic of the model.
#
# Then two grid-parallel passes finish the frame:
#
#   diffuse_decay : box-blur the trail (tex2D LINEAR neighbor taps) and multiply
#                   by a decay factor < 1. Reads the current array, writes the
#                   other, then we swap (ping-pong).
#   colorize      : color the trail by local gradient direction (hue) modulated
#                   by intensity, with a ridge boost + bloom halo, into the PBO.
#
#   PING-PONG (two single-channel float arrays)
#   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   move_agents reads + deposits into the CURRENT array (tex + surf of same arr).
#   diffuse_decay reads CURRENT (tex) -> writes OTHER (surf) -> swap.
#   colorize reads the new CURRENT (tex) -> OpenGL PBO.
#
# Why LINEAR + WRAP + normalized coords?
# --------------------------------------
# Addressing modes WRAP and MIRROR are only supported with normalized
# coordinates. WRAP makes the world a torus so agents and diffusion seamlessly
# cross the edges. LINEAR filtering is essentially free on the hardware and
# gives the agents smooth sub-texel gradient sensing. We sample at texel centers
# `(x + 0.5) / W` so neighbor offsets land on integer texel positions.
#
# Channel byte width in surf2Dwrite
# ---------------------------------
# `surf2Dwrite` takes the x coordinate in BYTES, not elements. The trail is a
# single-channel `float` surface, so the x offset is `x * sizeof(float)` = `x*4`.
# (Contrast a `float2` surface, which would need `x*8`.) Getting this wrong
# silently corrupts every Nth column.
#
# Per-agent state lives in a plain device Buffer
# ----------------------------------------------
# Agents are stored as a flat float32 array of length 3*N laid out as
# [x0, y0, h0, x1, y1, h1, ...]. We allocate it once with `dev.allocate` and
# pass the Buffer object straight to `launch` (matching saxpy.py / memory_ops.py,
# which pass Buffer objects directly rather than a raw pointer int).
#
# What you should see
# ===================
# A window of glowing neon filaments that grow, branch, and reorganize into a
# living transport network. Press 1/2/3 to switch behavior presets (different
# sensor geometry and turn speed give different morphologies), R to reseed the
# agents and clear the trail, and Escape to exit. The title shows the preset,
# agent count, and FPS.
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
    LegacyPinnedMemoryResource,
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
# Simulation parameters (feel free to change these)
# ---------------------------------------------------------------------------
WIDTH = 1024
HEIGHT = 1024
N_AGENTS = 1 << 21  # ~2.1 million agents
DEPOSIT = 0.2  # pheromone added to the trail per agent per frame (small so the
#              additive deposit accumulates meaningfully instead of instantly
#              saturating the field to 1.0)

# Named presets: (sensor_angle_rad, sensor_distance_px, turn_speed_rad, move_speed_px, decay, label).
# Different sensor geometry / turn speeds yield strikingly different networks.
PRESETS = {
    "1": (0.40, 9.0, 0.40, 1.0, 0.92, "veins"),
    "2": (0.80, 16.0, 0.25, 1.0, 0.90, "webs"),
    "3": (1.20, 5.0, 0.65, 1.5, 0.95, "swarm"),
}
DEFAULT_PRESET = "1"


# ============================= Helper functions =============================
#
# The functions below set up CUDA and OpenGL. If you're here to learn about
# CUDAArray/TextureObject/SurfaceObject/Buffer, skip ahead to main() -- the
# interesting part is there. These helpers exist so that main() reads like a
# short story instead of a wall of boilerplate.
# ============================================================================


def setup_cuda():
    """Compile the CUDA kernels and return (device, stream, kernels, configs).

    Returns a dict of kernels keyed by name and matching LaunchConfigs. The
    move pass is 1D over agents; the diffuse/colorize passes are 2D over pixels.
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
        name_expressions=("move_agents", "diffuse_decay", "colorize"),
    )

    kernels = {
        "move": mod.get_kernel("move_agents"),
        "diffuse": mod.get_kernel("diffuse_decay"),
        "colorize": mod.get_kernel("colorize"),
    }

    # 1D launch over agents.
    move_block = (256, 1, 1)
    move_grid = ((N_AGENTS + move_block[0] - 1) // move_block[0], 1, 1)
    move_config = LaunchConfig(grid=move_grid, block=move_block)

    # 2D launch over pixels (shared by diffuse and colorize).
    px_block = (16, 16, 1)
    px_grid = (
        (WIDTH + px_block[0] - 1) // px_block[0],
        (HEIGHT + px_block[1] - 1) // px_block[1],
        1,
    )
    px_config = LaunchConfig(grid=px_grid, block=px_block)

    configs = {"move": move_config, "diffuse": px_config, "colorize": px_config}

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
        caption="cuda.core CUDAArray/Texture/Surface/Buffer - Physarum",
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


def make_trail_arrays():
    """Allocate the two single-channel float ping-pong arrays for the trail map."""
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


def fill_agent_host(host_view, seed):
    """Fill a host-side float32 view (length 3*N) with random agents.

    Layout is [x0, y0, h0, x1, y1, h1, ...]: position in [0, W)x[0, H) and
    heading in [0, 2*pi).
    """
    rng = np.random.default_rng(seed)
    agents = host_view.reshape(N_AGENTS, 3)
    agents[:, 0] = rng.uniform(0.0, WIDTH, size=N_AGENTS)
    agents[:, 1] = rng.uniform(0.0, HEIGHT, size=N_AGENTS)
    agents[:, 2] = rng.uniform(0.0, 2.0 * np.pi, size=N_AGENTS)


def reseed_agents(stream, device_agents, pinned_agents, host_view, seed):
    """Refill the host staging view and copy it into the device agent Buffer.

    Reuses the already-allocated device and pinned buffers -- no reallocation.
    """
    fill_agent_host(host_view, seed)
    device_agents.copy_from(pinned_agents, stream=stream)


def clear_trail(stream, arr_a, arr_b, zeros):
    """Zero both trail arrays. CUDAArray.copy_from accepts a buffer-protocol host
    object directly (unlike Buffer.copy_from), so a NumPy zero array works."""
    arr_a.copy_from(zeros, stream=stream)
    arr_b.copy_from(zeros, stream=stream)


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
    #     The PBO is GPU memory owned by OpenGL. CUDA writes into it, OpenGL
    #     reads from it.
    pbo_id, _ = create_pixel_buffer(gl, WIDTH, HEIGHT)

    # --- Step 5: Register the PBO with CUDA ---
    resource = GraphicsResource.from_gl_buffer(pbo_id, flags="write_discard")

    # --- Step 6: Allocate the two ping-pong trail Arrays ---
    #     Single-channel float with is_surface_load_store=True so they can be
    #     bound as SurfaceObjects.
    #
    #   API MAP -- the four cuda.core objects that drive this simulation:
    #     * device Buffer (dev.allocate) holds raw agent state alongside the
    #       array/texture/surface stack.
    #     * TextureObject LINEAR+WRAP+normalized -> smooth, toroidal SENSE of the
    #       pheromone field.
    #     * SurfaceObject -> typed DEPOSIT writes into the same CUDAArray sensed
    #       as a texture (is_surface_load_store=True).
    arr_a, arr_b = make_trail_arrays()

    # --- Step 7: Pre-create the four bindless handles (once, kept alive) ---
    tex_a = make_texture(arr_a)
    tex_b = make_texture(arr_b)
    surf_a = SurfaceObject.from_array(arr_a)
    surf_b = SurfaceObject.from_array(arr_b)

    # --- Step 8: Allocate per-agent state in a plain device Buffer ---
    #     Flat float32 [x, y, heading] * N. We stage host data through a
    #     host-accessible pinned Buffer, then copy it into the device Buffer.
    #     Both buffers are allocated once and reused on reseed.
    agent_floats = 3 * N_AGENTS
    agent_bytes = agent_floats * 4
    device_agents = dev.allocate(agent_bytes, stream=stream)
    pinned_mr = LegacyPinnedMemoryResource()
    pinned_agents = pinned_mr.allocate(agent_bytes)
    host_view = np.from_dlpack(pinned_agents).view(np.float32)

    # Host-side zero image reused to clear the trail arrays.
    zeros = np.zeros((WIDTH, HEIGHT), dtype=np.float32)

    # --- Step 9: Seed initial agents + clear the trail ---
    state = {"current": "a", "preset": DEFAULT_PRESET, "seed": 0, "frame": 0}
    reseed_agents(stream, device_agents, pinned_agents, host_view, seed=state["seed"])
    clear_trail(stream, arr_a, arr_b, zeros)
    stream.sync()  # ensure the seed copy finishes before the first launch reads it

    # --- Step 10: Render loop ---
    start_time = time.monotonic()
    frame_count = 0
    fps_time = start_time

    def current_tex_surf():
        """Return (tex, surf) for the CURRENT trail array (read + deposit)."""
        if state["current"] == "a":
            return tex_a, surf_a
        return tex_b, surf_b

    def diffuse_read_write():
        """Return (tex_read_current, surf_write_other, next_current)."""
        if state["current"] == "a":
            return tex_a, surf_b, "b"
        return tex_b, surf_a, "a"

    @window.event
    def on_key_press(symbol, _modifiers):
        key = pyglet.window.key
        if symbol == key.ESCAPE:
            window.close()
            return
        if symbol == key.R:
            state["seed"] += 1
            state["frame"] = 0
            reseed_agents(stream, device_agents, pinned_agents, host_view, seed=state["seed"])
            clear_trail(stream, arr_a, arr_b, zeros)
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
        sensor_angle, sensor_dist, turn_speed, move_speed, decay, _label = PRESETS[state["preset"]]

        # (a) Move + deposit: 1D over agents. Reads and deposits into the
        #     CURRENT array (tex + surf of the same array).
        tex_cur, surf_cur = current_tex_surf()
        launch(
            stream,
            configs["move"],
            kernels["move"],
            device_agents,
            np.int32(N_AGENTS),
            np.uint64(tex_cur.handle),
            np.uint64(surf_cur.handle),
            np.int32(WIDTH),
            np.int32(HEIGHT),
            np.float32(sensor_angle),
            np.float32(sensor_dist),
            np.float32(turn_speed),
            np.float32(move_speed),
            np.float32(DEPOSIT),
            np.uint32(state["frame"]),
        )

        # (b) Diffuse + decay: 2D over pixels. Reads CURRENT, writes OTHER, swap.
        tex_read, surf_write, next_current = diffuse_read_write()
        launch(
            stream,
            configs["diffuse"],
            kernels["diffuse"],
            np.uint64(tex_read.handle),
            np.uint64(surf_write.handle),
            np.int32(WIDTH),
            np.int32(HEIGHT),
            np.float32(decay),
        )
        state["current"] = next_current

        # (c) Colorize the latest trail into the OpenGL PBO.
        tex_show = tex_a if state["current"] == "a" else tex_b
        with resource.map(stream=stream) as buf:
            launch(
                stream,
                configs["colorize"],
                kernels["colorize"],
                np.uint64(tex_show.handle),
                buf.handle,
                np.int32(WIDTH),
                np.int32(HEIGHT),
            )
        # Unmap happens automatically when the `with` block exits.

        # (d) Tell OpenGL to copy the PBO contents into our texture.
        copy_pbo_to_texture(gl, pbo_id, tex_id, WIDTH, HEIGHT)

        # (e) Draw the texture to the screen.
        draw_fullscreen_quad(gl, shader_prog, quad_vao, tex_id)

        state["frame"] += 1

        # FPS counter (shown in window title)
        frame_count += 1
        now = time.monotonic()
        if now - fps_time >= 1.0:
            fps = frame_count / (now - fps_time)
            label = PRESETS[state["preset"]][5]
            window.set_caption(
                "cuda.core CUDAArray/Texture/Surface/Buffer - Physarum"
                f" [{label}] ({WIDTH}x{HEIGHT}, {N_AGENTS:,} agents, {fps:.0f} FPS)"
                " | Buffer(agents) + TextureObject[LINEAR|WRAP|norm] sense"
                " + SurfaceObject deposit"
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
        pinned_agents.close()
        device_agents.close(stream)
        stream.close()

    pyglet.app.run(interval=0)


# ======================== GPU code (CUDA + GLSL) ============================
#
# These source strings are kept at the bottom of the file so they don't
# distract from the Python logic above.
#
#   - KERNEL_SOURCE contains three CUDA C++ kernels:
#       * move_agents   -- 1 thread per agent: senses the trail at three points
#                          via tex2D<float> (LINEAR + WRAP), rotates toward the
#                          strongest, steps forward with toroidal wrap, and
#                          deposits pheromone via surf2Dwrite (x offset in BYTES).
#       * diffuse_decay -- box-blur the trail via tex2D LINEAR neighbor taps and
#                          multiply by a decay factor < 1; ping-pong write.
#       * colorize      -- color the trail by the local gradient DIRECTION (hue
#                          via HSV) modulated by intensity, with a ridge boost
#                          and a wider-tap bloom halo for glowing veins, into
#                          RGBA bytes in the PBO.
#
#   - VERTEX_SHADER_SOURCE / FRAGMENT_SHADER_SOURCE are GLSL. They draw the
#     texture onto a rectangle covering the entire window. Nothing interesting.
#
# ============================================================================

KERNEL_SOURCE = r"""
// Per-agent xorshift32 RNG: cheap, good enough for turn jitter. Seeded per
// agent and per frame so the sequence differs every step.
__device__ __forceinline__ unsigned int xorshift32(unsigned int s) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
}

extern "C"
__global__
void move_agents(float* agents,
                 int n_agents,
                 cudaTextureObject_t tex,
                 cudaSurfaceObject_t surf,
                 int width, int height,
                 float sensor_angle,
                 float sensor_dist,
                 float turn_speed,
                 float move_speed,
                 float deposit,
                 unsigned int frame) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_agents) return;

    int base = i * 3;
    float x = agents[base + 0];
    float y = agents[base + 1];
    float heading = agents[base + 2];

    float inv_w = 1.0f / (float)width;
    float inv_h = 1.0f / (float)height;

    // Sample the trail at center / left / right of the heading. Normalized
    // coords (+0.5 texel center) are required for WRAP addressing.
    float ca = heading;
    float la = heading - sensor_angle;
    float ra = heading + sensor_angle;

    float cx = x + cosf(ca) * sensor_dist;
    float cy = y + sinf(ca) * sensor_dist;
    float lx = x + cosf(la) * sensor_dist;
    float ly = y + sinf(la) * sensor_dist;
    float rx = x + cosf(ra) * sensor_dist;
    float ry = y + sinf(ra) * sensor_dist;

    float sc = tex2D<float>(tex, (cx + 0.5f) * inv_w, (cy + 0.5f) * inv_h);
    float sl = tex2D<float>(tex, (lx + 0.5f) * inv_w, (ly + 0.5f) * inv_h);
    float sr = tex2D<float>(tex, (rx + 0.5f) * inv_w, (ry + 0.5f) * inv_h);

    // Per-agent jitter in [0, 1).
    unsigned int rng = xorshift32(((unsigned int)i + 1u) * 2654435761u + frame * 40503u);
    float jitter = (rng & 0xffffffu) / (float)0x1000000;

    // Steer toward the strongest sensor; random turn when ahead is ambiguous.
    if (sc > sl && sc > sr) {
        // keep going straight
    } else if (sc < sl && sc < sr) {
        // both sides better than center: turn randomly left or right
        heading += (jitter < 0.5f ? -turn_speed : turn_speed);
    } else if (sl > sr) {
        heading -= turn_speed;
    } else if (sr > sl) {
        heading += turn_speed;
    } else {
        // tie: small random wiggle
        heading += (jitter - 0.5f) * turn_speed;
    }

    // Step forward and wrap around the toroidal world.
    x += cosf(heading) * move_speed;
    y += sinf(heading) * move_speed;

    float fw = (float)width;
    float fh = (float)height;
    if (x < 0.0f) x += fw;
    if (x >= fw) x -= fw;
    if (y < 0.0f) y += fh;
    if (y >= fh) y -= fh;

    agents[base + 0] = x;
    agents[base + 1] = y;
    agents[base + 2] = heading;

    // Deposit pheromone at the new integer cell. surf2Dwrite x offset is in
    // BYTES: single-channel float => x * sizeof(float). Concurrent agents may
    // race on the same texel; that is acceptable for Physarum.
    int ix = (int)x;
    int iy = (int)y;
    if (ix < 0) ix = 0; else if (ix >= width) ix = width - 1;
    if (iy < 0) iy = 0; else if (iy >= height) iy = height - 1;

    float prev = surf2Dread<float>(surf, ix * (int)sizeof(float), iy);
    float val = prev + deposit;
    if (val > 1.0f) val = 1.0f;
    surf2Dwrite(val, surf, ix * (int)sizeof(float), iy);
}

extern "C"
__global__
void diffuse_decay(cudaTextureObject_t tex,
                   cudaSurfaceObject_t surf,
                   int width, int height,
                   float decay) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float inv_w = 1.0f / (float)width;
    float inv_h = 1.0f / (float)height;
    float cx = (x + 0.5f) * inv_w;
    float cy = (y + 0.5f) * inv_h;

    // 3x3 box blur via LINEAR neighbor taps; WRAP gives toroidal edges.
    float sum = 0.0f;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            sum += tex2D<float>(tex, cx + dx * inv_w, cy + dy * inv_h);
        }
    }
    float blurred = sum * (1.0f / 9.0f);

    float out = blurred * decay;
    if (out < 0.0f) out = 0.0f;
    if (out > 1.0f) out = 1.0f;

    surf2Dwrite(out, surf, x * (int)sizeof(float), y);
}

// HSV -> RGB (all components in [0, 1]). Standard six-sector conversion; used
// by colorize to turn the local trail-gradient direction into a hue.
__device__ __forceinline__ void hsv2rgb(float h, float s, float v,
                                        float* r, float* g, float* b) {
    h -= floorf(h);          // wrap hue into [0, 1)
    float hp = h * 6.0f;
    int sector = (int)hp;
    float f = hp - (float)sector;
    float p = v * (1.0f - s);
    float q = v * (1.0f - s * f);
    float t = v * (1.0f - s * (1.0f - f));
    switch (sector % 6) {
        case 0:  *r = v; *g = t; *b = p; break;
        case 1:  *r = q; *g = v; *b = p; break;
        case 2:  *r = p; *g = v; *b = t; break;
        case 3:  *r = p; *g = q; *b = v; break;
        case 4:  *r = t; *g = p; *b = v; break;
        default: *r = v; *g = p; *b = q; break;
    }
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

    float v = tex2D<float>(tex, cx, cy);
    if (v < 0.0f) v = 0.0f;
    if (v > 1.0f) v = 1.0f;

    // Local trail gradient from LINEAR+WRAP neighbor taps (toroidal, no edge
    // special-casing). Its direction sets the HUE so the network is colored by
    // the orientation of the veins instead of a single intensity ramp.
    float l = tex2D<float>(tex, cx - inv_w, cy);
    float rgt = tex2D<float>(tex, cx + inv_w, cy);
    float dn = tex2D<float>(tex, cx, cy - inv_h);
    float up = tex2D<float>(tex, cx, cy + inv_h);
    float gx = rgt - l;
    float gy = up - dn;
    float hue = atan2f(gy, gx) * (0.1591549f) + 0.5f;  // atan2/(2*pi) + 0.5 -> [0,1)

    // Soft glow/bloom: a wider ring of taps lifts a luminous halo around the
    // veins so they read as glowing rather than flat. Still WRAP-sampled.
    float bloom = 0.0f;
    bloom += tex2D<float>(tex, cx - 2.0f * inv_w, cy);
    bloom += tex2D<float>(tex, cx + 2.0f * inv_w, cy);
    bloom += tex2D<float>(tex, cx, cy - 2.0f * inv_h);
    bloom += tex2D<float>(tex, cx, cy + 2.0f * inv_h);
    bloom += l + rgt + dn + up;
    bloom *= 0.125f;  // average of the 8 surrounding taps

    // Intensity stays the dominant brightness driver so the reticular structure
    // survives; gradient magnitude sharpens ridges into bright luminous veins.
    float grad_mag = sqrtf(gx * gx + gy * gy);
    float ridge = grad_mag * 6.0f;
    if (ridge > 1.0f) ridge = 1.0f;

    // Saturation eases toward white on the brightest ridges (neon -> white-hot).
    float sat = 1.0f - 0.45f * v;

    // Brightness: core intensity (gamma-lifted) + ridge boost + bloom halo.
    float val = sqrtf(v) + 0.55f * ridge + 0.45f * bloom;
    if (val > 1.0f) val = 1.0f;

    float r, g, b;
    hsv2rgb(hue, sat, val, &r, &g, &b);

    // Lift the floor toward a deep blue-violet so empty space is not pure black,
    // giving the glow something to bleed into.
    r += 0.02f;
    g += 0.0f;
    b += 0.06f;
    if (r > 1.0f) r = 1.0f;
    if (g > 1.0f) g = 1.0f;
    if (b > 1.0f) b = 1.0f;

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
