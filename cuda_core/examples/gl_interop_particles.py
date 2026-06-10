# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# ################################################################################
#
# This example demonstrates cuda.core.GraphicsResource VBO interop together with
# CUDAArray, SurfaceObject, and TextureObject. Hundreds of thousands of points
# flow through an animated curl-noise velocity field. CUDA writes particle
# positions directly into an OpenGL Vertex Buffer Object (VBO), and OpenGL draws
# that same buffer as a glowing additive point cloud -- no PBO, no fullscreen
# quad, no pixel copy. Requires pyglet.
#
# ################################################################################

# What this example teaches
# =========================
# - How to register an OpenGL VBO (GL_ARRAY_BUFFER) with CUDA using
#   `GraphicsResource.from_gl_buffer(vbo_id, flags="none")` and treat the mapped
#   `buf.handle` as a device pointer to a particle array that CUDA both reads and
#   writes in place. This is the standout difference from every other interop
#   example here: those copy CUDA output into a PBO, upload it to a texture, and
#   draw a fullscreen quad. This one renders geometry straight out of the buffer
#   CUDA just wrote.
# - How to bake a smooth, periodic scalar potential into a 2D CUDAArray once (via
#   a SurfaceObject write kernel), then bind that array as a LINEAR + WRAP
#   normalized TextureObject and derive a divergence-free curl-noise velocity
#   field from finite differences of texture samples.
# - How to draw GL_POINTS directly from a CUDA-written VBO with additive blending
#   and shader-controlled point size for a luminous, flowing look.
#
# How it works
# ============
# We allocate one VBO holding N particles. Each particle is 4 floats:
#
#     [x, y, age, speed]   (stride = 16 bytes)
#
#   - x, y   : position in the [0, 1] x [0, 1] domain. The vertex shader maps
#              this to clip space with `pos * 2 - 1`. Keeping a single [0, 1]
#              domain means the kernel can sample the velocity texture with
#              normalized coordinates directly -- no scaling bugs.
#   - age    : seconds since this particle last (re)spawned. Drives color and
#              alpha; resets to 0 on respawn.
#   - speed  : normalized flow magnitude in [0, 1] at the particle's location
#              (the kernel maps gradient steepness through tanh). Drives the
#              color ramp so fast jets glow hotter than calm eddies.
#
# The GL vertex attributes read from the same buffer:
#   - "position" : 2 floats at offset 0
#   - "attribs"  : 2 floats (age, speed) at offset 8
#
# The CUDA kernel `advance_particles` indexes the buffer as `float4*` so its
# layout agrees with the host init array and the GL attribute pointers above.
#
#   VBO INTEROP (one buffer, CUDA writes -> OpenGL draws)
#   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   +-------------------+   map(stream)    +---------------------+
#   |   OpenGL VBO      | ---------------> |  advance_particles  |
#   | float4 per point  |                  |  (curl-noise flow)  |
#   | [x, y, age, speed]| <--------------- |  reads+writes pts   |
#   +-------------------+   unmap          +---------------------+
#           |
#           |  glDrawArrays(GL_POINTS)   (after unmap; GL cannot read a
#           v                             buffer while it is mapped to CUDA)
#       glowing point cloud on screen
#
# The velocity field is a curl of a baked scalar potential P(u, v):
#
#     velocity = ( dP/dv, -dP/du )
#
# Taking the curl of a scalar potential yields a divergence-free field, so
# particles swirl without piling up or thinning out. The potential is baked once
# into a single-channel float CUDAArray as a sum of periodic sinusoids, then
# sampled with LINEAR + WRAP + normalized coordinates. A time uniform scrolls the
# sample coordinates so the whole field slowly drifts and animates.
#
# Why flags="none" (not "write_discard")?
# ---------------------------------------
# The PBO examples register with "write_discard" because they overwrite every
# pixel each frame and never read the old contents. Here the kernel READS each
# particle's current position before writing the advanced one, so we must NOT
# tell CUDA the prior contents are garbage. We use "none".
#
# Single-channel surf2Dwrite byte offset
# --------------------------------------
# The potential array is single-channel `float` (4 bytes). `surf2Dwrite` takes
# the x coordinate in BYTES, so the offset is `x * sizeof(float)` = `x * 4`.
# (Contrast the float2 reaction-diffusion example, which uses `x * 8`.)
#
# What you should see
# ===================
# Luminous filaments of points swirling through an animated flow field, colored
# blue -> cyan -> white by speed and faded by age. Press R to respawn all
# particles, +/- to slow down / speed up the flow, and Escape to exit. The window
# title shows the particle count and FPS.
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
WIDTH = 900
HEIGHT = 900
N_PARTICLES = 1_000_000  # number of points in the cloud
FLOATS_PER_PARTICLE = 4  # [x, y, age, speed]
POTENTIAL_DIM = 256  # resolution of the baked potential texture (square)
DT = 1.0 / 60.0  # simulation time step per frame (seconds)
BASE_SPEED = 0.15  # base flow speed (domain units per second)
SPEED_STEP = 1.25  # multiplier applied by +/-
MAX_AGE = 4.0  # seconds before a particle respawns
POINT_SIZE = 2.4  # rendered point size in pixels


# ============================= Helper functions =============================
#
# The functions below set up CUDA and OpenGL. If you're here to learn about VBO
# interop, skip ahead to main() -- the interesting part is there. These helpers
# exist so that main() reads like a short story instead of a wall of
# boilerplate.
# ============================================================================


def setup_cuda():
    """Compile the CUDA kernels and return (device, stream, kernels, configs)."""
    dev = Device(0)
    dev.set_current()

    # SurfaceObject requires bindless surface objects (cuSurfObjectCreate),
    # which need compute capability >= 3.0.
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
        name_expressions=("bake_potential", "init_particles", "advance_particles"),
    )

    kernels = {
        "bake": mod.get_kernel("bake_potential"),
        "init": mod.get_kernel("init_particles"),
        "advance": mod.get_kernel("advance_particles"),
    }

    # The potential bake is 2D over POTENTIAL_DIM x POTENTIAL_DIM texels.
    block2d = (16, 16, 1)
    grid2d = (
        (POTENTIAL_DIM + block2d[0] - 1) // block2d[0],
        (POTENTIAL_DIM + block2d[1] - 1) // block2d[1],
        1,
    )
    # init/advance are 1D over N_PARTICLES.
    block1d = (256, 1, 1)
    grid1d = ((N_PARTICLES + block1d[0] - 1) // block1d[0], 1, 1)

    configs = {
        "bake": LaunchConfig(grid=grid2d, block=block2d),
        "init": LaunchConfig(grid=grid1d, block=block1d),
        "advance": LaunchConfig(grid=grid1d, block=block1d),
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
        caption="cuda.core VBO interop - Curl-Noise Particle Flow",
        vsync=False,
    )
    return window, _gl, pyglet


def create_particle_vbo(gl, shader_prog):
    """Create the particle VBO and its VAO, and wire up the vertex attributes.

    The VBO holds N_PARTICLES * 4 floats laid out as [x, y, age, speed] per
    particle. We initialize positions to a deterministic pseudo-random spread
    across the [0, 1] domain so there is something to see even before the first
    kernel launch; CUDA overwrites this every frame.

    Returns (vbo_gl_name, vao_gl_name).
    """
    # Host-side initial layout MUST match the kernel's float4 view and the GL
    # attribute pointers below: [x, y, age, speed] per particle.
    init = np.empty((N_PARTICLES, FLOATS_PER_PARTICLE), dtype=np.float32)
    rng = np.random.default_rng(12345)
    init[:, 0] = rng.random(N_PARTICLES, dtype=np.float32)  # x in [0, 1]
    init[:, 1] = rng.random(N_PARTICLES, dtype=np.float32)  # y in [0, 1]
    init[:, 2] = rng.random(N_PARTICLES, dtype=np.float32) * MAX_AGE  # staggered age
    init[:, 3] = 0.0  # speed
    init = np.ascontiguousarray(init)

    vao = ctypes.c_uint(0)
    gl.glGenVertexArrays(1, ctypes.byref(vao))
    gl.glBindVertexArray(vao.value)

    vbo = ctypes.c_uint(0)
    gl.glGenBuffers(1, ctypes.byref(vbo))
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo.value)
    gl.glBufferData(
        gl.GL_ARRAY_BUFFER,
        init.nbytes,
        init.ctypes.data_as(ctypes.c_void_p),
        gl.GL_DYNAMIC_DRAW,  # CUDA rewrites this buffer every frame
    )

    stride = FLOATS_PER_PARTICLE * 4  # 4 floats * 4 bytes = 16 bytes per particle

    pos_loc = gl.glGetAttribLocation(shader_prog.id, b"position")
    gl.glEnableVertexAttribArray(pos_loc)
    gl.glVertexAttribPointer(pos_loc, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(0))

    attr_loc = gl.glGetAttribLocation(shader_prog.id, b"attribs")
    gl.glEnableVertexAttribArray(attr_loc)
    gl.glVertexAttribPointer(attr_loc, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(8))

    gl.glBindVertexArray(0)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

    return vbo.value, vao.value


def create_shader(gl):
    """Build the point-cloud shader program (kept alive by the caller)."""
    from pyglet.graphics.shader import Shader, ShaderProgram

    vert = Shader(VERTEX_SHADER_SOURCE, "vertex")
    frag = Shader(FRAGMENT_SHADER_SOURCE, "fragment")
    shader_prog = ShaderProgram(vert, frag)

    # Additive blending so overlapping points accumulate into glow, and
    # shader-controlled point size (off by default in the core profile).
    gl.glEnable(gl.GL_BLEND)
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE)
    gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)
    gl.glDisable(gl.GL_DEPTH_TEST)

    return shader_prog


def make_potential_array():
    """Allocate the single-channel float CUDAArray that holds the baked potential.

    `is_surface_load_store=True` lets us write it once via a SurfaceObject and
    then read it as a TextureObject for smooth, wrapping, bilinear sampling.
    """
    return CUDAArray.from_descriptor(
        shape=(POTENTIAL_DIM, POTENTIAL_DIM),
        format=ArrayFormat.FLOAT32,
        num_channels=1,
        is_surface_load_store=True,
    )


def make_potential_texture(arr):
    """Bind `arr` as a TextureObject configured for LINEAR + WRAP + normalized."""
    res_desc = ResourceDescriptor.from_array(arr)
    tex_desc = TextureDescriptor(
        address_mode=AddressMode.WRAP,
        filter_mode=FilterMode.LINEAR,
        read_mode=ReadMode.ELEMENT_TYPE,
        # WRAP addressing only works with normalized coordinates.
        normalized_coords=True,
    )
    return TextureObject.from_descriptor(resource=res_desc, texture_descriptor=tex_desc)


def reset_particles(stream, kernels, configs, resource, seed):
    """Respawn every particle by launching init_particles on the mapped VBO.

    Reuses the same map() path the per-frame advance uses, so there is no host
    re-upload. The map brackets only the launch; GL must not touch the buffer
    while it is mapped.
    """
    with resource.map(stream=stream) as buf:
        launch(
            stream,
            configs["init"],
            kernels["init"],
            buf.handle,
            np.int32(N_PARTICLES),
            np.uint32(seed),
            np.float32(MAX_AGE),
        )


# ================================== main() ==================================


def main():
    # --- Step 1: Set up CUDA (compile kernels, create stream) ---
    dev, stream, kernels, configs = setup_cuda()

    # --- Step 2: Open a window ---
    window, gl, pyglet = create_window()

    # --- Step 3: Build the point-cloud shader and enable additive blending ---
    shader_prog = create_shader(gl)

    # --- Step 4: Create the particle VBO + VAO (the buffer CUDA writes into) ---
    vbo_id, vao_id = create_particle_vbo(gl, shader_prog)

    # =======================================================================
    # API MAP -- the four cuda.core interop objects this example hinges on
    # =======================================================================
    #   GraphicsResource.from_gl_buffer(VBO)
    #       Registers a GL VBO (NOT a PBO) so CUDA writes vertex positions,
    #       OpenGL then draws directly -- zero copy. The mapped buf.handle is a
    #       raw device pointer into the same float4 array OpenGL renders from.
    #   CUDAArray (single-channel float, is_surface_load_store=True)
    #       The backing storage for the baked scalar potential.
    #   SurfaceObject.from_array(pot_arr)
    #       Write view used ONCE at startup to bake the potential into the array.
    #   TextureObject (LINEAR + WRAP + normalized, 1ch)
    #       Read view: LINEAR+WRAP+normalized lets the kernel read the baked
    #       potential's gradient with smooth, tileable sampling -- the curl of
    #       that gradient is the divergence-free velocity field.
    # The texture handle is created once, kept alive, and wrapped in np.uint64
    # at launch; buf.handle is passed raw.
    # =======================================================================

    # --- Step 5: Register the VBO with CUDA ---
    #     flags="none": the kernel reads each particle before writing it back,
    #     so we must NOT discard the prior contents (that's why this is not
    #     "write_discard" like the PBO examples).
    resource = GraphicsResource.from_gl_buffer(vbo_id, flags="none")

    # --- Step 6: Allocate + bake the curl-noise potential, bind it as a texture ---
    pot_arr = make_potential_array()
    pot_surf = SurfaceObject.from_array(pot_arr)  # created once, kept alive
    pot_tex = make_potential_texture(pot_arr)  # created once, kept alive

    # Bake the scalar potential once via the SurfaceObject.
    launch(
        stream,
        configs["bake"],
        kernels["bake"],
        np.uint64(pot_surf.handle),
        np.int32(POTENTIAL_DIM),
        np.int32(POTENTIAL_DIM),
    )

    # --- Step 7: Seed the particles into the VBO ---
    state = {"seed": 1, "speed": BASE_SPEED, "t": 0.0}
    reset_particles(stream, kernels, configs, resource, state["seed"])

    # --- Step 8: Render loop ---
    start_time = time.monotonic()
    frame_count = 0
    fps_time = start_time

    @window.event
    def on_key_press(symbol, _modifiers):
        key = pyglet.window.key
        if symbol == key.ESCAPE:
            window.close()
            return
        if symbol == key.R:
            state["seed"] += 1
            reset_particles(stream, kernels, configs, resource, state["seed"])
            return
        if symbol in (key.PLUS, key.NUM_ADD, key.EQUAL):
            state["speed"] *= SPEED_STEP
            return
        if symbol in (key.MINUS, key.NUM_SUBTRACT):
            state["speed"] /= SPEED_STEP
            return

    @window.event
    def on_draw():
        nonlocal frame_count, fps_time

        # Black background so additive accumulation reads as glow.
        window.clear()

        state["t"] += DT

        # (a) Advance particles. The map brackets ONLY the CUDA launch -- OpenGL
        #     cannot read the buffer while it is mapped to CUDA.
        with resource.map(stream=stream) as buf:
            launch(
                stream,
                configs["advance"],
                kernels["advance"],
                buf.handle,  # raw device pointer to the float4 particle array
                np.uint64(pot_tex.handle),
                np.int32(N_PARTICLES),
                np.float32(DT),
                np.float32(state["speed"]),
                np.float32(state["t"]),
                np.float32(MAX_AGE),
                np.uint32(state["seed"]),
            )
        # Unmap happens automatically when the `with` block exits; only after
        # that may OpenGL draw from the buffer.

        # (b) Draw the particles straight from the VBO as GL_POINTS.
        gl.glUseProgram(shader_prog.id)
        max_age_loc = gl.glGetUniformLocation(shader_prog.id, b"max_age")
        gl.glUniform1f(max_age_loc, MAX_AGE)
        psize_loc = gl.glGetUniformLocation(shader_prog.id, b"point_size")
        gl.glUniform1f(psize_loc, POINT_SIZE)
        gl.glBindVertexArray(vao_id)
        gl.glDrawArrays(gl.GL_POINTS, 0, N_PARTICLES)
        gl.glBindVertexArray(0)
        gl.glUseProgram(0)

        # FPS counter (shown in window title)
        frame_count += 1
        now = time.monotonic()
        if now - fps_time >= 1.0:
            fps = frame_count / (now - fps_time)
            window.set_caption(
                "cuda.core VBO interop - Curl-Noise Particle Flow"
                f" ({N_PARTICLES:,} points, {fps:.0f} FPS,"
                f" speed x{state['speed'] / BASE_SPEED:.2f})"
                " | GraphicsResource(VBO) + TextureObject[LINEAR|WRAP|norm|1ch]"
            )
            frame_count = 0
            fps_time = now

    @window.event
    def on_close():
        # Release everything we opened, in reverse order.
        resource.close()
        pot_tex.close()
        pot_surf.close()
        pot_arr.close()
        stream.close()

    pyglet.app.run(interval=0)


# ======================== GPU code (CUDA + GLSL) ============================
#
# These source strings are kept at the bottom of the file so they don't distract
# from the Python logic above.
#
#   - KERNEL_SOURCE contains three CUDA C++ kernels:
#       * bake_potential    -- writes a smooth, periodic scalar potential into a
#                              single-channel float surface (once at startup).
#       * init_particles    -- (re)spawns every particle to a pseudo-random
#                              position with a staggered age. Operates on the
#                              mapped VBO as a float4 array.
#       * advance_particles -- reads each particle from the mapped VBO, samples
#                              the potential texture, computes a divergence-free
#                              curl velocity, integrates the position, handles
#                              wrap/respawn, and writes the particle back.
#
#   - VERTEX_SHADER_SOURCE / FRAGMENT_SHADER_SOURCE draw GL_POINTS from the VBO
#     with a soft round sprite colored by speed and faded by age.
#
# ============================================================================

KERNEL_SOURCE = r"""
// ---- shared helpers --------------------------------------------------------

// Cheap deterministic xorshift hash -> float in [0, 1).
__device__ __forceinline__ float hash01(unsigned int h) {
    h ^= h >> 16; h *= 0x7feb352du;
    h ^= h >> 15; h *= 0x846ca68bu;
    h ^= h >> 16;
    return (h & 0x00ffffffu) / (float)0x01000000;
}

__device__ __forceinline__ unsigned int seed_of(unsigned int idx, unsigned int salt) {
    return idx * 747796405u + salt * 2891336453u + 1u;
}

// ---- bake the scalar potential ---------------------------------------------
//
// A sum of periodic sinusoids over the unit square. Using full 2*pi*k periods
// makes the field seamless under WRAP addressing -- no visible edge.
extern "C"
__global__
void bake_potential(cudaSurfaceObject_t surf, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float u = (x + 0.5f) / (float)width;   // [0, 1)
    float v = (y + 0.5f) / (float)height;  // [0, 1)
    const float TWO_PI = 6.2831853f;

    float p = 0.0f;
    p += 1.00f * sinf(TWO_PI * (1.0f * u + 0.0f * v) + 0.3f);
    p += 0.70f * sinf(TWO_PI * (0.0f * u + 1.0f * v) + 1.7f);
    p += 0.55f * sinf(TWO_PI * (1.0f * u + 1.0f * v) + 2.1f);
    p += 0.45f * sinf(TWO_PI * (2.0f * u - 1.0f * v) + 0.9f);
    p += 0.30f * sinf(TWO_PI * (-1.0f * u + 2.0f * v) + 4.2f);
    p += 0.25f * sinf(TWO_PI * (3.0f * u + 2.0f * v) + 5.5f);

    // Single-channel float surface: x offset is in BYTES = x * sizeof(float).
    surf2Dwrite(p, surf, x * (int)sizeof(float), y);
}

// ---- (re)spawn particles ---------------------------------------------------
//
// The VBO is a flat array of float4 [x, y, age, speed] per particle.
extern "C"
__global__
void init_particles(float4* particles, int n,
                    unsigned int seed, float max_age) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    unsigned int s = seed_of((unsigned int)i, seed);
    float px = hash01(s + 11u);
    float py = hash01(s + 53u);
    // Stagger ages so respawns don't pulse in lockstep.
    float age = hash01(s + 97u) * max_age;
    particles[i] = make_float4(px, py, age, 0.0f);
}

// ---- advance particles through the curl-noise field ------------------------
extern "C"
__global__
void advance_particles(float4* particles,
                       cudaTextureObject_t pot,
                       int n, float dt, float speed,
                       float t, float max_age,
                       unsigned int seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float4 p = particles[i];
    float x = p.x;
    float y = p.y;
    float age = p.z;

    // Scroll the sample coordinates slowly with time so the field animates.
    float scroll = 0.03f * t;
    float su = x + scroll;
    float sv = y - 0.5f * scroll;

    // Curl of a scalar potential P is (dP/dv, -dP/du): divergence-free flow.
    // Estimate the gradient by central differences of texture samples. The
    // texture is LINEAR + WRAP + normalized, so wrapped reads are seamless.
    const float eps = 1.0f / 256.0f;
    float p_up = tex2D<float>(pot, su, sv + eps);
    float p_dn = tex2D<float>(pot, su, sv - eps);
    float p_rt = tex2D<float>(pot, su + eps, sv);
    float p_lt = tex2D<float>(pot, su - eps, sv);

    float dP_dv = (p_up - p_dn) / (2.0f * eps);
    float dP_du = (p_rt - p_lt) / (2.0f * eps);

    // Curl direction, then bound the magnitude. The raw analytic gradient of
    // the summed sinusoids runs ~0..20, which (times speed) would whip every
    // particle across the domain in well under a second and saturate the color
    // ramp. We split it: `dir` is the flow direction, and `flow` maps the
    // gradient steepness through tanh into [0, 1] so the field has slow eddies
    // and fast jets. The displacement is `speed * flow` domain-units/sec, so
    // `speed` is a true unit-per-second knob and `flow` drives the color ramp.
    float gx = dP_dv;
    float gy = -dP_du;
    float grad = sqrtf(gx * gx + gy * gy) + 1e-6f;
    float flow = tanhf(grad * 0.12f);  // 0 in calm regions, ->1 in steep jets
    float vx = speed * flow * (gx / grad);
    float vy = speed * flow * (gy / grad);

    // Store `flow` (the normalized speed in [0, 1]) as the color driver.
    float vmag = flow;

    // Integrate position.
    x += vx * dt;
    y += vy * dt;
    age += dt;

    // Respawn on age expiry or if a particle drifts out of the unit domain.
    bool respawn = (age >= max_age) || x < 0.0f || x > 1.0f || y < 0.0f || y > 1.0f;
    if (respawn) {
        // Jitter the seed by frame-ish state so respawns spread out over time.
        unsigned int s = seed_of((unsigned int)i, seed + (unsigned int)(t * 60.0f));
        x = hash01(s + 11u);
        y = hash01(s + 53u);
        age = 0.0f;
        vmag = 0.0f;
    }

    particles[i] = make_float4(x, y, age, vmag);
}
"""

# GLSL shaders -- draw GL_POINTS from the VBO. Position maps [0,1] -> clip space;
# color ramps blue -> cyan -> white by speed and fades with age. The fragment
# shader makes each point a soft round sprite for the glow.

VERTEX_SHADER_SOURCE = """#version 330 core
in vec2 position;   // x, y in [0, 1]
in vec2 attribs;    // age, speed
out float v_age;
out float v_speed;
uniform float max_age;
uniform float point_size;
void main() {
    gl_Position = vec4(position * 2.0 - 1.0, 0.0, 1.0);
    v_age = clamp(attribs.x / max_age, 0.0, 1.0);
    v_speed = attribs.y;
    // Subtle size-by-speed: fast jets render a touch larger so filaments read
    // as brighter, structured streaks. Reuses the existing speed attribute --
    // no struct change. Calm points keep the base size; never shrinks below it.
    gl_PointSize = point_size * (1.0 + 0.3 * clamp(v_speed, 0.0, 1.0));
}
"""

FRAGMENT_SHADER_SOURCE = """#version 330 core
in float v_age;
in float v_speed;
out vec4 fragColor;
void main() {
    // Soft round sprite: fade toward the edge of the point.
    vec2 d = gl_PointCoord - vec2(0.5);
    float r = length(d) * 2.0;
    float falloff = clamp(1.0 - r, 0.0, 1.0);
    falloff *= falloff;

    // Speed ramp: blue -> cyan -> white. v_speed is the normalized flow
    // magnitude in [0, 1] (see advance_particles), so it spans the ramp.
    float s = clamp(v_speed, 0.0, 1.0);
    vec3 cool = vec3(0.12, 0.40, 1.00);   // lifted enough that slow points still glow
    vec3 mid  = vec3(0.22, 0.85, 1.15);
    vec3 hot  = vec3(1.15, 1.15, 1.20);   // slightly >1 so only the densest cores clip
    vec3 color = (s < 0.5)
        ? mix(cool, mid, s * 2.0)
        : mix(mid, hot, (s - 0.5) * 2.0);

    // Fade in just after spawn and out near end of life.
    float life = (1.0 - v_age) * smoothstep(0.0, 0.08, v_age);
    float alpha = falloff * life * 0.7;   // density carries the glow; trim so cores don't fully clip

    fragColor = vec4(color, alpha);
}
"""


if __name__ == "__main__":
    main()
