# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# ################################################################################
#
# This example demonstrates cuda.core.Array, TextureObject, and SurfaceObject
# in combination with GraphicsResource for CUDA/OpenGL interop. A real-time
# Gerstner-wave ocean is rebuilt every frame: a heightmap Array is rewritten
# through a SurfaceObject, sampled through a TextureObject with LINEAR + WRAP
# filtering for normal estimation, and shaded with Phong + Fresnel sky
# reflection straight into an OpenGL PBO. Requires pyglet.
#
# ################################################################################

# What this example teaches
# =========================
# - How to use a CUDA Array as a typed heightmap that is simultaneously
#   written by one kernel (via SurfaceObject) and sampled by another (via
#   TextureObject) within the same frame.
# - How LINEAR filtering + WRAP addressing + normalized coordinates gives
#   essentially-free bilinear neighbor lookups for finite-difference normal
#   estimation on a tiling heightmap.
# - How to compose Array/TextureObject/SurfaceObject with GraphicsResource so
#   the entire render path never leaves the GPU.
#
# How it works
# ============
# Gerstner waves are a sum of N moving sinusoids with directional vectors --
# a classic ocean approximation that looks shockingly close to FFT ocean at a
# glance without any external library dependencies. For each heightmap texel:
#
#     h(x, z, t) = sum_i  A_i * sin( D_i . (x, z) * k_i  -  w_i * t  +  phi_i )
#
# where k_i = 2*pi / wavelength_i and w_i = sqrt(g * k_i) is the dispersion
# relation for deep-water gravity waves. We bake 12 waves with hand-picked
# directions / wavelengths / amplitudes / phases into the kernel as constant
# arrays. Weather presets just scale amplitude and speed at the host level.
#
#   PER FRAME (all on GPU)
#   ~~~~~~~~~~~~~~~~~~~~~~
#   +-----------------+   surf2Dwrite   +--------------+
#   |   update_height | --------------> |  heightmap   |
#   |     kernel      |                 |    Array     |
#   +-----------------+                 |  (FLOAT32)   |
#                                       +--------------+
#                                              |
#                                              | tex2D<float> (LINEAR + WRAP)
#                                              v
#                                       +-----------------+    write RGBA8
#                                       |  render_ocean   | ----------------> PBO
#                                       |     kernel      |
#                                       +-----------------+
#
# Why LINEAR + WRAP + normalized coords?
# --------------------------------------
# WRAP / MIRROR addressing modes require normalized coordinates (see the CUDA
# Programming Guide). The ocean naturally tiles, so WRAP gives free seamless
# horizon repetition. LINEAR filtering means our four-tap finite-difference
# normal estimate gets bilinear interpolation between texels for free, which
# smooths the lighting noticeably without a single extra ALU instruction.
#
# Channel byte width in surf2Dwrite
# ---------------------------------
# surf2Dwrite takes the x coordinate in BYTES, not in elements. For a
# single-channel float surface that means `x * sizeof(float)` = `x * 4`.
# Getting this wrong silently corrupts every other column.
#
# What you should see
# ===================
# A window showing a real-time animated ocean rendered with Phong shading and
# a Fresnel-modulated sky reflection. Drag with the left mouse button to
# orbit, scroll to zoom, press 1/2/3 to switch weather presets (calm /
# breezy / stormy), press P to pause animation, Escape to exit. Window title
# shows preset name and FPS.
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
    SurfaceObject,
    TextureDescriptor,
    TextureObject,
    launch,
)

# ---------------------------------------------------------------------------
# Window and heightmap dimensions (feel free to change these)
# ---------------------------------------------------------------------------
WIDTH = 1024
HEIGHT = 768
GRID = 512  # heightmap resolution (GRID x GRID texels)

# Weather presets: (amplitude_scale, speed_scale, label).
# These are applied as multiplicative scalars on top of the per-wave amplitude
# and angular-frequency arrays baked into the kernel, so a single compiled
# binary can render every preset.
PRESETS = {
    "1": (0.35, 0.7, "calm"),
    "2": (1.00, 1.0, "breezy"),
    "3": (1.85, 1.4, "stormy"),
}
DEFAULT_PRESET = "2"

# Initial camera (orbit-around-origin) parameters.
INITIAL_YAW = 0.6        # radians around world-y
INITIAL_PITCH = 0.35     # radians above the horizon (small positive = looking down)
INITIAL_DISTANCE = 5.0   # camera distance from origin
PITCH_LIMIT = 1.4        # clamp |pitch| to keep basis non-degenerate (< pi/2)
ZOOM_MIN = 1.5
ZOOM_MAX = 30.0


# ============================= Helper functions =============================
#
# The functions below set up CUDA and OpenGL. If you're here to learn about
# Array/TextureObject/SurfaceObject, skip ahead to main() -- the interesting
# part is there. These helpers exist so that main() reads like a short story
# instead of a wall of boilerplate.
# ============================================================================


def setup_cuda():
    """Compile the CUDA kernels and return (device, stream, kernels, configs).

    The two kernels live on different grids:
      - update_height runs over the heightmap (GRID x GRID texels).
      - render_ocean  runs over output pixels  (WIDTH x HEIGHT).
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

    # C++ compile so the templated tex2D<float> overload resolves.
    program_options = ProgramOptions(std="c++17", arch=f"sm_{dev.arch}")
    prog = Program(KERNEL_SOURCE, code_type="c++", options=program_options)
    mod = prog.compile(
        "cubin",
        name_expressions=("update_height", "render_ocean"),
    )

    kernels = {
        "update": mod.get_kernel("update_height"),
        "render": mod.get_kernel("render_ocean"),
    }

    block = (16, 16, 1)
    update_grid = (
        (GRID + block[0] - 1) // block[0],
        (GRID + block[1] - 1) // block[1],
        1,
    )
    render_grid = (
        (WIDTH + block[0] - 1) // block[0],
        (HEIGHT + block[1] - 1) // block[1],
        1,
    )
    configs = {
        "update": LaunchConfig(grid=update_grid, block=block),
        "render": LaunchConfig(grid=render_grid, block=block),
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
        caption="cuda.core Array/Texture/Surface - Gerstner Ocean",
        vsync=False,
    )
    return window, _gl, pyglet


def create_display_resources(gl, width, height):
    """Create the GL objects needed to show a texture on screen.

    Standard OpenGL boilerplate -- not CUDA-specific. Returns
    (shader_program, vao_id, tex_id). The shader_program is a pyglet
    ShaderProgram object (must be kept alive).
    """
    from pyglet.graphics.shader import Shader, ShaderProgram

    vert = Shader(VERTEX_SHADER_SOURCE, "vertex")
    frag = Shader(FRAGMENT_SHADER_SOURCE, "fragment")
    shader_prog = ShaderProgram(vert, frag)

    # Fullscreen quad (two triangles covering the entire window).
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

    tex = ctypes.c_uint(0)
    gl.glGenTextures(1, ctypes.byref(tex))
    gl.glBindTexture(gl.GL_TEXTURE_2D, tex.value)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
    gl.glTexImage2D(
        gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, width, height, 0,
        gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None,
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


def make_heightmap_array():
    """Allocate the single-channel float heightmap Array."""
    return Array.from_descriptor(
        shape=(GRID, GRID),
        format=ArrayFormat.FLOAT32,
        num_channels=1,
        surface_load_store=True,
    )


def make_height_texture(arr):
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


def orbit_camera_position(yaw, pitch, distance):
    """Convert (yaw, pitch, distance) to a world-space camera position.

    The camera orbits the origin looking at it. World up is +y. Pitch is the
    angle above the xz-plane: pitch=0 puts the camera on the horizon,
    pitch=+1.4 nearly directly overhead.
    """
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    cam_x = distance * cp * sy
    cam_y = distance * sp
    cam_z = distance * cp * cy
    return cam_x, cam_y, cam_z


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

    # --- Step 6: Allocate the heightmap Array and build its texture/surface ---
    #     We pre-create both the TextureObject (read path) and the
    #     SurfaceObject (write path) once and reuse them every frame. Creating
    #     them inside the per-frame loop would work but adds per-frame overhead
    #     and risks lifetime issues with async kernel launches.
    height_arr = make_heightmap_array()
    height_tex = make_height_texture(height_arr)
    height_surf = SurfaceObject.from_array(height_arr)

    # --- Step 7: Camera + animation state ---
    state = {
        "preset": DEFAULT_PRESET,
        "yaw": INITIAL_YAW,
        "pitch": INITIAL_PITCH,
        "distance": INITIAL_DISTANCE,
        "drag": False,
        "paused": False,
        "t_anim": 0.0,
        "t_prev": time.monotonic(),
    }

    # --- Step 8: Render loop ---
    frame_count = 0
    fps_time = state["t_prev"]

    @window.event
    def on_draw():
        nonlocal frame_count, fps_time

        window.clear()

        # Advance animation time only when not paused, so pausing freezes the
        # ocean exactly where it was rather than letting it lurch when resumed.
        now = time.monotonic()
        dt = now - state["t_prev"]
        state["t_prev"] = now
        if not state["paused"]:
            state["t_anim"] += dt
        t = state["t_anim"]

        amp_scale, speed_scale, _label = PRESETS[state["preset"]]

        # (a) Rebuild the heightmap for time t.
        launch(
            stream,
            configs["update"],
            kernels["update"],
            np.uint64(height_surf.handle),
            np.int32(GRID),
            np.int32(GRID),
            np.float32(t),
            np.float32(amp_scale),
            np.float32(speed_scale),
        )

        # (b) Render the scene: sample the heightmap through the texture,
        #     estimate normals via finite differences, shade with Phong +
        #     Fresnel sky reflection, write RGBA8 into the OpenGL PBO.
        cam_x, cam_y, cam_z = orbit_camera_position(
            state["yaw"], state["pitch"], state["distance"]
        )
        with resource.map(stream=stream) as buf:
            launch(
                stream,
                configs["render"],
                kernels["render"],
                np.uint64(height_tex.handle),
                buf.handle,
                np.int32(WIDTH),
                np.int32(HEIGHT),
                np.float32(cam_x),
                np.float32(cam_y),
                np.float32(cam_z),
                np.float32(t),
            )
        # Unmap happens automatically when the `with` block exits.

        # (c) PBO -> GL texture (GPU-to-GPU).
        copy_pbo_to_texture(gl, pbo_id, tex_id, WIDTH, HEIGHT)

        # (d) Draw the texture to the screen.
        draw_fullscreen_quad(gl, shader_prog, quad_vao, tex_id)

        # FPS counter (shown in window title)
        frame_count += 1
        if now - fps_time >= 1.0:
            fps = frame_count / (now - fps_time)
            label = PRESETS[state["preset"]][2]
            paused = " [paused]" if state["paused"] else ""
            window.set_caption(
                "cuda.core Array/Texture/Surface - Gerstner Ocean"
                f" [{label}]{paused} ({WIDTH}x{HEIGHT}, {fps:.0f} FPS)"
            )
            frame_count = 0
            fps_time = now

    # --- Mouse: drag to orbit, scroll to zoom ------------------------------
    @window.event
    def on_mouse_press(x, y, button, modifiers):
        if button == pyglet.window.mouse.LEFT:
            state["drag"] = True

    @window.event
    def on_mouse_release(x, y, button, modifiers):
        if button == pyglet.window.mouse.LEFT:
            state["drag"] = False

    @window.event
    def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
        if not (buttons & pyglet.window.mouse.LEFT):
            return
        # Rotate yaw on horizontal drag, pitch on vertical drag. The yaw
        # direction matches the camera moving with the cursor.
        state["yaw"] -= dx * 0.005
        state["pitch"] -= dy * 0.005
        # Clamp pitch to keep the camera basis non-degenerate (never look
        # straight down/up the world-y axis).
        if state["pitch"] > PITCH_LIMIT:
            state["pitch"] = PITCH_LIMIT
        if state["pitch"] < -PITCH_LIMIT:
            state["pitch"] = -PITCH_LIMIT

    @window.event
    def on_mouse_scroll(x, y, scroll_x, scroll_y):
        # Geometric zoom in camera distance; clamp to a sensible range.
        factor = 1.1 ** (-scroll_y)
        new_d = state["distance"] * factor
        state["distance"] = max(ZOOM_MIN, min(ZOOM_MAX, new_d))

    # --- Keyboard: 1/2/3 weather presets, P pauses, Escape exits ----------
    @window.event
    def on_key_press(symbol, modifiers):
        key = pyglet.window.key
        if symbol == key.ESCAPE:
            window.close()
            return
        if symbol == key.P:
            state["paused"] = not state["paused"]
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
    def on_close():
        # Release CUDA resources in reverse order of creation.
        resource.close()
        height_tex.close()
        height_surf.close()
        height_arr.close()
        stream.close()

    pyglet.app.run(interval=0)


# ======================== GPU code (CUDA + GLSL) ============================
#
# KERNEL_SOURCE contains two CUDA C++ kernels:
#   - update_height: per-heightmap-texel. Sums 12 Gerstner waves and writes
#                    one float per texel via SurfaceObject.
#   - render_ocean:  per-screen-pixel. Builds a camera ray, intersects the
#                    ocean plane (y=0), samples the heightmap via
#                    TextureObject (LINEAR + WRAP), estimates the normal via
#                    finite differences, and shades with Phong + Fresnel sky
#                    reflection. Misses go to a vertical sky gradient.
#
# VERTEX_SHADER_SOURCE / FRAGMENT_SHADER_SOURCE are plain GLSL that draws a
# texture on a fullscreen quad -- nothing CUDA-specific.
# ============================================================================

KERNEL_SOURCE = r"""
// ---------------------------------------------------------------------------
// Wave bank: 12 Gerstner-ish waves with hand-picked parameters.
//
// Wavelengths span 0.05 .. 1.0 world units. Amplitudes decrease with
// frequency so that long swells dominate and short ripples ride on top
// (a rough Phillips/JONSWAP-style envelope, but coarsely hand-tuned for
// visual punch rather than physical accuracy).
//
// Directions are spread non-uniformly around the unit circle to avoid the
// streaky-grid look you get from evenly-spaced directions.
// ---------------------------------------------------------------------------
__constant__ float c_dirx[12] = {
    1.000f,  0.866f,  0.500f,  0.000f, -0.500f, -0.866f,
   -1.000f, -0.940f, -0.500f,  0.174f,  0.643f,  0.940f
};
__constant__ float c_dirz[12] = {
    0.000f,  0.500f,  0.866f,  1.000f,  0.866f,  0.500f,
    0.000f,  0.342f,  0.866f,  0.985f,  0.766f,  0.342f
};
__constant__ float c_wavelen[12] = {
    1.000f, 0.730f, 0.520f, 0.380f, 0.260f, 0.190f,
    0.140f, 0.105f, 0.085f, 0.070f, 0.058f, 0.050f
};
__constant__ float c_amp[12] = {
    0.080f, 0.060f, 0.045f, 0.034f, 0.025f, 0.018f,
    0.013f, 0.010f, 0.0075f, 0.0055f, 0.0040f, 0.0030f
};
__constant__ float c_phase[12] = {
    0.00f, 1.20f, 2.10f, 0.40f, 3.70f, 5.10f,
    2.65f, 4.85f, 1.55f, 6.05f, 3.20f, 0.95f
};

// Deep-water dispersion: w = sqrt(g * k), with k = 2*pi / wavelength.
__device__ __forceinline__ float angular_freq(float wavelength) {
    const float G = 9.81f;
    float k = 6.2831853f / wavelength;
    return sqrtf(G * k);
}

// World extent (in world units) covered by one tile of the heightmap.
// The heightmap WRAPs, so the ocean tiles seamlessly every TILE world units.
__device__ __forceinline__ float tile_extent() { return 4.0f; }

// ---------------------------------------------------------------------------
// Tiny vec3 helpers. Kept inline + __forceinline__ so they stay free.
// ---------------------------------------------------------------------------
struct V3 { float x, y, z; };

__device__ __forceinline__ V3 v3(float x, float y, float z) {
    V3 r; r.x = x; r.y = y; r.z = z; return r;
}
__device__ __forceinline__ V3 v_add(V3 a, V3 b) {
    return v3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ __forceinline__ V3 v_sub(V3 a, V3 b) {
    return v3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ __forceinline__ V3 v_scale(V3 a, float s) {
    return v3(a.x * s, a.y * s, a.z * s);
}
__device__ __forceinline__ float v_dot(V3 a, V3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ __forceinline__ V3 v_cross(V3 a, V3 b) {
    return v3(a.y * b.z - a.z * b.y,
              a.z * b.x - a.x * b.z,
              a.x * b.y - a.y * b.x);
}
__device__ __forceinline__ V3 v_normalize(V3 a) {
    float inv = rsqrtf(fmaxf(v_dot(a, a), 1e-20f));
    return v_scale(a, inv);
}

// ---------------------------------------------------------------------------
// update_height: each thread computes one heightmap texel.
//
// Sums the 12 Gerstner waves at world position (x, z), using the
// amplitude_scale and speed_scale knobs to switch between weather presets
// without recompiling the kernel. Writes one float via surf2Dwrite.
// ---------------------------------------------------------------------------
extern "C" __global__
void update_height(cudaSurfaceObject_t surf,
                   int width, int height,
                   float t,
                   float amp_scale, float speed_scale) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height) return;

    // Map texel (ix, iy) to world position (x, z) inside one tile.
    float inv_w = 1.0f / (float)width;
    float inv_h = 1.0f / (float)height;
    float te = tile_extent();
    float wx = ((float)ix + 0.5f) * inv_w * te;
    float wz = ((float)iy + 0.5f) * inv_h * te;

    float h = 0.0f;
    #pragma unroll
    for (int i = 0; i < 12; ++i) {
        float k = 6.2831853f / c_wavelen[i];
        float w = angular_freq(c_wavelen[i]) * speed_scale;
        float arg = (c_dirx[i] * wx + c_dirz[i] * wz) * k - w * t + c_phase[i];
        h += c_amp[i] * sinf(arg);
    }
    h *= amp_scale;

    // Single-channel float surface: byte offset is x * sizeof(float).
    surf2Dwrite(h, surf, ix * (int)sizeof(float), iy);
}

// ---------------------------------------------------------------------------
// Sample the heightmap at a world position. Texture is normalized + WRAP,
// so we just divide world coords by tile_extent. WRAP gives us the tiling
// for free at the horizon.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float sample_height(cudaTextureObject_t tex,
                                               float wx, float wz) {
    float inv_te = 1.0f / tile_extent();
    return tex2D<float>(tex, wx * inv_te, wz * inv_te);
}

// ---------------------------------------------------------------------------
// Sky gradient: a vertical interpolation from a soft horizon to a deeper
// overhead blue. `up_angle` is in [-1, 1] (the y component of the ray dir).
// ---------------------------------------------------------------------------
__device__ __forceinline__ V3 sky_color(float up_angle) {
    // Clamp to [0, 1] so straight-down rays still get a horizon color.
    float a = fmaxf(0.0f, fminf(1.0f, up_angle));
    // Soft pale-blue horizon
    V3 horizon = v3(0.70f, 0.82f, 0.92f);
    // Deeper blue overhead
    V3 zenith  = v3(0.18f, 0.34f, 0.62f);
    // Curve so the gradient isn't linear -- horizon stays brighter longer.
    float t = powf(a, 0.6f);
    return v_add(v_scale(horizon, 1.0f - t), v_scale(zenith, t));
}

// ---------------------------------------------------------------------------
// render_ocean: each thread shades one screen pixel.
//
// 1. Reconstruct the camera basis from cam_pos (orbiting origin, world-up).
// 2. Build a perspective ray through the pixel.
// 3. Intersect ray with y = 0 plane; if it misses, return sky gradient.
// 4. Sample heightmap at hit point; finite-difference for the normal.
// 5. Phong diffuse + specular, blended with Fresnel sky reflection.
// 6. Write RGBA8 into the OpenGL PBO.
// ---------------------------------------------------------------------------
extern "C" __global__
void render_ocean(cudaTextureObject_t tex,
                  unsigned char* out,
                  int w, int h,
                  float cam_x, float cam_y, float cam_z,
                  float /*t*/) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= w || py >= h) return;

    // ---- Camera basis ----
    // Forward looks from cam_pos toward origin. World up is +y.
    // cam_y > 0 guarantees forward.y < 0 and the cross product with world-up
    // is well-defined (the pitch is clamped on the host side).
    V3 cam_pos = v3(cam_x, cam_y, cam_z);
    V3 forward = v_normalize(v_sub(v3(0.0f, 0.0f, 0.0f), cam_pos));
    V3 world_up = v3(0.0f, 1.0f, 0.0f);
    V3 right = v_normalize(v_cross(forward, world_up));
    V3 cam_up = v_cross(right, forward);

    // ---- Pixel ray (perspective) ----
    float aspect = (float)w / (float)h;
    float fov = 1.0472f;                 // 60 degrees vertical FoV
    float scale = tanf(fov * 0.5f);
    float ndc_x = (2.0f * ((float)px + 0.5f) / (float)w - 1.0f) * aspect * scale;
    float ndc_y = (1.0f - 2.0f * ((float)py + 0.5f) / (float)h) * scale;
    V3 dir = v_normalize(v_add(v_add(forward,
                                     v_scale(right, ndc_x)),
                               v_scale(cam_up, ndc_y)));

    // ---- Background sky if the ray misses the ocean plane ----
    // The ocean is the y=0 plane; we only count hits with rays going downward
    // (dir.y < 0). Anything else is sky. A small eps avoids near-horizontal
    // rays producing absurd hit distances.
    V3 col;
    const float HIT_EPS = 1e-3f;
    if (dir.y > -HIT_EPS) {
        col = sky_color(dir.y);
    } else {
        // ---- Hit the ocean plane ----
        float t_hit = -cam_y / dir.y;
        if (t_hit <= 0.0f) {
            // Camera under the surface -- treat as sky to avoid garbage.
            col = sky_color(dir.y);
        } else {
            V3 p = v_add(cam_pos, v_scale(dir, t_hit));

            // ---- Sample heightmap; estimate normal via finite differences ----
            // The heightmap tiles every tile_extent() world units (WRAP), so
            // we use a small world-space epsilon. Four taps -> central
            // differences in x and z.
            const float FD = 0.01f;
            float h_c = sample_height(tex, p.x,       p.z);
            float h_xp = sample_height(tex, p.x + FD, p.z);
            float h_xm = sample_height(tex, p.x - FD, p.z);
            float h_zp = sample_height(tex, p.x,      p.z + FD);
            float h_zm = sample_height(tex, p.x,      p.z - FD);
            float dh_dx = (h_xp - h_xm) / (2.0f * FD);
            float dh_dz = (h_zp - h_zm) / (2.0f * FD);
            // Normal of the surface y = h(x, z) is (-dh/dx, 1, -dh/dz).
            V3 N = v_normalize(v3(-dh_dx, 1.0f, -dh_dz));

            // ---- Lighting ----
            V3 L = v_normalize(v3(0.55f, 0.65f, 0.35f));   // sun: high+side
            V3 V = v_normalize(v_sub(cam_pos, p));         // view direction
            // Reflect L about N: R = 2*(N.L)*N - L
            float ndotl = fmaxf(0.0f, v_dot(N, L));
            V3 R = v_normalize(v_sub(v_scale(N, 2.0f * v_dot(N, L)), L));

            // Phong specular highlight on wave crests.
            float spec = powf(fmaxf(0.0f, v_dot(R, V)), 32.0f);

            // Diffuse: deep-sea blue-green.
            V3 deep = v3(0.04f, 0.18f, 0.28f);
            V3 shallow = v3(0.10f, 0.32f, 0.42f);
            // Tiny height-based shading bias so crests look slightly brighter.
            float tint = 0.5f + 0.5f * fmaxf(-1.0f, fminf(1.0f, h_c * 6.0f));
            V3 base = v_add(v_scale(deep, 1.0f - tint),
                            v_scale(shallow, tint));

            // Diffuse term + ambient.
            V3 diffuse = v_add(v_scale(base, 0.18f),
                               v_scale(base, 0.82f * ndotl));

            // Fresnel-modulated sky reflection. Sample the sky in the
            // reflected-view direction so reflections of overhead show
            // overhead colors, etc. View reflection: Rv = 2*(N.V)*N - V.
            float ndotv = fmaxf(0.0f, v_dot(N, V));
            V3 Rv = v_normalize(v_sub(v_scale(N, 2.0f * v_dot(N, V)), V));
            V3 reflected_sky = sky_color(fmaxf(0.0f, Rv.y));
            float F = powf(1.0f - ndotv, 5.0f);
            // Clamp Fresnel just in case of NaN-prone edge cases.
            if (F < 0.0f) F = 0.0f;
            if (F > 1.0f) F = 1.0f;

            // Blend: more reflection at grazing angles.
            V3 lit = v_add(v_scale(diffuse, 1.0f - F),
                           v_scale(reflected_sky, F));

            // Add specular highlight (sun color).
            V3 sun_col = v3(1.0f, 0.96f, 0.85f);
            col = v_add(lit, v_scale(sun_col, spec));
        }
    }

    // ---- Tonemap + write ----
    // Simple Reinhard-ish curve keeps highlights in [0, 1].
    col.x = col.x / (1.0f + col.x);
    col.y = col.y / (1.0f + col.y);
    col.z = col.z / (1.0f + col.z);

    int idx = (py * w + px) * 4;
    out[idx + 0] = (unsigned char)(fmaxf(0.0f, fminf(1.0f, col.x)) * 255.0f);
    out[idx + 1] = (unsigned char)(fmaxf(0.0f, fminf(1.0f, col.y)) * 255.0f);
    out[idx + 2] = (unsigned char)(fmaxf(0.0f, fminf(1.0f, col.z)) * 255.0f);
    out[idx + 3] = 255;
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
