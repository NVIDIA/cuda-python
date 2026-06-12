# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# ################################################################################
#
# This example demonstrates cuda.core's 3D CUDAArray + trilinear TextureObject by
# baking a procedural fractal-noise density volume once at startup and then
# ray-marching it every frame as participating media to render fluffy, sunlit,
# semi-transparent clouds. The SurfaceObject is used during the one-shot bake;
# the TextureObject (with LINEAR + WRAP + normalized coords) drives the per-frame
# volumetric ray march with Beer-Lambert absorption and self-shadowing. The
# whole pipeline stays on the GPU through GraphicsResource. Requires pyglet.
#
# ################################################################################

# What this example teaches
# =========================
# - How to allocate a 3D cuda.core.CUDAArray (cuArray3DCreate under the hood) and
#   bind it as both a SurfaceObject (for one-shot kernel writes via surf3Dwrite)
#   and a TextureObject (for hardware-accelerated trilinear tex3D sampling).
# - How to ray-march a baked scalar density volume as PARTICIPATING MEDIA: this
#   goes beyond gl_interop_sdf_volume.py (which renders a hard SDF surface). Here
#   the volume is fog: we accumulate color and transmittance front-to-back and
#   apply Beer-Lambert absorption, with a short secondary march toward the sun
#   for self-shadowing.
# - How to wire mouse + keyboard input into a pyglet/cuda.core interop loop.
#
# How it works
# ============
# A single-channel float (FLOAT32) 3D volume (96^3) is filled once at
# startup with fractal Brownian motion (fbm) built from a cheap integer-hash
# value noise:
#
#     fbm(p) = sum over octaves of amplitude * value_noise(p * frequency)
#     density = remap(fbm) with a coverage threshold
#
# The volume stores only the raw noise; the cloud SHAPING (coverage threshold +
# a vertical height falloff that fades density near the top and bottom of the
# box) is applied in the RENDER kernel, not baked. That lets us ANIMATE the
# clouds for free by scrolling the sample coordinate with a `time` uniform
# (cheaper than re-baking 96^3 every frame, which would stack a second 3D launch
# on top of the already heavy raymarch). WRAP addressing avoids clamping the
# scrolled coordinate at the box edge (the baked field is not perfectly
# tileable, so a faint density seam sweeps through slowly); the ray-vs-box bail
# is what keeps density zero outside the volume, so WRAP is safe here.
#
#   STARTUP (one-shot bake)
#   ~~~~~~~~~~~~~~~~~~~~~~~
#   1. Allocate 3D CUDAArray (96^3, FLOAT32 x1, is_surface_load_store=True).
#   2. Bind it as a SurfaceObject.
#   3. Launch `bake_density`: one thread per voxel writes fbm via surf3Dwrite.
#   4. Close the SurfaceObject; the CUDAArray stays alive.
#
#   EACH FRAME
#   ~~~~~~~~~~
#   1. resource.map() -> CUDA device pointer into the OpenGL PBO.
#   2. Launch `render_clouds` (one thread per pixel). It builds an orbit-camera
#      ray, intersects the [-1,1]^3 box, marches front-to-back sampling density
#      via tex3D<float> (LINEAR + WRAP + normalized coords), shades each sample
#      with a short sun-ward shadow march (Beer-Lambert), accumulates over an
#      analytic sky, and writes RGBA8 straight into the PBO.
#   3. Unmap, GPU-side copy PBO -> texture, draw fullscreen quad.
#
# Performance note
# ================
# This is the most compute-heavy example here: a primary march (up to ~96 steps)
# with a nested secondary shadow march (~6 steps) per sample is O(steps^2) work
# per pixel. To keep it interactive we use a modest 96^3 volume, cap the step
# counts, and EARLY-OUT once transmittance drops below ~0.01. Lower
# PRIMARY_STEPS / VOLUME_SIZE if your GPU struggles.
#
# Controls
# ========
#   Left mouse drag    orbit camera (dx -> yaw, dy -> pitch)
#   Arrow keys         orbit camera (keyboard alternative)
#   Mouse wheel        zoom (camera distance)
#   + / -              raise / lower the sun (changes light angle + sky glow)
#   [ / ]              decrease / increase cloud coverage (more / less cloud)
#   R                  reset camera + sun + coverage
#   Escape / close     quit
#
# The window title shows yaw, pitch, distance, sun height, coverage, and FPS.
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
# Configuration (feel free to change these)
# ---------------------------------------------------------------------------
WIDTH = 800
HEIGHT = 600
VOLUME_SIZE = 96  # 96^3 voxels; bake cost is one-shot. Lower if memory is tight.

# Camera defaults / clamps.
RESET_YAW = 0.6
RESET_PITCH = 0.25
RESET_DIST = 3.2
PITCH_MIN = -1.45  # stay inside (-pi/2, pi/2) so the up-vector stays sane.
PITCH_MAX = 1.45
DIST_MIN = 1.5
DIST_MAX = 9.0

# Lighting / shaping defaults and clamps.
RESET_SUN_HEIGHT = 0.55  # 0 = sun at horizon, 1 = sun overhead.
SUN_HEIGHT_MIN = 0.05
SUN_HEIGHT_MAX = 0.98
RESET_COVERAGE = 0.50  # higher = more cloud (lower density threshold).
COVERAGE_MIN = 0.20
COVERAGE_MAX = 0.85


# ============================= Helper functions =============================
#
# The functions below set up CUDA and OpenGL. If you're here to learn about
# 3D CUDAArray / TextureObject / SurfaceObject, skip ahead to main() -- the
# interesting part is there. These helpers exist so that main() reads like a
# short story instead of a wall of boilerplate.
# ============================================================================


def _check_compute_capability(dev):
    """3D arrays + bindless surface/texture objects require sm_30+."""
    cc = dev.compute_capability
    if cc.major < 3:
        print(
            f"This example requires compute capability >= 3.0, got sm_{cc.major}{cc.minor}.",
            file=sys.stderr,
        )
        sys.exit(1)


def setup_cuda():
    """Compile the two kernels and return (device, stream, kernels)."""
    dev = Device(0)
    dev.set_current()
    _check_compute_capability(dev)
    stream = dev.create_stream()

    # C++ is required so the templated tex3D<float> / surf3Dwrite<float>
    # overloads resolve. extern "C" on the kernel symbols keeps the function
    # names unmangled even when the rest of the TU is compiled as C++.
    program_options = ProgramOptions(std="c++17", arch=f"sm_{dev.arch}")
    prog = Program(KERNEL_SOURCE, code_type="c++", options=program_options)
    mod = prog.compile(
        "cubin",
        name_expressions=("bake_density", "render_clouds"),
    )
    kernels = {
        "bake": mod.get_kernel("bake_density"),
        "render": mod.get_kernel("render_clouds"),
    }
    return dev, stream, kernels


def make_volume_array():
    """Allocate the 3D density volume. Single-channel float, surface-capable.

    API MAP
    =======
    - 3D CUDAArray shape=(W,H,D): CUDAArray.from_descriptor allocates a 96^3
      single-channel array (cuArray3DCreate under the hood). This is the
      headline of the example: a true 3D, hardware-laid-out array sampled
      trilinearly from a kernel.
    - tex3D trilinear (FilterMode.LINEAR) + normalized coords: configured by
      make_volume_texture below; gives free hardware trilinear sampling, the
      thing that makes a smooth volumetric raymarch cheap.
    - surf3Dwrite typed store during the one-shot bake: bind the same CUDAArray
      as a SurfaceObject (is_surface_load_store=True) and write one density per
      voxel; the byte x-offset uses sizeof(float) because surf3Dwrite's x
      coordinate is in BYTES (y, z are in elements).
    """
    return CUDAArray.from_descriptor(
        shape=(VOLUME_SIZE, VOLUME_SIZE, VOLUME_SIZE),
        format=ArrayFormat.FLOAT32,
        num_channels=1,
        is_surface_load_store=True,
    )


def make_volume_texture(arr):
    """Bind `arr` as a TextureObject configured for LINEAR + WRAP + normalized.

    WRAP (not CLAMP) is the right choice here: the render kernel scrolls the
    sample coordinate by a time uniform to animate the clouds, and WRAP avoids
    clamping (smearing) the edge texels as the coordinate drifts past [0, 1].
    The baked field is not perfectly tileable, so a faint density seam sweeps
    through slowly as the scroll wraps -- a minor demo-grade artifact, not a
    crash. WRAP/MIRROR addressing modes require normalized coordinates. The
    ray-vs-box bail in the raymarch is what keeps density zero outside the
    [-1, 1]^3 volume, so wrapping the noise field never leaks cloud outside it.
    """
    res_desc = ResourceDescriptor.from_array(arr)
    tex_desc = TextureDescriptor(
        address_mode=AddressMode.WRAP,
        filter_mode=FilterMode.LINEAR,
        read_mode=ReadMode.ELEMENT_TYPE,
        normalized_coords=True,
    )
    return TextureObject.from_descriptor(resource=res_desc, texture_descriptor=tex_desc)


def bake_volume(stream, kernels, arr):
    """Run the one-shot bake kernel that fills the volume with fractal noise.

    The SurfaceObject lives only for the duration of this call; once the bake
    is enqueued and the kernel has captured the bindless handle into its
    arguments, we sync the stream before letting the SurfaceObject close.
    The CUDAArray itself outlives this scope -- it's the long-lived backing
    store for the render-loop TextureObject.
    """
    with SurfaceObject.from_array(arr) as bake_surf:
        block = (8, 8, 8)
        grid = (
            (VOLUME_SIZE + block[0] - 1) // block[0],
            (VOLUME_SIZE + block[1] - 1) // block[1],
            (VOLUME_SIZE + block[2] - 1) // block[2],
        )
        launch(
            stream,
            LaunchConfig(grid=grid, block=block),
            kernels["bake"],
            np.uint64(bake_surf.handle),
            np.int32(VOLUME_SIZE),
        )
        # Synchronize before the SurfaceObject context exits so the bindless
        # handle is still valid while the kernel runs.
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
        caption="cuda.core 3D CUDAArray - Volumetric Cloud Ray-Marcher",
        vsync=False,
    )
    return window, _gl, pyglet


def create_display_resources(gl, width, height):
    """Standard GL boilerplate: shader, fullscreen quad, empty texture.

    Not CUDA-specific; identical to the other gl_interop_* examples.
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
    # --- Step 1: Set up CUDA (compile kernels, create stream) ---
    dev, stream, kernels = setup_cuda()

    # --- Step 2: Allocate the 3D density volume and bake it once ---
    #     The CUDAArray is the long-lived backing store; it must outlive the
    #     render loop. The SurfaceObject is only needed for the one-shot bake
    #     and is closed before we ever bind a TextureObject to the same CUDAArray.
    arr = make_volume_array()
    bake_volume(stream, kernels, arr)

    # --- Step 3: Bind the volume as a trilinear TextureObject ---
    #     LINEAR + WRAP + normalized_coords gives free hardware trilinear
    #     filtering plus seamless wrapping for the animated coordinate scroll.
    volume_tex = make_volume_texture(arr)

    # --- Step 4: Open a window and set up the CUDA/GL bridge ---
    window, gl, pyglet = create_window()
    shader_prog, quad_vao, tex_id = create_display_resources(gl, WIDTH, HEIGHT)
    pbo_id, _ = create_pixel_buffer(gl, WIDTH, HEIGHT)
    resource = GraphicsResource.from_gl_buffer(pbo_id, flags="write_discard")

    # --- Step 5: Render loop state ---
    # Camera is orbit-style: yaw and pitch are angles, dist is the orbit
    # radius. sun_height drives the light direction + sky glow; coverage shapes
    # how much of the noise field reads as cloud. The render kernel turns these
    # into rays + shading itself.
    state = {
        "yaw": RESET_YAW,
        "pitch": RESET_PITCH,
        "dist": RESET_DIST,
        "sun_height": RESET_SUN_HEIGHT,
        "coverage": RESET_COVERAGE,
    }
    start_time = time.monotonic()
    frame_count = [0]
    fps_time = [start_time]
    last_fps = [0.0]

    block = (16, 16, 1)
    grid = (
        (WIDTH + block[0] - 1) // block[0],
        (HEIGHT + block[1] - 1) // block[1],
        1,
    )
    config = LaunchConfig(grid=grid, block=block)

    @window.event
    def on_draw():
        window.clear()
        elapsed = time.monotonic() - start_time

        # (a) Map the PBO so CUDA can write into it.
        with resource.map(stream=stream) as buf:
            # (b) Launch the volumetric raymarch kernel. Camera + lighting +
            #     shaping params are passed as scalars; the kernel builds the
            #     orbit eye, per-pixel ray, and clouds itself. `time` scrolls
            #     the noise sample coordinate to animate the clouds.
            launch(
                stream,
                config,
                kernels["render"],
                buf.handle,
                np.int32(WIDTH),
                np.int32(HEIGHT),
                np.uint64(volume_tex.handle),
                np.float32(state["yaw"]),
                np.float32(state["pitch"]),
                np.float32(state["dist"]),
                np.float32(state["sun_height"]),
                np.float32(state["coverage"]),
                np.float32(elapsed),
            )
        # (c) Unmap happens automatically; cuGraphicsUnmapResources serializes
        #     the CUDA work against subsequent OpenGL use.

        copy_pbo_to_texture(gl, pbo_id, tex_id, WIDTH, HEIGHT)
        draw_fullscreen_quad(gl, shader_prog, quad_vao, tex_id)

        frame_count[0] += 1
        now = time.monotonic()
        if now - fps_time[0] >= 0.5:
            last_fps[0] = frame_count[0] / (now - fps_time[0])
            frame_count[0] = 0
            fps_time[0] = now
            window.set_caption(
                "cuda.core 3D CUDAArray - Volumetric Cloud Ray-Marcher  "
                f"yaw={state['yaw']:+.2f} pitch={state['pitch']:+.2f} "
                f"dist={state['dist']:.2f} sun={state['sun_height']:.2f} "
                f"cov={state['coverage']:.2f}  "
                f"{last_fps[0]:.0f} FPS  |  "
                "3D CUDAArray[FLOAT32,1ch] + tex3D[LINEAR|WRAP|norm] + surf3D bake"
            )

    @window.event
    def on_mouse_drag(_x, _y, dx, dy, buttons, _modifiers):
        # Left-click drag orbits the camera. dx -> yaw, dy -> pitch.
        if not (buttons & pyglet.window.mouse.LEFT):
            return
        orbit_scale = 0.005
        state["yaw"] += dx * orbit_scale
        state["pitch"] += dy * orbit_scale
        if state["pitch"] < PITCH_MIN:
            state["pitch"] = PITCH_MIN
        elif state["pitch"] > PITCH_MAX:
            state["pitch"] = PITCH_MAX

    @window.event
    def on_mouse_scroll(_x, _y, _scroll_x, scroll_y):
        # Scroll wheel zoom: geometric so each tick feels uniform. Positive
        # scroll_y (wheel up) zooms in.
        if scroll_y == 0:
            return
        state["dist"] *= 0.9**scroll_y
        if state["dist"] < DIST_MIN:
            state["dist"] = DIST_MIN
        elif state["dist"] > DIST_MAX:
            state["dist"] = DIST_MAX

    @window.event
    def on_key_press(symbol, _modifiers):
        key = pyglet.window.key
        keyboard_orbit = 0.08
        if symbol == key.ESCAPE:
            window.close()
        elif symbol == key.R:
            state["yaw"] = RESET_YAW
            state["pitch"] = RESET_PITCH
            state["dist"] = RESET_DIST
            state["sun_height"] = RESET_SUN_HEIGHT
            state["coverage"] = RESET_COVERAGE
        elif symbol == key.LEFT:
            state["yaw"] -= keyboard_orbit
        elif symbol == key.RIGHT:
            state["yaw"] += keyboard_orbit
        elif symbol == key.UP:
            state["pitch"] = min(PITCH_MAX, state["pitch"] + keyboard_orbit)
        elif symbol == key.DOWN:
            state["pitch"] = max(PITCH_MIN, state["pitch"] - keyboard_orbit)
        elif symbol in (key.PLUS, key.EQUAL, key.NUM_ADD):
            state["sun_height"] = min(SUN_HEIGHT_MAX, state["sun_height"] + 0.05)
        elif symbol in (key.MINUS, key.UNDERSCORE, key.NUM_SUBTRACT):
            state["sun_height"] = max(SUN_HEIGHT_MIN, state["sun_height"] - 0.05)
        elif symbol == key.BRACKETLEFT:
            state["coverage"] = max(COVERAGE_MIN, state["coverage"] - 0.03)
        elif symbol == key.BRACKETRIGHT:
            state["coverage"] = min(COVERAGE_MAX, state["coverage"] + 0.03)

    @window.event
    def on_close():
        # Release CUDA resources in reverse construction order. The GL objects
        # clean up via pyglet on window close.
        resource.close()
        volume_tex.close()
        arr.close()
        stream.close()

    pyglet.app.run(interval=0)


# ======================== GPU code (CUDA + GLSL) ============================
#
# Two CUDA C++ kernels are concatenated into one program string so they share
# a single NVRTC compile. NOTE: with no GPU available at authoring time, the
# noise/raymarch math below is unverified at runtime -- it is kept deliberately
# conservative (integer-hash value noise, plain fbm, no STL / host-only calls)
# so it compiles cleanly under NVRTC c++17.
#
#   bake_density   -- one thread per voxel. Evaluates fractal Brownian motion
#                     (fbm) of a cheap integer-hash value noise and writes the
#                     raw scalar via surf3Dwrite. NOTE: surf3Dwrite's
#                     x coordinate is in BYTES; a FLOAT32 element is 4 bytes, so
#                     multiply by sizeof(float). y and z are in elements
#                     -- a classic CUDA gotcha.
#
#   render_clouds  -- one thread per screen pixel. Builds the orbit-camera ray,
#                     intersects the [-1, 1]^3 box, marches front-to-back
#                     sampling density via tex3D<float> (LINEAR + WRAP +
#                     normalized coords, coordinate scrolled by `time`), applies
#                     a coverage threshold + vertical height falloff, does a
#                     short sun-ward shadow march per sample (Beer-Lambert),
#                     accumulates color + transmittance, composites over an
#                     analytic sky, and writes RGBA8 into the PBO.
#
# GLSL shaders at the very bottom just draw a textured quad. Nothing CUDA-
# specific there.
#
# ============================================================================

KERNEL_SOURCE = r"""
// --------------------------------------------------------------------------
// Small inline helpers.
// --------------------------------------------------------------------------
__device__ __forceinline__ float clampf(float v, float a, float b) {
    return fminf(fmaxf(v, a), b);
}

__device__ __forceinline__ float dot3(float ax, float ay, float az,
                                      float bx, float by, float bz) {
    return ax * bx + ay * by + az * bz;
}

__device__ __forceinline__ float length3(float x, float y, float z) {
    return sqrtf(x * x + y * y + z * z);
}

__device__ __forceinline__ float lerpf(float a, float b, float t) {
    return a + (b - a) * t;
}

__device__ __forceinline__ float smoothstepf(float t) {
    // Hermite fade curve used both for noise interpolation and shaping.
    return t * t * (3.0f - 2.0f * t);
}

// --------------------------------------------------------------------------
// Cheap integer-hash value noise + fractal Brownian motion (fbm).
//
// hash3() turns an integer lattice point into a pseudo-random float in [0,1].
// value_noise() trilinearly interpolates the 8 lattice corners around a
// floating-point position with a smoothstep fade. fbm() sums several octaves
// of value_noise at doubling frequency / halving amplitude. All integer math,
// no tables, no host-only calls -- NVRTC-friendly.
// --------------------------------------------------------------------------
__device__ __forceinline__ float hash3(int ix, int iy, int iz) {
    unsigned int h = (unsigned int)ix * 374761393u +
                     (unsigned int)iy * 668265263u +
                     (unsigned int)iz * 2147483647u;
    h = (h ^ (h >> 13)) * 1274126177u;
    h = h ^ (h >> 16);
    return (float)(h & 0x00ffffffu) / (float)0x01000000u;  // [0, 1)
}

__device__ __forceinline__ float value_noise(float x, float y, float z) {
    float fx = floorf(x), fy = floorf(y), fz = floorf(z);
    int ix = (int)fx, iy = (int)fy, iz = (int)fz;
    float tx = smoothstepf(x - fx);
    float ty = smoothstepf(y - fy);
    float tz = smoothstepf(z - fz);

    float c000 = hash3(ix,     iy,     iz);
    float c100 = hash3(ix + 1, iy,     iz);
    float c010 = hash3(ix,     iy + 1, iz);
    float c110 = hash3(ix + 1, iy + 1, iz);
    float c001 = hash3(ix,     iy,     iz + 1);
    float c101 = hash3(ix + 1, iy,     iz + 1);
    float c011 = hash3(ix,     iy + 1, iz + 1);
    float c111 = hash3(ix + 1, iy + 1, iz + 1);

    float x00 = lerpf(c000, c100, tx);
    float x10 = lerpf(c010, c110, tx);
    float x01 = lerpf(c001, c101, tx);
    float x11 = lerpf(c011, c111, tx);
    float y0  = lerpf(x00, x10, ty);
    float y1  = lerpf(x01, x11, ty);
    return lerpf(y0, y1, tz);
}

__device__ __forceinline__ float fbm(float x, float y, float z) {
    float sum = 0.0f;
    float amp = 0.5f;
    float freq = 1.0f;
    #pragma unroll
    for (int o = 0; o < 5; ++o) {
        sum += amp * value_noise(x * freq, y * freq, z * freq);
        freq *= 2.0f;
        amp  *= 0.5f;
    }
    return sum;  // roughly in [0, 1)
}

// --------------------------------------------------------------------------
// bake_density: one thread per voxel writes raw fbm into the volume via a
//               SurfaceObject. The cloud SHAPING (coverage threshold + height
//               falloff) is applied later in render_clouds so the threshold and
//               fade stay fixed while the render kernel scrolls the coordinate
//               for animation.
//
//   surf is bound to a (size^3, FLOAT32 x 1) CUDAArray allocated with
//   is_surface_load_store=True.
//   surf3Dwrite's x coordinate is in BYTES; a FLOAT32 element is 4 bytes, so
//   multiply x by sizeof(float). y and z are in elements -- a classic CUDA
//   gotcha.
// --------------------------------------------------------------------------
extern "C" __global__
void bake_density(cudaSurfaceObject_t surf, int size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= size || y >= size || z >= size) return;

    // Voxel-center position mapped into a few noise cells so fbm has structure
    // across the volume. ~4 base cells across the volume gives puffy blobs.
    const float NOISE_SCALE = 4.0f;
    float fx = ((float)x + 0.5f) / (float)size;
    float fy = ((float)y + 0.5f) / (float)size;
    float fz = ((float)z + 0.5f) / (float)size;

    float n = fbm(fx * NOISE_SCALE, fy * NOISE_SCALE, fz * NOISE_SCALE);

    // FLOAT32 store: surf3Dwrite's x offset is in BYTES (x * sizeof(float)).
    surf3Dwrite(n, surf, x * (int)sizeof(float), y, z);
}

// --------------------------------------------------------------------------
// Density sampler: tex3D wants normalized coords in [0, 1]; the volume covers
// [-1, 1] in world space, so we remap with (p + 1) * 0.5 and add a time-based
// scroll (WRAP addressing wraps it without edge clamping). The raw fbm is then shaped into
// a cloud density with:
//   - a coverage threshold (higher `coverage` -> lower threshold -> more cloud)
//   - a vertical height falloff that fades density near the top and bottom of
//     the box so clouds float in a slab rather than filling the whole cube.
// Returns density >= 0 (0 = clear air).
// --------------------------------------------------------------------------
__device__ __forceinline__ float sample_density(cudaTextureObject_t tex,
                                                 float px, float py, float pz,
                                                 float coverage, float t) {
    // Slow horizontal drift + gentle vertical bob for evolving clouds.
    float u = (px + 1.0f) * 0.5f + t * 0.015f;
    float v = (py + 1.0f) * 0.5f + t * 0.004f;
    float w = (pz + 1.0f) * 0.5f + t * 0.010f;
    float n = tex3D<float>(tex, u, v, w);

    // Coverage threshold: subtract a threshold and rescale so values below it
    // become clear air. coverage in [0,1] maps to threshold in [~0.8, ~0.15].
    float threshold = lerpf(0.80f, 0.15f, coverage);
    float d = (n - threshold) / fmaxf(1.0f - threshold, 1e-3f);
    d = clampf(d, 0.0f, 1.0f);

    // Vertical height falloff: py in [-1, 1]. Fade to zero near the top/bottom
    // so clouds form a horizontal band. Peak density around py ~ -0.1.
    float h = clampf((py + 1.0f) * 0.5f, 0.0f, 1.0f);   // [0,1] bottom->top
    float falloff = smoothstepf(clampf(h * 4.0f, 0.0f, 1.0f)) *
                    smoothstepf(clampf((1.0f - h) * 2.5f, 0.0f, 1.0f));

    return d * falloff;
}

// --------------------------------------------------------------------------
// render_clouds: one thread per screen pixel. Volumetric ray march of the
// density volume as participating media.
//
// Camera math (orbit, look-at origin, world-up (0, 1, 0)) matches the SDF
// example. Per pixel:
//   1. Build the ray, intersect the [-1, 1]^3 AABB (slab method).
//   2. March front-to-back from the entry point. At each step sample density;
//      if positive, do a SHORT secondary march toward the sun to estimate how
//      much light reaches this sample (Beer-Lambert: exp(-sum*absorption)).
//   3. Accumulate color and transmittance front-to-back. Early-out when
//      transmittance < 0.01 (rest of the ray is occluded -> big speedup).
//   4. Composite the accumulated cloud color over an analytic sky gradient
//      (horizon-to-zenith blue + a sun glow), tonemap, write RGBA8.
// --------------------------------------------------------------------------
extern "C" __global__
void render_clouds(unsigned char* output,
                   int width,
                   int height,
                   cudaTextureObject_t tex,
                   float yaw,
                   float pitch,
                   float dist,
                   float sun_height,
                   float coverage,
                   float t) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // ---- Build the orbit camera basis ----------------------------------
    float cp = cosf(pitch), sp = sinf(pitch);
    float cyw = cosf(yaw),  syw = sinf(yaw);

    float ex = dist * cp * cyw;
    float ey = dist * sp;
    float ez = dist * cp * syw;

    float fl = length3(ex, ey, ez);
    if (fl < 1e-6f) fl = 1e-6f;
    float fx = -ex / fl, fy = -ey / fl, fz = -ez / fl;

    // right = normalize(cross(fwd, world_up)), world_up = (0, 1, 0).
    float rx = -fz;
    float ry = 0.0f;
    float rz = fx;
    float rl = length3(rx, ry, rz);
    if (rl < 1e-6f) rl = 1e-6f;
    rx /= rl; ry /= rl; rz /= rl;

    // up' = cross(right, fwd).
    float ux = ry * fz - rz * fy;
    float uy = rz * fx - rx * fz;
    float uz = rx * fy - ry * fx;

    // ---- Per-pixel ray direction ---------------------------------------
    float u_ndc = 2.0f * ((float)x + 0.5f) / (float)width  - 1.0f;
    float v_ndc = 2.0f * ((float)y + 0.5f) / (float)height - 1.0f;

    const float TAN_HALF = 0.41421356237309515f;       // tanf(45deg / 2)
    float aspect = (float)width / (float)height;

    float dx = fx + u_ndc * aspect * TAN_HALF * rx + v_ndc * TAN_HALF * ux;
    float dy = fy + u_ndc * aspect * TAN_HALF * ry + v_ndc * TAN_HALF * uy;
    float dz = fz + u_ndc * aspect * TAN_HALF * rz + v_ndc * TAN_HALF * uz;
    float dl = length3(dx, dy, dz);
    if (dl < 1e-6f) dl = 1e-6f;
    dx /= dl; dy /= dl; dz /= dl;

    // ---- Sun direction from sun_height ---------------------------------
    // sun_height in [0,1]: 0 -> near horizon, 1 -> overhead. Keep a fixed
    // azimuth so the light feels stable while orbiting.
    float sun_el = sun_height * 1.4707963f;            // up to ~84 degrees
    float se = sinf(sun_el), ce = cosf(sun_el);
    const float SUN_AZ = 0.7853981633974483f;          // 45 deg azimuth
    float lx = ce * cosf(SUN_AZ);
    float ly = se;
    float lz = ce * sinf(SUN_AZ);
    float ll = length3(lx, ly, lz);
    if (ll < 1e-6f) ll = 1e-6f;
    lx /= ll; ly /= ll; lz /= ll;

    // ---- Ray vs. the [-1, 1]^3 box (slab method) -----------------------
    float inv_dx = 1.0f / (fabsf(dx) > 1e-8f ? dx : (dx >= 0 ? 1e-8f : -1e-8f));
    float inv_dy = 1.0f / (fabsf(dy) > 1e-8f ? dy : (dy >= 0 ? 1e-8f : -1e-8f));
    float inv_dz = 1.0f / (fabsf(dz) > 1e-8f ? dz : (dz >= 0 ? 1e-8f : -1e-8f));
    float t1x = (-1.0f - ex) * inv_dx, t2x = ( 1.0f - ex) * inv_dx;
    float t1y = (-1.0f - ey) * inv_dy, t2y = ( 1.0f - ey) * inv_dy;
    float t1z = (-1.0f - ez) * inv_dz, t2z = ( 1.0f - ez) * inv_dz;
    float tNear = fmaxf(fmaxf(fminf(t1x, t2x), fminf(t1y, t2y)), fminf(t1z, t2z));
    float tFar  = fminf(fminf(fmaxf(t1x, t2x), fmaxf(t1y, t2y)), fmaxf(t1z, t2z));

    // Accumulators: front-to-back compositing. transmittance starts at 1
    // (fully clear); accumulated radiance starts at 0.
    float trans = 1.0f;
    float acc_r = 0.0f, acc_g = 0.0f, acc_b = 0.0f;

    // Cloud material + lighting constants.
    const float ABSORPTION   = 6.0f;    // primary extinction per unit density
    const float SUN_ABSORP   = 8.0f;    // shadow-ray extinction per unit density
    const float STEP_LEN     = 2.0f / 96.0f;   // ~one voxel at 96^3
    const int   PRIMARY_STEPS = 96;
    const int   SHADOW_STEPS   = 6;
    const float SHADOW_STEP_LEN = 0.06f;

    // Henyey-Greenstein forward-scattering phase function. g>0 biases scatter
    // toward the light direction, producing the bright "silver lining" rim when
    // the view ray points toward the sun. cos(theta) = dot(view_dir, sun_dir);
    // both are unit length here. phase = (1-g^2) / (4pi * (1+g^2-2g*cos)^1.5).
    // The constant 1/(4pi) factor is folded into the lighting scale below, so
    // we only keep the angular shape that drives the glow.
    const float HG_G = 0.6f;
    float cos_vl = dot3(dx, dy, dz, lx, ly, lz);
    float hg_denom = 1.0f + HG_G * HG_G - 2.0f * HG_G * cos_vl;
    float hg_phase = (1.0f - HG_G * HG_G) / (hg_denom * sqrtf(fmaxf(hg_denom, 1e-4f)));

    if (tFar > fmaxf(tNear, 0.0f)) {
        float tcur = fmaxf(tNear, 0.0f) + 1e-4f;

        #pragma unroll 1
        for (int i = 0; i < PRIMARY_STEPS; ++i) {
            if (tcur > tFar) break;

            float pxw = ex + tcur * dx;
            float pyw = ey + tcur * dy;
            float pzw = ez + tcur * dz;

            float density = sample_density(tex, pxw, pyw, pzw, coverage, t);

            if (density > 1e-3f) {
                // ---- Secondary march toward the sun for self-shadowing ----
                float shadow_sum = 0.0f;
                #pragma unroll
                for (int s = 1; s <= SHADOW_STEPS; ++s) {
                    float st = (float)s * SHADOW_STEP_LEN;
                    float sxw = pxw + lx * st;
                    float syw = pyw + ly * st;
                    float szw = pzw + lz * st;
                    // Stop sampling outside the box (no density there anyway).
                    if (fabsf(sxw) > 1.0f || fabsf(syw) > 1.0f || fabsf(szw) > 1.0f) {
                        break;
                    }
                    shadow_sum += sample_density(tex, sxw, syw, szw, coverage, t);
                }
                float sun_trans = expf(-shadow_sum * SUN_ABSORP * SHADOW_STEP_LEN);

                // Powder ("dark edge") term: thin cloud edges scatter less light
                // back than a naive 1-exp model predicts, so darken low-density
                // samples for fluffier, more rounded volumes. Saturates toward 1
                // in dense cloud (cores stay bright); only thin edges are dimmed.
                // Apply as a gentle modulation so cores keep full sunlight.
                float powder = 0.4f + 0.6f * (1.0f - expf(-density * 3.0f));

                // Beer-Lambert extinction for this slab of the primary ray.
                float slab_trans = expf(-density * ABSORPTION * STEP_LEN);
                float absorbed = trans * (1.0f - slab_trans);

                // Direct sunlight reaching this sample, shaped by the HG phase so
                // it spikes when looking toward the sun (silver lining). Add a
                // small ambient floor so shadowed cores stay bluish, not black.
                float sun_light = sun_trans * (0.4f + 1.6f * hg_phase) * powder;
                float lit = clampf(0.15f + sun_light, 0.0f, 1.6f);
                float cr = lerpf(0.42f, 1.05f, clampf(lit, 0.0f, 1.0f)) + 0.05f * fmaxf(lit - 1.0f, 0.0f);
                float cg = lerpf(0.48f, 0.99f, clampf(lit, 0.0f, 1.0f)) + 0.04f * fmaxf(lit - 1.0f, 0.0f);
                float cb = lerpf(0.62f, 0.92f, clampf(lit, 0.0f, 1.0f));

                acc_r += absorbed * cr;
                acc_g += absorbed * cg;
                acc_b += absorbed * cb;
                trans *= slab_trans;

                if (trans < 0.01f) break;   // remaining ray fully occluded
            }

            tcur += STEP_LEN;
        }
    }

    // ---- Analytic sky behind / through the clouds ----------------------
    // Vertical gradient from a pale horizon to a deeper zenith blue, plus a
    // soft sun glow where the ray direction aligns with the sun.
    float up_amt = clampf(0.5f * (dy + 1.0f), 0.0f, 1.0f);
    float sky_r = lerpf(0.70f, 0.18f, up_amt);
    float sky_g = lerpf(0.80f, 0.34f, up_amt);
    float sky_b = lerpf(0.92f, 0.62f, up_amt);

    // Sun glow + a crisp sun disk. The broad glow uses a moderate power; the
    // disk is a high-power lobe that reads as a bright, slightly warm sun.
    float sun_dot = clampf(dot3(dx, dy, dz, lx, ly, lz), 0.0f, 1.0f);
    float glow = powf(sun_dot, 64.0f);
    float disk = powf(sun_dot, 2048.0f);
    sky_r += glow * 0.8f + disk * 6.0f;
    sky_g += glow * 0.7f + disk * 5.4f;
    sky_b += glow * 0.5f + disk * 3.6f;

    // Composite: accumulated cloud radiance over the sky weighted by the
    // remaining transmittance.
    float r = acc_r + trans * sky_r;
    float g = acc_g + trans * sky_g;
    float b = acc_b + trans * sky_b;

    // Simple Reinhard tonemap to keep the sun glow from blowing out.
    r = r / (1.0f + r);
    g = g / (1.0f + g);
    b = b / (1.0f + b);
    // Mild gamma for a punchier image.
    r = powf(clampf(r, 0.0f, 1.0f), 0.85f);
    g = powf(clampf(g, 0.0f, 1.0f), 0.85f);
    b = powf(clampf(b, 0.0f, 1.0f), 0.85f);

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
