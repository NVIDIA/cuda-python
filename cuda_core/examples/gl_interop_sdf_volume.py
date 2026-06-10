# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# ################################################################################
#
# This example demonstrates cuda.core's 3D CUDAArray + trilinear TextureObject by
# baking a procedural Signed Distance Field (SDF) volume once at startup and
# then ray-marching it every frame to render an orbitable 3D scene. The
# SurfaceObject is used during the one-shot bake; the TextureObject (with
# LINEAR + CLAMP + normalized coords) drives the per-frame ray march. The
# whole pipeline stays on the GPU through GraphicsResource. Requires pyglet.
#
# ################################################################################

# What this example teaches
# =========================
# - How to allocate a 3D cuda.core.CUDAArray (cuArray3DCreate under the hood) and
#   bind it as both a SurfaceObject (for one-shot kernel writes) and a
#   TextureObject (for hardware-accelerated trilinear sampling).
# - How to ray-march a baked SDF volume from a CUDA kernel, sampling via
#   tex3D<float> and writing pixels straight into an OpenGL PBO.
# - How to wire mouse + keyboard input into a pyglet/cuda.core interop loop.
#
# How it works
# ============
# The signed distance field of a "gyroid intersected with a sphere" is baked
# once into a 128 x 128 x 128 single-channel float volume:
#
#     gyroid(p)   = sin(p.x*tau)cos(p.y*tau)
#                 + sin(p.y*tau)cos(p.z*tau)
#                 + sin(p.z*tau)cos(p.x*tau)
#     sdf_gyroid  = |gyroid(p)| - 0.20         # slab around the gyroid surface
#     sdf_sphere  = length(p) - 0.9            # bounding sphere
#     sdf(p)      = max(sdf_gyroid, sdf_sphere) # CSG intersection
#
# where p in [-1, 1]^3 is the voxel's world-space position.
#
# Each frame, the render kernel emits one ray per pixel from an orbiting
# camera, marches the volume in fixed voxel-sized steps (up to ~256), and on intersection
# computes a normal by central differences of tex3D, then applies a simple
# diffuse + ambient + specular shade. Misses fall back to a vertical sky
# gradient.
#
#   STARTUP (one-shot bake)
#   ~~~~~~~~~~~~~~~~~~~~~~~
#   1. Allocate 3D CUDAArray (128^3, FLOAT32 x1, is_surface_load_store=True).
#   2. Bind it as a SurfaceObject.
#   3. Launch `bake_sdf`: one thread per voxel writes the SDF via surf3Dwrite.
#   4. Close the SurfaceObject; the CUDAArray stays alive.
#
#   EACH FRAME
#   ~~~~~~~~~~
#   1. resource.map() -> CUDA device pointer into the OpenGL PBO.
#   2. Launch `render_sdf` (one thread per pixel). It samples the SDF via the
#      long-lived TextureObject (LINEAR + CLAMP + normalized coords) using
#      tex3D<float>. RGBA8 lands directly in the PBO.
#   3. Unmap, GPU-side copy PBO -> texture, draw fullscreen quad.
#
# Controls
# ========
#   Left mouse drag    orbit camera (dx -> yaw, dy -> pitch)
#   Mouse wheel        zoom (camera distance)
#   R                  reset camera (yaw=0, pitch=0.3, dist=2.5)
#   Escape / close     quit
#
# The window title shows yaw, pitch, distance, FPS, and ms/frame.
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
VOLUME_SIZE = 128  # 128^3 voxels; bake cost is one-shot.

# Camera defaults / clamps.
RESET_YAW = 0.0
RESET_PITCH = 0.3
RESET_DIST = 2.5
PITCH_MIN = -1.45  # stay inside (-pi/2, pi/2) so the up-vector stays sane.
PITCH_MAX = 1.45
DIST_MIN = 1.2
DIST_MAX = 8.0


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
        name_expressions=("bake_sdf", "render_sdf"),
    )
    kernels = {
        "bake": mod.get_kernel("bake_sdf"),
        "render": mod.get_kernel("render_sdf"),
    }
    return dev, stream, kernels


def make_volume_array():
    """Allocate the 3D SDF volume. Single-channel float, surface-capable."""
    return CUDAArray.from_descriptor(
        shape=(VOLUME_SIZE, VOLUME_SIZE, VOLUME_SIZE),
        format=ArrayFormat.FLOAT32,
        num_channels=1,
        is_surface_load_store=True,
    )


def make_volume_texture(arr):
    """Bind `arr` as a TextureObject configured for LINEAR + CLAMP + normalized.

    Normalized coords let the kernel sample as (u, v, w) in [0, 1]; CLAMP at
    the boundaries matches the rendering logic that bails out as soon as the
    march leaves the volume's [-1, 1]^3 box, so out-of-range sampling never
    pollutes a real hit.
    """
    res_desc = ResourceDescriptor.from_array(arr)
    tex_desc = TextureDescriptor(
        address_mode=AddressMode.CLAMP,
        filter_mode=FilterMode.LINEAR,
        read_mode=ReadMode.ELEMENT_TYPE,
        normalized_coords=True,
    )
    return TextureObject.from_descriptor(resource=res_desc, texture_descriptor=tex_desc)


def bake_volume(stream, kernels, arr):
    """Run the one-shot bake kernel that fills the volume with the SDF.

    The SurfaceObject lives only for the duration of this call; once the bake
    is enqueued and the kernel has captured the bindless handle into its
    arguments, we sync the stream before letting the SurfaceObject close.
    The CUDAArray itself outlives this scope -- it's the long-lived backing store
    for the render-loop TextureObject.
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
        caption="cuda.core 3D CUDAArray - SDF Volume Ray-Marcher",
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

    # --- Step 2: Allocate the 3D SDF volume and bake it once ---
    #     The CUDAArray is the long-lived backing store; it must outlive the
    #     render loop. The SurfaceObject is only needed for the one-shot bake
    #     and is closed before we ever bind a TextureObject to the same CUDAArray.
    arr = make_volume_array()
    bake_volume(stream, kernels, arr)

    # --- Step 3: Bind the volume as a trilinear TextureObject ---
    #     LINEAR + CLAMP + normalized_coords gives us free hardware trilinear
    #     filtering, which is exactly what we want for both the SDF samples
    #     in the ray march and the normal-finite-difference samples.
    volume_tex = make_volume_texture(arr)

    # --- Step 4: Open a window and set up the CUDA/GL bridge ---
    window, gl, pyglet = create_window()
    shader_prog, quad_vao, tex_id = create_display_resources(gl, WIDTH, HEIGHT)
    pbo_id, _ = create_pixel_buffer(gl, WIDTH, HEIGHT)
    resource = GraphicsResource.from_gl_buffer(pbo_id, flags="write_discard")

    # --- Step 5: Render loop state ---
    # Camera is orbit-style: yaw and pitch are angles, dist is the orbit
    # radius. The render kernel turns these into a (origin, basis) and
    # constructs per-pixel rays itself.
    cam = {
        "yaw": RESET_YAW,
        "pitch": RESET_PITCH,
        "dist": RESET_DIST,
    }
    frame_count = [0]
    fps_time = [time.monotonic()]
    last_fps = [0.0]
    last_frame_ms = [0.0]

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

        # (a) Map the PBO so CUDA can write into it.
        with resource.map(stream=stream) as buf:
            # (b) Launch the ray-march kernel. The camera params are passed
            #     as scalars; the kernel computes the orbit eye position and
            #     per-pixel ray direction itself.
            launch(
                stream,
                config,
                kernels["render"],
                buf.handle,
                np.int32(WIDTH),
                np.int32(HEIGHT),
                np.uint64(volume_tex.handle),
                np.float32(cam["yaw"]),
                np.float32(cam["pitch"]),
                np.float32(cam["dist"]),
            )
        # (c) Unmap happens automatically; cuGraphicsUnmapResources serializes
        #     the CUDA work against subsequent OpenGL use.

        copy_pbo_to_texture(gl, pbo_id, tex_id, WIDTH, HEIGHT)
        draw_fullscreen_quad(gl, shader_prog, quad_vao, tex_id)

        frame_count[0] += 1
        now = time.monotonic()
        if now - fps_time[0] >= 0.5:
            last_fps[0] = frame_count[0] / (now - fps_time[0])
            last_frame_ms[0] = 1000.0 / last_fps[0] if last_fps[0] > 0 else 0.0
            frame_count[0] = 0
            fps_time[0] = now
            window.set_caption(
                "cuda.core 3D CUDAArray - SDF Volume Ray-Marcher  "
                f"yaw={cam['yaw']:+.2f} pitch={cam['pitch']:+.2f} "
                f"dist={cam['dist']:.2f}  "
                f"{last_fps[0]:.0f} FPS  {last_frame_ms[0]:.2f} ms/frame"
            )

    @window.event
    def on_mouse_drag(_x, _y, dx, dy, buttons, _modifiers):
        # Left-click drag orbits the camera. dx -> yaw (sign convention chosen
        # so that dragging right rotates the scene right); dy -> pitch (drag
        # up tilts the camera up).
        if not (buttons & pyglet.window.mouse.LEFT):
            return
        orbit_scale = 0.005
        cam["yaw"] += dx * orbit_scale
        cam["pitch"] += dy * orbit_scale
        # Clamp pitch so the up-vector never flips (we use world-up (0,1,0)).
        if cam["pitch"] < PITCH_MIN:
            cam["pitch"] = PITCH_MIN
        elif cam["pitch"] > PITCH_MAX:
            cam["pitch"] = PITCH_MAX

    @window.event
    def on_mouse_scroll(_x, _y, _scroll_x, scroll_y):
        # Scroll wheel zoom: geometric so each tick feels uniform regardless
        # of current distance. Positive scroll_y (wheel up) zooms in.
        if scroll_y == 0:
            return
        cam["dist"] *= 0.9**scroll_y
        if cam["dist"] < DIST_MIN:
            cam["dist"] = DIST_MIN
        elif cam["dist"] > DIST_MAX:
            cam["dist"] = DIST_MAX

    @window.event
    def on_key_press(symbol, _modifiers):
        key = pyglet.window.key
        if symbol == key.ESCAPE:
            window.close()
        elif symbol == key.R:
            cam["yaw"] = RESET_YAW
            cam["pitch"] = RESET_PITCH
            cam["dist"] = RESET_DIST

    @window.event
    def on_close():
        # Release CUDA resources in reverse construction order. The GL
        # objects clean up via pyglet on window close.
        resource.close()
        volume_tex.close()
        arr.close()
        stream.close()

    pyglet.app.run(interval=0)


# ======================== GPU code (CUDA + GLSL) ============================
#
# Two CUDA C++ kernels are concatenated into one program string so they share
# a single NVRTC compile.
#
#   bake_sdf    -- one thread per voxel. Computes the SDF of an
#                  "abs(gyroid) - 0.20" surface intersected with a bounding
#                  sphere, then writes the scalar via surf3Dwrite. NOTE:
#                  surf3Dwrite's x coordinate is in BYTES, y and z in
#                  elements -- a classic CUDA gotcha.
#
#   render_sdf  -- one thread per screen pixel. Builds the orbit-camera ray,
#                  fixed-step-marches the volume via tex3D<float> on a trilinear-
#                  filtered, normalized-coord TextureObject, and shades the
#                  hit with diffuse + ambient + specular. Misses return a
#                  sky gradient. Writes RGBA8 directly into the OpenGL PBO.
#
# GLSL shaders at the very bottom just draw a textured quad. Nothing CUDA-
# specific there.
#
# ============================================================================

KERNEL_SOURCE = r"""
// --------------------------------------------------------------------------
// Small inline helpers. Keeping them __device__ __forceinline__ encourages
// the compiler to drop them inline and avoids any cross-TU linkage worries.
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

// --------------------------------------------------------------------------
// bake_sdf: one thread per voxel writes the SDF of a gyroid-intersect-sphere
//           into a single-channel float 3D CUDAArray via a SurfaceObject.
//
//   surf is bound to a (size^3, FLOAT32 x 1) CUDAArray allocated with
//   is_surface_load_store=True.
//   surf3Dwrite's x coordinate is in BYTES (multiply by sizeof(float));
//   y and z are in elements. Off-by-one on the byte conversion silently
//   corrupts every other column, so it's worth flagging explicitly.
// --------------------------------------------------------------------------
extern "C" __global__
void bake_sdf(cudaSurfaceObject_t surf, int size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= size || y >= size || z >= size) return;

    // Map the voxel index to world-space p in [-1, 1]^3 (texel centers).
    float fx = ((float)x + 0.5f) / (float)size;
    float fy = ((float)y + 0.5f) / (float)size;
    float fz = ((float)z + 0.5f) / (float)size;
    float px = fx * 2.0f - 1.0f;
    float py = fy * 2.0f - 1.0f;
    float pz = fz * 2.0f - 1.0f;

    // Gyroid frequency: 3 cycles across [-1, 1] gives a busy but not noisy
    // surface at 128^3 resolution. tau = 2 * pi * frequency.
    const float TAU = 6.2831853071795864f * 3.0f;

    float sx = sinf(px * TAU), cx = cosf(px * TAU);
    float sy = sinf(py * TAU), cy = cosf(py * TAU);
    float sz = sinf(pz * TAU), cz = cosf(pz * TAU);
    float gyroid     = sx * cy + sy * cz + sz * cx;
    // Slab thickness: the gyroid SDF is non-Lipschitz (its gradient scales
    // with TAU ~= 19), so the stored values along the surface are dense but
    // unreliable as a true distance metric. A wider slab (0.20 vs the
    // canonical 0.05) gives the fixed-step ray marcher in render_sdf enough
    // hit candidates per ray to render real geometry instead of mostly sky.
    float sdf_gyroid = fabsf(gyroid) - 0.20f;          // slab around iso-zero
    float sdf_sphere = length3(px, py, pz) - 0.9f;     // bounding sphere
    float sdf        = fmaxf(sdf_gyroid, sdf_sphere);  // CSG intersection

    // surf3Dwrite: x in BYTES (cast sizeof to int so 32-bit arithmetic works
    // even when x is large), y/z in elements.
    surf3Dwrite<float>(sdf, surf, x * (int)sizeof(float), y, z);
}

// --------------------------------------------------------------------------
// SDF sampler: tex3D wants normalized coords in [0, 1]; the volume covers
// [-1, 1] in world space, so we remap with `(p + 1) * 0.5`. Returns the
// raw stored SDF (a signed distance in world units).
// --------------------------------------------------------------------------
__device__ __forceinline__ float sample_sdf(cudaTextureObject_t tex,
                                            float px, float py, float pz) {
    return tex3D<float>(tex,
                        (px + 1.0f) * 0.5f,
                        (py + 1.0f) * 0.5f,
                        (pz + 1.0f) * 0.5f);
}

// --------------------------------------------------------------------------
// render_sdf: one thread per screen pixel. Builds the orbit camera, marches
// a ray through the SDF volume, and writes a shaded RGBA8 pixel to the PBO.
//
// Camera math (orbit, look-at origin, world-up (0, 1, 0)):
//   eye = dist * (cos(pitch)*cos(yaw), sin(pitch), cos(pitch)*sin(yaw))
//   fwd = normalize(target - eye)         (target = origin)
//   right = normalize(cross(fwd, up))
//   up'   = cross(right, fwd)
//   For a pixel at (u, v) in NDC ([-1, 1] x [-1, 1] with v=1 at the top),
//   dir = normalize(fwd + tan(fov/2) * (aspect * u * right + v * up'))
//
// Ray-march:
//   Fixed-step march: t += STEP, where STEP is set to roughly one voxel. The
//   gyroid SDF is non-Lipschitz, which makes classical sphere tracing
//   (t += sdf(p)) overshoot through thin slabs and miss almost every ray. A
//   uniform voxel-sized step is robust and cheap because the SDF is just a
//   tex3D lookup. We declare a HIT when sdf < HIT_EPS.
//
// Bounds bail: outside the [-1, 1]^3 box, return the sky.
// Normal: 6-sample central differences with eps ~ 1.5/VOLUME_SIZE so the
//         offsets are just over one voxel apart -- short enough to capture
//         local surface direction, long enough that trilinear filtering
//         actually moves the result.
// --------------------------------------------------------------------------
extern "C" __global__
void render_sdf(unsigned char* output,
                int width,
                int height,
                cudaTextureObject_t tex,
                float yaw,
                float pitch,
                float dist) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // ---- Build the orbit camera basis ----------------------------------
    float cp = cosf(pitch), sp = sinf(pitch);
    float cy = cosf(yaw),   sy = sinf(yaw);

    // Eye on a sphere of radius `dist` around the origin.
    float ex = dist * cp * cy;
    float ey = dist * sp;
    float ez = dist * cp * sy;

    // fwd = normalize(target - eye), target = origin -> fwd = -eye / |eye|.
    float fl = length3(ex, ey, ez);
    // Guard against the (clamped) dist being zero (not reachable, but cheap).
    if (fl < 1e-6f) fl = 1e-6f;
    float fx = -ex / fl, fy = -ey / fl, fz = -ez / fl;

    // right = normalize(cross(fwd, world_up)), world_up = (0, 1, 0).
    // cross((fx,fy,fz), (0,1,0)) = (fy*0 - fz*1, fz*0 - fx*0, fx*1 - fy*0)
    //                            = (-fz, 0, fx)
    float rx = -fz;
    float ry = 0.0f;
    float rz = fx;
    float rl = length3(rx, ry, rz);
    if (rl < 1e-6f) rl = 1e-6f;
    rx /= rl; ry /= rl; rz /= rl;

    // up' = cross(right, fwd). With right purely in the xz-plane, this is a
    // proper orthonormal up; recompute to keep the basis consistent.
    float ux = ry * fz - rz * fy;
    float uy = rz * fx - rx * fz;
    float uz = rx * fy - ry * fx;

    // ---- Per-pixel ray direction ---------------------------------------
    // NDC with v=1 at the TOP. With our PBO layout (y=0 written first ->
    // ends up at the bottom of the on-screen texture courtesy of the GL
    // shader's [0, 1] texcoord), v = 2*v_norm - 1 already maps row 0 of the
    // PBO to v = -1 (bottom of the image), which matches the camera's
    // up'-axis convention. No flip needed.
    float u_ndc = 2.0f * ((float)x + 0.5f) / (float)width  - 1.0f;
    float v_ndc = 2.0f * ((float)y + 0.5f) / (float)height - 1.0f;

    const float FOV_Y    = 0.7853981633974483f;        // 45 degrees
    const float TAN_HALF = 0.41421356237309515f;       // tanf(FOV_Y / 2)
    float aspect = (float)width / (float)height;

    float dx = fx + u_ndc * aspect * TAN_HALF * rx + v_ndc * TAN_HALF * ux;
    float dy = fy + u_ndc * aspect * TAN_HALF * ry + v_ndc * TAN_HALF * uy;
    float dz = fz + u_ndc * aspect * TAN_HALF * rz + v_ndc * TAN_HALF * uz;
    float dl = length3(dx, dy, dz);
    if (dl < 1e-6f) dl = 1e-6f;
    dx /= dl; dy /= dl; dz /= dl;

    // ---- Ray vs. the [-1, 1]^3 box (slab method) -----------------------
    // The camera always sits outside the volume (DIST_MIN >= 1.2 and the
    // orbit puts at least one component of the eye outside [-1, 1] for
    // typical framings), so we must first advance `t` to the AABB entry
    // before any in-volume sampling is meaningful. tNear is the entry
    // distance (clamped to >= 0 so we don't march backwards if the eye is
    // inside the box for some configuration); tFar is the exit distance.
    // If the slab interval is empty (tNear > tFar), the ray misses outright.
    float inv_dx = 1.0f / (fabsf(dx) > 1e-8f ? dx : (dx >= 0 ? 1e-8f : -1e-8f));
    float inv_dy = 1.0f / (fabsf(dy) > 1e-8f ? dy : (dy >= 0 ? 1e-8f : -1e-8f));
    float inv_dz = 1.0f / (fabsf(dz) > 1e-8f ? dz : (dz >= 0 ? 1e-8f : -1e-8f));
    float t1x = (-1.0f - ex) * inv_dx, t2x = ( 1.0f - ex) * inv_dx;
    float t1y = (-1.0f - ey) * inv_dy, t2y = ( 1.0f - ey) * inv_dy;
    float t1z = (-1.0f - ez) * inv_dz, t2z = ( 1.0f - ez) * inv_dz;
    float tNear = fmaxf(fmaxf(fminf(t1x, t2x), fminf(t1y, t2y)), fminf(t1z, t2z));
    float tFar  = fminf(fminf(fmaxf(t1x, t2x), fmaxf(t1y, t2y)), fmaxf(t1z, t2z));

    bool  hit = false;
    float hx = 0.0f, hy = 0.0f, hz = 0.0f;

    if (tFar > fmaxf(tNear, 0.0f)) {
        // ---- Fixed-step march through the SDF volume from the AABB entry
        // Sphere tracing relies on a Lipschitz-1 SDF: the magnitude of the
        // sample tells you a safe distance you can step without crossing
        // the surface. But the gyroid SDF here, |sx*cy + sy*cz + sz*cx|
        // - 0.20, has a gradient scaling with TAU ~= 19, so the stored
        // magnitude vastly over-reports the true distance. Sphere tracing
        // would routinely overshoot thin slab regions, leaving most rays
        // missing geometry that's actually there. A fixed-step march is
        // cheap (the SDF is just a tex3D lookup) and robust: each step
        // advances by one voxel, so any positive crossing of the iso-zero
        // surface lands inside a thin window where HIT_EPS catches it.
        //
        // 2 worldspace units / 256 steps = ~0.008 / step, slightly under
        // one voxel at 128^3 resolution.
        const int   MAX_STEPS = 256;
        const float STEP      = 1.0f / 128.0f;
        const float HIT_EPS   = 1.0e-3f;
        // Bias slightly inside the box so the very first sample isn't on
        // the boundary (CLAMP addressing makes the boundary sample valid,
        // but starting just inside avoids one wasted iteration).
        float t = fmaxf(tNear, 0.0f) + 1e-4f;
        float t_exit = tFar;

        #pragma unroll 1
        for (int i = 0; i < MAX_STEPS; ++i) {
            float pxw = ex + t * dx;
            float pyw = ey + t * dy;
            float pzw = ez + t * dz;

            float s = sample_sdf(tex, pxw, pyw, pzw);
            if (s < HIT_EPS) {
                hit = true;
                hx = pxw; hy = pyw; hz = pzw;
                break;
            }
            t += STEP;
            if (t > t_exit) break;
        }
    }

    // ---- Shade -----------------------------------------------------------
    float r, g, b;
    if (hit) {
        // Central-difference normal in world space. Each sample step is
        // ~1.17 voxels: short enough to capture local geometry, long enough
        // that trilinear filtering meaningfully moves the result.
        const float NEPS = 1.5f / 128.0f;
        float nx = sample_sdf(tex, hx + NEPS, hy, hz) -
                   sample_sdf(tex, hx - NEPS, hy, hz);
        float ny = sample_sdf(tex, hx, hy + NEPS, hz) -
                   sample_sdf(tex, hx, hy - NEPS, hz);
        float nz = sample_sdf(tex, hx, hy, hz + NEPS) -
                   sample_sdf(tex, hx, hy, hz - NEPS);
        float nl = length3(nx, ny, nz);
        if (nl < 1e-6f) nl = 1e-6f;
        nx /= nl; ny /= nl; nz /= nl;

        // Fixed key light (normalized world direction).
        const float LX = 0.5773502691896258f;          // (1,1,-1)/sqrt(3)
        const float LY = 0.5773502691896258f;
        const float LZ = -0.5773502691896258f;
        float diff = fmaxf(0.0f, dot3(nx, ny, nz, LX, LY, LZ));

        // Specular: Blinn-Phong half-vector exponent. View dir = -ray dir.
        float vx = -dx, vy = -dy, vz = -dz;
        float hx2 = LX + vx, hy2 = LY + vy, hz2 = LZ + vz;
        float hl  = length3(hx2, hy2, hz2);
        if (hl < 1e-6f) hl = 1e-6f;
        hx2 /= hl; hy2 /= hl; hz2 /= hl;
        float ndoth = fmaxf(0.0f, dot3(nx, ny, nz, hx2, hy2, hz2));
        float spec = powf(ndoth, 32.0f);

        // Base albedo varies with the hit position so the gyroid lattice
        // reads as a single material with smooth variation, not flat plastic.
        float base_r = 0.55f + 0.30f * nx;
        float base_g = 0.50f + 0.30f * ny;
        float base_b = 0.70f + 0.30f * nz;

        const float AMBIENT = 0.18f;
        r = base_r * (AMBIENT + 0.82f * diff) + 0.6f * spec;
        g = base_g * (AMBIENT + 0.82f * diff) + 0.6f * spec;
        b = base_b * (AMBIENT + 0.82f * diff) + 0.7f * spec;
    } else {
        // Sky: dark blue at the top, near-black at the bottom. The PBO's row
        // 0 is the bottom of the on-screen image (see the v_ndc comment),
        // so we use the y coordinate of the ray direction (close to v_ndc
        // in screen space) for the gradient.
        float sky = 0.5f * (dy + 1.0f);                // [0, 1] roughly
        sky = clampf(sky, 0.0f, 1.0f);
        r = 0.02f + 0.06f * sky;
        g = 0.03f + 0.10f * sky;
        b = 0.05f + 0.20f * sky;
    }

    r = clampf(r, 0.0f, 1.0f);
    g = clampf(g, 0.0f, 1.0f);
    b = clampf(b, 0.0f, 1.0f);

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
