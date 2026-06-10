# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# ################################################################################
#
# This example demonstrates cuda.core.CUDAArray, TextureObject, and
# GraphicsResource for CUDA/OpenGL interop. A vivid procedural background image
# is uploaded once into a 2D CUDAArray and bound as a TextureObject sampled with
# FilterMode.LINEAR + AddressMode.MIRROR + normalized coordinates. Each frame a
# `render_water` kernel evaluates an animated water surface analytically, refracts
# the view ray through it to perturb the background lookup UVs, adds shimmering
# caustic highlights, and writes RGBA8 straight into an OpenGL PBO. The effect is
# "looking through a sunlit pool". Requires pyglet.
#
# ################################################################################

# What this example teaches
# =========================
# - How to upload a host numpy image into a CUDAArray with `CUDAArray.copy_from`
#   (host layout (H, W, 4) uint8 row-major for an array allocated as
#   shape=(WIDTH, HEIGHT)) and bind it as a long-lived TextureObject.
# - Why FilterMode.LINEAR + AddressMode.MIRROR + normalized_coords=True is the
#   right pairing for a refraction effect: refracted UV lookups routinely fall
#   slightly outside [0, 1], and MIRROR returns a sensible mirrored pixel rather
#   than a clamped smear or a hard edge, while LINEAR keeps the warp smooth.
# - Why srgb=True is the correct read mode for an 8-bit color image: the texels
#   are decoded sRGB->linear on read, the kernel does its lighting and tonemap
#   in linear light, then re-encodes to sRGB on output (the gamma-correct
#   "sample in linear, tonemap, output" pipeline).
# - Why max_anisotropy is justified here: refraction samples the texture at
#   grazing, stretched angles, the case anisotropic filtering exists to clean
#   up.
# - That the animated water normal field is computed ANALYTICALLY in the kernel
#   (a sum of moving directional sine waves plus a few expanding circular
#   ripples), so there is no second CUDAArray and no SurfaceObject pass -- the
#   normal and its curvature are evaluated per pixel from a `time` uniform.
# - How to feed a small fixed ring of interactive click-ripples to the kernel
#   purely as scalar launch arguments (the demonstrated launch convention),
#   avoiding any custom device-buffer machinery.
#
# How it works
# ============
#   Startup (once):
#     +-------------------+   copy_from   +-----------+
#     | host numpy image  | ------------> | CUDAArray |  (UINT8 RGBA, vivid grid)
#     +-------------------+               +-----+-----+
#                                               |
#                                               v
#                                        +-------------+
#                                        | TextureObj  |  LINEAR + MIRROR + norm
#                                        +-------------+
#
#   Each frame (render_water kernel, 2D over the screen):
#     1. Evaluate the water height/normal at this pixel from the analytic wave
#        sum (directional waves + circular ripples) using the `time` uniform.
#     2. Refract: offset the background sample UV by `refract` * (the water
#        surface gradient) -- a cheap 2D approximation of bending the view ray.
#     3. Sample the background TextureObject at the perturbed UV (LINEAR +
#        MIRROR keeps it smooth and well-defined outside [0, 1]).
#     4. Caustics: brightness focuses where wavefronts converge. Approximate
#        with a sharpened power of the surface curvature (Laplacian), adding
#        bright cyan/white highlights. Add a depth tint (deeper = bluer) and a
#        specular sparkle from the normal versus a fixed light direction.
#     5. Tonemap and write RGBA8 into the OpenGL PBO. No PCIe traffic per frame.
#
# Why MIRROR (not WRAP or CLAMP)?
# -------------------------------
# WRAP and MIRROR both require normalized coordinates. WRAP tiles the image, so
# a refraction pushing past the right edge suddenly shows the far-left content
# (a visible seam). CLAMP smears the edge texel into a streak. MIRROR reflects
# the image at the boundary, which for a small refraction offset looks like the
# pool simply continuing -- the most natural choice here.
#
# What you should see
# ===================
# A colorful tiled background rippling as if seen through moving water, with
# bright caustic highlights skittering across it. Press +/- to change the
# refraction/ripple strength, click anywhere to spawn an expanding circular
# ripple at the cursor, and Escape to exit. The title shows FPS and the current
# ripple strength.
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
    TextureDescriptor,
    TextureObject,
    launch,
)

# ---------------------------------------------------------------------------
# Parameters (feel free to change these)
# ---------------------------------------------------------------------------
WIDTH = 800
HEIGHT = 600
BG_SIZE = 256  # the background CUDAArray is BG_SIZE x BG_SIZE RGBA8

# Interactive click-ripples. We keep a small fixed ring and pass each slot to
# the kernel as plain float scalars (matching the demonstrated launch
# convention -- no custom device buffers). A ripple with start time < 0 is
# inactive.
MAX_RIPPLES = 3
RIPPLE_LIFETIME = 4.0  # seconds before a click-ripple fully fades out

DEFAULT_STRENGTH = 1.0
STRENGTH_STEP = 0.15
MIN_STRENGTH = 0.0
MAX_STRENGTH = 3.0


# ============================= Helper functions =============================
#
# The functions below set up CUDA and OpenGL. If you're here to learn about
# CUDAArray/TextureObject, skip ahead to main() -- the interesting part is
# there. These helpers exist so main() reads like a short story instead of a
# wall of boilerplate.
# ============================================================================


def make_background_image(size):
    """Build a (size, size, 4) uint8 RGBA background designed to show refraction.

    Layout convention: CUDAArray.from_descriptor takes shape=(WIDTH, HEIGHT), so
    the host buffer fed to copy_from must be H rows of W elements (row-major),
    i.e. host.shape == (HEIGHT, WIDTH, 4). Here the image is square so the two
    agree, but the (y, x) indexing below is the load-bearing part.

    The pattern is deliberately vivid and high-frequency -- a grid of saturated
    hues with concentric rings -- so even small refraction offsets are obvious.
    """
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
    u = xs / size
    v = ys / size

    # Saturated, smoothly varying hues across the plane (a cheap HSV-ish wheel).
    r = 0.5 + 0.5 * np.sin(u * 6.2831853 * 2.0 + 0.0)
    g = 0.5 + 0.5 * np.sin(v * 6.2831853 * 2.0 + 2.0944)
    b = 0.5 + 0.5 * np.sin((u + v) * 6.2831853 * 2.0 + 4.1888)

    # Bright grid lines so the warp is legible.
    cells = 8.0
    gx = np.abs(((u * cells) % 1.0) - 0.5)
    gy = np.abs(((v * cells) % 1.0) - 0.5)
    grid = np.maximum(gx, gy)
    grid_line = (grid > 0.42).astype(np.float32)
    r = r * (1.0 - grid_line) + 1.0 * grid_line
    g = g * (1.0 - grid_line) + 1.0 * grid_line
    b = b * (1.0 - grid_line) + 1.0 * grid_line

    # A couple of concentric rings centered on the image to add curvature cues.
    cx, cy = 0.5, 0.5
    dist = np.sqrt((u - cx) ** 2 + (v - cy) ** 2)
    rings = 0.5 + 0.5 * np.sin(dist * 6.2831853 * 10.0)
    r = np.clip(r * 0.75 + rings * 0.25, 0.0, 1.0)
    g = np.clip(g * 0.75 + rings * 0.20, 0.0, 1.0)
    b = np.clip(b * 0.85 + rings * 0.15, 0.0, 1.0)

    img = np.zeros((size, size, 4), dtype=np.uint8)
    img[:, :, 0] = (r * 255.0).astype(np.uint8)
    img[:, :, 1] = (g * 255.0).astype(np.uint8)
    img[:, :, 2] = (b * 255.0).astype(np.uint8)
    img[:, :, 3] = 255
    return img


def setup_cuda():
    """Compile the kernel and return (device, stream, kernel, launch_config)."""
    dev = Device(0)
    dev.set_current()

    cc = dev.compute_capability
    if cc.major < 3:
        print(
            "This example requires a GPU with compute capability >= 3.0 for "
            f"bindless texture objects. Found sm_{cc.major}{cc.minor}.",
            file=sys.stderr,
        )
        sys.exit(1)

    stream = dev.create_stream()

    program_options = ProgramOptions(std="c++17", arch=f"sm_{dev.arch}")
    prog = Program(KERNEL_SOURCE, code_type="c++", options=program_options)
    mod = prog.compile("cubin", name_expressions=("render_water",))
    kernel = mod.get_kernel("render_water")

    block = (16, 16, 1)
    grid = (
        (WIDTH + block[0] - 1) // block[0],
        (HEIGHT + block[1] - 1) // block[1],
        1,
    )
    config = LaunchConfig(grid=grid, block=block)
    return dev, stream, kernel, config


def create_window():
    """Open a pyglet window. Returns (window, gl_module, pyglet_module)."""
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
        caption="cuda.core CUDAArray + TextureObject - Water Caustics",
        vsync=False,
    )
    return window, _gl, pyglet


def create_display_resources(gl, width, height):
    """Standard pyglet boilerplate: shader, fullscreen quad, screen texture."""
    from pyglet.graphics.shader import Shader, ShaderProgram

    vert = Shader(VERTEX_SHADER_SOURCE, "vertex")
    frag = Shader(FRAGMENT_SHADER_SOURCE, "fragment")
    shader_prog = ShaderProgram(vert, frag)

    quad_verts = np.array(
        [
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
    """Create the GL PBO that CUDA writes RGBA pixels into each frame."""
    pbo = ctypes.c_uint(0)
    gl.glGenBuffers(1, ctypes.byref(pbo))
    gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, pbo.value)
    nbytes = width * height * 4
    gl.glBufferData(gl.GL_PIXEL_UNPACK_BUFFER, nbytes, None, gl.GL_DYNAMIC_DRAW)
    gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)
    return pbo.value


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


MAX_ANISOTROPY = 8  # kept in lockstep with the API MAP comment + live caption


def make_background_texture(arr):
    """Bind `arr` as a TextureObject for LINEAR + MIRROR + normalized sampling.

    MIRROR (like WRAP) requires normalized coordinates. UINT8 source +
    NORMALIZED_FLOAT means tex2D<float4> returns each channel in [0, 1].

    API MAP: UINT8 RGBA CUDAArray sampled as TextureObject[LINEAR | MIRROR |
    NORMALIZED_FLOAT | srgb | max_anisotropy=8]; MIRROR handles refracted UVs
    that leave [0,1]; srgb does the gamma-correct decode; anisotropy cleans up
    grazing-angle sampling.

    Two TextureDescriptor features are showcased here on an 8-bit color image:

    - srgb=True: the background is UINT8 RGBA authored in perceptual space, so
      enabling sRGB->linear conversion on read is the correct thing to do --
      the kernel then does all of its lighting/tonemap math in linear light and
      re-encodes to sRGB on output (the final pow(c, 1/2.2) below). This is the
      gamma-correct "sample in linear, tonemap, output" pipeline.
    - max_anisotropy=8: refraction samples the texture at grazing, stretched
      angles, which is exactly the case anisotropic filtering is meant to clean
      up, so we request it on the background texture.
    """
    res_desc = ResourceDescriptor.from_array(arr)
    tex_desc = TextureDescriptor(
        address_mode=AddressMode.MIRROR,
        filter_mode=FilterMode.LINEAR,
        read_mode=ReadMode.NORMALIZED_FLOAT,
        # MIRROR/WRAP addressing modes require normalized coordinates.
        normalized_coords=True,
        # 8-bit color image -> decode sRGB to linear on read so the lighting and
        # tonemap math runs in linear light (re-encoded to sRGB on output).
        srgb=True,
        # Refraction samples at grazing/stretched angles; anisotropic filtering
        # cleans those up.
        max_anisotropy=MAX_ANISOTROPY,
    )
    return TextureObject.from_descriptor(resource=res_desc, texture_descriptor=tex_desc)


# ================================== main() ==================================


def main():
    # --- Step 1: Set up CUDA (compile kernel, create stream) ---
    dev, stream, kernel, config = setup_cuda()

    # --- Step 2: Open a window ---
    window, gl, pyglet = create_window()

    # --- Step 3: Create GL resources (shader, fullscreen quad, screen tex) ---
    shader_prog, quad_vao, screen_tex = create_display_resources(gl, WIDTH, HEIGHT)

    # --- Step 4: Create the PBO that CUDA will write into ---
    pbo_id = create_pixel_buffer(gl, WIDTH, HEIGHT)
    resource = GraphicsResource.from_gl_buffer(pbo_id, flags="write_discard")

    # --- Step 5: Allocate the background CUDAArray and upload the image once ---
    bg_arr = CUDAArray.from_descriptor(
        shape=(BG_SIZE, BG_SIZE),
        format=ArrayFormat.UINT8,
        num_channels=4,
    )
    host_image = make_background_image(BG_SIZE)
    bg_arr.copy_from(np.ascontiguousarray(host_image), stream=stream)
    stream.sync()

    # --- Step 6: Bind the CUDAArray as a long-lived TextureObject ---
    #     Created once and kept alive: `launch` is async, so a per-frame texture
    #     inside a closing `with` would destroy the handle before the kernel ran.
    bg_tex = make_background_texture(bg_arr)

    # Interactive state. Each ripple slot is (origin_x, origin_y, start_time) in
    # normalized screen coords / seconds; start_time < 0 means inactive.
    state = {
        "strength": DEFAULT_STRENGTH,
        "ripples": [[0.0, 0.0, -1.0] for _ in range(MAX_RIPPLES)],
        "next_slot": 0,
    }
    start_time = time.monotonic()

    @window.event
    def on_key_press(symbol, _modifiers):
        key = pyglet.window.key
        if symbol == key.ESCAPE:
            window.close()
        elif symbol in (key.PLUS, key.EQUAL, key.NUM_ADD):
            state["strength"] = min(MAX_STRENGTH, state["strength"] + STRENGTH_STEP)
        elif symbol in (key.MINUS, key.UNDERSCORE, key.NUM_SUBTRACT):
            state["strength"] = max(MIN_STRENGTH, state["strength"] - STRENGTH_STEP)

    @window.event
    def on_mouse_press(x, y, _button, _modifiers):
        # pyglet's origin is bottom-left, which matches our normalized UV
        # convention below (v increases upward). Record into the ring buffer.
        now = time.monotonic() - start_time
        slot = state["next_slot"]
        state["ripples"][slot] = [x / WIDTH, y / HEIGHT, now]
        state["next_slot"] = (slot + 1) % MAX_RIPPLES

    # --- Step 7: Render loop ---
    frame_count = 0
    fps_time = start_time

    @window.event
    def on_draw():
        nonlocal frame_count, fps_time

        now = time.monotonic()
        t = now - start_time

        window.clear()

        # Flatten the ripple ring into the scalar args the kernel expects:
        # for each slot, (origin_x, origin_y, age) where age < 0 == inactive.
        ripple_args = []
        for ox, oy, st in state["ripples"]:
            age = (t - st) if st >= 0.0 else -1.0
            if age >= RIPPLE_LIFETIME:
                age = -1.0
            ripple_args.extend((np.float32(ox), np.float32(oy), np.float32(age)))

        with resource.map(stream=stream) as buf:
            launch(
                stream,
                config,
                kernel,
                np.uint64(bg_tex.handle),
                buf.handle,
                np.int32(WIDTH),
                np.int32(HEIGHT),
                np.float32(t),
                np.float32(state["strength"]),
                np.float32(RIPPLE_LIFETIME),
                *ripple_args,
            )
        copy_pbo_to_texture(gl, pbo_id, screen_tex, WIDTH, HEIGHT)
        draw_fullscreen_quad(gl, shader_prog, quad_vao, screen_tex)

        frame_count += 1
        if now - fps_time >= 1.0:
            fps = frame_count / (now - fps_time)
            window.set_caption(
                "cuda.core CUDAArray + TextureObject - Water Caustics "
                f"(strength={state['strength']:.2f}, {fps:.0f} FPS) "
                f"| TextureObject[LINEAR|MIRROR|sRGB|aniso={MAX_ANISOTROPY}] UINT8 "
                "[+/- strength, click = ripple, Esc = quit]"
            )
            frame_count = 0
            fps_time = now

    @window.event
    def on_close():
        bg_tex.close()
        bg_arr.close()
        resource.close()
        stream.close()

    pyglet.app.run(interval=0)


# ============================== GPU code (kernel) ============================
#
# render_water samples a static background TextureObject (LINEAR + MIRROR +
# normalized coords) at refraction-perturbed UVs. The water surface and its
# normal/curvature are evaluated analytically from a `time` uniform -- there is
# no second array and no SurfaceObject. MAX_RIPPLES click-ripples arrive as
# (origin_x, origin_y, age) float triples; age < 0 marks an empty slot.
#
# The ripple count is compiled in via the MAX_RIPPLES define so the kernel's
# parameter list (host side) and the loop bound (device side) stay in lockstep.
# ============================================================================

KERNEL_SOURCE = (
    "#define MAX_RIPPLES "
    + str(MAX_RIPPLES)
    + "\n"
    + r"""
// Analytic water height field at normalized position p and time t. A sum of a
// few moving directional waves gives the base chop; the expanding circular
// ripples from clicks ride on top. Returns height; gradient/curvature are taken
// numerically by sampling this a few times (cheap and robust).
__device__ __forceinline__
float water_height(float px, float py, float t,
                   const float* rip_x, const float* rip_y,
                   const float* rip_age, float ripple_lifetime) {
    float h = 0.0f;

    // Directional waves: (dir_x, dir_y, freq, speed, amp).
    // Hand-picked so they never perfectly align (avoids an obvious repeat).
    const float waves[5][5] = {
        { 1.00f,  0.00f,  9.0f,  1.3f, 0.45f},
        { 0.20f,  0.98f, 12.0f,  1.0f, 0.35f},
        {-0.70f,  0.71f, 16.0f,  1.7f, 0.25f},
        { 0.80f, -0.60f, 22.0f,  2.1f, 0.18f},
        {-0.30f, -0.95f, 31.0f,  2.6f, 0.12f},
    };
    #pragma unroll
    for (int i = 0; i < 5; ++i) {
        float phase = (waves[i][0] * px + waves[i][1] * py) * waves[i][2]
                      + t * waves[i][3];
        h += waves[i][4] * sinf(phase);
    }

    // Expanding circular ripples from mouse clicks. Each is a decaying radial
    // wave packet whose ring radius grows with age.
    for (int r = 0; r < MAX_RIPPLES; ++r) {
        float age = rip_age[r];
        if (age < 0.0f) continue;
        float dx = px - rip_x[r];
        float dy = py - rip_y[r];
        float dist = sqrtf(dx * dx + dy * dy);
        float ring = dist * 40.0f - age * 8.0f;       // outward-moving ring
        float envelope = expf(-dist * 6.0f);           // localized in space
        float fade = 1.0f - (age / ripple_lifetime);   // fade over lifetime
        if (fade < 0.0f) fade = 0.0f;
        h += 0.9f * fade * envelope * sinf(ring);
    }
    return h;
}

extern "C"
__global__
void render_water(cudaTextureObject_t bg,
                  unsigned char* output,
                  int width, int height,
                  float t,
                  float strength,
                  float ripple_lifetime,
"""
    + "".join(
        f"                  float rip_x{i}, float rip_y{i}, float rip_age{i}"
        + (",\n" if i < MAX_RIPPLES - 1 else ") {\n")
        for i in range(MAX_RIPPLES)
    )
    + r"""
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // Pack the per-ripple scalars back into arrays so the helper can loop.
    float rip_x[MAX_RIPPLES];
    float rip_y[MAX_RIPPLES];
    float rip_age[MAX_RIPPLES];
"""
    + "".join(
        f"    rip_x[{i}] = rip_x{i}; rip_y[{i}] = rip_y{i}; rip_age[{i}] = rip_age{i};\n" for i in range(MAX_RIPPLES)
    )
    + r"""
    // Normalized screen position. v increases upward to match pyglet's
    // bottom-left mouse origin used when recording ripple coordinates.
    float u = (x + 0.5f) / (float)width;
    float v = 1.0f - (y + 0.5f) / (float)height;

    // Sample the water height field on a small stencil to get the surface
    // gradient (slope -> refraction) and Laplacian (curvature -> caustics).
    const float eps = 1.5f / (float)width;
    float hc = water_height(u, v, t, rip_x, rip_y, rip_age, ripple_lifetime);
    float hl = water_height(u - eps, v, t, rip_x, rip_y, rip_age, ripple_lifetime);
    float hr = water_height(u + eps, v, t, rip_x, rip_y, rip_age, ripple_lifetime);
    float hd = water_height(u, v - eps, t, rip_x, rip_y, rip_age, ripple_lifetime);
    float hu = water_height(u, v + eps, t, rip_x, rip_y, rip_age, ripple_lifetime);

    float gx = (hr - hl) / (2.0f * eps);   // d(height)/du
    float gy = (hu - hd) / (2.0f * eps);   // d(height)/dv
    // Discrete Laplacian (curvature). Divide by eps^2 so it is a true second
    // derivative -- without this the finite-difference sum is ~Laplacian*eps^2
    // (tiny), and the caustic term below would collapse to zero.
    float lap = (hl + hr + hd + hu - 4.0f * hc) / (eps * eps);

    // 2D refraction approximation: bend the background lookup by the surface
    // slope, scaled by the user `strength`. Small factor keeps it gentle.
    float refract = 0.015f * strength;
    float su = u - refract * gx;
    float sv = v - refract * gy;

    // Sample the background. LINEAR + MIRROR + normalized coords means the
    // perturbed (su, sv) can leave [0, 1] and still return a smooth, mirrored
    // pixel rather than a clamped streak or a hard seam. Because the texture was
    // bound with srgb=True, each channel is already decoded to LINEAR light
    // here -- so all the lighting/tonemap math below is physically sensible and
    // we only re-encode to sRGB at the very end.
    //
    // Chromatic dispersion: water bends short (blue) wavelengths more than long
    // (red) ones, so we sample R/G/B at slightly different refraction offsets.
    // This gives caustic edges and warped grid lines faint rainbow fringes.
    float disp = 0.30f * refract;                // dispersion spread, in UV
    float base_r = tex2D<float4>(bg, su - disp * gx, sv - disp * gy).x;
    float base_b = tex2D<float4>(bg, su + disp * gx, sv + disp * gy).z;
    float4 base = tex2D<float4>(bg, su, sv);   // green keeps the unsplit UV
    base.x = base_r;
    base.z = base_b;

    // Surface normal from the gradient (z component points out of the water).
    float nx = -gx;
    float ny = -gy;
    float nz = 1.0f;
    float ninv = rsqrtf(nx * nx + ny * ny + nz * nz);
    nx *= ninv; ny *= ninv; nz *= ninv;

    // Caustics: light focuses where the wavefront converges (negative
    // curvature). Raise a sharpened function of the curvature to a power to get
    // tight bright filaments, then add as a cyan/white highlight.
    // The wave-sum Laplacian peaks around O(150-200), so this factor lands
    // `focus` near O(1) at a converging wavefront.
    float focus = -lap * 0.005f;
    if (focus < 0.0f) focus = 0.0f;
    float caustic = focus * focus * focus;       // sharpen into thin filaments
    caustic *= (0.6f + 0.8f * strength);
    if (caustic > 1.5f) caustic = 1.5f;

    // Specular sparkle: normal vs a fixed light direction.
    float lx = 0.4f, ly = 0.5f, lz = 0.768f;     // normalized-ish light dir
    float spec = nx * lx + ny * ly + nz * lz;
    if (spec < 0.0f) spec = 0.0f;
    spec = powf(spec, 48.0f);

    // Animated light shafts / god-rays: angled bright bands that drift and
    // breathe over time, as if sunlight were cutting down through the water.
    // Built purely from (u, v, t) -- no extra launch args. The shafts are
    // gated by the surface slope so they ripple with the waves and the water
    // curvature concentrates them into bright filaments where the wavefront
    // focuses, reinforcing the caustics.
    float shaft_dir = u * 7.5f + v * 3.0f;       // angled across the screen
    float shafts = 0.5f + 0.5f * sinf(shaft_dir + t * 0.7f + 1.5f * gx);
    shafts *= 0.5f + 0.5f * sinf(shaft_dir * 0.37f - t * 0.4f);
    shafts = powf(shafts, 3.0f);                 // crush into thin shafts
    float godray = shafts * (0.18f + 0.45f * focus);

    // Depth tint: deeper troughs read bluer/darker, crests slightly brighter.
    float depth = 0.5f + 0.5f * hc;              // ~[0, 1]
    float tint_r = 0.85f + 0.15f * depth;
    float tint_g = 0.92f + 0.08f * depth;
    float tint_b = 1.05f - 0.10f * depth;

    // Composite in LINEAR light. Caustics get a faint warm/cool split and the
    // god-rays a sunlit warm bias so the bright filaments read as light, not
    // just blown-out white.
    float cr = base.x * tint_r + caustic * 0.95f + spec * 0.9f + godray * 1.10f;
    float cg = base.y * tint_g + caustic * 1.00f + spec * 0.9f + godray * 1.00f;
    float cb = base.z * tint_b + caustic * 1.05f + spec * 1.0f + godray * 0.80f;

    // Simple Reinhard tonemap so highlights roll off instead of clipping hard.
    cr = cr / (1.0f + cr);
    cg = cg / (1.0f + cg);
    cb = cb / (1.0f + cb);

    // Encode LINEAR -> sRGB on output. This is the matching half of the
    // srgb=True decode on the texture read: we sampled and lit in linear, and
    // now re-encode for the 8-bit RGBA8 PBO. The ~1/2.2 exponent is the
    // gamma-correct encode (and also lifts the midtones the linear decode
    // darkened, so the pool reads luminous rather than murky).
    cr = powf(cr, 1.0f / 2.2f);
    cg = powf(cg, 1.0f / 2.2f);
    cb = powf(cb, 1.0f / 2.2f);

    int idx = (y * width + x) * 4;
    output[idx + 0] = (unsigned char)(fminf(cr, 1.0f) * 255.0f);
    output[idx + 1] = (unsigned char)(fminf(cg, 1.0f) * 255.0f);
    output[idx + 2] = (unsigned char)(fminf(cb, 1.0f) * 255.0f);
    output[idx + 3] = 255;
}
"""
)

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
