# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# ################################################################################
#
# This example demonstrates the cuda.core texture/surface stack used to build a
# bloom / glow post-effect entirely on the GPU. An animated HDR-ish scene is
# rendered into the base level of a MipmappedArray; the mip pyramid is then
# built level by level via SurfaceObject writes (each level reads the one above
# through its own LINEAR TextureObject); finally a single mipmapped
# TextureObject samples several LODs with tex2DLod to composite a soft bloom on
# top of the sharp scene. Requires pyglet.
#
# ################################################################################

# What this example teaches
# =========================
# The least-demonstrated corner of the texture/surface API: the two halves of a
# mip pyramid round-trip.
#
# - BUILD side: MipmappedArray.get_level(i) returns a NON-OWNING CUDAArray view
#   of level i. Bind each level as its own SurfaceObject and have a kernel write
#   into it. We downsample by reading level i-1 through a per-level LINEAR
#   TextureObject (one bilinear tap == a 2x2 box average) and storing into
#   level i through that level's SurfaceObject. This is a mip chain built
#   *on the GPU*, not by the driver.
# - SAMPLE side: ONE mipmapped TextureObject (FilterMode.LINEAR +
#   mipmap_filter_mode=LINEAR, normalized coords) bound to the whole pyramid via
#   ResourceDescriptor.from_mipmapped_array lets a single tex2DLod<float4> read
#   any level -- the blurred coarse levels are exactly the glow.
#
# How it works
# ============
# Bloom is "blur the bright parts, add them back." A mip pyramid is a ready-made
# multi-scale blur: each coarser level is a halved, box-filtered copy of the
# level below, so reading a high LOD is reading a heavily blurred image.
#
#     level 0: 512 x 512   <- sharp animated scene (the emitters)
#     level 1: 256 x 256       (downsampled via SurfaceObject write)
#     level 2: 128 x 128
#     ...
#     level L-1: small        <- the softest, widest glow
#
#   PER FRAME (render loop)
#   ~~~~~~~~~~~~~~~~~~~~~~~
#   1. render_scene  -- writes an animated scene of moving bright emitters into
#                       level 0 through its SurfaceObject (float4 RGBA, values
#                       can exceed 1.0 in the hot spots).
#   2. downsample    -- for i in 1..L-1, read level i-1 through its LINEAR
#                       TextureObject and write level i through its
#                       SurfaceObject. A single LINEAR tap at the midpoint of
#                       the parent's 2x2 footprint *is* the box average.
#   3. composite     -- one mipmapped TextureObject; tex2DLod at lod 0 gives the
#                       sharp scene, and a weighted sum of lods 1..L-1 gives the
#                       bloom. Tonemap with 1 - exp(-c*x) and write RGBA8 to the
#                       OpenGL PBO.
#
#   surf2Dwrite indexes x in BYTES, so a float4 write uses x * sizeof(float4)
#   (= x * 16). Getting this wrong silently corrupts every fourth column.
#
# What you should see
# ===================
# Several colored emitters orbiting on a dark background, each wrapped in a soft
# glow. Bright cores bleed light into their surroundings.
#
#   +  /  =           bloom strength += 0.15
#   -                 bloom strength -= 0.15
#   [                 bloom threshold -= 0.05 (more of the scene glows)
#   ]                 bloom threshold += 0.05 (only the brightest glow)
#   ,  /  .           mipmap_level_bias -= / += 0.25 (sharper / softer glow)
#   ;  /  '           LODs summed -= / += 1 (the live max-LOD clamp)
#   B                 toggle bloom on / off (makes the effect obvious)
#   R                 reset all controls
#   Escape / close    quit
#
# The window title shows FPS plus the live mipmap LOD-selection config
# (MipmappedArray level count, trilinear tex2DLod bias / clamp / LODs) and the
# bloom strength, threshold, and on/off state.
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
    Device,
    FilterMode,
    GraphicsResource,
    LaunchConfig,
    MipmappedArray,
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
BASE_SIZE = 512  # Mip base-level edge length (power of two so levels halve cleanly).
MAX_LEVELS = 7  # Modest cap on pyramid depth; bounded by log2(BASE_SIZE)+1.
NUM_EMITTERS = 7

BLOOM_STRENGTH_STEP = 0.15
BLOOM_THRESHOLD_STEP = 0.05


# ============================= Helper functions =============================
#
# The functions below set up CUDA, OpenGL, and the mip pyramid. If you're here
# to learn about MipmappedArray / per-level SurfaceObject writes / mipmapped
# TextureObject sampling, skip straight to main() -- the interesting part is
# there. These helpers keep main() reading like a short story.
# ============================================================================


def _check_compute_capability(dev):
    """Surface load/store + mipmapped arrays require sm_30+."""
    cc = dev.compute_capability
    if cc.major < 3:
        print(
            f"This example requires compute capability >= 3.0, got sm_{cc.major}{cc.minor}.",
            file=sys.stderr,
        )
        sys.exit(1)


def setup_cuda():
    """Compile the three kernels and return (device, stream, kernels).

    kernels is a dict with keys "render_scene", "downsample", "composite".
    """
    dev = Device(0)
    dev.set_current()
    _check_compute_capability(dev)
    stream = dev.create_stream()

    program_options = ProgramOptions(std="c++17", arch=f"sm_{dev.arch}")
    prog = Program(KERNEL_SOURCE, code_type="c++", options=program_options)
    mod = prog.compile(
        "cubin",
        name_expressions=("render_scene", "downsample", "composite"),
    )
    kernels = {
        "render_scene": mod.get_kernel("render_scene"),
        "downsample": mod.get_kernel("downsample"),
        "composite": mod.get_kernel("composite"),
    }
    return dev, stream, kernels


def make_level_grid(level_size, block):
    """2D launch grid covering a (level_size x level_size) image."""
    return (
        (level_size + block[0] - 1) // block[0],
        (level_size + block[1] - 1) // block[1],
        1,
    )


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
        caption="cuda.core MipmappedArray - GPU mip-pyramid bloom",
        vsync=False,
    )
    return window, _gl, pyglet


def create_display_resources(gl, width, height):
    """Standard GL boilerplate: a shader program, a fullscreen quad, and an
    empty texture that we'll repeatedly fill from a PBO. Not CUDA-specific.

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

    # --- Step 2: Allocate the mip pyramid (single allocation, all levels) ---
    #     is_surface_load_store=True is required so each level can back a
    #     SurfaceObject for kernel-side writes. We cap the depth at MAX_LEVELS;
    #     each level halves until 1x1 at most.
    num_levels = min(int(math.log2(BASE_SIZE)) + 1, MAX_LEVELS)
    mm = MipmappedArray.from_descriptor(
        shape=(BASE_SIZE, BASE_SIZE),
        format=ArrayFormat.FLOAT32,
        num_channels=4,
        num_levels=num_levels,
        is_surface_load_store=True,
    )

    # --- Step 3: Pre-create per-level handles ONCE and keep them alive ---
    #     For every level we build a SurfaceObject (to write into it) and a
    #     non-mipmapped LINEAR TextureObject (so the downsample kernel can read
    #     the level above with hardware bilinear). get_level(i) returns a
    #     NON-OWNING view -- the storage belongs to `mm`, which we keep alive.
    #     Building these per-frame would be wasteful and, worse, a handle closed
    #     before its async launch runs would dangle.
    level_sizes = [BASE_SIZE >> i for i in range(num_levels)]
    level_arrays = [mm.get_level(i) for i in range(num_levels)]  # keep views alive

    src_tex_desc = TextureDescriptor(
        address_mode=AddressMode.CLAMP,
        filter_mode=FilterMode.LINEAR,  # one bilinear tap == 2x2 box average
        read_mode=ReadMode.ELEMENT_TYPE,
        normalized_coords=False,  # integer/pixel coordinates for the box tap
    )
    level_surfaces = [SurfaceObject.from_array(arr) for arr in level_arrays]
    level_textures = [
        TextureObject.from_descriptor(
            resource=ResourceDescriptor.from_array(arr),
            texture_descriptor=src_tex_desc,
        )
        for arr in level_arrays
    ]

    # --- Step 4: One mipmapped TextureObject over the WHOLE pyramid ---
    #     This is the sample side: tex2DLod can fetch any LOD from it, so the
    #     composite kernel reads the sharp scene (lod 0) and the blurred glow
    #     (lods 1..L-1) through this single handle. WRAP/MIRROR need normalized
    #     coords; we use CLAMP + normalized so a level's edge does not bleed in.
    #
    #   API MAP -- the mip pyramid round-trip
    #   =====================================
    #   BUILD on the GPU:   MipmappedArray.from_descriptor(...) allocates the
    #                       whole chain; mm.get_level(i) hands back a NON-OWNING
    #                       CUDAArray view of each level that we bind to a
    #                       per-level SurfaceObject and write into (the loop in
    #                       on_draw). The driver never builds the mips -- we do.
    #   READ it back:       ResourceDescriptor.from_mipmapped_array(mm) wraps the
    #                       SAME chain in ONE mipmapped TextureObject. tex2DLod
    #                       then samples any LOD with trilinear filtering.
    #   LOD selection knobs (TextureDescriptor):
    #     mipmap_filter_mode=LINEAR  -> trilinear: blend BETWEEN the two nearest
    #                                   integer LODs (vs NEAREST = snap to one).
    #     mipmap_level_bias          -> constant added to the requested LOD.
    #     min/max_mipmap_level_clamp -> clamp the effective LOD to a range.
    #   These descriptor fields are baked at construction (the texture is created
    #   ONCE, per the invariants). To demonstrate them INTERACTIVELY, the
    #   composite kernel folds the SAME bias/clamp math into its explicit
    #   tex2DLod `lod` argument -- live keys move bias / max-LOD without ever
    #   rebuilding the texture, while the descriptor encodes the static defaults.
    mip_tex_desc = TextureDescriptor(
        address_mode=AddressMode.CLAMP,
        filter_mode=FilterMode.LINEAR,
        read_mode=ReadMode.ELEMENT_TYPE,
        normalized_coords=True,
        mipmap_filter_mode=FilterMode.LINEAR,  # trilinear between levels
        mipmap_level_bias=0.0,
        min_mipmap_level_clamp=0.0,
        max_mipmap_level_clamp=float(num_levels - 1),
    )
    mip_tex = TextureObject.from_descriptor(
        resource=ResourceDescriptor.from_mipmapped_array(mm),
        texture_descriptor=mip_tex_desc,
    )

    # --- Step 5: Open a window and set up the GL/CUDA bridge ---
    window, gl, pyglet = create_window()
    shader_prog, quad_vao, tex_id = create_display_resources(gl, WIDTH, HEIGHT)
    pbo_id, _ = create_pixel_buffer(gl, WIDTH, HEIGHT)
    resource = GraphicsResource.from_gl_buffer(pbo_id, flags="write_discard")

    # --- Step 6: Render loop state + launch configs ---
    state = {
        "strength": 1.8,  # bloom intensity multiplier
        "threshold": 0.6,  # only luminance above this contributes to bloom
        "bloom_on": True,
        # --- Live LOD-selection controls (folded into the tex2DLod loop) ---
        "bias": 0.5,  # mipmap_level_bias added to each bloom tap's LOD
        "num_lods": max(1, num_levels - 1),  # how many LODs the bloom sums
        "min_clamp": 0.0,  # min_mipmap_level_clamp (shown; static default)
    }
    max_clamp = float(num_levels - 1)  # max_mipmap_level_clamp ceiling
    start_time = time.monotonic()
    frame_count = [0]
    fps_time = [start_time]

    block = (16, 16, 1)
    # The composite kernel covers the WIDTHxHEIGHT screen.
    composite_config = LaunchConfig(grid=make_level_grid_screen(block), block=block)

    @window.event
    def on_draw():
        window.clear()
        t = time.monotonic() - start_time

        # (a) Render the animated HDR-ish scene into level 0's surface.
        launch(
            stream,
            LaunchConfig(grid=make_level_grid(BASE_SIZE, block), block=block),
            kernels["render_scene"],
            np.uint64(level_surfaces[0].handle),
            np.int32(BASE_SIZE),
            np.int32(BASE_SIZE),
            np.float32(t),
            np.int32(NUM_EMITTERS),
        )

        # (b) Build the pyramid on the GPU: each level i reads level i-1 via its
        #     LINEAR TextureObject and writes level i via its SurfaceObject.
        for i in range(1, num_levels):
            dst_size = level_sizes[i]
            launch(
                stream,
                LaunchConfig(grid=make_level_grid(dst_size, block), block=block),
                kernels["downsample"],
                np.uint64(level_textures[i - 1].handle),  # read parent level
                np.uint64(level_surfaces[i].handle),  # write this level
                np.int32(dst_size),
            )

        # (c) Composite: one mipmapped texture, sample several LODs, tonemap,
        #     and write RGBA8 straight into the PBO.
        with resource.map(stream=stream) as buf:
            launch(
                stream,
                composite_config,
                kernels["composite"],
                buf.handle,
                np.int32(WIDTH),
                np.int32(HEIGHT),
                np.uint64(mip_tex.handle),
                np.float32(state["strength"]),
                np.float32(state["threshold"]),
                np.int32(state["num_lods"]),  # # of bloom LODs summed (max-clamp)
                np.float32(state["bias"]),  # mipmap_level_bias folded into tex2DLod
                np.float32(max_clamp),  # max_mipmap_level_clamp ceiling
                np.int32(1 if state["bloom_on"] else 0),
            )
        # Unmap happens automatically when the `with` block exits.

        copy_pbo_to_texture(gl, pbo_id, tex_id, WIDTH, HEIGHT)
        draw_fullscreen_quad(gl, shader_prog, quad_vao, tex_id)

        frame_count[0] += 1
        now = time.monotonic()
        if now - fps_time[0] >= 1.0:
            fps = frame_count[0] / (now - fps_time[0])
            window.set_caption(
                f"GPU mip-pyramid bloom ({WIDTH}x{HEIGHT}, {fps:.0f} FPS) | "
                f"MipmappedArray[{num_levels} lvls] + tex2DLod[trilinear, "
                f"bias={state['bias']:+.2f}, "
                f"clamp={state['min_clamp']:.0f}..{max_clamp:.0f}, "
                f"lods={state['num_lods']}] | "
                f"bloom={state['strength']:.2f} "
                f"thr={state['threshold']:.2f} "
                f"{'ON' if state['bloom_on'] else 'OFF'}"
            )
            frame_count[0] = 0
            fps_time[0] = now

    @window.event
    def on_key_press(symbol, _modifiers):
        key = pyglet.window.key
        if symbol == key.ESCAPE:
            window.close()
        elif symbol in (key.PLUS, key.EQUAL, key.NUM_ADD):
            state["strength"] = min(8.0, state["strength"] + BLOOM_STRENGTH_STEP)
        elif symbol in (key.MINUS, key.NUM_SUBTRACT):
            state["strength"] = max(0.0, state["strength"] - BLOOM_STRENGTH_STEP)
        elif symbol == key.BRACKETLEFT:
            state["threshold"] = max(0.0, state["threshold"] - BLOOM_THRESHOLD_STEP)
        elif symbol == key.BRACKETRIGHT:
            state["threshold"] = min(4.0, state["threshold"] + BLOOM_THRESHOLD_STEP)
        elif symbol == key.COMMA:
            state["bias"] = max(-float(num_levels - 1), state["bias"] - 0.25)
        elif symbol == key.PERIOD:
            state["bias"] = min(float(num_levels - 1), state["bias"] + 0.25)
        elif symbol == key.SEMICOLON:
            state["num_lods"] = max(1, state["num_lods"] - 1)
        elif symbol == key.APOSTROPHE:
            state["num_lods"] = min(num_levels - 1, state["num_lods"] + 1)
        elif symbol == key.B:
            state["bloom_on"] = not state["bloom_on"]
        elif symbol == key.R:
            state["strength"] = 1.8
            state["threshold"] = 0.6
            state["bloom_on"] = True
            state["bias"] = 0.5
            state["num_lods"] = max(1, num_levels - 1)

    @window.event
    def on_close():
        # Release CUDA-side resources in reverse construction order. GL objects
        # clean up via pyglet on window close. `mm` is closed LAST because the
        # per-level surfaces/textures reference its (non-owning) level views.
        resource.close()
        mip_tex.close()
        for tex in level_textures:
            tex.close()
        for surf in level_surfaces:
            surf.close()
        mm.close()
        stream.close()

    pyglet.app.run(interval=0)


def make_level_grid_screen(block):
    """2D launch grid covering the WIDTH x HEIGHT screen."""
    return (
        (WIDTH + block[0] - 1) // block[0],
        (HEIGHT + block[1] - 1) // block[1],
        1,
    )


# ======================== GPU code (CUDA + GLSL) ============================
#
# Three CUDA kernels are concatenated into one program string so they share a
# single NVRTC compile. All three operate on float4 RGBA pixels.
#
#   render_scene -- writes an animated scene of moving bright emitters into mip
#                   level 0 via a SurfaceObject. Hot cores exceed 1.0 so the
#                   bloom has something to bleed. NOTE: surf2Dwrite's x is in
#                   BYTES, so we multiply by sizeof(float4) (= 16).
#
#   downsample   -- reads level L-1 through a LINEAR TextureObject and writes
#                   level L through a SurfaceObject. With LINEAR filtering and
#                   non-normalized coords, ONE tap at the midpoint of the
#                   parent's 2x2 footprint -- (2x + 1.0, 2y + 1.0) -- equals the
#                   4-texel box average. (A POINT-sampled +0.5 offset would be
#                   a single texel, NOT the average; the +1.0 midpoint is the
#                   crux of this example.)
#
#   composite    -- samples the WHOLE pyramid through one mipmapped texture.
#                   tex2DLod(...,0) is the sharp scene; a weighted sum of
#                   tex2DLod(...,lod) for lod 1..maxLod is the blurred glow.
#                   We threshold the glow's luminance, scale by `strength`,
#                   add the sharp scene, tonemap with 1-exp(-x), write RGBA8.
#
# GLSL shaders at the very bottom just draw a textured quad. Nothing CUDA-
# specific there.
#
# ============================================================================

KERNEL_SOURCE = r"""
__device__ __forceinline__ float clampf(float v, float a, float b) {
    return fminf(fmaxf(v, a), b);
}

__device__ __forceinline__ float luminance(float4 c) {
    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

// --------------------------------------------------------------------------
// render_scene: animated bright emitters on a dark background -> level 0.
//
// `surf` is a SurfaceObject bound to mip level 0 (float4 RGBA). Each emitter
// orbits the center and contributes a sharp colored core whose intensity can
// exceed 1.0, giving the bloom pass something to bleed.
// --------------------------------------------------------------------------
extern "C" __global__
void render_scene(cudaSurfaceObject_t surf, int width, int height,
                  float t, int num_emitters) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float u = ((float)x + 0.5f) / (float)width;
    float v = ((float)y + 0.5f) / (float)height;

    // Faint moving background wash so the frame is never fully black.
    float bg = 0.04f + 0.02f * sinf(6.2831853f * (u + v) + t * 0.5f);
    float3 color = make_float3(bg * 0.4f, bg * 0.5f, bg * 0.9f);

    // Accumulate emitters: each orbits the center on its own radius/phase.
    for (int i = 0; i < num_emitters; ++i) {
        float fi = (float)i;
        float phase = t * (0.4f + 0.12f * fi) + fi * 2.3998f;  // golden-ish spread
        float radius = 0.18f + 0.06f * fi / fmaxf(1.0f, (float)(num_emitters - 1));
        float ex = 0.5f + radius * cosf(phase);
        float ey = 0.5f + radius * sinf(phase * 1.13f);

        float dx = u - ex;
        float dy = v - ey;
        float d2 = dx * dx + dy * dy;

        // Tight bright core (Gaussian) plus a gentle per-emitter pulse so the
        // HDR peak breathes and the bloom halo visibly swells. 1/sigma^2 sets
        // the core size; the smaller multiplier here widens the hot spot a bit
        // so coarse LODs pick up plenty of energy to bleed.
        float pulse = 0.75f + 0.25f * sinf(t * (1.3f + 0.17f * fi) + fi);
        float core = expf(-d2 * 3200.0f);
        float hot = 3.0f * pulse * core;  // peak well above 1.0 -> blooms strongly

        // Per-emitter hue cycling through R/G/B-ish triplets.
        float hue = fi * 1.0471975f + t * 0.2f;  // 60 deg steps + slow drift
        float3 tint = make_float3(
            0.5f + 0.5f * sinf(hue),
            0.5f + 0.5f * sinf(hue + 2.0943951f),
            0.5f + 0.5f * sinf(hue + 4.1887902f));

        color.x += hot * tint.x;
        color.y += hot * tint.y;
        color.z += hot * tint.z;
    }

    float4 px = make_float4(color.x, color.y, color.z, 1.0f);

    // surf2Dwrite indexes x in BYTES: float4 is 16 bytes.
    surf2Dwrite<float4>(px, surf, x * (int)sizeof(float4), y);
}

// --------------------------------------------------------------------------
// downsample: halve the parent level into this level via a single LINEAR tap.
//
// `src` is a LINEAR-filtered TextureObject bound to the parent level (L-1).
// `dst` is a SurfaceObject bound to this level (L). dst_size is L's edge.
//
// With non-normalized coords, tex2D returns texel (i,j) when sampled at
// (i+0.5, j+0.5). For output texel (x,y) the parent 2x2 footprint covers
// parent texels (2x,2y), (2x+1,2y), (2x,2y+1), (2x+1,2y+1). The midpoint of
// those four centers is (2x+1.0, 2y+1.0); LINEAR filtering there blends all
// four at weight 0.25 each -- exactly the box average. (NOT +0.5, which would
// land on one texel center and return a single texel.)
// --------------------------------------------------------------------------
extern "C" __global__
void downsample(cudaTextureObject_t src,
                cudaSurfaceObject_t dst,
                int dst_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_size || y >= dst_size) return;

    float fx = 2.0f * (float)x + 1.0f;
    float fy = 2.0f * (float)y + 1.0f;

    float4 px = tex2D<float4>(src, fx, fy);

    surf2Dwrite<float4>(px, dst, x * (int)sizeof(float4), y);
}

// --------------------------------------------------------------------------
// composite: sharp scene + multi-LOD bloom, tonemapped, into the PBO.
//
// `mip_tex` is ONE mipmapped TextureObject over the whole pyramid. tex2DLod at
// lod 0 is the sharp scene; lods 1..max_lod are progressively blurrier copies
// that form the glow. We threshold each blurred sample's luminance so only the
// bright parts bloom, weight coarser (wider) levels a bit less, scale by
// `strength`, add the sharp scene, and tonemap.
// --------------------------------------------------------------------------
extern "C" __global__
void composite(unsigned char *output,
               int width,
               int height,
               cudaTextureObject_t mip_tex,
               float strength,
               float threshold,
               int num_lods,
               float bias,
               float max_lod,
               int bloom_on) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float u = ((float)x + 0.5f) / (float)width;
    float v = ((float)y + 0.5f) / (float)height;

    // Sharp scene from the base level. The base sample stays at lod 0 -- bias is
    // applied only to the bloom taps below, so the scene never blurs.
    float4 scene = tex2DLod<float4>(mip_tex, u, v, 0.0f);
    float3 hdr = make_float3(scene.x, scene.y, scene.z);

    if (bloom_on) {
        // Sum the blurred levels. Each coarser level covers a wider area, so we
        // taper its weight to keep the glow soft rather than flat.
        //
        // This loop is where the live LOD-selection knobs live: `num_lods` is the
        // max-clamp (how high up the pyramid we read), and `bias` is the
        // mipmap_level_bias folded into the explicit tex2DLod `lod` argument.
        // We clamp the effective LOD to [0, max_lod] so a positive bias can never
        // index past the top of the pyramid.
        float3 bloom = make_float3(0.0f, 0.0f, 0.0f);
        float weight_sum = 0.0f;
        for (int lod = 1; lod <= num_lods; ++lod) {
            float eff_lod = clampf((float)lod + bias, 0.0f, max_lod);
            float4 s = tex2DLod<float4>(mip_tex, u, v, eff_lod);
            // Soft-knee threshold: keep only the energy above `threshold`.
            float lum = luminance(s);
            float excess = fmaxf(lum - threshold, 0.0f);
            float keep = (lum > 1e-4f) ? (excess / lum) : 0.0f;

            float w = 1.0f / (float)lod;  // finer blurred levels weigh more
            bloom.x += w * keep * s.x;
            bloom.y += w * keep * s.y;
            bloom.z += w * keep * s.z;
            weight_sum += w;
        }
        if (weight_sum > 0.0f) {
            float inv = strength / weight_sum;
            hdr.x += bloom.x * inv;
            hdr.y += bloom.y * inv;
            hdr.z += bloom.z * inv;
        }
    }

    // Tonemap HDR -> [0,1] with a simple exposure curve, then to 8-bit.
    float r = 1.0f - expf(-hdr.x);
    float g = 1.0f - expf(-hdr.y);
    float b = 1.0f - expf(-hdr.z);

    int idx = (y * width + x) * 4;
    output[idx + 0] = (unsigned char)(clampf(r, 0.0f, 1.0f) * 255.0f);
    output[idx + 1] = (unsigned char)(clampf(g, 0.0f, 1.0f) * 255.0f);
    output[idx + 2] = (unsigned char)(clampf(b, 0.0f, 1.0f) * 255.0f);
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
