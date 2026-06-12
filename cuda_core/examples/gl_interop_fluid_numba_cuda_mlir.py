# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# ################################################################################
#
# A numba-cuda port of gl_interop_fluid.py (Jos Stam "Stable Fluids").
#
# The original example is a cuda.core showcase: each field is a `cudaArray`
# bound both as a TextureObject (cached, hardware-filtered READS) and a
# SurfaceObject (raw WRITES), and semi-Lagrangian advection is a single
# `tex2D<float2>` LINEAR fetch at a fractional back-traced coordinate.
#
# numba-cuda has NO texture or surface support (no tex2D, no surf2Dwrite, no
# cudaArray binding). So this port shows what the SAME solver looks like when
# every field is an ordinary linear device array and the hardware bilinear
# filter is written by hand (see `sample_*` below). The physics is identical;
# only the memory model and the read path change.
#
# What changes vs. the cuda.core version
# =======================================
#   cuda.core / CUDA C++                  ->  numba-cuda (this file)
#   --------------------------------------    -----------------------------------
#   CUDAArray(num_channels=2) as texture  ->  cuda.device_array((H, W, 2), f32)
#   tex2D<float2>(tex, u, v)  [HW LINEAR] ->  sample_vec(arr, px, py)  [manual lerp]
#   AddressMode.CLAMP                     ->  index clamp inside sample_*()
#   surf2Dwrite(v, surf, x*8, y)          ->  arr[y, x, 0] = v.x; arr[y, x, 1] = v.y
#   TextureObject + SurfaceObject pair    ->  one device array; ping-pong by swap
#   GraphicsResource PBO (zero-copy)      ->  copy_to_host + glTexSubImage2D
#
# This file is intentionally self-contained: numba-cuda + pyglet + numpy only.
# It does not import cuda.core. Run it next to gl_interop_fluid.py to compare.
#
# /// script
# dependencies = ["numba-cuda-mlir", "pyglet", "numpy"]
# ///

import colorsys
import ctypes
import math
import random
import sys
import time

import numpy as np

# This port targets the MLIR-based numba-cuda backend (`numba-cuda-mlir`), which
# tracks the current cuda.bindings / cuda.core API. It exposes the same
# `cuda.jit` / `cuda.grid` / device-array surface as stock `numba.cuda`, so the
# kernels below are unchanged. Fall back to stock `numba.cuda` for environments
# that only ship the classic backend.
try:
    from numba_cuda_mlir import cuda
except ImportError:
    from numba import cuda

# --------------------------------------------------------------------------- #
# Simulation parameters (same values as gl_interop_fluid.py)
# --------------------------------------------------------------------------- #
WIDTH = 512
HEIGHT = 512
DT = 1.0
PRESSURE_ITERS = 30
VELOCITY_DISSIPATION = 0.999
DYE_DISSIPATION = 0.994
SPLAT_RADIUS = 24.0
SPLAT_FORCE = 6.0
SPLAT_DYE = 1.0
CURL_SEED = 2.5
VORTICITY = 0.28

AUTO_EMIT = True
BURST_INTERVAL = 0.45
BURSTS_PER_EVENT = 2
BURST_RADIUS = 42.0
BURST_FORCE = 18.0
BURST_DYE = 1.2

REF_FPS = 60.0

# =============================== Device helpers ============================= #
#
# These replace the texture unit. A cudaArray bound LINEAR + CLAMP + normalized
# turns one `tex2D` call into a hardware bilinear read with edge clamping. With
# linear device memory we do the same arithmetic explicitly: locate the four
# texel centers around the (fractional) sample point, clamp each to the edge,
# and blend. Sampling at an integer pixel center returns the stored value
# exactly, which is why the stencil kernels (divergence/jacobi/gradient) can use
# the same sampler the advection kernels use.


@cuda.jit(device=True, inline=True)
def _clampi(i, n):
    # AddressMode.CLAMP: out-of-range coordinates read the border texel.
    if i < 0:
        return 0
    if i > n - 1:
        return n - 1
    return i


@cuda.jit(device=True, inline=True)
def _bilinear_setup(px, py, w, h):
    # Shared front half of every sampler: pixel-center coords -> (corner
    # indices, fractional weights). px/py are in pixel space where an integer
    # value addresses a texel center, matching the (i + 0.5)/N convention the
    # C++ version feeds to tex2D.
    x0 = int(math.floor(px))
    y0 = int(math.floor(py))
    fx = px - x0
    fy = py - y0
    x0c = _clampi(x0, w)
    x1c = _clampi(x0 + 1, w)
    y0c = _clampi(y0, h)
    y1c = _clampi(y0 + 1, h)
    return x0c, x1c, y0c, y1c, fx, fy


@cuda.jit(device=True, inline=True)
def sample_scalar(fld, px, py, w, h):
    # Equivalent of tex2D<float>(tex, u, v) with LINEAR + CLAMP.
    x0c, x1c, y0c, y1c, fx, fy = _bilinear_setup(px, py, w, h)
    top = fld[y0c, x0c] * (1.0 - fx) + fld[y0c, x1c] * fx
    bot = fld[y1c, x0c] * (1.0 - fx) + fld[y1c, x1c] * fx
    return top * (1.0 - fy) + bot * fy


@cuda.jit(device=True, inline=True)
def sample_vec(fld, px, py, w, h):
    # Equivalent of tex2D<float2>(tex, u, v). Returns (vx, vy) as a tuple --
    # numba device functions can return tuples, so this reads almost like the
    # float2 the C++ version returns.
    x0c, x1c, y0c, y1c, fx, fy = _bilinear_setup(px, py, w, h)
    g = 1.0 - fx
    h0 = 1.0 - fy
    vx = (
        (fld[y0c, x0c, 0] * g + fld[y0c, x1c, 0] * fx) * h0
        + (fld[y1c, x0c, 0] * g + fld[y1c, x1c, 0] * fx) * fy
    )
    vy = (
        (fld[y0c, x0c, 1] * g + fld[y0c, x1c, 1] * fx) * h0
        + (fld[y1c, x0c, 1] * g + fld[y1c, x1c, 1] * fx) * fy
    )
    return vx, vy


@cuda.jit(device=True, inline=True)
def sample_color(fld, px, py, w, h):
    # Equivalent of tex2D<float4>(tex, u, v). Returns (r, g, b, a).
    x0c, x1c, y0c, y1c, fx, fy = _bilinear_setup(px, py, w, h)
    g = 1.0 - fx
    h0 = 1.0 - fy
    out0 = (fld[y0c, x0c, 0] * g + fld[y0c, x1c, 0] * fx) * h0 + (
        fld[y1c, x0c, 0] * g + fld[y1c, x1c, 0] * fx
    ) * fy
    out1 = (fld[y0c, x0c, 1] * g + fld[y0c, x1c, 1] * fx) * h0 + (
        fld[y1c, x0c, 1] * g + fld[y1c, x1c, 1] * fx
    ) * fy
    out2 = (fld[y0c, x0c, 2] * g + fld[y0c, x1c, 2] * fx) * h0 + (
        fld[y1c, x0c, 2] * g + fld[y1c, x1c, 2] * fx
    ) * fy
    out3 = (fld[y0c, x0c, 3] * g + fld[y0c, x1c, 3] * fx) * h0 + (
        fld[y1c, x0c, 3] * g + fld[y1c, x1c, 3] * fx
    ) * fy
    return out0, out1, out2, out3


@cuda.jit(device=True, inline=True)
def curl_at(vel, px, py, w, h):
    # Scalar 2D curl w = dVy/dx - dVx/dy via central differences.
    lx, ly = sample_vec(vel, px - 1.0, py, w, h)
    rx, ry = sample_vec(vel, px + 1.0, py, w, h)
    dx_, dy_ = sample_vec(vel, px, py - 1.0, w, h)
    ux, uy = sample_vec(vel, px, py + 1.0, w, h)
    return 0.5 * ((ry - ly) - (ux - dx_))


# ================================= Kernels ================================== #
#
# One thread per cell. cuda.grid(2) returns (x, y) with x fastest, matching the
# blockIdx.x*blockDim.x+threadIdx.x layout of the C++ kernels. Arrays are
# indexed [y, x] (row-major), so writes go to fld[y, x].


@cuda.jit
def seed_field(vel, dye, prs, w, h, curl, seed):
    x, y = cuda.grid(2)
    if x >= w or y >= h:
        return

    cx = w * 0.5
    cy = h * 0.5
    rx = (x - cx) / cx
    ry = (y - cy) / cy
    vx = -ry * curl
    vy = rx * curl

    # Same integer hash as the C++ version; mask to 32 bits to emulate uint32.
    hsh = (x * 374761393 + y * 668265263 + seed * 2246822519) & 0xFFFFFFFF
    hsh = ((hsh ^ (hsh >> 13)) * 1274126177) & 0xFFFFFFFF
    hsh = (hsh ^ (hsh >> 16)) & 0xFFFFFFFF
    noise = ((hsh & 0xFFFF) / 65535.0) - 0.5

    vel[y, x, 0] = vx + noise * 0.2
    vel[y, x, 1] = vy + noise * 0.2
    dye[y, x, 0] = 0.0
    dye[y, x, 1] = 0.0
    dye[y, x, 2] = 0.0
    dye[y, x, 3] = 0.0
    prs[y, x] = 0.0


@cuda.jit
def splat(vel, dye, w, h, mx, my, fx, fy, radius, dr, dg, db, inject):
    x, y = cuda.grid(2)
    if x >= w or y >= h:
        return
    if inject == 0:
        return

    dx = x - mx
    dy = y - my
    falloff = math.exp(-(dx * dx + dy * dy) / (radius * radius))
    if falloff < 1e-3:
        return

    # In-place read-modify-write; each thread owns its own cell, so no race.
    vel[y, x, 0] += fx * falloff
    vel[y, x, 1] += fy * falloff
    dye[y, x, 0] += dr * falloff
    dye[y, x, 1] += dg * falloff
    dye[y, x, 2] += db * falloff
    dye[y, x, 3] = 1.0


@cuda.jit
def advect_velocity(vel_in, vel_out, w, h, dt, diss):
    x, y = cuda.grid(2)
    if x >= w or y >= h:
        return
    vx, vy = sample_vec(vel_in, float(x), float(y), w, h)
    # Back-trace the cell center one timestep against the local velocity.
    px = x - dt * vx
    py = y - dt * vy
    ax, ay = sample_vec(vel_in, px, py, w, h)
    vel_out[y, x, 0] = ax * diss
    vel_out[y, x, 1] = ay * diss


@cuda.jit
def vorticity_confinement(vel_in, vel_out, w, h, dt, eps):
    x, y = cuda.grid(2)
    if x >= w or y >= h:
        return
    fx = float(x)
    fy = float(y)

    wc = curl_at(vel_in, fx, fy, w, h)
    wl = curl_at(vel_in, fx - 1.0, fy, w, h)
    wr = curl_at(vel_in, fx + 1.0, fy, w, h)
    wd = curl_at(vel_in, fx, fy - 1.0, w, h)
    wu = curl_at(vel_in, fx, fy + 1.0, w, h)

    gx = 0.5 * (abs(wr) - abs(wl))
    gy = 0.5 * (abs(wu) - abs(wd))
    length = math.sqrt(gx * gx + gy * gy) + 1e-5
    nx = gx / length
    ny = gy / length

    vx, vy = sample_vec(vel_in, fx, fy, w, h)
    vel_out[y, x, 0] = vx + eps * dt * (ny * wc)
    vel_out[y, x, 1] = vy + eps * dt * (-nx * wc)


@cuda.jit
def divergence(vel, div_out, w, h):
    x, y = cuda.grid(2)
    if x >= w or y >= h:
        return
    lx, ly = sample_vec(vel, x - 1.0, float(y), w, h)
    rx, ry = sample_vec(vel, x + 1.0, float(y), w, h)
    dx_, dy_ = sample_vec(vel, float(x), y - 1.0, w, h)
    ux, uy = sample_vec(vel, float(x), y + 1.0, w, h)
    div_out[y, x] = 0.5 * ((rx - lx) + (uy - dy_))


@cuda.jit
def pressure_jacobi(prs_in, div, prs_out, w, h, clear):
    x, y = cuda.grid(2)
    if x >= w or y >= h:
        return
    pl = 0.0
    pr = 0.0
    pd = 0.0
    pu = 0.0
    if clear == 0:
        pl = sample_scalar(prs_in, x - 1.0, float(y), w, h)
        pr = sample_scalar(prs_in, x + 1.0, float(y), w, h)
        pd = sample_scalar(prs_in, float(x), y - 1.0, w, h)
        pu = sample_scalar(prs_in, float(x), y + 1.0, w, h)
    d = sample_scalar(div, float(x), float(y), w, h)
    prs_out[y, x] = (pl + pr + pd + pu - d) * 0.25


@cuda.jit
def subtract_gradient(prs, vel, w, h):
    x, y = cuda.grid(2)
    if x >= w or y >= h:
        return
    pl = sample_scalar(prs, x - 1.0, float(y), w, h)
    pr = sample_scalar(prs, x + 1.0, float(y), w, h)
    pd = sample_scalar(prs, float(x), y - 1.0, w, h)
    pu = sample_scalar(prs, float(x), y + 1.0, w, h)
    vel[y, x, 0] -= 0.5 * (pr - pl)
    vel[y, x, 1] -= 0.5 * (pu - pd)


@cuda.jit
def advect_dye(dye_in, vel, dye_out, w, h, dt, diss):
    x, y = cuda.grid(2)
    if x >= w or y >= h:
        return
    vx, vy = sample_vec(vel, float(x), float(y), w, h)
    px = x - dt * vx
    py = y - dt * vy
    r, g, b, a = sample_color(dye_in, px, py, w, h)
    dye_out[y, x, 0] = r * diss
    dye_out[y, x, 1] = g * diss
    dye_out[y, x, 2] = b * diss
    dye_out[y, x, 3] = a * diss


@cuda.jit
def colorize(dye, out, w, h):
    # Filmic 1 - exp(-c) tonemap into an RGBA8 buffer (flat, row-major).
    x, y = cuda.grid(2)
    if x >= w or y >= h:
        return
    r, g, b, a = sample_color(dye, float(x), float(y), w, h)
    gain = 1.3
    rr = 1.0 - math.exp(-max(r, 0.0) * gain)
    gg = 1.0 - math.exp(-max(g, 0.0) * gain)
    bb = 1.0 - math.exp(-max(b, 0.0) * gain)
    idx = (y * w + x) * 4
    out[idx + 0] = np.uint8(rr * 255.0)
    out[idx + 1] = np.uint8(gg * 255.0)
    out[idx + 2] = np.uint8(bb * 255.0)
    out[idx + 3] = np.uint8(255)


# ============================== Display (GL) ================================ #
#
# Pure OpenGL boilerplate, unchanged in spirit from gl_interop_fluid.py. The
# only difference: instead of a CUDA-registered PBO, the colorized frame is
# copied to host and uploaded with glTexSubImage2D. For a 512x512 demo that is
# ~1 MB/frame and not the bottleneck; see the note at the bottom of the file
# for the zero-copy upgrade.

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
void main() { fragColor = texture(tex, v_texcoord); }
"""


def create_window():
    try:
        import pyglet
        from pyglet.gl import gl as _gl
    except ImportError:
        print("This example requires pyglet >= 2.0:  pip install pyglet", file=sys.stderr)
        sys.exit(1)
    window = pyglet.window.Window(
        WIDTH, HEIGHT, caption="numba-cuda - Stable Fluids", vsync=False
    )
    return window, _gl, pyglet


def create_display_resources(gl):
    from pyglet.graphics.shader import Shader, ShaderProgram

    shader_prog = ShaderProgram(
        Shader(VERTEX_SHADER_SOURCE, "vertex"), Shader(FRAGMENT_SHADER_SOURCE, "fragment")
    )
    quad = np.array(
        [-1, -1, 0, 0, 1, -1, 1, 0, 1, 1, 1, 1, -1, -1, 0, 0, 1, 1, 1, 1, -1, 1, 0, 1],
        dtype=np.float32,
    )
    vao = ctypes.c_uint(0)
    gl.glGenVertexArrays(1, ctypes.byref(vao))
    gl.glBindVertexArray(vao.value)
    vbo = ctypes.c_uint(0)
    gl.glGenBuffers(1, ctypes.byref(vbo))
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo.value)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, quad.nbytes, quad.ctypes.data_as(ctypes.c_void_p), gl.GL_STATIC_DRAW)
    stride = 16
    pos = gl.glGetAttribLocation(shader_prog.id, b"position")
    gl.glEnableVertexAttribArray(pos)
    gl.glVertexAttribPointer(pos, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(0))
    tc = gl.glGetAttribLocation(shader_prog.id, b"texcoord")
    gl.glEnableVertexAttribArray(tc)
    gl.glVertexAttribPointer(tc, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(8))
    gl.glBindVertexArray(0)

    tex = ctypes.c_uint(0)
    gl.glGenTextures(1, ctypes.byref(tex))
    gl.glBindTexture(gl.GL_TEXTURE_2D, tex.value)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
    gl.glTexImage2D(
        gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, WIDTH, HEIGHT, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None
    )
    return shader_prog, vao.value, tex.value


def upload_and_draw(gl, shader_prog, vao_id, tex_id, host_rgba):
    gl.glBindTexture(gl.GL_TEXTURE_2D, tex_id)
    gl.glTexSubImage2D(
        gl.GL_TEXTURE_2D, 0, 0, 0, WIDTH, HEIGHT, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE,
        host_rgba.ctypes.data_as(ctypes.c_void_p),
    )
    gl.glUseProgram(shader_prog.id)
    gl.glBindVertexArray(vao_id)
    gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)
    gl.glBindVertexArray(0)
    gl.glUseProgram(0)


# ================================== main() ================================== #


def main():
    if not cuda.is_available():
        print("No CUDA GPU available to numba.", file=sys.stderr)
        sys.exit(1)

    window, gl, pyglet = create_window()
    shader_prog, quad_vao, tex_id = create_display_resources(gl)

    stream = cuda.stream()
    block = (16, 16)
    grid = ((WIDTH + 15) // 16, (HEIGHT + 15) // 16)

    # Fields as linear device arrays. velocity = (H, W, 2), dye = (H, W, 4),
    # pressure/divergence = (H, W). The a/b pairs ping-pong; we swap references.
    f32 = np.float32
    vel_a = cuda.device_array((HEIGHT, WIDTH, 2), f32)
    vel_b = cuda.device_array((HEIGHT, WIDTH, 2), f32)
    prs_a = cuda.device_array((HEIGHT, WIDTH), f32)
    prs_b = cuda.device_array((HEIGHT, WIDTH), f32)
    div = cuda.device_array((HEIGHT, WIDTH), f32)
    dye_a = cuda.device_array((HEIGHT, WIDTH, 4), f32)
    dye_b = cuda.device_array((HEIGHT, WIDTH, 4), f32)

    # Colorized frame: device buffer + a pinned host buffer for fast copyback.
    rgba_dev = cuda.device_array(WIDTH * HEIGHT * 4, np.uint8)
    rgba_host = cuda.pinned_array(WIDTH * HEIGHT * 4, np.uint8)

    fields = {"vel": [vel_a, vel_b], "prs": [prs_a, prs_b], "dye": [dye_a, dye_b]}

    seed_field[grid, block, stream](vel_a, dye_a, prs_a, WIDTH, HEIGHT, CURL_SEED, 0)
    stream.synchronize()

    mouse = {"down": False, "x": 0.0, "y": 0.0, "dx": 0.0, "dy": 0.0}
    state = {"seed": 0, "next_burst": 0.0}
    start_time = time.monotonic()
    clock = {"last": start_time}
    frame = {"n": 0, "t": start_time}

    def w2s(x, y):
        return float(x), float(HEIGHT - 1 - y)

    @window.event
    def on_key_press(symbol, _mods):
        key = pyglet.window.key
        if symbol == key.ESCAPE:
            window.close()
        elif symbol == key.R:
            state["seed"] += 1
            seed_field[grid, block, stream](
                fields["vel"][0], fields["dye"][0], fields["prs"][0], WIDTH, HEIGHT, CURL_SEED, state["seed"]
            )

    @window.event
    def on_mouse_press(x, y, *_):
        mouse["down"] = True
        mouse["x"], mouse["y"] = w2s(x, y)
        mouse["dx"] = mouse["dy"] = 0.0

    @window.event
    def on_mouse_release(*_):
        mouse["down"] = False
        mouse["dx"] = mouse["dy"] = 0.0

    @window.event
    def on_mouse_drag(x, y, dx, dy, *_):
        mouse["down"] = True
        mouse["x"], mouse["y"] = w2s(x, y)
        mouse["dx"] = float(dx)
        mouse["dy"] = float(-dy)

    @window.event
    def on_draw():
        window.clear()
        now_t = time.monotonic()
        elapsed = now_t - start_time
        dt_real = now_t - clock["last"]
        clock["last"] = now_t
        step = min(max(dt_real * REF_FPS, 0.0), 3.0)
        dt_adv = DT * step
        vel_diss = VELOCITY_DISSIPATION**step
        dye_diss = DYE_DISSIPATION**step

        vel, dye, prs = fields["vel"], fields["dye"], fields["prs"]

        # (a) advect velocity along itself
        advect_velocity[grid, block, stream](vel[0], vel[1], WIDTH, HEIGHT, dt_adv, vel_diss)
        vel.reverse()

        # (b) splat mouse velocity + colored dye (in-place on live buffers)
        inject = 1 if mouse["down"] else 0
        mr, mg, mb = colorsys.hsv_to_rgb((elapsed * 0.15) % 1.0, 0.85, 1.0)
        splat[grid, block, stream](
            vel[0], dye[0], WIDTH, HEIGHT, mouse["x"], mouse["y"],
            mouse["dx"] * SPLAT_FORCE, mouse["dy"] * SPLAT_FORCE, SPLAT_RADIUS,
            mr * SPLAT_DYE, mg * SPLAT_DYE, mb * SPLAT_DYE, inject,
        )

        # (b2) auto-bursts when idle
        if AUTO_EMIT and not mouse["down"] and elapsed >= state["next_burst"]:
            state["next_burst"] = elapsed + BURST_INTERVAL
            for _ in range(BURSTS_PER_EVENT):
                bx = random.uniform(0.12, 0.88) * WIDTH
                by = random.uniform(0.12, 0.88) * HEIGHT
                ang = random.uniform(0.0, 2.0 * math.pi)
                br, bg, bb = colorsys.hsv_to_rgb(random.random(), 0.9, 1.0)
                splat[grid, block, stream](
                    vel[0], dye[0], WIDTH, HEIGHT, bx, by,
                    math.cos(ang) * BURST_FORCE, math.sin(ang) * BURST_FORCE, BURST_RADIUS,
                    br * BURST_DYE, bg * BURST_DYE, bb * BURST_DYE, 1,
                )

        # (b3) vorticity confinement (ping-pong: reads neighbors)
        if VORTICITY > 0.0:
            vorticity_confinement[grid, block, stream](vel[0], vel[1], WIDTH, HEIGHT, dt_adv, VORTICITY)
            vel.reverse()

        # (c) divergence of the live velocity
        divergence[grid, block, stream](vel[0], div, WIDTH, HEIGHT)

        # (d) pressure solve: first pass clears, then Jacobi-iterate
        pressure_jacobi[grid, block, stream](prs[0], div, prs[1], WIDTH, HEIGHT, 1)
        prs.reverse()
        for _ in range(PRESSURE_ITERS - 1):
            pressure_jacobi[grid, block, stream](prs[0], div, prs[1], WIDTH, HEIGHT, 0)
            prs.reverse()

        # (e) subtract pressure gradient (in-place on live velocity)
        subtract_gradient[grid, block, stream](prs[0], vel[0], WIDTH, HEIGHT)

        # (f) advect dye along the divergence-free velocity
        advect_dye[grid, block, stream](dye[0], vel[0], dye[1], WIDTH, HEIGHT, dt_adv, dye_diss)
        dye.reverse()

        # (g) colorize -> (h) copy to host -> (i) upload + draw
        colorize[grid, block, stream](dye[0], rgba_dev, WIDTH, HEIGHT)
        rgba_dev.copy_to_host(rgba_host, stream=stream)
        stream.synchronize()
        upload_and_draw(gl, shader_prog, quad_vao, tex_id, rgba_host)

        mouse["dx"] = mouse["dy"] = 0.0

        frame["n"] += 1
        now = time.monotonic()
        if now - frame["t"] >= 1.0:
            fps = frame["n"] / (now - frame["t"])
            window.set_caption(
                f"numba-cuda - Stable Fluids ({WIDTH}x{HEIGHT}, {fps:.0f} FPS, "
                f"{PRESSURE_ITERS} pressure iters) | manual bilinear, linear device arrays"
            )
            frame["n"] = 0
            frame["t"] = now

    pyglet.app.run(interval=0)


if __name__ == "__main__":
    main()

# ============================ Notes / next steps ============================ #
#
# Zero-copy display (optional upgrade)
# ------------------------------------
# This sketch copies each frame host-side. To match the original's zero-copy
# CUDA->GL path you would register the GL PBO with CUDA and write the colorize
# kernel straight into the mapped device pointer. numba does not wrap GL interop
# itself, so you would call cuda.bindings driver functions
# (cuGraphicsGLRegisterBuffer / cuGraphicsMapResources /
# cuGraphicsResourceGetMappedPointer) and wrap the returned pointer as a numba
# DeviceNDArray via numba.cuda.cudadrv.driver.MemoryPointer. At that point
# reusing cuda.core's GraphicsResource (bridged with cuda.external_stream) is
# usually less code than reimplementing it.
#
# What is genuinely lost vs. the cuda.core/C++ version
# ----------------------------------------------------
# - Hardware bilinear filtering: sample_*() does it in software (more FLOPs).
# - The texture cache's 2D-locality optimization for the gather-heavy advection
#   and stencil reads. Linear global memory still caches via L1/L2, but the
#   texture path is purpose-built for this access pattern.
# These cost some throughput; for a 512x512 interactive demo it is unnoticeable.
