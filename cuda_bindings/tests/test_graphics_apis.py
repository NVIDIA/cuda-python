# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import ctypes
import ctypes.util
import os
import sys

import pyglet
import pytest

from cuda.bindings import runtime as cudart

# pyglet raises these when GL context/window creation fails. Matched by type
# name (not by class) because importing pyglet.gl / pyglet.window at
# module top triggers pyglet's shadow-window creation, which fails on
# headless machines before _configure_pyglet_headless() has set the
# headless option. GLException is intentionally excluded: pyglet
# raises it after any GL call that reports an error (GL_INVALID_ENUM,
# etc.), so catching it would hide real bugs in our own GL
# allocation code as "GL unavailable" skips.
_GL_UNAVAILABLE_EXC_NAMES = frozenset(
    {
        "NoSuchDisplayException",
        "NoSuchConfigException",
        "NoSuchScreenModeException",
        "WindowException",
        "ContextException",
    }
)


def _is_gl_unavailable(exc):
    if type(exc).__module__.startswith("pyglet") and type(exc).__name__ in _GL_UNAVAILABLE_EXC_NAMES:
        return True
    # Windows CI runners may lack opengl32.dll; pyglet's WGL backend raises
    # FileNotFoundError from ctypes.windll.opengl32. On newer Python
    # (3.12+) ctypes.LibraryLoader catches that and re-raises
    # AttributeError(dll_name). Match narrowly on the dll name so a
    # different FileNotFoundError or AttributeError from our own code
    # does not match.
    return isinstance(exc, (FileNotFoundError, AttributeError)) and "opengl32" in str(exc)


def _configure_pyglet_headless():
    """On headless Linux: enable EGL mode or skip if EGL is absent."""
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        if ctypes.util.find_library("EGL") is None:
            pytest.skip("No DISPLAY and no EGL runtime available for headless context.")
        pyglet.options["headless"] = True


def _open_gl_window():
    """Open a hidden window (or configure EGL headless). Returns the window or None."""
    if not pyglet.options.get("headless"):
        # Hidden window path (WGL on Windows, GLX/WLS on Linux)
        from pyglet import gl

        config = gl.Config(double_buffer=False)
        win = pyglet.window.Window(visible=False, config=config)
        win.switch_to()
        return win
    else:
        # Headless EGL path; pyglet will arrange a pbuffer-like headless context
        from pyglet.gl import headless  # noqa: F401

        return None


def _allocate_gl_texture(win):
    """Allocate a 2-D RGBA8 texture. Caller must have a current GL context."""
    from pyglet.gl import gl as _gl

    tex_id = _gl.GLuint(0)
    _gl.glGenTextures(1, ctypes.byref(tex_id))
    target = _gl.GL_TEXTURE_2D
    _gl.glBindTexture(target, tex_id.value)
    _gl.glTexParameteri(target, _gl.GL_TEXTURE_MIN_FILTER, _gl.GL_NEAREST)
    _gl.glTexParameteri(target, _gl.GL_TEXTURE_MAG_FILTER, _gl.GL_NEAREST)
    width, height = 16, 16
    _gl.glTexImage2D(target, 0, _gl.GL_RGBA8, width, height, 0, _gl.GL_RGBA, _gl.GL_UNSIGNED_BYTE, None)
    return tex_id, target


@contextlib.contextmanager
def _gl_context():
    """Yield ``(tex_id, tex_target)`` with a current GL context, or skip if GL is unavailable."""
    _configure_pyglet_headless()

    try:
        win = _open_gl_window()
    except Exception as e:
        if _is_gl_unavailable(e):
            pytest.skip(f"Could not create GL context: {type(e).__name__}: {e}")
        raise

    try:
        tex_id, target = _allocate_gl_texture(win)
        yield int(tex_id.value), int(target)
    finally:
        # Best-effort cleanup
        try:
            from pyglet.gl import gl as _gl

            if tex_id.value:
                _gl.glDeleteTextures(1, ctypes.byref(tex_id))
        except Exception:  # noqa: S110
            pass
        try:
            if win is not None:
                win.close()
        except Exception:  # noqa: S110
            pass


@pytest.mark.parametrize(
    "flags",
    [
        cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsNone,
        cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard,
    ],
)
def test_cuda_gl_register_image_smoketest(flags):
    with _gl_context() as (tex_id, tex_target):
        # Register
        err, resource = cudart.cudaGraphicsGLRegisterImage(tex_id, tex_target, flags)
        name = cudart.cudaGetErrorName(err)[1].decode()

        # Map error expectations by environment:
        # - success: we actually exercised the API
        # - operating-system: typical when the driver/runtime refuses interop (e.g., no GPU/driver in CI container)
        acceptable = {"cudaSuccess", "cudaErrorOperatingSystem"}

        assert name in acceptable, f"cudaGraphicsGLRegisterImage returned {name}"
        if name == "cudaSuccess":
            assert int(resource) != 0
            # Unregister to be tidy
            cudart.cudaGraphicsUnregisterResource(resource)


def test_cuda_register_image_invalid():
    """Exercise cudaGraphicsGLRegisterImage with dummy handle only using CUDA runtime API."""
    fake_gl_texture_id = 1
    fake_gl_target = 0x0DE1
    flags = cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard

    err, resource = cudart.cudaGraphicsGLRegisterImage(fake_gl_texture_id, fake_gl_target, flags)
    err_name = cudart.cudaGetErrorName(err)[1].decode()
    err_str = cudart.cudaGetErrorString(err)[1].decode()

    if err == 0:
        cudart.cudaGraphicsUnregisterResource(resource)
        raise AssertionError("Expected error from invalid GL texture ID")
