# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import ctypes
import ctypes.util
import os
import sys

import pytest

from cuda.bindings import runtime as cudart


def _create_gl_texture():
    """Create a hidden/headless GL context plus a tiny texture to register.

    Returns ``(win, tex_id, target)``. Raises on any pyglet/GL failure.
    Tries:
      1) Windows: hidden WGL window (no EGL)
      2) Linux with DISPLAY/wayland: hidden window
      3) Linux headless: EGL headless if available
    """
    pyglet = pytest.importorskip("pyglet")

    # Prefer non-headless when a display is available; it's more portable and avoids EGL.
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        if ctypes.util.find_library("EGL") is None:
            pytest.skip("No DISPLAY and no EGL runtime available for headless context.")
        pyglet.options["headless"] = True

    # Create a minimal offscreen/hidden context
    win = None
    if not pyglet.options.get("headless"):
        # Hidden window path (WGL on Windows, GLX/WLS on Linux)
        from pyglet import gl

        config = gl.Config(double_buffer=False)
        win = pyglet.window.Window(visible=False, config=config)
        win.switch_to()
    else:
        # Headless EGL path; pyglet will arrange a pbuffer-like headless context
        from pyglet.gl import headless  # noqa: F401

    # Make a tiny texture so we have a real GL object to register
    from pyglet.gl import gl as _gl

    tex_id = _gl.GLuint(0)
    target = _gl.GL_TEXTURE_2D
    _gl.glGenTextures(1, ctypes.byref(tex_id))
    _gl.glBindTexture(target, tex_id.value)
    _gl.glTexParameteri(target, _gl.GL_TEXTURE_MIN_FILTER, _gl.GL_NEAREST)
    _gl.glTexParameteri(target, _gl.GL_TEXTURE_MAG_FILTER, _gl.GL_NEAREST)
    width, height = 16, 16
    _gl.glTexImage2D(target, 0, _gl.GL_RGBA8, width, height, 0, _gl.GL_RGBA, _gl.GL_UNSIGNED_BYTE, None)

    return win, tex_id, target


@contextlib.contextmanager
def _gl_context():
    """Yield a (tex_id, tex_target) with a current GL context, or skip."""
    win = None
    tex_id = None
    try:
        try:
            win, tex_id, target = _create_gl_texture()
        except Exception as e:
            # Convert any pyglet/GL creation failure into a clean skip.
            # The yield below is deliberately NOT inside this handler:
            # @contextmanager re-raises the with-body's exception at the yield,
            # so catching there would turn a failing assertion in the test into
            # a skip, and the test could never fail.
            pytest.skip(f"Could not create GL context/texture: {type(e).__name__}: {e}")

        yield int(tex_id.value), int(target)
    finally:
        # Best-effort cleanup
        try:
            from pyglet.gl import gl as _gl

            if tex_id is not None and tex_id.value:
                _gl.glDeleteTextures(1, ctypes.byref(tex_id))
        except Exception:  # noqa: S110
            pass
        try:
            if win is not None:
                win.close()
        except Exception:  # noqa: S110
            pass


@pytest.mark.agent_authored(model="claude-opus-5")
def test_gl_context_lets_body_failures_fail(monkeypatch):
    """A failing assertion inside the with-body must not be reported as a skip."""

    class FakeTexId:
        value = 7

    monkeypatch.setattr(sys.modules[__name__], "_create_gl_texture", lambda: (None, FakeTexId(), 0x0DE1))

    with pytest.raises(AssertionError, match="body failure"), _gl_context() as (tex_id, tex_target):
        assert (tex_id, tex_target) == (7, 0x0DE1)
        raise AssertionError("body failure")


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
