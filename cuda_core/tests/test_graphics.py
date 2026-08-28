# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import contextlib
import ctypes
import ctypes.util
import gc
import os
import sys

import numpy as np
import pyglet
import pytest
from cuda_python_test_helpers.graphics import is_gl_context_unavailable

from cuda.core import (
    Buffer,
    GraphicsResource,
)
from cuda.core._utils.cuda_utils import CUDAError
from cuda.core.utils import StridedMemoryView

# TODO(seberg): Maybe some of these tests can be made threadable?
pytestmark = pytest.mark.thread_unsafe(reason="OpenGL context not threadable")

# ---------------------------------------------------------------------------
# GL context + buffer/texture helpers
# ---------------------------------------------------------------------------


# cuGraphicsGLRegister{Buffer,Image} returns CUDA_ERROR_OPERATING_SYSTEM on
# environments where the driver refuses CUDA-GL interop (e.g. WSL). Treat
# that as an acceptable skip, mirroring cuda_bindings/tests/test_graphics_apis.py.
def _register_gl_buffer(gl_buf, *, flags=None, stream=None):
    try:
        return GraphicsResource.from_gl_buffer(gl_buf, flags=flags, stream=stream)
    except CUDAError as exc:
        if "CUDA_ERROR_OPERATING_SYSTEM" in str(exc):
            pytest.skip(f"CUDA-GL interop refused by driver: {exc}")
        raise


def _register_gl_image(tex_id, target):
    try:
        return GraphicsResource.from_gl_image(tex_id, target)
    except CUDAError as exc:
        if "CUDA_ERROR_OPERATING_SYSTEM" in str(exc):
            pytest.skip(f"CUDA-GL interop refused by driver: {exc}")
        raise


def _configure_pyglet_headless():
    """On headless Linux: enable EGL mode or skip if EGL is absent."""
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        if ctypes.util.find_library("EGL") is None:
            pytest.skip("No DISPLAY and no EGL runtime available for headless context.")
        pyglet.options["headless"] = True


def _open_gl_window():
    """Open a hidden window (or configure EGL headless). Returns the window or None."""
    if not pyglet.options.get("headless"):
        from pyglet import gl

        config = gl.Config(double_buffer=False)
        win = pyglet.window.Window(visible=False, config=config)
        win.switch_to()
        return win
    else:
        from pyglet.gl import headless  # noqa: F401

        return None


def _allocate_gl_buffer(win, nbytes):
    """Allocate a GL buffer. Caller must have a current GL context."""
    from pyglet.gl import gl as _gl

    buf_id = _gl.GLuint(0)
    _gl.glGenBuffers(1, ctypes.byref(buf_id))
    _gl.glBindBuffer(_gl.GL_ARRAY_BUFFER, buf_id.value)
    _gl.glBufferData(_gl.GL_ARRAY_BUFFER, nbytes, None, _gl.GL_DYNAMIC_DRAW)
    return buf_id


def _allocate_gl_texture(win, width, height):
    """Allocate a 2-D RGBA8 texture. Caller must have a current GL context."""
    from pyglet.gl import gl as _gl

    tex_id = _gl.GLuint(0)
    _gl.glGenTextures(1, ctypes.byref(tex_id))
    target = _gl.GL_TEXTURE_2D
    _gl.glBindTexture(target, tex_id.value)
    _gl.glTexParameteri(target, _gl.GL_TEXTURE_MIN_FILTER, _gl.GL_NEAREST)
    _gl.glTexParameteri(target, _gl.GL_TEXTURE_MAG_FILTER, _gl.GL_NEAREST)
    _gl.glTexImage2D(target, 0, _gl.GL_RGBA8, width, height, 0, _gl.GL_RGBA, _gl.GL_UNSIGNED_BYTE, None)
    return tex_id, target


@contextlib.contextmanager
def _gl_context_and_buffer(nbytes=1024):
    """Yield ``(gl_buffer_name, nbytes)`` with a current GL context, or skip if GL is unavailable."""
    _configure_pyglet_headless()

    try:
        win = _open_gl_window()
    except Exception as e:
        if is_gl_context_unavailable(e):
            pytest.skip(f"Could not create GL context: {type(e).__name__}: {e}")
        raise

    try:
        buf_id = _allocate_gl_buffer(win, nbytes)
        yield int(buf_id.value), nbytes
    finally:
        try:
            from pyglet.gl import gl as _gl

            if buf_id.value:
                _gl.glDeleteBuffers(1, ctypes.byref(buf_id))
        except Exception:  # noqa: S110
            pass
        try:
            if win is not None:
                win.close()
        except Exception:  # noqa: S110
            pass


@contextlib.contextmanager
def _gl_context_and_texture(width=16, height=16):
    """Yield ``(tex_id, tex_target)`` with a current GL context, or skip if GL is unavailable."""
    _configure_pyglet_headless()

    try:
        win = _open_gl_window()
    except Exception as e:
        if is_gl_context_unavailable(e):
            pytest.skip(f"Could not create GL context: {type(e).__name__}: {e}")
        raise

    try:
        tex_id, target = _allocate_gl_texture(win, width, height)
        yield int(tex_id.value), int(target)
    finally:
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


# ---------------------------------------------------------------------------
# Register flags parsing tests
# ---------------------------------------------------------------------------


def test_parse_none():
    from cuda.core._graphics import _parse_register_flags

    assert _parse_register_flags(None) == 0


def test_parse_single_string():
    from cuda.core._graphics import _parse_register_flags

    assert _parse_register_flags("read_only") == 1
    assert _parse_register_flags("write_discard") == 2


def test_parse_combined_flags():
    from cuda.core._graphics import _parse_register_flags

    result = _parse_register_flags(("surface_load_store", "read_only"))
    assert result == 4 | 1


def test_parse_invalid_raises():
    from cuda.core._graphics import _parse_register_flags

    with pytest.raises(ValueError, match="Unknown register flag"):
        _parse_register_flags("bogus")


# ---------------------------------------------------------------------------
# GraphicsResource instantiation guard
# ---------------------------------------------------------------------------


def test_direct_init_raises():
    with pytest.raises(RuntimeError, match="cannot be instantiated directly"):
        GraphicsResource()


# ---------------------------------------------------------------------------
# GL buffer registration tests
# ---------------------------------------------------------------------------


def test_register_default_flags(init_cuda):
    with _gl_context_and_buffer() as (gl_buf, nbytes):
        resource = _register_gl_buffer(gl_buf)
        assert resource.handle != 0
        assert resource.resource_handle == resource.handle
        assert not isinstance(resource, Buffer)
        assert not resource.is_mapped
        resource.close()


def test_register_write_discard(init_cuda):
    with _gl_context_and_buffer() as (gl_buf, nbytes):
        resource = _register_gl_buffer(gl_buf, flags="write_discard")
        assert resource.handle != 0
        resource.close()


def test_close_is_idempotent(init_cuda):
    with _gl_context_and_buffer() as (gl_buf, nbytes):
        resource = _register_gl_buffer(gl_buf)
        assert not resource.is_closed
        resource.close()
        assert resource.is_closed
        assert bool(resource) is True  # Preserve backward-compatible truthiness after close.
        resource.close()  # Should not raise


# ---------------------------------------------------------------------------
# GL image registration tests
# ---------------------------------------------------------------------------


def test_register_image(init_cuda):
    with _gl_context_and_texture() as (tex_id, target):
        resource = _register_gl_image(tex_id, target)
        assert resource.handle != 0
        assert not resource.is_mapped
        resource.close()


# ---------------------------------------------------------------------------
# Map / unmap tests
# ---------------------------------------------------------------------------


def test_map_returns_buffer(init_cuda):
    with _gl_context_and_buffer(nbytes=4096) as (gl_buf, nbytes):
        stream = init_cuda.create_stream()
        resource = _register_gl_buffer(gl_buf, flags="write_discard")
        mapped = resource.map(stream=stream)
        assert resource.is_mapped
        assert isinstance(mapped, Buffer)
        assert mapped is not resource
        assert mapped.size > 0
        assert mapped.handle != 0
        assert resource.handle != mapped.handle
        resource.unmap(stream=stream)
        assert mapped.handle == 0
        assert not resource.is_mapped
        resource.close()


def test_context_manager_unmaps(init_cuda):
    with _gl_context_and_buffer(nbytes=4096) as (gl_buf, nbytes):
        stream = init_cuda.create_stream()
        resource = _register_gl_buffer(gl_buf, flags="write_discard")
        with resource.map(stream=stream) as buf:
            assert isinstance(buf, Buffer)
            assert resource.is_mapped
            assert buf.size > 0
            assert buf.handle != 0
        assert buf.handle == 0
        assert not resource.is_mapped
        resource.close()


def test_context_manager_unmaps_on_exception(init_cuda):
    with _gl_context_and_buffer(nbytes=4096) as (gl_buf, nbytes):
        stream = init_cuda.create_stream()
        resource = _register_gl_buffer(gl_buf, flags="write_discard")
        with pytest.raises(ValueError, match="test error"), resource.map(stream=stream) as _buf:
            assert resource.is_mapped
            raise ValueError("test error")
        # Must be unmapped even after exception
        assert not resource.is_mapped
        resource.close()


def test_strided_memory_view_from_mapped_buffer(init_cuda):
    """End-to-end: register, map, create StridedMemoryView."""
    nbytes = 256 * 4  # 256 float32 elements
    with _gl_context_and_buffer(nbytes=nbytes) as (gl_buf, _):
        stream = init_cuda.create_stream()
        resource = _register_gl_buffer(gl_buf, flags="write_discard")
        with resource.map(stream=stream) as buf:
            view = StridedMemoryView.from_buffer(buf, shape=(256,), dtype=np.dtype(np.float32))
            assert view.ptr == int(buf.handle)
            assert view.shape == (256,)
            assert view.is_device_accessible
        resource.close()


def test_from_gl_buffer_with_stream_context_manager(init_cuda):
    """Register + auto-map via from_gl_buffer(stream=), then create StridedMemoryView."""
    nbytes = 256 * 4  # 256 float32 elements
    with _gl_context_and_buffer(nbytes=nbytes) as (gl_buf, _):
        stream = init_cuda.create_stream()
        with _register_gl_buffer(gl_buf, stream=stream) as buf:
            assert isinstance(buf, Buffer)
            assert buf.size == nbytes
            view = StridedMemoryView.from_buffer(buf, shape=(256,), dtype=np.dtype(np.float32))
            assert view.ptr == int(buf.handle)
            assert view.shape == (256,)
            assert view.is_device_accessible
        assert buf.handle == 0
        assert buf.size == 0


def test_resource_context_manager_auto_closes(init_cuda):
    with _gl_context_and_buffer(nbytes=4096) as (gl_buf, _):
        with _register_gl_buffer(gl_buf, flags="write_discard") as resource:
            assert isinstance(resource, GraphicsResource)
            assert resource.handle != 0
            assert not resource.is_mapped
        assert resource.handle == 0


def test_resource_context_manager_can_map_inside_scope(init_cuda):
    with _gl_context_and_buffer(nbytes=4096) as (gl_buf, _):
        stream = init_cuda.create_stream()
        with _register_gl_buffer(gl_buf, flags="write_discard").map(stream=stream) as buf:
            assert isinstance(buf, Buffer)
            assert buf.handle != 0


def test_chained_map_context_manager_unmaps(init_cuda):
    with _gl_context_and_buffer(nbytes=4096) as (gl_buf, _):
        stream = init_cuda.create_stream()
        with _register_gl_buffer(gl_buf, flags="write_discard").map(stream=stream) as buf:
            assert isinstance(buf, Buffer)
            assert buf.handle != 0
            assert buf.size > 0
        assert buf.handle == 0
        assert buf.size == 0


def test_map_with_stream(init_cuda):
    with _gl_context_and_buffer(nbytes=4096) as (gl_buf, nbytes):
        stream = init_cuda.create_stream()
        resource = _register_gl_buffer(gl_buf, flags="write_discard")
        with resource.map(stream=stream) as buf:
            assert buf.size > 0
        resource.close()


def test_map_requires_explicit_stream(init_cuda):
    with _gl_context_and_buffer(nbytes=4096) as (gl_buf, _):
        resource = _register_gl_buffer(gl_buf, flags="write_discard")
        try:
            with pytest.raises(TypeError, match=r"keyword-only argument"):
                resource.map()
        finally:
            resource.close()


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


def test_double_map_raises(init_cuda):
    with _gl_context_and_buffer() as (gl_buf, nbytes):
        stream = init_cuda.create_stream()
        resource = _register_gl_buffer(gl_buf)
        resource.map(stream=stream)
        with pytest.raises(RuntimeError, match="already mapped"):
            resource.map(stream=stream)
        resource.unmap()
        resource.close()


def test_unmap_without_map_raises(init_cuda):
    with _gl_context_and_buffer() as (gl_buf, nbytes):
        resource = _register_gl_buffer(gl_buf)
        with pytest.raises(RuntimeError, match="not mapped"):
            resource.unmap()
        resource.close()


def test_map_after_close_raises(init_cuda):
    with _gl_context_and_buffer() as (gl_buf, nbytes):
        stream = init_cuda.create_stream()
        resource = _register_gl_buffer(gl_buf)
        resource.close()
        with pytest.raises(RuntimeError, match="has been closed"):
            resource.map(stream=stream)


def test_unmap_after_close_raises(init_cuda):
    with _gl_context_and_buffer() as (gl_buf, nbytes):
        resource = _register_gl_buffer(gl_buf)
        resource.close()
        with pytest.raises(RuntimeError, match="has been closed"):
            resource.unmap()


def test_close_while_mapped(init_cuda):
    """close() should unmap before unregistering."""
    with _gl_context_and_buffer() as (gl_buf, nbytes):
        stream = init_cuda.create_stream()
        resource = _register_gl_buffer(gl_buf, flags="write_discard")
        buf = resource.map(stream=stream)
        assert resource.is_mapped
        resource.close()  # Should unmap + unregister without error
        assert not resource.is_mapped
        assert buf.handle == 0


def test_buffer_close_updates_resource_state(init_cuda):
    with _gl_context_and_buffer() as (gl_buf, _):
        stream = init_cuda.create_stream()
        resource = _register_gl_buffer(gl_buf, flags="write_discard")
        buf = resource.map(stream=stream)
        assert resource.is_mapped
        buf.close()
        assert not resource.is_mapped


# ---------------------------------------------------------------------------
# GC / repr tests
# ---------------------------------------------------------------------------


def test_gc_cleanup(init_cuda):
    """Creating and dropping a resource should not leak."""
    with _gl_context_and_buffer() as (gl_buf, nbytes):
        resource = _register_gl_buffer(gl_buf)
        assert resource.handle != 0
        del resource
        gc.collect()
        # If we get here without a CUDA error, cleanup succeeded.


def test_repr(init_cuda):
    with _gl_context_and_buffer() as (gl_buf, nbytes):
        resource = _register_gl_buffer(gl_buf)
        r = repr(resource)
        assert "GraphicsResource" in r
        assert "0x" in r
        resource.close()


def test_repr_closed(init_cuda):
    with _gl_context_and_buffer() as (gl_buf, nbytes):
        resource = _register_gl_buffer(gl_buf)
        resource.close()
        r = repr(resource)
        assert "closed" in r


def test_graphics_resource_is_not_a_buffer(init_cuda):
    with _gl_context_and_buffer() as (gl_buf, nbytes):
        resource = _register_gl_buffer(gl_buf)
        assert not isinstance(resource, Buffer)
        resource.close()
