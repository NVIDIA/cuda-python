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
import pytest
from cuda.core import (
    Buffer,
    Device,
    GraphicsResource,
    StridedMemoryView,
)

# ---------------------------------------------------------------------------
# GL context + buffer helpers
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _gl_context_and_buffer(nbytes=1024):
    """
    Create a hidden GL context and a GL buffer of *nbytes* bytes.
    Yields ``(gl_buffer_name, nbytes)`` or skips if GL is unavailable.
    """
    pyglet = pytest.importorskip("pyglet")

    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        if ctypes.util.find_library("EGL") is None:
            pytest.skip("No DISPLAY and no EGL runtime available for headless context.")
        pyglet.options["headless"] = True

    win = None
    buf_id = None
    try:
        if not pyglet.options.get("headless"):
            from pyglet import gl

            config = gl.Config(double_buffer=False)
            win = pyglet.window.Window(visible=False, config=config)
            win.switch_to()
        else:
            from pyglet.gl import headless  # noqa: F401

        from pyglet.gl import gl as _gl

        buf_id = _gl.GLuint(0)
        _gl.glGenBuffers(1, ctypes.byref(buf_id))
        _gl.glBindBuffer(_gl.GL_ARRAY_BUFFER, buf_id.value)
        _gl.glBufferData(_gl.GL_ARRAY_BUFFER, nbytes, None, _gl.GL_DYNAMIC_DRAW)

        yield int(buf_id.value), nbytes

    except Exception as e:
        pytest.skip(f"Could not create GL context/buffer: {type(e).__name__}: {e}")
    finally:
        try:
            from pyglet.gl import gl as _gl

            if buf_id is not None and buf_id.value:
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
    """
    Create a hidden GL context and a GL texture.
    Yields ``(tex_id, tex_target)``.
    """
    pyglet = pytest.importorskip("pyglet")

    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        if ctypes.util.find_library("EGL") is None:
            pytest.skip("No DISPLAY and no EGL runtime available for headless context.")
        pyglet.options["headless"] = True

    win = None
    tex_id = None
    try:
        if not pyglet.options.get("headless"):
            from pyglet import gl

            config = gl.Config(double_buffer=False)
            win = pyglet.window.Window(visible=False, config=config)
            win.switch_to()
        else:
            from pyglet.gl import headless  # noqa: F401

        from pyglet.gl import gl as _gl

        tex_id = _gl.GLuint(0)
        _gl.glGenTextures(1, ctypes.byref(tex_id))
        target = _gl.GL_TEXTURE_2D
        _gl.glBindTexture(target, tex_id.value)
        _gl.glTexParameteri(target, _gl.GL_TEXTURE_MIN_FILTER, _gl.GL_NEAREST)
        _gl.glTexParameteri(target, _gl.GL_TEXTURE_MAG_FILTER, _gl.GL_NEAREST)
        _gl.glTexImage2D(
            target,
            0,
            _gl.GL_RGBA8,
            width,
            height,
            0,
            _gl.GL_RGBA,
            _gl.GL_UNSIGNED_BYTE,
            None,
        )

        yield int(tex_id.value), int(target)

    except Exception as e:
        pytest.skip(f"Could not create GL context/texture: {type(e).__name__}: {e}")
    finally:
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


# ---------------------------------------------------------------------------
# Register flags parsing tests
# ---------------------------------------------------------------------------


class TestRegisterFlags:
    def test_parse_none(self):
        from cuda.core._graphics import _parse_register_flags

        assert _parse_register_flags(None) == 0

    def test_parse_single_string(self):
        from cuda.core._graphics import _parse_register_flags

        assert _parse_register_flags("read_only") == 1
        assert _parse_register_flags("write_discard") == 2

    def test_parse_combined_flags(self):
        from cuda.core._graphics import _parse_register_flags

        result = _parse_register_flags(("surface_load_store", "read_only"))
        assert result == 4 | 1

    def test_parse_invalid_raises(self):
        from cuda.core._graphics import _parse_register_flags

        with pytest.raises(ValueError, match="Unknown register flag"):
            _parse_register_flags("bogus")


# ---------------------------------------------------------------------------
# GraphicsResource instantiation guard
# ---------------------------------------------------------------------------


class TestGraphicsResourceInit:
    def test_direct_init_raises(self):
        with pytest.raises(RuntimeError, match="cannot be instantiated directly"):
            GraphicsResource()


# ---------------------------------------------------------------------------
# GL buffer registration tests
# ---------------------------------------------------------------------------


class TestFromGLBuffer:
    def test_register_default_flags(self):
        with _gl_context_and_buffer() as (gl_buf, nbytes):
            resource = GraphicsResource.from_gl_buffer(gl_buf)
            assert resource.handle != 0
            assert not resource.is_mapped
            resource.close()

    def test_register_write_discard(self):
        with _gl_context_and_buffer() as (gl_buf, nbytes):
            resource = GraphicsResource.from_gl_buffer(gl_buf, flags="write_discard")
            assert resource.handle != 0
            resource.close()

    def test_close_is_idempotent(self):
        with _gl_context_and_buffer() as (gl_buf, nbytes):
            resource = GraphicsResource.from_gl_buffer(gl_buf)
            resource.close()
            resource.close()  # Should not raise


# ---------------------------------------------------------------------------
# GL image registration tests
# ---------------------------------------------------------------------------


class TestFromGLImage:
    def test_register_image(self):
        with _gl_context_and_texture() as (tex_id, target):
            resource = GraphicsResource.from_gl_image(tex_id, target)
            assert resource.handle != 0
            assert not resource.is_mapped
            resource.close()


# ---------------------------------------------------------------------------
# Map / unmap tests
# ---------------------------------------------------------------------------


class TestMapUnmap:
    def test_map_returns_buffer(self):
        with _gl_context_and_buffer(nbytes=4096) as (gl_buf, nbytes):
            resource = GraphicsResource.from_gl_buffer(gl_buf, flags="write_discard")
            mapped = resource.map()
            assert resource.is_mapped
            # mapped is a _MappedBufferContext; its .handle and .size delegate to Buffer
            assert mapped.size > 0
            assert mapped.handle != 0
            resource.unmap()
            assert not resource.is_mapped
            resource.close()

    def test_context_manager_unmaps(self):
        with _gl_context_and_buffer(nbytes=4096) as (gl_buf, nbytes):
            resource = GraphicsResource.from_gl_buffer(gl_buf, flags="write_discard")
            with resource.map() as buf:
                assert isinstance(buf, Buffer)
                assert resource.is_mapped
                assert buf.size > 0
            assert not resource.is_mapped
            resource.close()

    def test_context_manager_unmaps_on_exception(self):
        with _gl_context_and_buffer(nbytes=4096) as (gl_buf, nbytes):
            resource = GraphicsResource.from_gl_buffer(gl_buf, flags="write_discard")
            with pytest.raises(ValueError, match="test error"), resource.map() as _buf:
                assert resource.is_mapped
                raise ValueError("test error")
            # Must be unmapped even after exception
            assert not resource.is_mapped
            resource.close()

    def test_strided_memory_view_from_mapped_buffer(self):
        """End-to-end: register, map, create StridedMemoryView."""
        nbytes = 256 * 4  # 256 float32 elements
        with _gl_context_and_buffer(nbytes=nbytes) as (gl_buf, _):
            resource = GraphicsResource.from_gl_buffer(gl_buf, flags="write_discard")
            with resource.map() as buf:
                view = StridedMemoryView.from_buffer(buf, shape=(256,), dtype=np.float32)
                assert view.ptr == int(buf.handle)
                assert view.shape == (256,)
                assert view.is_device_accessible
            resource.close()

    def test_map_with_stream(self):
        with _gl_context_and_buffer(nbytes=4096) as (gl_buf, nbytes):
            dev = Device(0)
            dev.set_current()
            stream = dev.create_stream()
            resource = GraphicsResource.from_gl_buffer(gl_buf, flags="write_discard")
            with resource.map(stream=stream) as buf:
                assert buf.size > 0
            resource.close()


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_double_map_raises(self):
        with _gl_context_and_buffer() as (gl_buf, nbytes):
            resource = GraphicsResource.from_gl_buffer(gl_buf)
            resource.map()
            with pytest.raises(RuntimeError, match="already mapped"):
                resource.map()
            resource.unmap()
            resource.close()

    def test_unmap_without_map_raises(self):
        with _gl_context_and_buffer() as (gl_buf, nbytes):
            resource = GraphicsResource.from_gl_buffer(gl_buf)
            with pytest.raises(RuntimeError, match="not mapped"):
                resource.unmap()
            resource.close()

    def test_map_after_close_raises(self):
        with _gl_context_and_buffer() as (gl_buf, nbytes):
            resource = GraphicsResource.from_gl_buffer(gl_buf)
            resource.close()
            with pytest.raises(RuntimeError, match="has been closed"):
                resource.map()

    def test_unmap_after_close_raises(self):
        with _gl_context_and_buffer() as (gl_buf, nbytes):
            resource = GraphicsResource.from_gl_buffer(gl_buf)
            resource.close()
            with pytest.raises(RuntimeError, match="has been closed"):
                resource.unmap()

    def test_close_while_mapped(self):
        """close() should unmap before unregistering."""
        with _gl_context_and_buffer() as (gl_buf, nbytes):
            resource = GraphicsResource.from_gl_buffer(gl_buf, flags="write_discard")
            resource.map()
            assert resource.is_mapped
            resource.close()  # Should unmap + unregister without error
            assert not resource.is_mapped


# ---------------------------------------------------------------------------
# GC / repr tests
# ---------------------------------------------------------------------------


class TestMisc:
    def test_gc_cleanup(self):
        """Creating and dropping a resource should not leak."""
        with _gl_context_and_buffer() as (gl_buf, nbytes):
            resource = GraphicsResource.from_gl_buffer(gl_buf)
            assert resource.handle != 0
            del resource
            gc.collect()
            # If we get here without a CUDA error, cleanup succeeded.

    def test_repr(self):
        with _gl_context_and_buffer() as (gl_buf, nbytes):
            resource = GraphicsResource.from_gl_buffer(gl_buf)
            r = repr(resource)
            assert "GraphicsResource" in r
            assert "0x" in r
            resource.close()

    def test_repr_closed(self):
        with _gl_context_and_buffer() as (gl_buf, nbytes):
            resource = GraphicsResource.from_gl_buffer(gl_buf)
            resource.close()
            r = repr(resource)
            assert "closed" in r
