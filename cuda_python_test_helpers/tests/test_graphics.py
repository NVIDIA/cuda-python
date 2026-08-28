# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from cuda_python_test_helpers.graphics import is_gl_context_unavailable


class _PygletError(Exception):
    pass


# Simulate a pyglet-namespaced exception by name.
def _make_pyglet_exc(name):
    cls = type(name, (_PygletError,), {})
    cls.__module__ = "pyglet.window"
    return cls


@pytest.mark.parametrize(
    "exc",
    [
        _make_pyglet_exc("NoSuchDisplayException")("x"),
        _make_pyglet_exc("NoSuchConfigException")("x"),
        _make_pyglet_exc("NoSuchScreenModeException")("x"),
        _make_pyglet_exc("WindowException")("x"),
        _make_pyglet_exc("ContextException")("x"),
        FileNotFoundError("Could not find module 'opengl32' (or one of its dependencies)."),
        AttributeError("opengl32"),
        ImportError('Library "GL" not found.'),
        ImportError('Library "EGL" not found.'),
    ],
)
def test_is_gl_context_unavailable_accepts_genuine(exc):
    assert is_gl_context_unavailable(exc) is True


@pytest.mark.parametrize(
    "exc",
    [
        # pyglet exception names that are not context-creation failures
        _make_pyglet_exc("GLException")("GL_INVALID_ENUM"),
        _make_pyglet_exc("ImageException")("x"),
        # Built-in exceptions that do not mention opengl32 / GL library
        TypeError("bug"),
        AttributeError("'NoneType' object has no attribute 'Config'"),
        FileNotFoundError("No such file: /tmp/missing"),
        ImportError("No module named 'foo'"),
        OSError("disk full"),
        RuntimeError("bug"),
    ],
)
def test_is_gl_context_unavailable_rejects_unrelated(exc):
    assert is_gl_context_unavailable(exc) is False
