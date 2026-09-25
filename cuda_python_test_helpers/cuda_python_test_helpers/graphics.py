# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared GL helpers for graphics interop tests.

Both ``cuda_core`` and ``cuda_bindings`` graphics tests need to skip
when the GL backend cannot be made current, and that decision must not
hide real bugs in the tests' own GL allocation code. This module owns the
shared predicate so the two test suites stay in sync.

This module intentionally does **not** import ``pyglet`` at module load time:
importing ``pyglet.gl`` / ``pyglet.window`` triggers pyglet's shadow-window
creation, which fails on headless machines before the test has had a chance to
set ``pyglet.options["headless"]``.
"""

import contextlib
import ctypes


def select_headless_egl_device_for_cuda(cuda_device_ordinal: int) -> int | None:
    """Return the pyglet ``headless_device`` index matching *cuda_device_ordinal*.

    ``CUDA_VISIBLE_DEVICES`` reorders CUDA's device enumeration but has no
    effect on EGL's, so on headless multi-GPU systems EGL platform device
    ``N`` does not necessarily correspond to CUDA device ``N`` (the CUDA
    Programming Guide notes ``CUDA_VISIBLE_DEVICES`` does not select the
    default EGL device). The ``EGL_NV_device_cuda``
    extension (``eglQueryDeviceAttribEXT`` with ``EGL_CUDA_DEVICE_NV``)
    reports the CUDA ordinal each EGL device corresponds to, so it can be
    used to pick the EGL device that matches the CUDA device already
    selected via ``cuda.core``.

    Must be called after ``pyglet.options["headless"]`` is set and before a
    headless GL context is created. Returns ``None`` if the extension is
    unavailable or no EGL device reports *cuda_device_ordinal*; the caller
    should then fall back to pyglet's default (device 0).
    """
    egl_cuda_device_nv = 0x323A

    try:
        from pyglet.libs.egl import egl, eglext
        from pyglet.libs.egl.lib import link_EGL

        egl_query_device_attrib_ext = link_EGL(
            "eglQueryDeviceAttribEXT",
            egl.EGLBoolean,
            [eglext.EGLDeviceEXT, egl.EGLint, ctypes.POINTER(ctypes.c_ssize_t)],
        )

        num_devices = egl.EGLint()
        if not eglext.eglQueryDevicesEXT(0, None, ctypes.byref(num_devices)) or num_devices.value <= 0:
            return None

        devices = (eglext.EGLDeviceEXT * num_devices.value)()
        if not eglext.eglQueryDevicesEXT(num_devices.value, devices, ctypes.byref(num_devices)):
            return None

        for index in range(num_devices.value):
            queried_ordinal = ctypes.c_ssize_t(-1)
            found = egl_query_device_attrib_ext(devices[index], egl_cuda_device_nv, ctypes.byref(queried_ordinal))
            if found and queried_ordinal.value == cuda_device_ordinal:
                return index
    except Exception:
        return None

    return None


def open_gl_window():
    """Open a hidden window after the caller has configured pyglet.

    In headless mode, initialize pyglet's headless GL backend and return None.
    If window construction fails, clean up any partially-created windows and
    restore the GL context that was current before the attempt.
    """
    import pyglet

    if not pyglet.options.get("headless"):
        from pyglet import gl

        config = gl.Config(double_buffer=False)
        previous_context = gl.current_context
        previous_windows = set(pyglet.app.windows)
        try:
            win = pyglet.window.Window(visible=False, config=config)
        except Exception:
            for window in set(pyglet.app.windows) - previous_windows:
                with contextlib.suppress(Exception):
                    window.close()
            if previous_context is not None:
                with contextlib.suppress(Exception):
                    previous_context.set_current()
            raise
        try:
            win.switch_to()
        except Exception:
            with contextlib.suppress(Exception):
                win.close()
            raise
        return win

    from pyglet.gl import headless  # noqa: F401

    return None


_GL_CONTEXT_UNAVAILABLE_EXC_NAMES = frozenset(
    {
        "NoSuchDisplayException",
        "NoSuchConfigException",
        "NoSuchScreenModeException",
        "WindowException",
        "ContextException",
        # Pyglet's headless display raises MissingFunctionException when libEGL
        # exists but eglQueryDevicesEXT / eglGetPlatformDisplayEXT entry points do not.
        "MissingFunctionException",
    }
)

# pyglet raises these from pyglet/lib.py when libGL/libEGL cannot be loaded.
_PYGLET_GL_LIBRARY_IMPORT_ERRORS = frozenset(
    {
        'Library "GL" not found.',
        'Library "EGL" not found.',
    }
)


def is_gl_context_unavailable(exc: BaseException) -> bool:
    """Return True if *exc* means "no GL context could be created".

    Returns False for any other exception, so a real bug in the caller's
    own GL allocation code (e.g. a ``GLException`` from an invalid-enum GL
    call, a ``TypeError`` from wrong argument types) propagates and
    fails the test rather than being hidden as a skip.
    """
    exc_type = type(exc)
    if exc_type.__module__.startswith("pyglet") and exc_type.__name__ in _GL_CONTEXT_UNAVAILABLE_EXC_NAMES:
        return True

    # Windows CI runners may lack opengl32.dll; pyglet's WGL backend raises
    # FileNotFoundError from ctypes.windll.opengl32. On newer Python
    # (3.12+) ctypes.LibraryLoader catches that and re-raises
    # AttributeError(dll_name). Match narrowly on the dll name so a
    # different FileNotFoundError or AttributeError from our own code
    # does not match.
    if isinstance(exc, (FileNotFoundError, AttributeError)) and "opengl32" in str(exc):
        return True

    # Linux without libGL/libEGL: pyglet raises ImportError with the
    # exact messages above from pyglet/lib.py. A different ImportError
    # from our own code does not match.
    return isinstance(exc, ImportError) and str(exc) in _PYGLET_GL_LIBRARY_IMPORT_ERRORS
