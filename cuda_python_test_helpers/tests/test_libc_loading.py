# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The C-runtime load must not be able to break test collection.

``cuda_python_test_helpers`` is registered as a pytest plugin by
``cuda_core/tests/conftest.py``, so anything that raises at its import time
stops the whole cuda_core suite from being collected.
"""

from __future__ import annotations

import ctypes
import importlib
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import cuda_python_test_helpers as helpers


@pytest.fixture
def reimport_helpers(monkeypatch):
    """Re-execute the package with a filtered ``ctypes.CDLL``, then restore it."""

    def _reimport(blocked):
        real_cdll = ctypes.CDLL

        def fake_cdll(name, *args, **kwargs):
            if name in blocked:
                raise OSError(f"{name}: cannot open shared object file (simulated)")
            return real_cdll(name, *args, **kwargs)

        monkeypatch.setattr(ctypes, "CDLL", fake_cdll)
        return importlib.reload(helpers)

    yield _reimport
    monkeypatch.undo()
    importlib.reload(helpers)


@pytest.mark.agent_authored(model="claude-opus-5")
def test_libc_exposes_a_working_memcmp():
    assert helpers.libc.memcmp(b"ab", b"ab", 2) == 0
    assert helpers.libc.memcmp(b"ab", b"ac", 2) != 0


@pytest.mark.agent_authored(model="claude-opus-5")
@pytest.mark.skipif(helpers.IS_WINDOWS, reason="the glibc soname is not used on Windows")
def test_import_survives_without_the_glibc_soname(reimport_helpers):
    """`libc.so.6` is glibc-specific and absent on musl (Alpine) and macOS.

    It used to be loaded unconditionally on every non-Windows platform, so the
    import raised OSError there -- even though the package computes IS_LINUX
    right above and only one helper (`memcmp`, used by
    cuda_core/tests/helpers/buffers.py) needs the library at all.
    """
    reloaded = reimport_helpers({"libc.so.6"})

    assert reloaded.libc.memcmp(b"ab", b"ab", 2) == 0
    assert reloaded.libc.memcmp(b"ab", b"ac", 2) != 0


@pytest.mark.agent_authored(model="claude-opus-5")
@pytest.mark.skipif(helpers.IS_WINDOWS, reason="the glibc soname is not used on Windows")
def test_error_names_what_was_tried_when_no_c_runtime_loads(reimport_helpers, monkeypatch):
    """A genuinely unloadable C runtime must still fail, with a useful message."""
    monkeypatch.setattr(ctypes.util, "find_library", lambda _name: None)

    with pytest.raises(OSError, match=r"could not load the C runtime.*libc\.so\.6"):
        reimport_helpers({"libc.so.6"})
