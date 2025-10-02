# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import builtins
import sys
import types
import warnings

import pytest
from cuda.core.experimental import _linker as linker


@pytest.fixture(autouse=True)
def fresh_env(monkeypatch):
    """
    Put the module under test into a predictable state:
      - neutralize global caches,
      - provide a minimal 'driver' object,
      - make 'handle_return' a passthrough,
      - stabilize platform for deterministic messages.
    """

    class FakeDriver:
        # Something realistic but not used by the logic under test
        def cuDriverGetVersion(self):
            return 12090

    monkeypatch.setattr(linker, "_driver", None, raising=False)
    monkeypatch.setattr(linker, "_nvjitlink", None, raising=False)
    monkeypatch.setattr(linker, "_driver_ver", None, raising=False)
    monkeypatch.setattr(linker, "driver", FakeDriver(), raising=False)
    monkeypatch.setattr(linker, "handle_return", lambda x: x, raising=False)

    # Normalize platform-dependent wording (if any)
    monkeypatch.setattr(sys, "platform", "linux", raising=False)

    # Ensure a clean sys.modules slate for our synthetic packages
    for modname in list(sys.modules):
        if modname.startswith("cuda.bindings.nvjitlink") or modname == "cuda.bindings" or modname == "cuda":
            sys.modules.pop(modname, None)

    yield

    # Cleanup any stubs we added
    for modname in list(sys.modules):
        if modname.startswith("cuda.bindings.nvjitlink") or modname == "cuda.bindings" or modname == "cuda":
            sys.modules.pop(modname, None)


def _install_public_nvjitlink_stub():
    """
    Provide enough structure so that:
      - `from cuda.bindings import nvjitlink` succeeds
      - `from cuda.bindings._internal import nvjitlink as inner_nvjitlink` succeeds
    We don't care about the contents of inner_nvjitlink because tests stub
    `_nvjitlink_has_version_symbol()` directly.
    """
    # Make 'cuda' a package
    cuda_pkg = sys.modules.get("cuda") or types.ModuleType("cuda")
    cuda_pkg.__path__ = []  # mark as package
    sys.modules["cuda"] = cuda_pkg

    # Make 'cuda.bindings' a package
    bindings_pkg = sys.modules.get("cuda.bindings") or types.ModuleType("cuda.bindings")
    bindings_pkg.__path__ = []  # mark as package
    sys.modules["cuda.bindings"] = bindings_pkg

    # Public-facing nvjitlink module
    sys.modules["cuda.bindings.nvjitlink"] = types.ModuleType("cuda.bindings.nvjitlink")

    # Make 'cuda.bindings._internal' a package
    internal_pkg = sys.modules.get("cuda.bindings._internal") or types.ModuleType("cuda.bindings._internal")
    internal_pkg.__path__ = []  # mark as package
    sys.modules["cuda.bindings._internal"] = internal_pkg

    # Dummy inner nvjitlink module (imported but not actually used by tests)
    inner_nvjitlink_mod = types.ModuleType("cuda.bindings._internal.nvjitlink")
    # (optional) a no-op placeholder so attributes exist if accessed accidentally
    inner_nvjitlink_mod._inspect_function_pointer = lambda *_args, **_kw: True
    sys.modules["cuda.bindings._internal.nvjitlink"] = inner_nvjitlink_mod


def _collect_runtime_warnings(record):
    return [w for w in record if issubclass(w.category, RuntimeWarning)]


def _block_nvjitlink_import(monkeypatch):
    """Force 'from cuda.bindings import nvjitlink' to fail, regardless of sys.path."""
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        # Handle both 'from cuda.bindings import nvjitlink' and direct submodule imports
        target = "cuda.bindings.nvjitlink"
        if name == target or (name == "cuda.bindings" and fromlist and "nvjitlink" in fromlist):
            raise ModuleNotFoundError(target)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_warns_when_python_nvjitlink_missing(monkeypatch):
    """
    Case 1: 'from cuda.bindings import nvjitlink' fails -> bindings missing.
    Expect a RuntimeWarning stating that cuda.bindings.nvjitlink is not available,
    and that we fall back to cuLink* (function returns True).
    """
    # Ensure nothing is preloaded and actively block future imports.
    sys.modules.pop("cuda.bindings.nvjitlink", None)
    sys.modules.pop("cuda.bindings", None)
    sys.modules.pop("cuda", None)
    _block_nvjitlink_import(monkeypatch)

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        ret = linker._decide_nvjitlink_or_driver()

    assert ret is True
    warns = _collect_runtime_warnings(rec)
    assert len(warns) == 1
    msg = str(warns[0].message)
    assert "cuda.bindings.nvjitlink is not available" in msg
    assert "therefore the culink APIs will be used instead" in msg
    assert "recent version of cuda-bindings." in msg


def test_warns_when_nvjitlink_symbol_probe_raises(monkeypatch):
    """
    Case 2: Bindings present, but symbol probe raises RuntimeError -> 'not available'.
    Expect a RuntimeWarning mentioning 'libnvJitLink.so* is not available' and fallback.
    """
    _install_public_nvjitlink_stub()

    def raising_probe(_inner):
        raise RuntimeError("simulated: nvJitLink symbol unavailable")

    monkeypatch.setattr(linker, "_nvjitlink_has_version_symbol", raising_probe, raising=True)

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        ret = linker._decide_nvjitlink_or_driver()

    assert ret is True
    warns = _collect_runtime_warnings(rec)
    assert len(warns) == 1
    msg = str(warns[0].message)
    assert "libnvJitLink.so* is not available" in msg
    assert "cuda.bindings.nvjitlink is not usable" in msg
    assert "and the culink APIs will be used instead" in msg
    assert "recent version of nvJitLink." in msg


def test_warns_when_nvjitlink_too_old(monkeypatch):
    """
    Case 3: Bindings present, probe returns False -> 'too old (<12.3)'.
    Expect a RuntimeWarning mentioning 'too old (<12.3)' and fallback.
    """
    _install_public_nvjitlink_stub()
    monkeypatch.setattr(linker, "_nvjitlink_has_version_symbol", lambda _inner: False, raising=True)

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        ret = linker._decide_nvjitlink_or_driver()

    assert ret is True
    warns = _collect_runtime_warnings(rec)
    assert len(warns) == 1
    msg = str(warns[0].message)
    assert "libnvJitLink.so* is too old (<12.3)" in msg
    assert "cuda.bindings.nvjitlink is not usable" in msg
    assert "and the culink APIs will be used instead" in msg
    assert "recent version of nvJitLink." in msg


def test_uses_nvjitlink_when_available_and_ok(monkeypatch):
    """
    Sanity: Bindings present and probe returns True → no warning, use nvJitLink.
    """
    _install_public_nvjitlink_stub()
    monkeypatch.setattr(linker, "_nvjitlink_has_version_symbol", lambda _inner: True, raising=True)

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        ret = linker._decide_nvjitlink_or_driver()

    assert ret is False  # do NOT fall back
    warns = _collect_runtime_warnings(rec)
    assert not warns
