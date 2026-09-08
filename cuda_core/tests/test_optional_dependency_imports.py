# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from cuda.core import _linker, _program
from cuda.pathfinder import DynamicLibNotFoundError

# The autouse fixture below resets module-level import state for every test in this
# file, so the whole module is thread-unsafe -- not just the tests that monkeypatch.
pytestmark = pytest.mark.thread_unsafe(reason="resets cuda.core._program / _linker optional-import globals")


@pytest.fixture(autouse=True)
def restore_optional_import_state():
    saved_nvvm_module = _program._nvvm_module
    saved_nvvm_attempted = _program._nvvm_import_attempted
    saved_driver = _linker._driver
    saved_inited = _linker._inited
    saved_use_nvjitlink = _linker._use_nvjitlink_backend

    _program._nvvm_module = None
    _program._nvvm_import_attempted = False
    _linker._driver = None
    _linker._inited = False
    _linker._use_nvjitlink_backend = None

    yield

    _program._nvvm_module = saved_nvvm_module
    _program._nvvm_import_attempted = saved_nvvm_attempted
    _linker._driver = saved_driver
    _linker._inited = saved_inited
    _linker._use_nvjitlink_backend = saved_use_nvjitlink


@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_get_nvvm_module_rejects_old_bindings(monkeypatch):
    """NVVM import requires cuda-bindings >= 12.9.0 and caches a failed attempt."""
    calls = 0

    def old_binding_version():
        nonlocal calls
        calls += 1
        return (12, 8, 0)

    monkeypatch.setattr(_program, "binding_version", old_binding_version)

    with pytest.raises(RuntimeError, match="cuda-bindings >= 12.9.0"):
        _program._get_nvvm_module()
    with pytest.raises(RuntimeError, match="previous import attempt failed"):
        _program._get_nvvm_module()
    assert calls == 1


def test_get_nvvm_module_reraises_nested_module_not_found(monkeypatch):
    monkeypatch.setattr(_program, "binding_version", lambda: (12, 9, 0))

    def fake__optional_cuda_import(modname, probe_function=None):
        assert modname == "cuda.bindings.nvvm"
        assert probe_function is not None
        err = ModuleNotFoundError("No module named 'not_a_real_dependency'")
        err.name = "not_a_real_dependency"
        raise err

    monkeypatch.setattr(_program, "_optional_cuda_import", fake__optional_cuda_import)

    with pytest.raises(ModuleNotFoundError, match="not_a_real_dependency") as excinfo:
        _program._get_nvvm_module()
    assert excinfo.value.name == "not_a_real_dependency"


def test_get_nvvm_module_reports_missing_nvvm_module(monkeypatch):
    monkeypatch.setattr(_program, "binding_version", lambda: (12, 9, 0))

    def fake__optional_cuda_import(modname, probe_function=None):
        assert modname == "cuda.bindings.nvvm"
        assert probe_function is not None
        return None

    monkeypatch.setattr(_program, "_optional_cuda_import", fake__optional_cuda_import)

    with pytest.raises(RuntimeError, match="cuda.bindings.nvvm"):
        _program._get_nvvm_module()


def test_get_nvvm_module_handles_missing_libnvvm(monkeypatch):
    monkeypatch.setattr(_program, "binding_version", lambda: (12, 9, 0))

    def fake__optional_cuda_import(modname, probe_function=None):
        assert modname == "cuda.bindings.nvvm"
        assert probe_function is not None
        return None

    monkeypatch.setattr(_program, "_optional_cuda_import", fake__optional_cuda_import)

    with pytest.raises(RuntimeError, match="libnvvm"):
        _program._get_nvvm_module()


def test_decide_nvjitlink_or_driver_reraises_nested_module_not_found(monkeypatch):
    def fake__optional_cuda_import(modname, probe_function=None):
        assert modname == "cuda.bindings.nvjitlink"
        assert probe_function is None
        err = ModuleNotFoundError("No module named 'not_a_real_dependency'")
        err.name = "not_a_real_dependency"
        raise err

    monkeypatch.setattr(_linker, "_optional_cuda_import", fake__optional_cuda_import)

    with pytest.raises(ModuleNotFoundError, match="not_a_real_dependency") as excinfo:
        _linker._decide_nvjitlink_or_driver()
    assert excinfo.value.name == "not_a_real_dependency"


def test_decide_nvjitlink_or_driver_falls_back_when_module_missing(monkeypatch):
    def fake__optional_cuda_import(modname, probe_function=None):
        assert modname == "cuda.bindings.nvjitlink"
        assert probe_function is None
        return None

    monkeypatch.setattr(_linker, "_optional_cuda_import", fake__optional_cuda_import)

    with pytest.warns(RuntimeWarning, match="cuda.bindings.nvjitlink is not available"):
        use_driver_backend = _linker._decide_nvjitlink_or_driver()

    assert use_driver_backend is True
    assert _linker._use_nvjitlink_backend is False


@pytest.mark.agent_authored(model="grok-4.5")
def test_decide_nvjitlink_or_driver_falls_back_when_dylib_missing(monkeypatch):
    """Missing nvJitLink dylib must fall back via DynamicLibNotFoundError."""

    def raise_missing(_nvjitlink):
        raise DynamicLibNotFoundError("libnvJitLink missing")

    def fake__optional_cuda_import(modname, probe_function=None):
        assert modname == "cuda.bindings.nvjitlink"
        assert probe_function is None
        return object()

    monkeypatch.setattr(_linker, "_optional_cuda_import", fake__optional_cuda_import)
    monkeypatch.setattr(_linker, "_nvjitlink_has_version_symbol", raise_missing)

    with pytest.warns(RuntimeWarning, match="cuda.bindings.nvjitlink is not available"):
        use_driver_backend = _linker._decide_nvjitlink_or_driver()

    assert use_driver_backend is True
    assert _linker._use_nvjitlink_backend is False


@pytest.mark.agent_authored(model="grok-4.5")
def test_decide_nvjitlink_or_driver_falls_back_when_nvjitlink_too_old(monkeypatch):
    def fake__optional_cuda_import(modname, probe_function=None):
        assert modname == "cuda.bindings.nvjitlink"
        assert probe_function is None
        return object()

    monkeypatch.setattr(_linker, "_optional_cuda_import", fake__optional_cuda_import)
    monkeypatch.setattr(_linker, "_nvjitlink_has_version_symbol", lambda _nvjitlink: False)

    with pytest.warns(RuntimeWarning, match="too old \\(<12.3\\)"):
        use_driver_backend = _linker._decide_nvjitlink_or_driver()

    assert use_driver_backend is True
    assert _linker._use_nvjitlink_backend is False


@pytest.mark.agent_authored(model="grok-4.5")
def test_decide_nvjitlink_or_driver_selects_nvjitlink_when_version_symbol_present(monkeypatch):
    def fake__optional_cuda_import(modname, probe_function=None):
        assert modname == "cuda.bindings.nvjitlink"
        assert probe_function is None
        return object()

    monkeypatch.setattr(_linker, "_optional_cuda_import", fake__optional_cuda_import)
    monkeypatch.setattr(_linker, "_nvjitlink_has_version_symbol", lambda _nvjitlink: True)

    use_driver_backend = _linker._decide_nvjitlink_or_driver()

    assert use_driver_backend is False
    assert _linker._use_nvjitlink_backend is True


@pytest.mark.agent_authored(model="grok-4.5")
def test_decide_nvjitlink_or_driver_does_not_call_version(monkeypatch):
    """Regression guard for #2408: must not call module.version()."""
    called = {"version": False, "inspect": False}

    class FakeModule:
        def version(self):
            called["version"] = True
            raise AssertionError("module.version() must not be used for nvJitLink probing")

    def fake_has_version(_nvjitlink):
        called["inspect"] = True
        return True

    def fake__optional_cuda_import(modname, probe_function=None):
        assert modname == "cuda.bindings.nvjitlink"
        assert probe_function is None
        return FakeModule()

    monkeypatch.setattr(_linker, "_optional_cuda_import", fake__optional_cuda_import)
    monkeypatch.setattr(_linker, "_nvjitlink_has_version_symbol", fake_has_version)

    assert _linker._decide_nvjitlink_or_driver() is False
    assert called["inspect"] is True
    assert called["version"] is False
