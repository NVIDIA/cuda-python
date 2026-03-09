# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import types

import pytest

from cuda.core import _linker, _program


@pytest.fixture(autouse=True)
def restore_optional_import_state():
    saved_nvvm_module = _program._nvvm_module
    saved_nvvm_attempted = _program._nvvm_import_attempted
    saved_driver = _linker._driver
    saved_driver_ver = _linker._driver_ver
    saved_inited = _linker._inited
    saved_use_nvjitlink = _linker._use_nvjitlink_backend

    _program._nvvm_module = None
    _program._nvvm_import_attempted = False
    _linker._driver = None
    _linker._driver_ver = None
    _linker._inited = False
    _linker._use_nvjitlink_backend = False

    yield

    _program._nvvm_module = saved_nvvm_module
    _program._nvvm_import_attempted = saved_nvvm_attempted
    _linker._driver = saved_driver
    _linker._driver_ver = saved_driver_ver
    _linker._inited = saved_inited
    _linker._use_nvjitlink_backend = saved_use_nvjitlink


def _patch_driver_version(monkeypatch, version=13000):
    monkeypatch.setattr(
        _linker,
        "driver",
        types.SimpleNamespace(cuDriverGetVersion=lambda: version),
    )
    monkeypatch.setattr(_linker, "handle_return", lambda value: value)


def test_get_nvvm_module_reraises_nested_module_not_found(monkeypatch):
    monkeypatch.setattr(_program, "get_binding_version", lambda: (12, 9))

    def fake_optional_cuda_import(modname, probe_function=None):
        assert modname == "cuda.bindings.nvvm"
        assert probe_function is not None
        err = ModuleNotFoundError("No module named 'not_a_real_dependency'")
        err.name = "not_a_real_dependency"
        raise err

    monkeypatch.setattr(_program, "optional_cuda_import", fake_optional_cuda_import)

    with pytest.raises(ModuleNotFoundError, match="not_a_real_dependency") as excinfo:
        _program._get_nvvm_module()
    assert excinfo.value.name == "not_a_real_dependency"


def test_get_nvvm_module_reports_missing_nvvm_module(monkeypatch):
    monkeypatch.setattr(_program, "get_binding_version", lambda: (12, 9))

    def fake_optional_cuda_import(modname, probe_function=None):
        assert modname == "cuda.bindings.nvvm"
        assert probe_function is not None
        return None

    monkeypatch.setattr(_program, "optional_cuda_import", fake_optional_cuda_import)

    with pytest.raises(RuntimeError, match="cuda.bindings.nvvm"):
        _program._get_nvvm_module()


def test_get_nvvm_module_handles_missing_libnvvm(monkeypatch):
    monkeypatch.setattr(_program, "get_binding_version", lambda: (12, 9))

    def fake_optional_cuda_import(modname, probe_function=None):
        assert modname == "cuda.bindings.nvvm"
        assert probe_function is not None
        return None

    monkeypatch.setattr(_program, "optional_cuda_import", fake_optional_cuda_import)

    with pytest.raises(RuntimeError, match="libnvvm"):
        _program._get_nvvm_module()


def test_decide_nvjitlink_or_driver_reraises_nested_module_not_found(monkeypatch):
    _patch_driver_version(monkeypatch)

    def fake_optional_cuda_import(modname, probe_function=None):
        assert modname == "cuda.bindings.nvjitlink"
        assert probe_function is not None
        err = ModuleNotFoundError("No module named 'not_a_real_dependency'")
        err.name = "not_a_real_dependency"
        raise err

    monkeypatch.setattr(_linker, "optional_cuda_import", fake_optional_cuda_import)

    with pytest.raises(ModuleNotFoundError, match="not_a_real_dependency") as excinfo:
        _linker._decide_nvjitlink_or_driver()
    assert excinfo.value.name == "not_a_real_dependency"


def test_decide_nvjitlink_or_driver_falls_back_when_module_missing(monkeypatch):
    _patch_driver_version(monkeypatch)

    def fake_optional_cuda_import(modname, probe_function=None):
        assert modname == "cuda.bindings.nvjitlink"
        assert probe_function is not None
        return None

    monkeypatch.setattr(_linker, "optional_cuda_import", fake_optional_cuda_import)

    with pytest.warns(RuntimeWarning, match="cuda.bindings.nvjitlink is not available"):
        use_driver_backend = _linker._decide_nvjitlink_or_driver()

    assert use_driver_backend is True
    assert _linker._use_nvjitlink_backend is False
