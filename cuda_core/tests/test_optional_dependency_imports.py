# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from cuda.core import _linker, _program


@pytest.fixture(autouse=True)
def restore_optional_import_state():
    saved_nvvm_module = _program._nvvm_module
    saved_nvvm_attempted = _program._nvvm_import_attempted
    saved_probed = _linker._nvjitlink_probed
    saved_version = _linker._nvjitlink_version
    saved_warned = _linker._nvjitlink_missing_warned

    _program._nvvm_module = None
    _program._nvvm_import_attempted = False
    _linker._nvjitlink_probed = False
    _linker._nvjitlink_version = None
    _linker._nvjitlink_missing_warned = False

    yield

    _program._nvvm_module = saved_nvvm_module
    _program._nvvm_import_attempted = saved_nvvm_attempted
    _linker._nvjitlink_probed = saved_probed
    _linker._nvjitlink_version = saved_version
    _linker._nvjitlink_missing_warned = saved_warned


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


def test_probe_nvjitlink_reraises_nested_module_not_found(monkeypatch):
    def fake__optional_cuda_import(modname, probe_function=None):
        assert modname == "cuda.bindings.nvjitlink"
        assert probe_function is not None
        err = ModuleNotFoundError("No module named 'not_a_real_dependency'")
        err.name = "not_a_real_dependency"
        raise err

    monkeypatch.setattr(_linker, "_optional_cuda_import", fake__optional_cuda_import)

    with pytest.raises(ModuleNotFoundError, match="not_a_real_dependency") as excinfo:
        _linker._probe_nvjitlink()
    assert excinfo.value.name == "not_a_real_dependency"


def test_probe_nvjitlink_warns_and_returns_none_when_module_missing(monkeypatch):
    def fake__optional_cuda_import(modname, probe_function=None):
        assert modname == "cuda.bindings.nvjitlink"
        assert probe_function is not None
        return None

    monkeypatch.setattr(_linker, "_optional_cuda_import", fake__optional_cuda_import)

    with pytest.warns(RuntimeWarning, match="cuda.bindings.nvjitlink is not available"):
        probe_result = _linker._probe_nvjitlink()

    assert probe_result is None
    assert _linker._nvjitlink_version is None
    assert _linker._nvjitlink_probed is True
