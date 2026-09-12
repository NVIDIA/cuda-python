# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import types

import pytest

import cuda.pathfinder._optional_cuda_import as optional_import_mod
from cuda.pathfinder import DynamicLibNotFoundError
from cuda.pathfinder._optional_cuda_import import _optional_cuda_import


def test__optional_cuda_import_returns_module_when_available(monkeypatch):
    fake_module = types.SimpleNamespace(__name__="cuda.bindings.nvvm")
    monkeypatch.setattr(optional_import_mod.importlib, "import_module", lambda _name: fake_module)

    result = _optional_cuda_import("cuda.bindings.nvvm")

    assert result is fake_module


def test__optional_cuda_import_returns_none_when_module_missing(monkeypatch):
    def fake_import_module(name):
        err = ModuleNotFoundError("No module named 'cuda.bindings.nvvm'")
        err.name = name
        raise err

    monkeypatch.setattr(optional_import_mod.importlib, "import_module", fake_import_module)

    result = _optional_cuda_import("cuda.bindings.nvvm")

    assert result is None


@pytest.mark.agent_authored(model="claude-opus-5")
def test__optional_cuda_import_returns_none_when_parent_package_missing(monkeypatch):
    def fake_import_module(_name):
        # What CPython reports for `import cuda.bindings.nvjitlink` when the
        # cuda.bindings package itself is not installed: the outermost missing
        # name, not the fully qualified one that was requested.
        err = ModuleNotFoundError("No module named 'cuda.bindings'")
        err.name = "cuda.bindings"
        raise err

    monkeypatch.setattr(optional_import_mod.importlib, "import_module", fake_import_module)

    assert _optional_cuda_import("cuda.bindings.nvjitlink") is None


@pytest.mark.agent_authored(model="claude-opus-5")
def test__optional_cuda_import_returns_none_for_a_really_uninstalled_package():
    """Same as above, but through the real import machinery instead of a stub."""
    assert _optional_cuda_import("cuda_pathfinder_no_such_package.sub.mod") is None


@pytest.mark.agent_authored(model="claude-opus-5")
def test__optional_cuda_import_reraises_for_a_string_prefix_that_is_not_an_ancestor(monkeypatch):
    """``cuda.bindings`` is a string prefix of ``cuda.bindings_extra``, not a parent of it."""

    def fake_import_module(_name):
        err = ModuleNotFoundError("No module named 'cuda.bindings'")
        err.name = "cuda.bindings"
        raise err

    monkeypatch.setattr(optional_import_mod.importlib, "import_module", fake_import_module)

    with pytest.raises(ModuleNotFoundError, match="cuda.bindings"):
        _optional_cuda_import("cuda.bindings_extra.mod")


def test__optional_cuda_import_reraises_nested_module_not_found(monkeypatch):
    def fake_import_module(_name):
        err = ModuleNotFoundError("No module named 'not_a_real_dependency'")
        err.name = "not_a_real_dependency"
        raise err

    monkeypatch.setattr(optional_import_mod.importlib, "import_module", fake_import_module)

    with pytest.raises(ModuleNotFoundError, match="not_a_real_dependency") as excinfo:
        _optional_cuda_import("cuda.bindings.nvvm")
    assert excinfo.value.name == "not_a_real_dependency"


def test__optional_cuda_import_returns_none_when_probe_finds_missing_dynamic_lib(monkeypatch):
    fake_module = types.SimpleNamespace(__name__="cuda.bindings.nvvm")
    monkeypatch.setattr(optional_import_mod.importlib, "import_module", lambda _name: fake_module)

    def probe(_module):
        raise DynamicLibNotFoundError("libnvvm missing")

    result = _optional_cuda_import("cuda.bindings.nvvm", probe_function=probe)

    assert result is None


def test__optional_cuda_import_reraises_non_pathfinder_probe_error(monkeypatch):
    fake_module = types.SimpleNamespace(__name__="cuda.bindings.nvvm")
    monkeypatch.setattr(optional_import_mod.importlib, "import_module", lambda _name: fake_module)

    def probe(_module):
        raise RuntimeError("unexpected probe failure")

    with pytest.raises(RuntimeError, match="unexpected probe failure"):
        _optional_cuda_import("cuda.bindings.nvvm", probe_function=probe)
