# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the cuda-bindings PEP 517 build wrapper."""

import importlib.util
import zipfile
from pathlib import Path
from unittest import mock

import pytest


def _load_build_hooks():
    build_hooks_path = Path(__file__).parent.parent / "build_hooks.py"
    spec = importlib.util.spec_from_file_location("_cuda_bindings_test_build_hooks", build_hooks_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build_hooks = _load_build_hooks()


@pytest.mark.agent_authored(model="gpt-5.6")
def test_editable_wheel_uses_data_only_source_root_pth(tmp_path, monkeypatch):
    project_root = Path(build_hooks.__file__).resolve().parent
    bindings_build = mock.Mock()
    monkeypatch.chdir(project_root)
    monkeypatch.setattr(build_hooks, "_build_cuda_bindings", bindings_build)

    wheel_name = build_hooks.build_editable(tmp_path, {"debug": False})

    bindings_build.assert_called_once_with(debug=False)
    with zipfile.ZipFile(tmp_path / wheel_name) as wheel:
        pth_files = [name for name in wheel.namelist() if name.endswith(".pth")]
        assert len(pth_files) == 1
        assert wheel.read(pth_files[0]).decode("utf-8") == f"{project_root}\n"


@pytest.mark.agent_authored(model="gpt-5.6")
def test_regular_wheel_does_not_include_editable_path(tmp_path, monkeypatch):
    project_root = Path(build_hooks.__file__).resolve().parent
    bindings_build = mock.Mock()
    monkeypatch.chdir(project_root)
    monkeypatch.setattr(build_hooks, "_build_cuda_bindings", bindings_build)

    wheel_name = build_hooks.build_wheel(tmp_path, {"debug": False})

    bindings_build.assert_called_once_with(debug=False)
    with zipfile.ZipFile(tmp_path / wheel_name) as wheel:
        assert not any(name.endswith(".pth") for name in wheel.namelist())
