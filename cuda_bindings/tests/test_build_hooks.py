# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the cuda-bindings PEP 517 build wrapper."""

import base64
import csv
import hashlib
import importlib.util
import io
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


def _record_row(path, contents):
    digest = base64.urlsafe_b64encode(hashlib.sha256(contents).digest()).rstrip(b"=").decode("ascii")
    return path, f"sha256={digest}", len(contents)


def _assert_valid_record(wheel):
    record_path = next(name for name in wheel.namelist() if name.endswith(".dist-info/RECORD"))
    rows = list(csv.reader(io.StringIO(wheel.read(record_path).decode("utf-8"), newline="")))
    records = {path: (digest, size) for path, digest, size in rows}

    assert set(records) == set(wheel.namelist())
    for name in wheel.namelist():
        digest, size = records[name]
        if name == record_path:
            assert (digest, size) == ("", "")
        else:
            contents = wheel.read(name)
            assert (name, digest, int(size)) == _record_row(name, contents)


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
        _assert_valid_record(wheel)


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
        _assert_valid_record(wheel)
