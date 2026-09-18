# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import zipfile
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parent.parent
REPO = TOOLS.parent.parent
FLOOR_MODULE = REPO / "cuda_core" / "cuda" / "core" / "_bindings_floor.py"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tool = _load("cuda_core_bindings_floor", TOOLS / "cuda_core_bindings_floor.py")
floor_module = _load("cuda_core_bindings_floor_module", FLOOR_MODULE)


def _expected(major):
    return floor_module.format_version(floor_module.CUDA_BINDINGS_FLOOR[major])


def _wheel(tmp_path, entries):
    path = tmp_path / "cuda_core-1.3.0-cp312-cp312-linux_x86_64.whl"
    with zipfile.ZipFile(path, "w") as zf:
        for name in entries:
            zf.writestr(name, FLOOR_MODULE.read_text(encoding="utf-8"))
    return path


@pytest.mark.agent_authored(model="claude-fable-5-1")
@pytest.mark.parametrize("major", [12, 13])
def test_reads_the_merged_wheel_layout(tmp_path, major):
    wheel = _wheel(tmp_path, ["cuda/core/cu12/_bindings_floor.py", "cuda/core/cu13/_bindings_floor.py"])
    assert tool.floor_from_wheel(wheel, major) == _expected(major)


@pytest.mark.agent_authored(model="claude-fable-5-1")
def test_reads_a_single_major_wheel(tmp_path):
    wheel = _wheel(tmp_path, ["cuda/core/_bindings_floor.py"])
    assert tool.floor_from_wheel(wheel, 13) == _expected(13)


@pytest.mark.agent_authored(model="claude-fable-5-1")
def test_rejects_a_wheel_without_the_module(tmp_path):
    wheel = _wheel(tmp_path, [])
    with pytest.raises(SystemExit, match="contains no _bindings_floor.py"):
        tool.floor_from_wheel(wheel, 13)


@pytest.mark.agent_authored(model="claude-fable-5-1")
def test_rejects_an_unsupported_major(tmp_path):
    wheel = _wheel(tmp_path, ["cuda/core/_bindings_floor.py"])
    with pytest.raises(SystemExit, match="CUDA 11 is not a supported major"):
        tool.floor_from_wheel(wheel, 11)


@pytest.mark.agent_authored(model="claude-fable-5-1")
def test_cli_prints_the_floor(tmp_path, capsys):
    wheel = _wheel(tmp_path, ["cuda/core/_bindings_floor.py"])
    assert tool.main(["--wheel", str(wheel), "--major", "13"]) == 0
    assert capsys.readouterr().out.strip() == _expected(13)
