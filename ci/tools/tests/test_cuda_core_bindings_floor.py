# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import zipfile
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parent.parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tool = _load("cuda_core_bindings_floor", TOOLS / "cuda_core_bindings_floor.py")

# METADATA as setuptools writes it: pip's normalized specifier order, extras in quotes.
METADATA = """\
Metadata-Version: 2.4
Name: cuda-core
Version: 1.3.0
Requires-Dist: cuda-pathfinder>=1.4.2
Requires-Dist: numpy
Provides-Extra: cu12
Requires-Dist: cuda-bindings[all]<13,>=12.9.8; extra == "cu12"
Requires-Dist: cuda-toolkit==12.*; extra == "cu12"
Provides-Extra: cu13
Requires-Dist: cuda-bindings[all]<14,>=13.4.1; extra == "cu13"
Requires-Dist: cuda-toolkit==13.*; extra == "cu13"
"""


def _wheel(tmp_path, metadata=METADATA):
    path = tmp_path / "cuda_core-1.3.0-cp312-cp312-linux_x86_64.whl"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("cuda/core/__init__.py", "")
        if metadata is not None:
            zf.writestr("cuda_core-1.3.0.dist-info/METADATA", metadata)
    return path


@pytest.mark.agent_authored(model="claude-fable-5-1")
@pytest.mark.parametrize(("major", "floor"), [(12, "12.9.8"), (13, "13.4.1")])
def test_reads_the_floor_of_each_extra(tmp_path, major, floor):
    assert tool.floor_from_wheel(_wheel(tmp_path), major) == floor


@pytest.mark.agent_authored(model="claude-fable-5-1")
def test_accepts_other_spellings_of_the_requirement():
    metadata = 'Requires-Dist: cuda-bindings>=13.4.1,==13.*; extra == "cu13"\n'
    assert tool.floor_from_metadata(metadata, 13) == "13.4.1"
    metadata = "Requires-Dist: cuda-bindings [all] >= 13.4.1, < 14 ; extra == 'cu13'\n"
    assert tool.floor_from_metadata(metadata, 13) == "13.4.1"


@pytest.mark.agent_authored(model="claude-fable-5-1")
def test_ignores_lookalike_names_and_other_extras():
    metadata = (
        'Requires-Dist: cuda-bindings-extra>=1.0; extra == "cu13"\n'
        'Requires-Dist: cuda-bindings[all]<13,>=12.9.8; extra == "cu12"\n'
        'Requires-Dist: cuda-bindings[all]<14,>=13.4.1; extra == "cu13"\n'
    )
    assert tool.floor_from_metadata(metadata, 13) == "13.4.1"


@pytest.mark.agent_authored(model="claude-fable-5-1")
def test_rejects_a_missing_extra(tmp_path):
    with pytest.raises(SystemExit, match="no cuda-bindings requirement for the cu11 extra"):
        tool.floor_from_wheel(_wheel(tmp_path), 11)


@pytest.mark.agent_authored(model="claude-fable-5-1")
def test_rejects_a_requirement_without_a_floor():
    with pytest.raises(SystemExit, match="without a floor for the cu13 extra"):
        tool.floor_from_metadata('Requires-Dist: cuda-bindings==13.*; extra == "cu13"\n', 13)


@pytest.mark.agent_authored(model="claude-fable-5-1")
def test_rejects_a_floor_of_another_major():
    with pytest.raises(SystemExit, match="floor of another major for the cu13 extra"):
        tool.floor_from_metadata('Requires-Dist: cuda-bindings>=12.9.8,<14; extra == "cu13"\n', 13)


@pytest.mark.agent_authored(model="claude-fable-5-1")
def test_rejects_a_zip_that_is_not_a_wheel(tmp_path):
    with pytest.raises(SystemExit, match="contains 0 METADATA files"):
        tool.floor_from_wheel(_wheel(tmp_path, metadata=None), 13)


@pytest.mark.agent_authored(model="claude-fable-5-1")
def test_cli_prints_the_floor(tmp_path, capsys):
    assert tool.main(["--wheel", str(_wheel(tmp_path)), "--major", "13"]) == 0
    assert capsys.readouterr().out.strip() == "13.4.1"
