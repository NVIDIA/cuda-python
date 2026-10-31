# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import runpy
import shutil
from pathlib import Path
from unittest.mock import Mock

import pytest
import setuptools
import setuptools_scm
from packaging.requirements import Requirement
from packaging.version import Version


@pytest.fixture
def standalone_source(tmp_path, monkeypatch):
    package = Path(__file__).resolve().parents[3] / "cuda_python"
    for name in ("setup.py", "pyproject.toml"):
        shutil.copyfile(package / name, tmp_path / name)
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.mark.parametrize(("requested_major", "expected_major"), ((None, 13), ("12", 12)))
@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_source_build_selects_bindings_and_major_specific_dependencies(
    standalone_source, monkeypatch, requested_major, expected_major
):
    if requested_major is None:
        monkeypatch.delenv("CUDA_PYTHON_BUILD_MAJOR", raising=False)
    else:
        monkeypatch.setenv("CUDA_PYTHON_BUILD_MAJOR", requested_major)
    setup = Mock()
    monkeypatch.setattr(setuptools, "setup", setup)
    monkeypatch.setattr(
        setuptools_scm,
        "get_version",
        lambda **options: options["git_describe_command"][-1].removeprefix("v").replace("*", "0"),
    )

    runpy.run_path(str(standalone_source / "setup.py"))

    metadata = setup.call_args.kwargs
    assert Version(metadata["version"]).major == expected_major
    dependencies = {req.name: req for req in map(Requirement, metadata["install_requires"])}
    expected = {"cuda-bindings", "cuda-core", "cuda-pathfinder"} if expected_major >= 13 else {"cuda-bindings"}
    assert dependencies.keys() == expected
    assert metadata["version"] in dependencies["cuda-bindings"].specifier


@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_maintenance_sdist_keeps_its_dependencies_without_a_build_selector(standalone_source, monkeypatch):
    (standalone_source / "PKG-INFO").write_text(
        "Metadata-Version: 2.4\nName: cuda-python\nVersion: 12.9.11\n", encoding="utf-8"
    )
    monkeypatch.delenv("CUDA_PYTHON_BUILD_MAJOR", raising=False)
    setup = Mock()
    monkeypatch.setattr(setuptools, "setup", setup)

    runpy.run_path(str(standalone_source / "setup.py"))

    metadata = setup.call_args.kwargs
    assert Version(metadata["version"]).major == 12
    assert {Requirement(value).name for value in metadata["install_requires"]} == {"cuda-bindings"}
