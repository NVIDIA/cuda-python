# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import runpy
from pathlib import Path
from unittest.mock import Mock

import pytest
import setuptools
import setuptools_scm

SETUP = Path(__file__).resolve().parents[3] / "cuda_python" / "setup.py"


@pytest.mark.parametrize("major", (None, "12", "13"))
@pytest.mark.parametrize("development", (False, True))
@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_source_build_selects_matching_bindings_and_current_major_components(monkeypatch, major, development):
    if major is None:
        monkeypatch.delenv("CUDA_PYTHON_BUILD_MAJOR", raising=False)
    else:
        monkeypatch.setenv("CUDA_PYTHON_BUILD_MAJOR", major)
    selected_major = major or "13"
    version = "12.9.11" if selected_major == "12" else "13.4.2"
    if development:
        version += ".dev1"
    get_version = Mock(return_value=version)
    setup = Mock()
    monkeypatch.setattr(setuptools_scm, "get_version", get_version)
    monkeypatch.setattr(setuptools, "setup", setup)

    runpy.run_path(str(SETUP))

    selector = "v12.9.*" if selected_major == "12" else "v13.4.*"
    assert get_version.call_args.kwargs["git_describe_command"][-1] == selector
    matcher = "==" if development else "~="
    requirements = [f"cuda-bindings{matcher}{version}"]
    if selected_major == "13":
        requirements += ["cuda-core~=1.2.0", "cuda-pathfinder~=1.1"]
    assert setup.call_args.kwargs == {
        "version": version,
        "install_requires": requirements,
        "extras_require": {"all": [f"cuda-bindings[all]{matcher}{version}"]},
    }


@pytest.mark.parametrize("major", ("", "11", "14", "cu13"))
@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_source_build_rejects_an_invalid_explicit_major(monkeypatch, major):
    monkeypatch.setenv("CUDA_PYTHON_BUILD_MAJOR", major)

    with pytest.raises(ValueError, match="CUDA_PYTHON_BUILD_MAJOR must be 12 or 13"):
        runpy.run_path(str(SETUP))
