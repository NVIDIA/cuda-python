# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys

import pytest

from ci.tools.install_bindings_test_wheel import pip_install_command


@pytest.mark.agent_authored(model="gpt-5.6-sol")
@pytest.mark.parametrize("with_all", [False, True])
def test_dependency_group_install_command(tmp_path, with_all):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[dependency-groups]\ntest = ["pytest"]\n', encoding="utf-8")
    wheel = tmp_path / "cuda_bindings-13.3.0-py3-none-any.whl"

    command = pip_install_command(wheel, pyproject, with_all)

    requirement = f"{wheel}[all]" if with_all else str(wheel)
    assert command == [
        sys.executable,
        "-m",
        "pip",
        "install",
        requirement,
        "--group",
        f"{pyproject}:test",
    ]


@pytest.mark.agent_authored(model="gpt-5.6-sol")
@pytest.mark.parametrize("with_all", [False, True])
def test_legacy_extra_install_command(tmp_path, with_all):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[project]\nname = "cuda-bindings"\n[project.optional-dependencies]\ntest = ["pytest"]\n',
        encoding="utf-8",
    )
    wheel = tmp_path / "cuda_bindings-12.9.8-py3-none-any.whl"

    command = pip_install_command(wheel, pyproject, with_all)

    extras = "all,test" if with_all else "test"
    assert command == [sys.executable, "-m", "pip", "install", f"{wheel}[{extras}]"]


@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_missing_test_dependencies_are_rejected(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "cuda-bindings"\n', encoding="utf-8")

    with pytest.raises(ValueError, match="neither a test dependency group nor a test extra"):
        pip_install_command(tmp_path / "wheel.whl", pyproject, False)
