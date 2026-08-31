# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cuda12_scm_version import pretend_version, read_fallback_version

SHA = "abcdef0123456789"


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)  # noqa: S603, S607


def make_repo(tmp_path: Path) -> tuple[Path, Path]:
    config = tmp_path / "cuda_bindings_12" / "pyproject.toml"
    config.parent.mkdir()
    config.write_text('[tool.setuptools_scm]\nfallback_version = "12.9.8.dev0"\n', encoding="utf-8")
    git(tmp_path, "init")
    git(tmp_path, "config", "user.name", "CUDA Python CI")
    git(tmp_path, "config", "user.email", "cuda-python@nvidia.com")
    git(tmp_path, "config", "commit.gpgsign", "false")
    git(tmp_path, "add", config.relative_to(tmp_path).as_posix())
    git(tmp_path, "commit", "-m", "initial")
    return tmp_path, config


@pytest.mark.agent_authored(model="gpt-5")
def test_uses_configured_fallback_before_first_main_cuda12_tag(tmp_path):
    repo, config = make_repo(tmp_path)

    assert pretend_version(repo, SHA, config) == "12.9.8.dev0+gabcdef0"


@pytest.mark.agent_authored(model="gpt-5")
def test_reachable_cuda12_release_disables_override(tmp_path):
    repo, config = make_repo(tmp_path)
    git(repo, "tag", "v12.9.0")
    assert pretend_version(repo, SHA, config) == "12.9.8.dev0+gabcdef0"
    git(repo, "tag", "v12.9.7")
    assert pretend_version(repo, SHA, config) == "12.9.8.dev0+gabcdef0"
    git(repo, "tag", "v12.9.8a1")
    assert pretend_version(repo, SHA, config) == "12.9.8.dev0+gabcdef0"

    git(repo, "tag", "v12.9.8")
    assert pretend_version(repo, SHA, config) is None
    git(repo, "commit", "--allow-empty", "-m", "post-release")
    assert pretend_version(repo, SHA, config) is None


@pytest.mark.agent_authored(model="gpt-5")
def test_rejects_non_cuda12_development_fallback(tmp_path):
    config = tmp_path / "pyproject.toml"
    config.write_text('fallback_version = "13.0.0"\n', encoding="utf-8")

    with pytest.raises(ValueError, match="CUDA 12.9 development fallback"):
        read_fallback_version(config)
