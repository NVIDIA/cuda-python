# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from ci.tools import bindings_config
from ci.tools.bindings_scm_version import main, pretend_version, read_fallback_version

SHA = "abcdef0123456789"
RELEASED_12 = bindings_config.BindingsLine(
    line_id="released-12",
    source_dir="cuda_bindings_12",
    toolkit_version="12.9.1",
    tag_regex=r"^(?P<version>v12\.9\.\d+(?:\.post\d+)?)$",
)
ALTERNATE_13 = bindings_config.BindingsLine(
    line_id="alternate-13",
    source_dir="alternate_bindings",
    toolkit_version="13.2.0",
    tag_regex=r"^(?P<version>v13\.2\.\d+(?:(?:a|b|rc)\d+)?(?:\.post\d+)?)$",
)


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)  # noqa: S603, S607


def make_repo(
    tmp_path: Path,
    line: bindings_config.BindingsLine = RELEASED_12,
    fallback_version: str = "12.9.8.dev0",
) -> tuple[Path, Path]:
    config = tmp_path / line.source_dir / "pyproject.toml"
    config.parent.mkdir(parents=True)
    config.write_text(
        (f"[tool.setuptools_scm]\nfallback_version = \"{fallback_version}\"\ntag_regex = '{line.tag_regex}'\n"),
        encoding="utf-8",
    )
    git(tmp_path, "init")
    git(tmp_path, "config", "user.name", "CUDA Python CI")
    git(tmp_path, "config", "user.email", "cuda-python@nvidia.com")
    git(tmp_path, "config", "commit.gpgsign", "false")
    git(tmp_path, "add", config.relative_to(tmp_path).as_posix())
    git(tmp_path, "commit", "-m", "initial")
    return tmp_path, config


@pytest.mark.agent_authored(model="gpt-5.6")
def test_uses_configured_fallback_before_first_release_tag(tmp_path):
    repo, config = make_repo(tmp_path)

    assert config == repo / RELEASED_12.source_dir / "pyproject.toml"
    assert pretend_version(repo, SHA, RELEASED_12) == "12.9.8.dev0+gabcdef0"


@pytest.mark.agent_authored(model="gpt-5.6")
def test_reachable_release_disables_override(tmp_path):
    repo, _ = make_repo(tmp_path)
    git(repo, "tag", "v12.9.0")
    assert pretend_version(repo, SHA, RELEASED_12) == "12.9.8.dev0+gabcdef0"
    git(repo, "tag", "v12.9.7")
    assert pretend_version(repo, SHA, RELEASED_12) == "12.9.8.dev0+gabcdef0"
    git(repo, "tag", "v12.9.8a1")
    assert pretend_version(repo, SHA, RELEASED_12) == "12.9.8.dev0+gabcdef0"

    git(repo, "tag", "v12.9.8")
    assert pretend_version(repo, SHA, RELEASED_12) is None
    git(repo, "commit", "--allow-empty", "-m", "post-release")
    assert pretend_version(repo, SHA, RELEASED_12) is None


@pytest.mark.agent_authored(model="gpt-5.6")
def test_reachable_post_release_disables_override(tmp_path):
    repo, _ = make_repo(tmp_path)
    git(repo, "tag", "v12.9.8.post1")

    assert pretend_version(repo, SHA, RELEASED_12) is None


@pytest.mark.agent_authored(model="gpt-5.6")
def test_rejects_development_fallback_for_another_ctk_target(tmp_path):
    config = tmp_path / "pyproject.toml"
    config.write_text('[tool.setuptools_scm]\nfallback_version = "13.0.0.dev0"\n', encoding="utf-8")

    with pytest.raises(ValueError, match="CUDA 12.9 development fallback"):
        read_fallback_version(config, RELEASED_12.ctk_target)


@pytest.mark.agent_authored(model="gpt-5.6")
def test_configured_line_uses_its_source_and_scm_tag_regex(tmp_path):
    repo, config = make_repo(tmp_path, ALTERNATE_13, "13.2.2.dev0")
    assert config == repo / "alternate_bindings" / "pyproject.toml"

    git(repo, "tag", "v12.9.99")
    git(repo, "tag", "v13.3.99")
    git(repo, "tag", "v13.2.1")
    assert pretend_version(repo, SHA, ALTERNATE_13) == "13.2.2.dev0+gabcdef0"

    git(repo, "tag", "v13.2.2a1")
    assert pretend_version(repo, SHA, ALTERNATE_13) is None


@pytest.mark.agent_authored(model="gpt-5.6")
def test_cli_selects_released_12_from_requested_repo_root(tmp_path, capsys, monkeypatch):
    configured_line = bindings_config.load_config().get_line("released-12")
    make_repo(tmp_path, configured_line)
    load_config = bindings_config.load_config
    seen_roots = []

    def record_repo_root(path, repo_root):
        seen_roots.append(repo_root)
        return load_config(path)

    monkeypatch.setattr(bindings_config, "load_config", record_repo_root)

    result = main(["--repo-root", str(tmp_path), "--line-id", "released-12", "--sha", SHA])

    assert result == 0
    assert seen_roots == [tmp_path]
    assert capsys.readouterr().out.strip() == "12.9.8.dev0+gabcdef0"
