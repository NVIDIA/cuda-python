# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from packaging.version import Version

from ci.tools import bindings_config
from ci.tools.bindings_scm_version import bootstrap_version

PACKAGE = bindings_config.BindingsPackage("cuda_bindings_12", (12, 9, 1), "maintenance")


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()  # noqa: S603, S607


def make_repo(tmp_path: Path) -> tuple[Path, str]:
    config = tmp_path / PACKAGE.package_root / "pyproject.toml"
    config.parent.mkdir(parents=True)
    config.write_text(
        "[tool.setuptools_scm]\n"
        'git_describe_command = ["git", "describe", "--tags", "--long", "--match", "v12.9.*"]\n'
        '[tool.cuda-python]\nscm-bootstrap-release = "12.9.11"\n',
        encoding="utf-8",
    )
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.name", "CUDA Python CI")
    git(tmp_path, "config", "user.email", "cuda-python@nvidia.com")
    git(tmp_path, "config", "commit.gpgSign", "false")
    git(tmp_path, "config", "tag.gpgSign", "false")
    (tmp_path / "README").write_text("old main baseline\n", encoding="utf-8")
    git(tmp_path, "add", "README")
    git(tmp_path, "commit", "-qm", "old baseline")
    git(tmp_path, "tag", "v12.9.0")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-qm", "import package")
    git(tmp_path, "commit", "--allow-empty", "-qm", "activate maintenance package")
    return tmp_path, git(tmp_path, "rev-parse", "HEAD")


@pytest.mark.agent_authored(model="gpt-6")
def test_bootstrap_creates_only_a_local_ancestor_tag_and_reports_scm(tmp_path, capsys):
    repo, sha = make_repo(tmp_path)
    version = bootstrap_version(repo, sha, PACKAGE)
    assert Version(version).release == (12, 9, 11)
    assert Version(version).is_devrelease
    assert git(repo, "rev-parse", "v12.9.11a0.dev0") == git(repo, "rev-parse", "HEAD^")
    assert "git describe: v12.9.11a0.dev0-1-g" in capsys.readouterr().err
    assert bootstrap_version(repo, sha, PACKAGE) == version


@pytest.mark.parametrize("tag", ("v12.9.10", "v12.9.11a0"))
@pytest.mark.agent_authored(model="gpt-6")
def test_appropriate_reachable_tags_keep_standard_scm_progression(tmp_path, tag):
    repo, sha = make_repo(tmp_path)
    git(repo, "tag", tag)
    assert bootstrap_version(repo, sha, PACKAGE) == tag.removeprefix("v")
    git(repo, "commit", "--allow-empty", "-qm", "after retained tag")
    version = bootstrap_version(repo, git(repo, "rev-parse", "HEAD"), PACKAGE)
    assert ".dev1" in version
    assert "v12.9.11a0.dev0" not in git(repo, "tag", "--list").splitlines()


@pytest.mark.agent_authored(model="gpt-6")
def test_unreachable_synthetic_tag_is_preserved_and_fails(tmp_path):
    repo, sha = make_repo(tmp_path)
    git(repo, "commit", "--allow-empty", "-qm", "unrelated future preview")
    future = git(repo, "rev-parse", "HEAD")
    git(repo, "tag", "v12.9.11a0.dev0")
    git(repo, "checkout", "--detach", sha)
    with pytest.raises(subprocess.CalledProcessError):
        bootstrap_version(repo, sha, PACKAGE)
    assert git(repo, "rev-parse", "v12.9.11a0.dev0") == future
