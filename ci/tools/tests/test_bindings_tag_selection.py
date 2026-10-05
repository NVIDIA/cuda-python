# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise setuptools-scm with the bindings packages' source-build selectors."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import tomllib

REPO_ROOT = Path(__file__).resolve().parents[3]


def _scm_config(package_root: str) -> dict[str, object]:
    with (REPO_ROOT / package_root / "pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)["tool"]["setuptools_scm"]


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)  # noqa: S603, S607


@pytest.mark.parametrize(
    ("package_root", "tag"),
    (("cuda_bindings_12", "v12.9.8.post2"), ("cuda_bindings", "v13.4.1rc2")),
)
@pytest.mark.agent_authored(model="gpt-6-sol")
def test_default_scm_parser_preserves_release_suffixes(tmp_path, package_root, tag):
    setuptools_scm = pytest.importorskip("setuptools_scm")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "commit.gpgSign", "false")
    _git(tmp_path, "config", "tag.gpgSign", "false")
    (tmp_path / "README").write_text("test repository\n", encoding="utf-8")
    _git(tmp_path, "add", "README")
    _git(
        tmp_path,
        "-c",
        "user.name=CUDA Python CI",
        "-c",
        "user.email=cuda-python@nvidia.com",
        "commit",
        "-qm",
        "initial",
    )
    _git(tmp_path, "tag", tag)

    version = setuptools_scm.get_version(
        root=tmp_path,
        git_describe_command=_scm_config(package_root)["git_describe_command"],
    )

    assert version == tag.removeprefix("v")
