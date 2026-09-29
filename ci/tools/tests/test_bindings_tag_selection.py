# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check the build-side tag selectors shared by bindings and the metapackage."""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest
import tomllib
from packaging.version import Version

from ci.tools.bindings_config import load_config

REPO_ROOT = Path(__file__).resolve().parents[3]


def _literal_assignment(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in {path}")


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


@pytest.mark.parametrize(
    ("package_root", "tag_match"),
    (("cuda_bindings_12", "v12.9.[1-9]*"), ("cuda_bindings", "v13.4.*")),
)
@pytest.mark.agent_authored(model="gpt-6-sol")
def test_bindings_builds_use_default_tag_parser_and_select_their_release_line(package_root, tag_match):
    scm = _scm_config(package_root)
    package = load_config().get_package(package_root)

    assert "tag_regex" not in scm
    assert scm["git_describe_command"][-1] == tag_match
    assert tag_match.startswith(f"v{package.ctk_target}.")


@pytest.mark.agent_authored(model="gpt-6-sol")
def test_metapackage_uses_the_same_tag_selectors_and_default_parser():
    setup_path = REPO_ROOT / "cuda_python" / "setup.py"
    selectors = _literal_assignment(setup_path, "SCM_DESCRIBE_MATCH_BY_MAJOR")

    assert "tag_regex" not in setup_path.read_text(encoding="utf-8")
    for major, package_root in (("12", "cuda_bindings_12"), ("13", "cuda_bindings")):
        assert selectors[major] == _scm_config(package_root)["git_describe_command"][-1]


@pytest.mark.agent_authored(model="gpt-6-sol")
def test_maintenance_fallback_is_shared_by_bindings_metapackage_and_pixi():
    fallback = _literal_assignment(REPO_ROOT / "cuda_python" / "setup.py", "MAINTENANCE_FALLBACK_VERSION")
    with (REPO_ROOT / "cuda_bindings_12" / "pixi.toml").open("rb") as stream:
        pixi_version = tomllib.load(stream)["package"]["version"]

    assert fallback == _scm_config("cuda_bindings_12")["fallback_version"] == pixi_version
    assert Version(fallback).dev is not None
    assert Version(fallback).release[:2] == (12, 9)
    assert Version(fallback) > Version("12.9.9")
