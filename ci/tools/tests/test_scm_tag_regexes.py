# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
import tomllib

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


@pytest.mark.parametrize(
    ("package", "tag", "version"),
    (
        ("cuda_core", "cuda-core-v1.2.3.post1", "v1.2.3.post1"),
        ("cuda_pathfinder", "cuda-pathfinder-v1.2.3.post1", "v1.2.3.post1"),
        ("cuda_bindings", "v13.3.1.post1", "v13.3.1.post1"),
        ("cuda_bindings_12", "v12.9.8.post1", "v12.9.8.post1"),
    ),
)
@pytest.mark.agent_authored(model="gpt-5.6")
def test_release_package_scm_regex_preserves_post_suffix(package, tag, version):
    with (REPO_ROOT / package / "pyproject.toml").open("rb") as stream:
        pattern = tomllib.load(stream)["tool"]["setuptools_scm"]["tag_regex"]

    match = re.fullmatch(pattern, tag)

    assert match is not None
    assert match.group("version") == version


@pytest.mark.parametrize(
    ("package_root", "tag", "expected"),
    (
        ("cuda_bindings", "v13.3.1", "13.3.1"),
        ("cuda_bindings", "v13.3.1a2", "13.3.1a2"),
        ("cuda_bindings", "v13.3.1b2", "13.3.1b2"),
        ("cuda_bindings", "v13.3.1rc2", "13.3.1rc2"),
        ("cuda_bindings", "v13.3.1.post2", "13.3.1.post2"),
        ("cuda_bindings", "v13.3.1.dev2", "13.3.1.dev2"),
        ("cuda_bindings", "v13.3.1rc2.dev3", "13.3.1rc2.dev3"),
        ("cuda_bindings_12", "v12.9.8", "12.9.8"),
        ("cuda_bindings_12", "v12.9.8.post2", "12.9.8.post2"),
        ("cuda_bindings_12", "v12.9.8a2", None),
        ("cuda_bindings_12", "v12.9.8rc2", None),
        ("cuda_bindings_12", "v12.9.8.dev2", None),
    ),
)
@pytest.mark.agent_authored(model="gpt-5.6")
def test_bindings_registry_uses_each_package_scm_regex(package_root, tag, expected):
    version = load_config().get_package(package_root).version_from_tag(tag)

    assert (str(version) if version is not None else None) == expected


@pytest.mark.agent_authored(model="gpt-5.6")
def test_current_bindings_scm_metadata_is_minor_specific():
    with (REPO_ROOT / "cuda_bindings" / "pyproject.toml").open("rb") as stream:
        scm = tomllib.load(stream)["tool"]["setuptools_scm"]

    assert re.fullmatch(scm["tag_regex"], "v13.3.2") is not None
    assert re.fullmatch(scm["tag_regex"], "v13.4.0") is None
    assert scm["git_describe_command"][-1] == "v13.3.*"


@pytest.mark.agent_authored(model="gpt-5.6")
def test_metapackage_scm_metadata_matches_bindings_sources():
    setup_path = REPO_ROOT / "cuda_python" / "setup.py"
    tag_regexes = _literal_assignment(setup_path, "SCM_TAG_REGEX_BY_MAJOR")
    describe_matches = _literal_assignment(setup_path, "SCM_DESCRIBE_MATCH_BY_MAJOR")

    for major, source_dir in (("12", "cuda_bindings_12"), ("13", "cuda_bindings")):
        with (REPO_ROOT / source_dir / "pyproject.toml").open("rb") as stream:
            scm = tomllib.load(stream)["tool"]["setuptools_scm"]

        assert tag_regexes[major] == scm["tag_regex"]
        assert describe_matches[major] == scm["git_describe_command"][-1]
