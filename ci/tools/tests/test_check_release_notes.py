# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ci.tools.check_release_notes import check_release_notes, main, notes_path, parse_version_from_tag


def write_notes(root: Path, package: str, version: str, content: str = "Release notes.") -> Path:
    path = root / notes_path(package, version)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def resolved_12_package(package_root: str = "cuda_bindings") -> dict[str, object]:
    return {
        "package_root": package_root,
        "toolkit_version": "12.9.1",
        "release_version": "12.9.8",
        "release_registry_origin": "control",
    }


@pytest.mark.parametrize(
    ("tag", "component", "version"),
    (
        ("v13.3.0", "cuda-bindings", "13.3.0"),
        ("v13.3.0rc1", "cuda-bindings", "13.3.0rc1"),
        ("v13.3.0.dev1", "cuda-bindings", "13.3.0.dev1"),
        ("v12.9.8.post1", "cuda-python", "12.9.8.post1"),
        ("cuda-core-v1.1.1", "cuda-core", "1.1.1"),
        ("cuda-pathfinder-v1.8.1", "cuda-pathfinder", "1.8.1"),
    ),
)
@pytest.mark.agent_authored(model="gpt-5.6")
def test_parse_version_from_tag(tag, component, version):
    assert parse_version_from_tag(tag, component) == version


@pytest.mark.parametrize(
    ("tag", "component"),
    (
        ("not-a-tag", "cuda-core"),
        ("v1.0.0/../evil", "cuda-bindings"),
        ("cuda-core-v1.0.0", "cuda-pathfinder"),
        ("vv13.3.0", "cuda-bindings"),
        ("cuda-core-vv1.0.0", "cuda-core"),
        ("cuda-core-v1!2.0.0", "cuda-core"),
        ("cuda-core-v1.0.0-1", "cuda-core"),
        ("cuda-core-v01.0.0", "cuda-core"),
        ("cuda-core-v1.0", "cuda-core"),
    ),
)
@pytest.mark.agent_authored(model="gpt-5.6")
def test_parse_version_rejects_invalid_or_mismatched_tags(tag, component):
    assert parse_version_from_tag(tag, component) is None


@pytest.mark.parametrize(
    ("tag", "package", "version"),
    (
        ("v13.3.0", "cuda_bindings", "13.3.0"),
        ("v12.9.8", "cuda_bindings_12", "12.9.8"),
    ),
)
@pytest.mark.agent_authored(model="gpt-5.6")
def test_bindings_notes_follow_current_and_maintenance_packages(tmp_path, tag, package, version):
    write_notes(tmp_path, package, version)

    assert check_release_notes(tag, "cuda-bindings", tmp_path) == []


@pytest.mark.agent_authored(model="gpt-5.6")
def test_resolved_legacy_package_uses_legacy_package_root(tmp_path):
    write_notes(tmp_path, "cuda_bindings", "12.9.8")

    problems = check_release_notes(
        "v12.9.8",
        "cuda-bindings",
        tmp_path,
        resolved_12_package(),
    )

    assert problems == []


@pytest.mark.agent_authored(model="gpt-5.6")
def test_present_missing_and_empty_notes(tmp_path):
    write_notes(tmp_path, "cuda_core", "1.1.1")
    assert check_release_notes("cuda-core-v1.1.1", "cuda-core", tmp_path) == []

    missing = check_release_notes("cuda-pathfinder-v1.8.1", "cuda-pathfinder", tmp_path)
    assert missing == [(notes_path("cuda_pathfinder", "1.8.1"), "missing")]

    write_notes(tmp_path, "cuda_python", "13.3.0", content="")
    empty = check_release_notes("v13.3.0", "cuda-python", tmp_path)
    assert empty == [(notes_path("cuda_python", "13.3.0"), "empty")]


@pytest.mark.agent_authored(model="gpt-5.6")
def test_post_release_needs_no_notes(tmp_path):
    assert check_release_notes("v12.9.8.post1", "cuda-bindings", tmp_path) == []


@pytest.mark.agent_authored(model="gpt-5.6")
def test_main_accepts_resolved_package_and_reports_missing_notes(tmp_path, capsys):
    package = resolved_12_package()
    args = [
        "--git-tag",
        "v12.9.8",
        "--component",
        "cuda-bindings",
        "--repo-root",
        str(tmp_path),
        "--bindings-package",
        json.dumps(package),
    ]

    assert main(args) == 1
    assert "cuda_bindings/docs/source/release/12.9.8-notes.rst" in capsys.readouterr().err

    write_notes(tmp_path, "cuda_bindings", "12.9.8")
    assert main(args) == 0


@pytest.mark.agent_authored(model="gpt-5.6")
def test_main_rejects_unsafe_resolved_package_root(tmp_path, capsys):
    args = [
        "--git-tag",
        "v12.9.8",
        "--component",
        "cuda-bindings",
        "--repo-root",
        str(tmp_path),
        "--bindings-package",
        json.dumps(resolved_12_package("../outside")),
    ]

    assert main(args) == 2
    assert "normalized repository-relative POSIX path" in capsys.readouterr().err
