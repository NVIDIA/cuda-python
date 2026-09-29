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
        ("v13.4.1", "cuda-bindings", "13.4.1"),
        ("v13.4.1rc1", "cuda-bindings", "13.4.1rc1"),
        ("v13.4.1.dev1", "cuda-bindings", "13.4.1.dev1"),
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
        ("v13.4.1", "cuda_bindings", "13.4.1"),
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


@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_resolved_legacy_prerelease_uses_the_scm_release_version(tmp_path):
    package = resolved_12_package()
    package["toolkit_version"] = "13.1.0"
    package["release_version"] = "13.2.0"
    write_notes(tmp_path, "cuda_bindings", "13.2.0")

    assert check_release_notes("v13.2.0rc1", "cuda-bindings", tmp_path, package) == []


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


def legacy_release_args(tmp_path: Path, component: str, package: dict[str, object] | None) -> list[str]:
    control_config = tmp_path / "control" / "ci" / "versions.yml"
    control_config.parent.mkdir(parents=True, exist_ok=True)
    control_config.write_text(
        (Path(__file__).parents[2] / "versions.yml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    args = [
        "--git-tag",
        "v12.9.8",
        "--component",
        component,
        "--repo-root",
        str(tmp_path / "tagged"),
        "--legacy-notes-root",
        str(tmp_path / "control"),
    ]
    if package is not None:
        args.extend(("--bindings-package", json.dumps(package)))
    return args


@pytest.mark.parametrize(
    ("component", "package"), (("cuda-bindings", "cuda_bindings_12"), ("cuda-python", "cuda_python"))
)
@pytest.mark.agent_authored(model="gpt-6-astra")
def test_legacy_release_uses_matching_control_notes_with_warning(tmp_path, capsys, component, package):
    notes = write_notes(tmp_path / "control", package, "12.9.8")

    assert main(legacy_release_args(tmp_path, component, resolved_12_package())) == 0

    warning = capsys.readouterr().err
    assert "WARNING: legacy tag v12.9.8 has no tagged release notes" in warning
    assert str(notes) in warning


@pytest.mark.parametrize("notes_tree", ("tagged", "control"))
@pytest.mark.agent_authored(model="gpt-6-astra")
def test_legacy_metapackage_without_separate_notes_uses_matching_bindings_notes(tmp_path, capsys, notes_tree):
    package = "cuda_bindings" if notes_tree == "tagged" else "cuda_bindings_12"
    notes = write_notes(tmp_path / notes_tree, package, "12.9.8")

    assert main(legacy_release_args(tmp_path, "cuda-python", resolved_12_package())) == 0

    warning = capsys.readouterr().err
    assert "WARNING: historical cuda-python release has no separate metapackage notes" in warning
    assert str(notes) in warning


@pytest.mark.parametrize("component", ("cuda-bindings", "cuda-python"))
@pytest.mark.parametrize("context", ("absent", "tag", "no-notes-root"))
@pytest.mark.agent_authored(model="gpt-6-astra")
def test_control_notes_cannot_bypass_strict_release_check(tmp_path, capsys, component, context):
    write_notes(tmp_path / "control", "cuda_bindings_12", "12.9.8")
    write_notes(tmp_path / "control", "cuda_python", "12.9.8")
    package = None if context == "absent" else resolved_12_package()
    if context == "tag":
        package["release_registry_origin"] = "tag"
    args = legacy_release_args(tmp_path, component, package)
    if context == "no-notes-root":
        index = args.index("--legacy-notes-root")
        del args[index : index + 2]

    assert main(args) == 1
    assert "ERROR: missing or empty release notes" in capsys.readouterr().err


@pytest.mark.parametrize("component", ("cuda-bindings", "cuda-python"))
@pytest.mark.parametrize("notes_version", ("12.9.7", "12.9.9"))
@pytest.mark.agent_authored(model="gpt-6-astra")
def test_legacy_release_rejects_other_version_notes(tmp_path, capsys, component, notes_version):
    write_notes(tmp_path / "control", "cuda_bindings_12", notes_version)
    write_notes(tmp_path / "control", "cuda_python", notes_version)

    assert main(legacy_release_args(tmp_path, component, resolved_12_package())) == 1
    assert "No matching nonempty historical release notes found" in capsys.readouterr().err


@pytest.mark.parametrize(("component", "package"), (("cuda-bindings", "cuda_bindings"), ("cuda-python", "cuda_python")))
@pytest.mark.parametrize("empty_tree", ("tagged", "control"))
@pytest.mark.agent_authored(model="gpt-6-astra")
def test_legacy_fallback_does_not_hide_empty_component_notes(tmp_path, capsys, component, package, empty_tree):
    control_package = "cuda_bindings_12" if component == "cuda-bindings" else package
    write_notes(tmp_path / "control", control_package, "12.9.8")
    if component == "cuda-python":
        write_notes(tmp_path / "tagged", "cuda_bindings", "12.9.8")
    empty_package = package if empty_tree == "tagged" else control_package
    write_notes(tmp_path / empty_tree, empty_package, "12.9.8", content="")

    assert main(legacy_release_args(tmp_path, component, resolved_12_package())) == 1
    assert "ERROR: missing or empty release notes" in capsys.readouterr().err


@pytest.mark.parametrize("component", ("cuda-bindings", "cuda-python"))
@pytest.mark.agent_authored(model="gpt-6-astra")
def test_legacy_fallback_rejects_resolver_version_mismatch(tmp_path, capsys, component):
    write_notes(tmp_path / "control", "cuda_bindings_12", "12.9.8")
    write_notes(tmp_path / "control", "cuda_python", "12.9.8")
    package = resolved_12_package()
    package["release_version"] = "12.9.9"

    assert main(legacy_release_args(tmp_path, component, package)) == 2
    assert "does not match release tag" in capsys.readouterr().err
