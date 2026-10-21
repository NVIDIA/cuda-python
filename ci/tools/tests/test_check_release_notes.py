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


def resolved_12_package(package_root: str = "cuda_bindings_12") -> dict[str, object]:
    return {
        "package_root": package_root,
        "toolkit_version": "12.9.1",
        "release_version": "12.9.8",
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
        ("v13.4.1", "cuda_bindings_13", "13.4.1"),
        ("v12.9.8", "cuda_bindings_12", "12.9.8"),
    ),
)
@pytest.mark.agent_authored(model="gpt-5.6")
def test_bindings_notes_follow_current_and_maintenance_packages(tmp_path, tag, package, version):
    write_notes(tmp_path, package, version)

    assert check_release_notes(tag, "cuda-bindings", tmp_path) == []


@pytest.mark.parametrize("version", ("12.9.8", "12.9.8rc1", "12.9.8.dev1"))
@pytest.mark.agent_authored(model="gpt-6")
def test_resolved_package_selects_exact_version_and_tagged_root(tmp_path, version):
    package_root = "cuda_bindings_12_maintenance"
    package = resolved_12_package(package_root)
    package["release_version"] = version
    write_notes(tmp_path, package_root, version)

    problems = check_release_notes(
        f"v{version}",
        "cuda-bindings",
        tmp_path,
        package,
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
    assert "cuda_bindings_12/docs/source/release/12.9.8-notes.rst" in capsys.readouterr().err

    write_notes(tmp_path, "cuda_bindings_12", "12.9.8")
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


def release_args(repo_root: Path, component: str, package: dict[str, object]) -> list[str]:
    args = [
        "--git-tag",
        "v12.9.8",
        "--component",
        component,
        "--repo-root",
        str(repo_root),
        "--bindings-package",
        json.dumps(package),
    ]
    return args


@pytest.mark.parametrize(
    ("component", "package"), (("cuda-bindings", "cuda_bindings_12"), ("cuda-python", "cuda_python"))
)
@pytest.mark.parametrize("problem", ("missing", "empty"))
@pytest.mark.agent_authored(model="gpt-6")
def test_notes_in_another_checkout_do_not_satisfy_release_check(
    tmp_path, capsys, monkeypatch, component, package, problem
):
    write_notes(tmp_path / "other", package, "12.9.8")
    monkeypatch.chdir(tmp_path / "other")
    if problem == "empty":
        write_notes(tmp_path / "tagged", package, "12.9.8", content="")

    assert main(release_args(tmp_path / "tagged", component, resolved_12_package())) == 1
    assert f"{notes_path(package, '12.9.8')} ({problem})" in capsys.readouterr().err


@pytest.mark.agent_authored(model="gpt-6")
def test_metapackage_requires_its_own_notes(tmp_path, capsys):
    write_notes(tmp_path, "cuda_bindings_12", "12.9.8")

    assert main(release_args(tmp_path, "cuda-python", resolved_12_package())) == 1
    assert "cuda_python/docs/source/release/12.9.8-notes.rst (missing)" in capsys.readouterr().err

    write_notes(tmp_path, "cuda_python", "12.9.8")
    assert main(release_args(tmp_path, "cuda-python", resolved_12_package())) == 0


@pytest.mark.parametrize("component", ("cuda-bindings", "cuda-python"))
@pytest.mark.parametrize("notes_version", ("12.9.7", "12.9.9"))
@pytest.mark.agent_authored(model="gpt-6")
def test_release_rejects_other_version_notes(tmp_path, capsys, component, notes_version):
    write_notes(tmp_path, "cuda_bindings_12", notes_version)
    write_notes(tmp_path, "cuda_python", notes_version)

    assert main(release_args(tmp_path, component, resolved_12_package())) == 1
    assert "ERROR: missing or empty release notes" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("tag", "resolved_version"),
    (
        ("v12.9.8", "12.9.9"),
        ("v12.9.8rc1", "12.9.8"),
        ("v12.9.8.dev1", "12.9.8"),
        ("v12.9.8", "12.9.8.post1"),
    ),
)
@pytest.mark.agent_authored(model="gpt-6")
def test_release_rejects_resolver_version_mismatch(tmp_path, capsys, tag, resolved_version):
    write_notes(tmp_path, "cuda_bindings_12", resolved_version)
    package = resolved_12_package()
    package["release_version"] = resolved_version
    args = release_args(tmp_path, "cuda-bindings", package)
    args[args.index("--git-tag") + 1] = tag

    assert main(args) == 2
    assert "does not match release tag" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("component", "tag"),
    (
        ("cuda-bindings", "v12.9.8.post1"),
        ("cuda-python", "v12.9.8.post1"),
        ("cuda-core", "cuda-core-v1.1.1.post1"),
    ),
)
@pytest.mark.agent_authored(model="gpt-6")
def test_main_skips_post_release_without_notes(tmp_path, capsys, component, tag):
    package = resolved_12_package()
    package["release_version"] = "12.9.8.post1"
    args = release_args(tmp_path, component, package)
    args[args.index("--git-tag") + 1] = tag

    assert main(args) == 0
    assert "skipping release-notes check" in capsys.readouterr().out


@pytest.mark.parametrize("version", ["1.5.2", "1.5.2.post1"])
@pytest.mark.parametrize("contents", ["", "Release notes."])
@pytest.mark.agent_authored(model="gpt-6.1")
def test_pathfinder_notes_allow_empty_files_and_require_post_notes(tmp_path, monkeypatch, version, contents):
    notes = tmp_path / "cuda_pathfinder" / "docs" / "source" / "release" / f"{version}-notes.rst"
    notes.parent.mkdir(parents=True)
    notes.write_text(contents)
    output = tmp_path / "output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))

    assert (
        main(
            ["--git-tag", f"cuda-pathfinder-v{version}", "--component", "cuda-pathfinder", "--repo-root", str(tmp_path)]
        )
        == 0
    )
    assert output.read_text() == f"version={version}\n"


@pytest.mark.parametrize("version", ["1.5.2", "1.5.2.post1"])
@pytest.mark.agent_authored(model="gpt-6.1")
def test_pathfinder_missing_notes_fail_without_version_output(tmp_path, monkeypatch, version):
    output = tmp_path / "output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))

    assert (
        main(
            ["--git-tag", f"cuda-pathfinder-v{version}", "--component", "cuda-pathfinder", "--repo-root", str(tmp_path)]
        )
        == 1
    )
    assert not output.exists()


@pytest.mark.parametrize(
    "component,tag,version,package",
    [
        ("cuda-core", "cuda-core-v1.1.1", "1.1.1", "cuda_core"),
        ("cuda-core", "cuda-core-v1.1.1.post1", "1.1.1.post1", "cuda_core"),
        ("cuda-bindings", "v12.9.8", "12.9.8", "cuda_bindings_12"),
        ("cuda-bindings", "v13.4.1", "13.4.1", "cuda_bindings_13"),
        ("cuda-bindings", "v12.9.8.post1", "12.9.8.post1", "cuda_bindings_12"),
        ("cuda-python", "v13.3.0", "13.3.0", "cuda_python"),
        ("cuda-python", "v12.9.8.post1", "12.9.8.post1", "cuda_python"),
    ],
)
@pytest.mark.agent_authored(model="gpt-6.1")
def test_successful_checks_expose_version_for_every_policy(tmp_path, monkeypatch, component, tag, version, package):
    if ".post" not in version:
        write_notes(tmp_path, package, version)
    output = tmp_path / "output"
    output.write_text("existing=value\n")
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))

    assert main(["--git-tag", tag, "--component", component, "--repo-root", str(tmp_path)]) == 0
    assert output.read_text() == f"existing=value\nversion={version}\n"


@pytest.mark.parametrize("tag", ["invalid", "cuda-core-v1.5.2"])
@pytest.mark.agent_authored(model="gpt-6.1")
def test_invalid_pathfinder_tags_do_not_expose_version(tmp_path, monkeypatch, tag):
    output = tmp_path / "output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))

    assert main(["--git-tag", tag, "--component", "cuda-pathfinder", "--repo-root", str(tmp_path)]) == 2
    assert not output.exists()


@pytest.mark.parametrize(
    ("component", "tag", "package", "version"),
    (
        ("cuda-bindings", "v12.9.8", "cuda_bindings_12", "12.9.8"),
        ("cuda-python", "v12.9.8", "cuda_python", "12.9.8"),
        ("cuda-core", "cuda-core-v1.1.1", "cuda_core", "1.1.1"),
    ),
)
@pytest.mark.parametrize("problem", ("missing", "empty"))
@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_missing_or_empty_notes_do_not_expose_version(tmp_path, monkeypatch, component, tag, package, version, problem):
    if problem == "empty":
        write_notes(tmp_path, package, version, content="")
    output = tmp_path / "output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))

    assert main(["--git-tag", tag, "--component", component, "--repo-root", str(tmp_path)]) == 1
    assert not output.exists()
