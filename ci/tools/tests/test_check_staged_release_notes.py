# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from ci.tools.check_staged_release_notes import missing_release_note_pairs


def git(repo: Path, *args: str) -> None:
    subprocess.run(  # noqa: S603
        ["git", "-C", str(repo), *args],  # noqa: S607
        capture_output=True,
        check=True,
    )


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-q")
    return tmp_path


def write_note(repo: Path, package: str, version: str) -> str:
    relative = f"{package}/docs/source/release/{version}-notes.rst"
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("Release notes.\n", encoding="utf-8")
    return relative


@pytest.mark.parametrize(
    "package,version",
    [
        ("cuda_bindings_12", "12.9.11"),
        ("cuda_bindings_13", "13.5.1"),
        ("cuda_bindings_14", "14.0.0"),
        ("cuda_bindings_15", "15.0.0"),
    ],
)
@pytest.mark.agent_authored(model="gpt-6")
def test_numeric_notes_require_a_tracked_metapackage_counterpart(repo, package, version):
    bindings = write_note(repo, package, version)
    git(repo, "add", bindings)
    counterpart = f"cuda_python/docs/source/release/{version}-notes.rst"

    assert missing_release_note_pairs(repo) == [counterpart]

    write_note(repo, "cuda_python", version)
    assert missing_release_note_pairs(repo) == [counterpart]

    git(repo, "add", counterpart)
    assert missing_release_note_pairs(repo) == []


@pytest.mark.parametrize(
    "package,version",
    [
        ("cuda_bindings_12", "12.9.11"),
        ("cuda_bindings_13", "13.5.1"),
        ("cuda_bindings_14", "14.0.0"),
        ("cuda_bindings_15", "15.0.0"),
    ],
)
@pytest.mark.agent_authored(model="gpt-6")
def test_metapackage_notes_require_a_tracked_bindings_counterpart(repo, package, version):
    git(repo, "add", write_note(repo, "cuda_python", version))
    expected = f"{package}/docs/source/release/{version}-notes.rst"

    assert missing_release_note_pairs(repo) == [expected]

    counterpart = write_note(repo, package, version)
    assert missing_release_note_pairs(repo) == [expected]

    git(repo, "add", counterpart)
    assert missing_release_note_pairs(repo) == []


@pytest.mark.agent_authored(model="gpt-6")
def test_templates_and_prerelease_notes_do_not_require_counterparts(repo):
    for package, version in [
        ("cuda_bindings_12", "12.X.Y"),
        ("cuda_bindings_13", "13.X.Y"),
        ("cuda_bindings_13", "13.5.1rc1"),
        ("cuda_python", "12.X.Y"),
        ("cuda_python", "13.5.1rc1"),
    ]:
        git(repo, "add", write_note(repo, package, version))

    assert missing_release_note_pairs(repo) == []


@pytest.mark.parametrize(
    "package,other_package,version",
    [
        ("cuda_bindings_12", "cuda_bindings_13", "12.9.11"),
        ("cuda_bindings_13", "cuda_bindings_12", "13.5.1"),
    ],
)
@pytest.mark.agent_authored(model="gpt-6")
def test_wrong_major_note_does_not_satisfy_the_metapackage_counterpart(repo, package, other_package, version):
    git(repo, "add", write_note(repo, "cuda_python", version))
    git(repo, "add", write_note(repo, other_package, version))
    counterpart = f"{package}/docs/source/release/{version}-notes.rst"

    assert missing_release_note_pairs(repo) == [counterpart]

    write_note(repo, package, version)
    assert missing_release_note_pairs(repo) == [counterpart]

    git(repo, "add", counterpart)
    assert missing_release_note_pairs(repo) == []


@pytest.mark.parametrize("package", ["cuda_bindings_13", "cuda_python"])
@pytest.mark.parametrize("version", ["11.8.7", "12.9.10", "13.5.0"])
@pytest.mark.agent_authored(model="gpt-6")
def test_releases_before_the_pairing_policy_do_not_require_counterparts(repo, package, version):
    git(repo, "add", write_note(repo, package, version))

    assert missing_release_note_pairs(repo) == []


@pytest.mark.agent_authored(model="gpt-6")
def test_staged_counterpart_deletion_is_detected_even_if_the_file_remains(repo):
    bindings = write_note(repo, "cuda_bindings_12", "12.9.11")
    counterpart = write_note(repo, "cuda_python", "12.9.11")
    git(repo, "add", bindings, counterpart)
    git(repo, "rm", "--cached", counterpart)

    assert missing_release_note_pairs(repo) == [counterpart]
