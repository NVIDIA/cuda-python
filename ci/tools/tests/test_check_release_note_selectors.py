# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from ci.tools.check_staged_release_notes import version_selector_errors

SELECTORS = (
    "cuda_python/docs/nv-versions.json",
    "cuda_bindings_13/docs/nv-versions.json",
    "cuda_core/docs/nv-versions.json",
    "cuda_pathfinder/docs/nv-versions.json",
)


def git(repo: Path, *args: str) -> None:
    subprocess.run(  # noqa: S603
        ["git", "-C", str(repo), *args],  # noqa: S607
        capture_output=True,
        check=True,
    )


def stage_file(repo: Path, relative: str, contents: str) -> str:
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")
    git(repo, "add", relative)
    return relative


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-q")
    for selector in SELECTORS:
        stage_file(tmp_path, selector, "[]\n")
    return tmp_path


@pytest.mark.parametrize(
    "package,version,selector,url",
    [
        (
            "cuda_bindings_12",
            "12.9.11",
            "cuda_bindings_13/docs/nv-versions.json",
            "https://nvidia.github.io/cuda-python/cuda-bindings/12.9.11/",
        ),
        (
            "cuda_bindings_13",
            "13.5.1",
            "cuda_bindings_13/docs/nv-versions.json",
            "https://nvidia.github.io/cuda-python/cuda-bindings/13.5.1/",
        ),
        (
            "cuda_python",
            "12.9.11",
            "cuda_python/docs/nv-versions.json",
            "https://nvidia.github.io/cuda-python/12.9.11/",
        ),
        (
            "cuda_core",
            "1.3.2",
            "cuda_core/docs/nv-versions.json",
            "https://nvidia.github.io/cuda-python/cuda-core/1.3.2/",
        ),
        (
            "cuda_pathfinder",
            "1.9.2",
            "cuda_pathfinder/docs/nv-versions.json",
            "https://nvidia.github.io/cuda-python/cuda-pathfinder/1.9.2/",
        ),
    ],
)
@pytest.mark.agent_authored(model="gpt-6")
def test_release_notes_require_their_version_and_exact_url_in_the_correct_selector(
    repo, package, version, selector, url
):
    note = stage_file(repo, f"{package}/docs/source/release/{version}-notes.rst", "Release notes.\n")

    for entries in ([], [{"version": version, "url": url.removesuffix("/")}]):
        stage_file(repo, selector, json.dumps(entries))
        errors = version_selector_errors(repo)
        assert len(errors) == 1
        assert all(value in errors[0] for value in (note, version, selector))

    stage_file(repo, selector, json.dumps([{"version": version, "url": url}]))
    assert version_selector_errors(repo) == []


@pytest.mark.parametrize(
    "package,version",
    [
        ("cuda_bindings_12", "12.9.10"),
        ("cuda_bindings_13", "13.5.0"),
        ("cuda_python", "12.9.10"),
        ("cuda_python", "13.5.0"),
        ("cuda_core", "1.3.1"),
        ("cuda_pathfinder", "1.9.1"),
    ],
)
@pytest.mark.agent_authored(model="gpt-6")
def test_pre_policy_release_notes_do_not_require_selector_entries(repo, package, version):
    stage_file(repo, f"{package}/docs/source/release/{version}-notes.rst", "Release notes.\n")

    assert version_selector_errors(repo) == []


@pytest.mark.agent_authored(model="gpt-6")
def test_unstaged_json_correction_does_not_replace_the_staged_selector(repo):
    selector = "cuda_core/docs/nv-versions.json"
    stage_file(repo, "cuda_core/docs/source/release/1.3.2-notes.rst", "Release notes.\n")
    stage_file(
        repo,
        selector,
        json.dumps([{"version": "1.3.2", "url": "https://nvidia.github.io/cuda-python/cuda-core/latest/"}]),
    )
    staged_errors = version_selector_errors(repo)
    assert len(staged_errors) == 1
    (repo / selector).write_text(
        json.dumps([{"version": "1.3.2", "url": "https://nvidia.github.io/cuda-python/cuda-core/1.3.2/"}]),
        encoding="utf-8",
    )

    assert version_selector_errors(repo) == staged_errors

    git(repo, "add", selector)
    assert version_selector_errors(repo) == []


@pytest.mark.agent_authored(model="gpt-6")
def test_staged_selector_deletion_fails_even_without_release_notes(repo):
    selector = "cuda_core/docs/nv-versions.json"
    git(repo, "rm", "--cached", selector)
    assert (repo / selector).is_file()

    with pytest.raises(subprocess.CalledProcessError):
        version_selector_errors(repo)
