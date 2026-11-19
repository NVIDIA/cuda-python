# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
from pathlib import Path

import pytest
import yaml

WORKFLOW = Path(__file__).resolve().parents[3] / ".github/workflows/ci.yml"


@pytest.fixture
def tag_release_notes_script():
    steps = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]["should-skip"]["steps"]
    return next(step["run"] for step in steps if step.get("id") == "tag-release-notes")


def run_check(script, root, tag):
    return subprocess.run(  # noqa: S603 - execute the checked-in workflow script in a disposable fixture.
        ["/bin/bash", "--noprofile", "--norc", "-euo", "pipefail", "-c", script],
        cwd=root,
        env={"PATH": "/usr/bin:/bin", "RELEASE_TAG": tag},
        capture_output=True,
        text=True,
        timeout=10,
    )


def create_note(root, package, version):
    note = root / package / "docs/source/release" / f"{version}-notes.rst"
    note.parent.mkdir(parents=True, exist_ok=True)
    note.touch()
    return note


@pytest.mark.parametrize(
    "tag,package,version",
    [
        ("v12.9.11", "cuda_bindings_12", "12.9.11"),
        ("v13.5.1", "cuda_bindings_13", "13.5.1"),
        ("cuda-core-v1.3.2", "cuda_core", "1.3.2"),
        ("cuda-pathfinder-v1.9.2", "cuda_pathfinder", "1.9.2"),
    ],
)
@pytest.mark.agent_authored(model="GPT-6")
def test_public_tag_requires_only_its_own_release_note(tag_release_notes_script, tmp_path, tag, package, version):
    result = run_check(tag_release_notes_script, tmp_path, tag)
    assert result.returncode != 0

    create_note(tmp_path, package, version)
    result = run_check(tag_release_notes_script, tmp_path, tag)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.agent_authored(model="GPT-6")
def test_release_note_under_the_wrong_bindings_root_does_not_satisfy_the_tag(tag_release_notes_script, tmp_path):
    create_note(tmp_path, "cuda_bindings_13", "12.9.11")

    result = run_check(tag_release_notes_script, tmp_path, "v12.9.11")

    assert result.returncode != 0


@pytest.mark.parametrize("tag", ["v12.9.11a0", "v13.5.1a1.dev2", "cuda-pathfinder-v1.9.2.dev3"])
@pytest.mark.agent_authored(model="GPT-6")
def test_internal_alpha_and_development_tags_do_not_require_notes(tag_release_notes_script, tmp_path, tag):
    result = run_check(tag_release_notes_script, tmp_path, tag)

    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "tag,package,base_version,exact_version",
    [
        ("v13.5.1b1", "cuda_bindings_13", "13.5.1", "13.5.1b1"),
        ("cuda-core-v1.3.2rc1", "cuda_core", "1.3.2", "1.3.2rc1"),
        ("cuda-pathfinder-v1.9.2.post1", "cuda_pathfinder", "1.9.2", "1.9.2.post1"),
    ],
)
@pytest.mark.agent_authored(model="GPT-6")
def test_public_suffix_tags_require_an_exact_note(
    tag_release_notes_script, tmp_path, tag, package, base_version, exact_version
):
    create_note(tmp_path, package, base_version)
    result = run_check(tag_release_notes_script, tmp_path, tag)
    assert result.returncode != 0

    create_note(tmp_path, package, exact_version)
    result = run_check(tag_release_notes_script, tmp_path, tag)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("tag", ["v12.9.11.post1", "cuda-core-v1.3.2.post1"])
@pytest.mark.agent_authored(model="GPT-6")
def test_non_pathfinder_post_tags_do_not_require_notes(tag_release_notes_script, tmp_path, tag):
    result = run_check(tag_release_notes_script, tmp_path, tag)

    assert result.returncode == 0, result.stdout + result.stderr
