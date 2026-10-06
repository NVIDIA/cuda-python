# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import check_pr_release_notes
from check_pr_release_notes import EXIT_GH_FAILED, EXIT_PROBLEMS, ChangedFile, changed_files, find_problems, main

NOTE = "cuda_bindings/releasenotes/fix-thing-0123456789abcdef.yaml"
SOURCE = "cuda_bindings/cuda/bindings/driver.pyx"


def _files(*paths, status="modified"):
    return [ChangedFile(path, status) for path in paths]


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
@pytest.mark.parametrize(
    "path",
    [
        "cuda_bindings/cuda/bindings/driver.pyx",
        "cuda_bindings/cuda/bindings/_internal/nvml_linux.pyx",
        "cuda_bindings/pyproject.toml",
        "cuda_bindings/setup.py",
        "cuda_bindings/build_hooks.py",
        "cuda_bindings/MANIFEST.in",
    ],
)
def test_source_changes_need_a_note(path):
    problems = find_problems(_files(path), [])
    assert len(problems) == 1
    assert "cuda-bindings" in problems[0]


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
@pytest.mark.parametrize(
    "path",
    [
        "cuda_bindings/docs/source/install.rst",
        "cuda_bindings/tests/test_nvml.py",
        "cuda_bindings/examples/basic.py",
        "cuda_bindings/pixi.toml",
        "cuda_bindings/pixi.lock",
        "cuda_bindings/README.md",
        "cuda_bindings/AGENTS.md",
        "cuda_bindings/releasenotes/README.md",
        "cuda_bindings/cudax/other.py",  # not under cuda_bindings/cuda/
        ".github/workflows/ci.yml",
        "ci/tools/check_pr_release_notes.py",
        "cuda_core/tests/test_device.py",
        "cuda_core/pixi.toml",
        "cuda_pathfinder/tests/test_find.py",
        "cuda_pathfinder/pixi.toml",
        "cuda_pathfinder/docs/source/index.rst",
        "cuda_python/docs/source/conf.py",
        "cuda_python/README.md",
        "pixi.toml",
    ],
)
def test_other_changes_need_no_note(path):
    assert find_problems(_files(path), []) == []


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
@pytest.mark.parametrize(
    "path",
    ["cuda_core/cuda/core/_device.pyx", "cuda_core/pyproject.toml", "cuda_core/setup.py", "cuda_core/MANIFEST.in"],
)
def test_cuda_core_source_changes_need_a_core_note(path):
    problems = find_problems(_files(path), [])
    assert len(problems) == 1
    assert "cuda-core" in problems[0]


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
@pytest.mark.parametrize("path", ["cuda_pathfinder/cuda/pathfinder/__init__.py", "cuda_pathfinder/pyproject.toml"])
def test_cuda_pathfinder_source_changes_need_a_pathfinder_note(path):
    problems = find_problems(_files(path), [])
    assert len(problems) == 1
    assert "cuda-pathfinder" in problems[0]
    assert find_problems(_files(path, "cuda_pathfinder/releasenotes/x-0123456789abcdef.yaml"), []) == []


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
@pytest.mark.parametrize("path", ["cuda_python/setup.py", "cuda_python/pyproject.toml"])
def test_cuda_python_source_changes_need_a_python_note(path):
    problems = find_problems(_files(path), [])
    assert len(problems) == 1
    assert "cuda-python" in problems[0]
    assert find_problems(_files(path, "cuda_python/releasenotes/x-0123456789abcdef.yaml"), []) == []


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
def test_each_touched_package_needs_its_own_note():
    changed = _files(SOURCE, "cuda_core/cuda/core/_device.pyx", NOTE)
    problems = find_problems(changed, [])
    assert len(problems) == 1
    assert "cuda-core" in problems[0]
    changed += _files("cuda_core/releasenotes/other-0123456789abcdef.yaml")
    assert find_problems(changed, []) == []


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
@pytest.mark.parametrize("status", ["added", "modified", "renamed", "copied", "changed"])
def test_adding_or_editing_a_note_satisfies_the_check(status):
    assert find_problems([ChangedFile(SOURCE, "modified"), ChangedFile(NOTE, status)], ["bug"]) == []


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
def test_deleting_a_note_does_not_satisfy_the_check():
    changed = [ChangedFile(SOURCE, "modified"), ChangedFile(NOTE, "removed")]
    assert len(find_problems(changed, [])) == 1


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
def test_renaming_a_note_away_does_not_satisfy_the_check():
    # The new path is outside notes/; the old path (previous_filename) is not a note either.
    changed = [ChangedFile(SOURCE, "modified"), ChangedFile("docs/old-note.yaml", "renamed", NOTE)]
    assert len(find_problems(changed, [])) == 1
    # Renaming one *into* notes/ is a note.
    changed = [ChangedFile(SOURCE, "modified"), ChangedFile(NOTE, "renamed", "somewhere/else.yaml")]
    assert find_problems(changed, []) == []


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
@pytest.mark.parametrize(
    "path",
    [
        "cuda_bindings/releasenotes/.gitkeep",
        "cuda_bindings/releasenotes/README.md",
        "cuda_bindings/releasenotes/fix-0123456789abcdef.yaml.orig",
        "cuda_bindings/releasenotes/config.yaml",
        "cuda_bindings/releasenotes/fix.yaml",  # not named like a note (no random suffix)
        "cuda_bindings/releasenotes/sub/fix-0123456789abcdef.yaml",  # not directly in the directory
        "cuda_core/releasenotes/fix-0123456789abcdef.yaml",  # another package's note
    ],
)
def test_only_yaml_files_in_the_packages_notes_dir_are_notes(path):
    assert len(find_problems(_files(SOURCE, path), [])) == 1


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
def test_renaming_a_source_file_out_of_the_sources_still_needs_a_note():
    changed = [ChangedFile("cuda_bindings/docs/driver.pyx", "renamed", SOURCE)]
    assert len(find_problems(changed, [])) == 1
    assert find_problems([*changed, ChangedFile(NOTE, "added")], []) == []


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
def test_a_file_name_with_a_newline_cannot_fake_a_note():
    # A single file whose *name* contains a newline and a note path. Parsed as JSON records it is
    # one source-like name, not two files.
    fake = f"x\n{NOTE}"
    assert len(find_problems([ChangedFile(SOURCE, "modified"), ChangedFile(fake, "added")], [])) == 1


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
def test_config_or_readme_in_the_notes_dir_is_not_a_note():
    changed = _files(SOURCE, "cuda_bindings/releasenotes/config.yaml")
    assert len(find_problems(changed, [])) == 1


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
@pytest.mark.parametrize("label", ["skip-release-note", "Skip-Release-Note"])
def test_skip_label(label):
    assert find_problems(_files(SOURCE), ["bug", label]) == []


def _labels(*names):
    return json.dumps([{"name": name} for name in names])


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
def test_main_exit_codes(monkeypatch, capsys):
    calls = []

    def fake_changed_files(repo, pr_number):
        calls.append((repo, pr_number))
        return _files(SOURCE)

    monkeypatch.setattr(check_pr_release_notes, "changed_files", fake_changed_files)

    argv = ["--repo", "o/r", "--pr", "7"]
    assert main([*argv, "--labels-json", _labels("bug")]) == EXIT_PROBLEMS
    assert capsys.readouterr().out.startswith("- **Missing release note**")
    assert main([*argv, "--labels-json", _labels("bug", "skip-release-note")]) == 0
    assert capsys.readouterr().out == ""
    assert calls == [("o/r", "7"), ("o/r", "7")]

    monkeypatch.setattr(check_pr_release_notes, "changed_files", lambda *_args: _files("README.md"))
    assert main([*argv, "--labels-json", "[]"]) == 0


def _fake_gh(monkeypatch, *, stdout="", stderr="", returncode=0):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr=stderr)

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
def test_changed_files_parses_json_records_one_per_line(monkeypatch):
    tricky = f"x\n{NOTE}"
    lines = [
        {"filename": "a.py", "status": "modified", "previous_filename": None},
        {"filename": "new/b.py", "status": "renamed", "previous_filename": "old/b.py"},
        {"filename": tricky, "status": "added", "previous_filename": None},
    ]
    calls = _fake_gh(monkeypatch, stdout="\n".join(json.dumps(line) for line in lines) + "\n\n")
    assert changed_files("o/r", "7") == [
        ChangedFile("a.py", "modified", None),
        ChangedFile("new/b.py", "renamed", "old/b.py"),
        ChangedFile(tricky, "added", None),
    ]
    cmd, kwargs = calls[0]
    assert cmd[:4] == ["gh", "api", "--paginate", "repos/o/r/pulls/7/files"]
    assert cmd[cmd.index("--jq") + 1] == check_pr_release_notes.CHANGED_FILES_JQ
    assert kwargs["capture_output"] is True


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
def test_changed_files_jq_emits_one_compact_json_object_per_file():
    # `jq` is not required by the tests; just pin down what is asked of gh.
    assert "tojson" in check_pr_release_notes.CHANGED_FILES_JQ
    assert "previous_filename" in check_pr_release_notes.CHANGED_FILES_JQ


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
def test_gh_failure_reports_gh_stderr_and_exits_2(monkeypatch, capsys):
    _fake_gh(monkeypatch, stderr="HTTP 404: Not Found (https://api.github.com/repos/o/r/pulls/7/files)", returncode=1)
    assert main(["--repo", "o/r", "--pr", "7", "--labels-json", "[]"]) == EXIT_GH_FAILED
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "HTTP 404: Not Found" in captured.err
    assert "PR #7 in o/r" in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
def test_gh_missing_or_garbled_output_is_a_clean_error(monkeypatch, capsys):
    def missing(cmd, **kwargs):
        raise FileNotFoundError(2, "No such file or directory", "gh")

    monkeypatch.setattr(subprocess, "run", missing)
    assert main(["--repo", "o/r", "--pr", "7", "--labels-json", "[]"]) == EXIT_GH_FAILED
    assert "could not run `gh`" in capsys.readouterr().err

    _fake_gh(monkeypatch, stdout="not json\n")
    assert main(["--repo", "o/r", "--pr", "7", "--labels-json", "[]"]) == EXIT_GH_FAILED
    assert "unexpected output" in capsys.readouterr().err
