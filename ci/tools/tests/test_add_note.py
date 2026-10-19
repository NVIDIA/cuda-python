# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "toolshed"))
sys.path.insert(0, str(REPO_ROOT / "cuda_python" / "docs" / "exts"))
import add_note
import release_ranges


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
def test_the_note_types_are_the_sections():
    assert set(add_note.NOTE_TYPES) == set(release_ranges.SECTIONS)


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
def test_the_template_is_only_todo_text():
    assert add_note.TEMPLATE.startswith("TODO")
    assert "SPDX" not in add_note.TEMPLATE


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
@pytest.mark.parametrize(
    ("name", "component"),
    [
        ("cuda-core", "cuda-core"),
        ("cuda_core", "cuda-core"),
        ("core", "cuda-core"),
        ("Bindings", "cuda-bindings"),
        ("cuda_pathfinder", "cuda-pathfinder"),
        ("python", "cuda-python"),
    ],
)
def test_package_names(name, component):
    assert add_note.find_package(name) == component


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
def test_an_unknown_package_lists_the_choices():
    with pytest.raises(ValueError, match="cuda-bindings, cuda-core, cuda-pathfinder, cuda-python"):
        add_note.find_package("numpy")


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
@pytest.mark.parametrize(
    ("description", "slug"),
    [
        ("Fix the Foo (bar) crash!", "fix-the-foo-bar-crash"),
        ("  leading and trailing  ", "leading-and-trailing"),
        ("x" * 100, "x" * add_note.MAX_SLUG_LENGTH),
        ("Ünïcode and symbols #1", "n-code-and-symbols-1"),
    ],
)
def test_slugify(description, slug):
    assert add_note.slugify(description) == slug


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
@pytest.mark.parametrize("description", ["", "   ", "!!!"])
def test_a_description_without_letters_or_digits_is_an_error(description):
    with pytest.raises(ValueError, match="no letters or digits"):
        add_note.slugify(description)


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
def test_create_note_makes_a_valid_file_in_the_packages_directory(tmp_path):
    path = add_note.create_note("core", "fix", "Fix the thing", repo_root=tmp_path)
    assert path.parent == tmp_path / release_ranges.PACKAGES["cuda-core"].notes_dir
    assert release_ranges.NOTE_NAME_RE.fullmatch(path.name)
    assert path.name.startswith("fix-fix-the-thing-")
    assert path.read_text(encoding="utf-8") == add_note.TEMPLATE
    # Two notes with the same description do not collide.
    assert add_note.create_note("core", "fix", "Fix the thing", repo_root=tmp_path) != path


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
def test_main_works_from_any_directory_and_reports_errors(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(add_note, "REPO_ROOT", tmp_path)
    assert add_note.main(["cuda-pathfinder", "feature", "A new note"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("Created cuda_pathfinder/releasenotes/feature-a-new-note-")
    assert len(list((tmp_path / "cuda_pathfinder" / "releasenotes").glob("*.rst"))) == 1

    assert add_note.main(["core", "other", "!!!"]) == 2
    assert "no letters or digits" in capsys.readouterr().err


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
def test_edit_runs_the_editor_on_the_new_file(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(add_note, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(add_note.subprocess, "run", lambda cmd, **_kwargs: calls.append(cmd) or _Done())
    monkeypatch.delenv("VISUAL", raising=False)
    monkeypatch.setenv("EDITOR", "myeditor --wait")
    assert add_note.main(["core", "other", "Edit me", "--edit"]) == 0
    assert calls[0][:2] == ["myeditor", "--wait"]
    assert calls[0][2].endswith(".rst")

    monkeypatch.delenv("EDITOR")
    assert add_note.main(["core", "other", "Edit me too", "--edit"]) == 2


class _Done:
    returncode = 0


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
def test_the_package_choices_come_from_the_packages_and_short_names_still_work(capsys):
    with pytest.raises(SystemExit) as exit_info:
        add_note.main(["numpy", "other", "x"])
    assert exit_info.value.code == 2
    error = capsys.readouterr().err
    assert "invalid choice: 'cuda-numpy'" in error
    for component in release_ranges.PACKAGES:
        assert component in error


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
def test_an_unknown_note_type_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="unknown note type 'bug'"):
        add_note.create_note("core", "bug", "Fix the thing", repo_root=tmp_path)
