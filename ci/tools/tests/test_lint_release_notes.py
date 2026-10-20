# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "ci" / "tools"))
sys.path.insert(0, str(REPO_ROOT / "toolshed"))
sys.path.insert(0, str(REPO_ROOT / "cuda_python" / "docs" / "exts"))
import add_note
import lint_release_notes
import release_ranges

NAME = "fix-something-0123456789abcdef.rst"


def _lint(tmp_path, text, name=NAME):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return lint_release_notes.lint_note(path)


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
@pytest.mark.parametrize("note_type", sorted(lint_release_notes.KNOWN_TYPES))
def test_valid_notes_pass(tmp_path, note_type):
    text = "A change.\n\nOver two paragraphs.\n"
    assert _lint(tmp_path, text, name=f"{note_type}-something-0123456789abcdef.rst") == []


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "is empty"),
        ("\n  \n", "is empty"),
        ("TODO fill this in\n", "TODO"),
        ("TODO: fill this in\n", "TODO"),
    ],
)
def test_obvious_mistakes_are_reported(tmp_path, text, message):
    problems = _lint(tmp_path, text)
    assert len(problems) == 1
    assert message in problems[0]


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
def test_an_unknown_type_is_reported(tmp_path):
    problems = _lint(tmp_path, "A fix.\n", name="bug-something-0123456789abcdef.rst")
    assert len(problems) == 1
    assert "unknown note type 'bug'" in problems[0]


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
@pytest.mark.parametrize(
    "name",
    [
        "config.yaml",
        "README.md",
        ".gitkeep",
        "fixes.rst",
        "something-0123456789abcdef.rst",
        "Fixes-something-0123456789abcdef.rst",
        "fix-something-0123456789abcdef.yaml",
        "fix-something-xyz.rst",
        "fix-0123456789abcdef.rst",
    ],
)
def test_only_notes_belong_in_the_directory(tmp_path, name):
    problems = _lint(tmp_path, "A fix.\n", name=name)
    assert len(problems) == 1
    assert "not named like a note" in problems[0]


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
def test_a_freshly_created_note_fails_until_it_is_filled_in(tmp_path):
    path = add_note.create_note("bindings", "fix", "Something new", repo_root=tmp_path)
    problems = lint_release_notes.lint_note(path)
    assert len(problems) == 1
    assert "TODO" in problems[0]

    path.write_text("Something was fixed.\n", encoding="utf-8")
    assert lint_release_notes.lint_note(path) == []


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
def test_lint_package_reports_every_file_and_ignores_a_missing_directory(tmp_path):
    package = release_ranges.PACKAGES["cuda-core"]
    assert lint_release_notes.lint_package(tmp_path, package) == []

    notes_dir = tmp_path / package.notes_dir
    notes_dir.mkdir(parents=True)
    (notes_dir / NAME).write_text("A fix.\n", encoding="utf-8")
    (notes_dir / "config.yaml").write_text("sections: []\n", encoding="utf-8")
    (notes_dir / "oops-other-0123456789abcdef.rst").write_text("A note.\n", encoding="utf-8")
    problems = lint_release_notes.lint_package(tmp_path, package)
    assert sorted(path.name for path, _message in problems) == ["config.yaml", "oops-other-0123456789abcdef.rst"]


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
def test_the_notes_in_this_checkout_are_clean(capsys):
    assert lint_release_notes.main() == 0, capsys.readouterr().err
