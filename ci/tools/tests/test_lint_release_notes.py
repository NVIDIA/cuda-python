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

NAME = "fix-something-0123456789abcdef.yaml"


def _lint(tmp_path, text, name=NAME):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return lint_release_notes.lint_note(path)


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
@pytest.mark.parametrize(
    "text",
    [
        "fixes:\n  - A fix.\n",
        "---\nfixes:\n  - |\n    A fix.\n    Over two lines.\nissues:\n  - A known issue.\n",
        "prelude: >\n  An introduction.\nfeatures:\n  - A feature.\n",
        "prelude:\n  - An introduction.\n",
    ],
)
def test_valid_notes_pass(tmp_path, text):
    assert _lint(tmp_path, text) == []


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "must be a YAML mapping"),
        ("- fixes\n", "must be a YAML mapping"),
        ("fixes: [unclosed\n", "not valid YAML"),
        ("fix:\n  - Typo in the section name.\n", "unknown section 'fix'"),
        ("fixes: A bare string.\n", "must be a list of entries"),
        ("fixes: []\n", "non-empty list"),
        ("fixes:\n", "non-empty list"),
        ("fixes:\n  - ''\n", "not text"),
        ("fixes:\n  - 5\n", "not text"),
        ("fixes:\n  - TODO fill this in\n", "TODO"),
        ("features:\n  - A feature.\nfixes:\n  - TODO fill this in\n", "'fixes' still has"),
        ("fixes:\n  - TODO: unquoted colon is a mapping\n", "not text"),
        ('"fixes":\n  - A fix.\n', "plain `fixes:`"),
        ("'prelude': An introduction.\n", "plain `prelude:`"),
        ("  fixes:\n    - Indented key.\n", "plain `fixes:`"),
        ('"issues":\n  - A known issue.\n', "plain `issues:`"),
        ("'issues':\n  - A known issue.\n", "plain `issues:`"),
        ("issues :\n  - A known issue.\n", "plain `issues:`"),
    ],
)
def test_obvious_mistakes_are_reported(tmp_path, text, message):
    problems = _lint(tmp_path, text)
    assert len(problems) == 1
    assert message in problems[0]


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
@pytest.mark.parametrize(
    "name",
    [
        "config.yaml",
        "README.md",
        ".gitkeep",
        "fix.yaml",
        "Fix-0123456789abcdef.yaml",
        "fix-0123456789abcdef.yml",
        "fix-xyz.yaml",
    ],
)
def test_only_notes_belong_in_the_directory(tmp_path, name):
    problems = _lint(tmp_path, "fixes:\n  - A fix.\n", name=name)
    assert len(problems) == 1
    assert "not named like a note" in problems[0]


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
def test_a_freshly_created_note_fails_until_it_is_filled_in(tmp_path):
    path = add_note.create_note("bindings", "Something new", repo_root=tmp_path)
    problems = lint_release_notes.lint_note(path)
    assert len(problems) == len(release_ranges.SECTIONS)
    assert all("TODO" in problem for problem in problems)

    path.write_text("---\nfixes:\n  - Something was fixed.\n", encoding="utf-8")
    assert lint_release_notes.lint_note(path) == []


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
def test_lint_package_reports_every_file_and_ignores_a_missing_directory(tmp_path):
    package = release_ranges.PACKAGES["cuda-core"]
    assert lint_release_notes.lint_package(tmp_path, package) == []

    notes_dir = tmp_path / package.notes_dir
    notes_dir.mkdir(parents=True)
    (notes_dir / NAME).write_text("fixes:\n  - A fix.\n", encoding="utf-8")
    (notes_dir / "config.yaml").write_text("sections: []\n", encoding="utf-8")
    (notes_dir / "other-0123456789abcdef.yaml").write_text("oops: 1\n", encoding="utf-8")
    problems = lint_release_notes.lint_package(tmp_path, package)
    assert sorted(path.name for path, _message in problems) == ["config.yaml", "other-0123456789abcdef.yaml"]


@pytest.mark.agent_authored(model="claude-sonnet-5-5")
def test_the_notes_in_this_checkout_are_clean(capsys):
    assert lint_release_notes.main() == 0, capsys.readouterr().err
