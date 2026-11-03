# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the copyright-year rule in toolshed/check_spdx.py.

The rule must give the same answer however the hook is invoked (``git commit``,
``pre-commit run --all-files``, CI), so each test builds a throwaway git repository
with controlled commit dates and runs the script against it. Years are derived from
the current year so the tests do not expire.

    python -m pytest ci/tools/tests/test_check_spdx.py -v --noconftest
"""

from __future__ import annotations

import datetime
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("pathspec", reason="check_spdx.py requires pathspec")

REPO = Path(__file__).resolve().parents[3]
CHECKER = REPO / "toolshed" / "check_spdx.py"
LICENSE = REPO / "LICENSE"

CURRENT_YEAR = datetime.datetime.now(tz=datetime.UTC).year
SAMPLE = "toolshed/sample.py"

CONTENT = (
    "# SPDX-FileCopyrightText: Copyright (c) {years} NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n"
    "# SPDX-License-Identifier: Apache-2.0\n"
    "\n"
    "pass\n"
)


def git(repo, *args, year=None):
    env = dict(os.environ)
    env.update(
        GIT_AUTHOR_NAME="Test",
        GIT_AUTHOR_EMAIL="test@example.com",
        GIT_COMMITTER_NAME="Test",
        GIT_COMMITTER_EMAIL="test@example.com",
    )
    if year is not None:
        date = f"{year}-06-15T12:00:00+0000"
        env.update(GIT_AUTHOR_DATE=date, GIT_COMMITTER_DATE=date)
    subprocess.run(  # noqa: S603
        ["git", "-c", "commit.gpgsign=false", *args],  # noqa: S607
        cwd=repo,
        env=env,
        check=True,
        capture_output=True,
    )


def commit(repo, relpath, years, *, year):
    path = repo / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(CONTENT.format(years=years), encoding="utf-8")
    git(repo, "add", relpath)
    git(repo, "commit", "-m", f"add {relpath}", year=year)
    return path


def run_checker(repo, *args):
    # The script reads LICENSE and runs git relative to the working directory,
    # as pre-commit does from the repository root.
    return subprocess.run(  # noqa: S603
        [sys.executable, str(CHECKER), *args, SAMPLE],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init")
    shutil.copy(LICENSE, tmp_path / "LICENSE")
    git(tmp_path, "add", "LICENSE")
    git(tmp_path, "commit", "-m", "init", year=CURRENT_YEAR - 5)
    return tmp_path


@pytest.mark.agent_authored(model="claude-sonnet-5.5")
@pytest.mark.parametrize(
    ("commit_age", "copyright_age", "passes"),
    [
        # Not modified this year, so last year's copyright is still correct.
        pytest.param(1, 1, True, id="untouched-since-last-year"),
        # The bug in #2273: committed this year, nothing staged, stale copyright.
        pytest.param(0, 1, False, id="committed-this-year-stale-copyright"),
    ],
)
def test_committed_file_is_checked_against_its_last_commit_year(repo, commit_age, copyright_age, passes):
    commit(repo, SAMPLE, str(CURRENT_YEAR - copyright_age), year=CURRENT_YEAR - commit_age)

    result = run_checker(repo)

    assert (result.returncode == 0) is passes, result.stdout + result.stderr
    assert ("OUTDATED" in result.stdout) is (not passes)


@pytest.mark.agent_authored(model="claude-sonnet-5.5")
def test_staged_edit_requires_current_year_and_fix_writes_it(repo):
    # The `git commit` path: the edited file is staged when the hook runs.
    path = commit(repo, SAMPLE, str(CURRENT_YEAR - 3), year=CURRENT_YEAR - 2)
    path.write_text(path.read_text(encoding="utf-8") + "# edited\n", encoding="utf-8")
    git(repo, "add", SAMPLE)

    check = run_checker(repo)
    assert check.returncode == 1, check.stdout + check.stderr
    assert f"(expected {CURRENT_YEAR})" in check.stdout

    # --fix reports that it changed something, as the hook always has.
    fix = run_checker(repo, "--fix")
    assert fix.returncode == 1, fix.stdout + fix.stderr
    assert f"Copyright (c) {CURRENT_YEAR - 3}-{CURRENT_YEAR} " in path.read_text(encoding="utf-8")

    # The rewrite is stable: a second run accepts the file.
    assert run_checker(repo).returncode == 0


@pytest.mark.agent_authored(model="claude-sonnet-5.5")
def test_fix_writes_last_commit_year_not_current_year(repo):
    path = commit(repo, SAMPLE, str(CURRENT_YEAR - 3), year=CURRENT_YEAR - 2)

    result = run_checker(repo, "--fix")

    # --fix reports that it changed something, as the hook always has.
    assert result.returncode == 1, result.stdout + result.stderr
    assert f"Copyright (c) {CURRENT_YEAR - 3}-{CURRENT_YEAR - 2} " in path.read_text(encoding="utf-8")
