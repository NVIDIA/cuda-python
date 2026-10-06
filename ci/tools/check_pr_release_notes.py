# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check that a pull request that changes a package's sources adds or edits a release note.

Usage:
    python check_pr_release_notes.py --repo OWNER/REPO --pr NUMBER --labels-json '[{"name": ...}]'

The changed files are listed with the ``gh`` CLI, which needs ``GH_TOKEN``, as
JSON records (path, status, and the old path of a rename) so that unusual file
names cannot be mistaken for other paths.  Packages are those in
``release_ranges.PACKAGES``; a PR needs a note for each package whose
``source_paths`` it touches (as a new or an old path), unless it carries the
``skip-release-note`` label.  Anything outside ``source_paths`` (docs, tests,
CI, pixi files, ...) never needs a note.  A note is a file named like ``release_ranges.NOTE_NAME_RE``
in the package's ``releasenotes/`` that the PR adds, modifies or renames to;
deleting or renaming one away does not count.

Prints one markdown bullet per problem to stdout and exits with status 3 if
there are any.  Failing to list the files prints the reason to stderr and exits
with status 2, so that callers can tell problems from a crash.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cuda_python" / "docs" / "exts"))
import release_ranges

SKIP_LABEL = "skip-release-note"

# `gh pr view --json files` would stop at 100 files; the files API pages through all of them.
# One compact JSON object per line: newlines in a file name are escaped in JSON, so they cannot
# split a record.
CHANGED_FILES_JQ = ".[] | {filename, status, previous_filename} | tojson"
EXIT_PROBLEMS = 3
EXIT_GH_FAILED = 2

# Statuses of the files API in which the file exists afterwards.
EXISTING_STATUSES = frozenset({"added", "modified", "renamed", "copied", "changed"})


class ChangedFile(NamedTuple):
    filename: str
    status: str
    previous_filename: str | None = None


class GhError(RuntimeError):
    """The list of changed files could not be obtained."""


def _is_source(path: str, package: release_ranges.NotesPackage) -> bool:
    for source in package.source_paths:
        if source.endswith("/"):
            if path.startswith(source):
                return True
        elif path == source:
            return True
    return False


def touches_sources(changed: list[ChangedFile], package: release_ranges.NotesPackage) -> bool:
    # A file renamed out of the sources changes them as much as one renamed into them.
    for file in changed:
        paths = [file.filename]
        if file.previous_filename is not None:
            paths.append(file.previous_filename)
        if any(_is_source(path, package) for path in paths):
            return True
    return False


def has_note(changed: list[ChangedFile], package: release_ranges.NotesPackage) -> bool:
    notes_prefix = f"{package.notes_dir}/"
    return any(
        file.status in EXISTING_STATUSES
        and file.filename.startswith(notes_prefix)
        and release_ranges.NOTE_NAME_RE.fullmatch(file.filename[len(notes_prefix) :])
        for file in changed
    )


def find_problems(changed: list[ChangedFile], labels: list[str]) -> list[str]:
    if SKIP_LABEL in (label.lower() for label in labels):
        return []
    problems = []
    for component, package in release_ranges.PACKAGES.items():
        if touches_sources(changed, package) and not has_note(changed, package):
            problems.append(
                f"**Missing release note** for `{component}`: this PR changes its sources, so add or edit a "
                f"note in `{package.notes_dir}/` "
                f'(see "Release notes" in CONTRIBUTING.md), '
                f"or apply the `{SKIP_LABEL}` label if none is needed."
            )
    return problems


def changed_files(repo: str, pr_number: str) -> list[ChangedFile]:
    """List the files a PR changes, with their status and the old path of renamed files."""
    cmd = ["gh", "api", "--paginate", f"repos/{repo}/pulls/{pr_number}/files", "--jq", CHANGED_FILES_JQ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603
    except OSError as e:
        raise GhError(f"could not run `gh`: {e}") from e
    if result.returncode != 0:
        raise GhError(f"`gh api` failed with status {result.returncode}:\n{result.stderr.strip()}")
    files = []
    for line in result.stdout.splitlines():
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as e:
            raise GhError(f"unexpected output from `gh api`: {line!r}") from e
        files.append(ChangedFile(record["filename"], record["status"], record.get("previous_filename")))
    return files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="OWNER/REPO the PR belongs to")
    parser.add_argument("--pr", required=True, help="the PR number")
    parser.add_argument("--labels-json", required=True, help="the PR's labels as a JSON list of objects with a name")
    args = parser.parse_args(argv)

    labels = [label["name"] for label in json.loads(args.labels_json)]
    try:
        files = changed_files(args.repo, args.pr)
    except GhError as e:
        print(f"ERROR: could not list the files of PR #{args.pr} in {args.repo}: {e}", file=sys.stderr)
        return EXIT_GH_FAILED
    problems = find_problems(files, labels)
    for problem in problems:
        print(f"- {problem}")
    if problems:
        return EXIT_PROBLEMS
    return 0


if __name__ == "__main__":
    sys.exit(main())
