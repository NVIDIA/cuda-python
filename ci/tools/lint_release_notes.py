# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check the release note files of every package in ``release_ranges.PACKAGES`` for obvious mistakes.

Run from anywhere; it checks this checkout.  Each file in a package's ``releasenotes`` directory must:

* be named like the files ``toolshed/add_note.py`` creates
  (``<type>-<short-description>-<16 hex digits>.rst``);
* have a type that is one of the ``release_ranges.SECTIONS`` keys (including ``prelude``);
* not be empty;
* not still contain the ``TODO`` text of the template.

Prints one line per problem and exits with status 1 if there are any.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "cuda_python" / "docs" / "exts"))
import release_ranges  # noqa: E402

KNOWN_TYPES = set(release_ranges.SECTIONS)


def lint_note(path: Path) -> list[str]:
    """Return the problems of one note file, as messages without the file name."""
    note_type = release_ranges.note_type(path.name)
    if note_type is None:
        return ["not named like a note (<type>-short-description-<16 hex digits>.rst); only notes belong here"]
    if note_type not in KNOWN_TYPES:
        return [f"unknown note type {note_type!r}; use one of: {', '.join(sorted(KNOWN_TYPES))}"]
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        return [f"not valid UTF-8 text: {e}"]
    if not text.strip():
        return ["is empty"]
    if text.lstrip().startswith("TODO"):
        return ["still has the template's TODO text"]
    return []


def lint_package(repo_root: Path, package: release_ranges.NotesPackage) -> list[tuple[Path, str]]:
    notes_dir = repo_root / package.notes_dir
    if not notes_dir.is_dir():
        return []
    problems = []
    for path in sorted(notes_dir.iterdir()):
        for message in lint_note(path):
            problems.append((path, message))
    return problems


def main() -> int:
    failed = False
    for package in release_ranges.PACKAGES.values():
        for path, message in lint_package(REPO_ROOT, package):
            print(f"{path.relative_to(REPO_ROOT).as_posix()}: {message}", file=sys.stderr)
            failed = True
    if failed:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
