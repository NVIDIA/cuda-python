# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.10"
# dependencies = ["pyyaml"]
# ///

"""Check the release note files of every package in ``release_ranges.PACKAGES`` for obvious mistakes.

Run from anywhere; it checks this checkout.  Each file in a package's ``releasenotes`` directory must:

* be named like the files ``toolshed/add_note.py`` creates (``short-description-<16 hex digits>.yaml``);
* be a YAML mapping of known section keys (``release_ranges.SECTIONS`` and ``prelude``) to a
  non-empty list of non-empty entries (``prelude`` may also be a single string);
* spell each section key plainly (``fixes:`` at the start of a line, no quotes), which is how the docs find them;
* not still contain the ``TODO`` text of the template.

Prints one line per problem and exits with status 1 if there are any.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "cuda_python" / "docs" / "exts"))
import release_ranges  # noqa: E402

KNOWN_KEYS = {release_ranges.PRELUDE, *(key for key, _title in release_ranges.SECTIONS)}


def lint_note(path: Path) -> list[str]:
    """Return the problems of one note file, as messages without the file name."""
    if not release_ranges.NOTE_NAME_RE.fullmatch(path.name):
        return ["not named like a note (short-description-<16 hex digits>.yaml); only notes belong here"]
    try:
        text = path.read_text(encoding="utf-8")
        data = yaml.safe_load(text)
    except (yaml.YAMLError, UnicodeDecodeError) as e:
        return [f"not valid YAML: {e}"]
    if not isinstance(data, dict) or not data:
        return ["must be a YAML mapping of sections to entries, with at least one section"]

    problems = []
    for key, value in data.items():
        if key not in KNOWN_KEYS:
            problems.append(f"unknown section {key!r}; use one of: {', '.join(sorted(KNOWN_KEYS))}")
            continue
        if not re.search(release_ranges.section_key_re(key), text, re.MULTILINE):
            problems.append(f"write the {key!r} key as plain `{key}:` at the start of a line (no quotes or spaces)")
            continue
        entries = value
        if isinstance(value, str):
            if key != release_ranges.PRELUDE:
                problems.append(f"section {key!r} must be a list of entries")
                continue
            entries = [value]
        if not isinstance(entries, list) or not entries:
            problems.append(f"section {key!r} must be a non-empty list of entries (delete it if unused)")
            continue
        for entry in entries:
            if not isinstance(entry, str) or not entry.strip():
                problems.append(f"section {key!r} has an entry that is not text")
            elif entry.lstrip().startswith("TODO"):
                problems.append(f"section {key!r} still has the template's TODO text")
    return problems


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
