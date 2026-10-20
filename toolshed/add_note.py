#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.10"
# ///

"""Create a new release note for one of the packages.

Usage:
    python toolshed/add_note.py PACKAGE TYPE SHORT-DESCRIPTION [--edit]

PACKAGE is one of the packages that have release notes (``cuda-bindings``, ``cuda-core``,
``cuda-pathfinder`` or ``cuda-python``; the ``cuda-`` prefix is optional and ``_`` also works).
TYPE is the kind of note: one of the keys of ``release_ranges.SECTIONS`` (``feature``,
``upgrade``, ..., ``prelude``); it becomes the start of the file name.
SHORT-DESCRIPTION is a few words that become part of the file name.  It can be run from any
directory: the note is always created in the package's ``releasenotes`` directory of this checkout.

Edit the new file and replace the ``TODO`` text with a reStructuredText entry.  A note has one
type; to record two kinds of change, create two notes.
See "Release notes" in CONTRIBUTING.md.
"""

from __future__ import annotations

import argparse
import os
import re
import secrets
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "cuda_python" / "docs" / "exts"))
import release_ranges  # noqa: E402

TEMPLATE = "TODO: Describe the change in reStructuredText.\n"

NOTE_TYPES = tuple(release_ranges.SECTIONS)

MAX_SLUG_LENGTH = 60


def normalize_package(name: str) -> str:
    """Spell a user-supplied package name as a component name: ``Core`` -> ``cuda-core``."""
    component = name.strip().lower().replace("_", "-")
    if not component.startswith("cuda-"):
        component = f"cuda-{component}"
    return component


def find_package(name: str) -> str:
    """Return the component name (``cuda-core``) for a user-supplied package name."""
    component = normalize_package(name)
    if component not in release_ranges.PACKAGES:
        choices = ", ".join(sorted(release_ranges.PACKAGES))
        raise ValueError(f"unknown package {name!r}; choose one of: {choices}")
    return component


def slugify(description: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", description.lower()).strip("-")[:MAX_SLUG_LENGTH].strip("-")
    if not slug:
        raise ValueError(f"the description {description!r} has no letters or digits to make a file name from")
    return slug


def create_note(package: str, note_type: str, description: str, repo_root: Path | None = None) -> Path:
    if note_type not in NOTE_TYPES:
        raise ValueError(f"unknown note type {note_type!r}; choose one of: {', '.join(NOTE_TYPES)}")
    if repo_root is None:
        repo_root = REPO_ROOT
    component = find_package(package)
    notes_dir = repo_root / release_ranges.PACKAGES[component].notes_dir
    notes_dir.mkdir(parents=True, exist_ok=True)
    path = notes_dir / f"{note_type}-{slugify(description)}-{secrets.token_hex(8)}.rst"
    path.write_text(TEMPLATE, encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "package",
        type=normalize_package,
        choices=sorted(release_ranges.PACKAGES),
        help="the package to add a release note to (the cuda- prefix is optional)",
    )
    parser.add_argument("type", choices=NOTE_TYPES, help="the kind of note, which is the section it is listed under")
    parser.add_argument("description", help="a few words describing the change, used in the file name")
    parser.add_argument("--edit", action="store_true", help="open the new file in $VISUAL or $EDITOR")
    args = parser.parse_args(argv)

    try:
        path = create_note(args.package, args.type, args.description)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    print(f"Created {path.relative_to(REPO_ROOT).as_posix()}")

    if args.edit:
        editor = os.environ.get("VISUAL") or os.environ.get("EDITOR")
        if not editor:
            print("ERROR: set $VISUAL or $EDITOR to use --edit.", file=sys.stderr)
            return 2
        return subprocess.run([*shlex.split(editor), str(path)], check=False).returncode  # noqa: S603
    return 0


if __name__ == "__main__":
    sys.exit(main())
