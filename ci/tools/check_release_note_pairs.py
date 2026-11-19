# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Require numeric bindings and metapackage release notes to be paired."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BINDINGS_NOTE = re.compile(r"cuda_bindings_[^/]+/docs/source/release/([0-9]+\.[0-9]+\.[0-9]+)-notes\.rst")
METAPACKAGE_NOTES = "cuda_python/docs/source/release"
METAPACKAGE_NOTE = re.compile(rf"{METAPACKAGE_NOTES}/([0-9]+\.[0-9]+\.[0-9]+)-notes\.rst")


def _requires_pairing(version: str) -> bool:
    """Enforce pairing starting from releases 12.9.11 and 13.5.1."""
    major, minor, patch = map(int, version.split("."))
    return (major == 12 and (minor, patch) >= (9, 11)) or (major == 13 and (minor, patch) >= (5, 1)) or major >= 14


def missing_release_note_pairs(repo_root: Path) -> list[str]:
    """Return missing release-note counterparts in the Git index."""
    listing = subprocess.run(  # noqa: S603
        [  # noqa: S607
            "git",
            "ls-files",
            "-z",
            "--",
            "cuda_bindings_*/docs/source/release/*-notes.rst",
            f"{METAPACKAGE_NOTES}/*-notes.rst",
        ],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        check=True,
        encoding="utf-8",
    ).stdout
    indexed: set[str] = set()
    for path in listing.split("\0"):
        match = BINDINGS_NOTE.fullmatch(path) or METAPACKAGE_NOTE.fullmatch(path)
        if match and _requires_pairing(match[1]):
            indexed.add(path)

    missing: set[str] = set()
    for path in indexed:
        if match := BINDINGS_NOTE.fullmatch(path):
            counterpart = f"{METAPACKAGE_NOTES}/{match[1]}-notes.rst"
            if counterpart not in indexed:
                missing.add(counterpart)

    for path in indexed:
        if match := METAPACKAGE_NOTE.fullmatch(path):
            version = match[1]
            major = int(version.split(".", 1)[0])
            counterpart = f"cuda_bindings_{major}/docs/source/release/{version}-notes.rst"
            if counterpart not in indexed:
                missing.add(counterpart)
    return sorted(missing)


def main() -> int:
    """Report release notes that would be missing from the committed tree."""
    missing = missing_release_note_pairs(ROOT)
    if not missing:
        return 0

    print("Missing release-note counterparts in the Git index:", file=sys.stderr)
    for path in missing:
        print(f"  - {path}", file=sys.stderr)
    print("Add and stage matching cuda-python and cuda-bindings notes for each numeric release.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
