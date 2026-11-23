# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check staged bindings and metapackage release-note pairs."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
METAPACKAGE_NOTES = "cuda_python/docs/source/release"
RELEASE_NOTE = re.compile(r"(cuda_bindings_[^/]+|cuda_python)/docs/source/release/([0-9]+\.[0-9]+\.[0-9]+)-notes\.rst")


def _requires_pairing(version: str) -> bool:
    """Enforce pairing starting from releases 12.9.11 and 13.5.1."""
    major, minor, patch = map(int, version.split("."))
    return (major == 12 and (minor, patch) >= (9, 11)) or (major == 13 and (minor, patch) >= (5, 1)) or major >= 14


def _build_indexed_release_notes(repo_root: Path) -> dict[str, tuple[str, str]]:
    """Collect the package and version of each release note covered by the checks."""
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
    indexed_release_notes: dict[str, tuple[str, str]] = {}
    for path in listing.split("\0"):
        if match := RELEASE_NOTE.fullmatch(path):
            package, version = match.groups()
            if _requires_pairing(version):
                indexed_release_notes[path] = package, version
    return indexed_release_notes


def missing_release_note_pairs(indexed_release_notes: dict[str, tuple[str, str]]) -> list[str]:
    """Return missing release-note counterparts in the Git index."""
    missing: set[str] = set()
    for package, version in indexed_release_notes.values():
        if package.startswith("cuda_bindings_"):
            counterpart = f"{METAPACKAGE_NOTES}/{version}-notes.rst"
            if counterpart not in indexed_release_notes:
                missing.add(counterpart)

    for package, version in indexed_release_notes.values():
        if package == "cuda_python":
            major = int(version.split(".", 1)[0])
            counterpart = f"cuda_bindings_{major}/docs/source/release/{version}-notes.rst"
            if counterpart not in indexed_release_notes:
                missing.add(counterpart)
    return sorted(missing)


def main() -> int:
    """Report missing release-note pairs in the Git index."""
    indexed_release_notes = _build_indexed_release_notes(ROOT)
    errors = missing_release_note_pairs(indexed_release_notes)
    if not errors:
        return 0

    print("Release-note checks failed in the Git index:", file=sys.stderr)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    print("Add and stage matching cuda-bindings and cuda-python release notes.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
