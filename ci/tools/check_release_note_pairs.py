# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Require numeric bindings release notes to have matching metapackage notes."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BINDINGS_NOTE = re.compile(r"cuda_bindings_[^/]+/docs/source/release/([0-9]+\.[0-9]+\.[0-9]+)-notes\.rst")
METAPACKAGE_NOTES = "cuda_python/docs/source/release"

# These historical releases predate the pairing check. Do not add new releases.
LEGACY_UNPAIRED_VERSIONS = frozenset(
    {
        "11.4.0",
        "11.5.0",
        "11.6.0",
        "11.6.1",
        "11.7.0",
        "11.7.1",
        "11.8.0",
        "11.8.1",
        "11.8.2",
        "11.8.3",
        "11.8.4",
        "11.8.5",
        "12.0.0",
        "12.1.0",
        "12.2.0",
        "12.2.1",
        "12.3.0",
        "12.4.0",
        "12.5.0",
        "12.6.0",
        "12.9.8",
        "12.9.9",
        "12.9.10",
        "13.4.2",
        "13.4.3",
    }
)


def missing_release_note_pairs(repo_root: Path) -> list[str]:
    """Return missing or empty metapackage notes in the Git index."""
    listing = subprocess.run(  # noqa: S603
        [  # noqa: S607
            "git",
            "ls-files",
            "--stage",
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
    indexed: dict[str, str] = {}
    for entry in listing.split("\0"):
        if entry:
            metadata, path = entry.split("\t", 1)
            indexed[path] = metadata.split()[1]

    required = {
        f"{METAPACKAGE_NOTES}/{match[1]}-notes.rst"
        for path in indexed
        if (match := BINDINGS_NOTE.fullmatch(path)) and match[1] not in LEGACY_UNPAIRED_VERSIONS
    }
    object_ids = {indexed[path] for path in required if path in indexed}
    sizes: dict[str, int] = {}
    if object_ids:
        objects = subprocess.run(
            ["git", "cat-file", "--batch-check=%(objectname) %(objectsize)"],  # noqa: S607
            cwd=repo_root,
            input="\n".join(sorted(object_ids)) + "\n",
            stdout=subprocess.PIPE,
            check=True,
            encoding="utf-8",
        ).stdout
        for line in objects.splitlines():
            object_id, size = line.split()
            sizes[object_id] = int(size)

    return sorted(path for path in required if path not in indexed or sizes[indexed[path]] == 0)


def main() -> int:
    """Report release notes that would be missing from the committed tree."""
    missing = missing_release_note_pairs(ROOT)
    if not missing:
        return 0

    print("Missing or empty cuda-python release notes in the Git index:", file=sys.stderr)
    for path in missing:
        print(f"  - {path}", file=sys.stderr)
    print("Add and stage matching metapackage notes for each numeric bindings release.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
