# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The packages with per-change note files, the note sections, and which release each note belongs to.

This module has no third-party dependencies so that the docs extension (``release_notes``), the
release gate (``ci/tools/check_release_notes.py``), the PR check, the linter and
``toolshed/add_note.py`` can all share it.  See the ``release_notes`` docstring for the rules.
"""

from __future__ import annotations

import re
from typing import NamedTuple

Version = tuple[int, int, int]

#: The sections of a note, in the order and with the titles they have on a release page.  A note
#: is a YAML mapping from these keys to lists of reStructuredText entries.  Known issues come last.
#: ``prelude`` is not here: it is an untitled introduction at the top of a page.
SECTIONS: tuple[tuple[str, str], ...] = (
    ("features", "New Features"),
    ("upgrade", "Breaking Changes"),
    ("deprecations", "Deprecation Notes"),
    ("critical", "Critical Issues"),
    ("security", "Security Issues"),
    ("fixes", "Bug Fixes"),
    ("other", "Other Notes"),
    ("issues", "Known Issues"),
)
PRELUDE = "prelude"


def section_key_re(key: str) -> str:
    """The pattern a section key must match, written plainly at the start of a line.

    The docs find the notes with known issues by grepping for ``section_key_re("issues")``, so
    ``lint_release_notes.py`` rejects other spellings of any key (``"issues":``, ``issues :``).
    """
    return rf"^{re.escape(key)}:"


#: The file name of a note: a short description and a random suffix, as ``toolshed/add_note.py``
#: makes them.
NOTE_NAME_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*-[0-9a-f]{16}\.yaml")


class NotesPackage(NamedTuple):
    #: Directory with the package's note files (``*.yaml``), relative to the repo root.
    notes_dir: str
    #: Prefix of the package's release tags, e.g. ``v`` for ``v13.5.0``.
    tag_prefix: str
    #: First release whose notes are note files; earlier ones are hand-written pages.
    first_version: Version
    #: Changes to these paths (a trailing ``/`` marks a directory) need a release note.
    #: Everything else (docs, tests, CI, pixi files, ...) does not.
    source_paths: tuple[str, ...]


#: Packages whose release notes are built from note files, by release component name.
PACKAGES: dict[str, NotesPackage] = {
    "cuda-core": NotesPackage(
        "cuda_core/releasenotes",
        "cuda-core-v",
        (1, 3, 0),
        (
            "cuda_core/cuda/",
            "cuda_core/pyproject.toml",
            "cuda_core/setup.py",
            "cuda_core/build_hooks.py",
            "cuda_core/MANIFEST.in",
        ),
    ),
    # Releases before 1.9.0, including any 1.8.x patch releases, keep hand-written notes. Starting at a
    # minor release (not a patch) keeps the first release with note files the minor release of its line.
    "cuda-pathfinder": NotesPackage(
        "cuda_pathfinder/releasenotes",
        "cuda-pathfinder-v",
        (1, 9, 0),
        ("cuda_pathfinder/cuda/", "cuda_pathfinder/pyproject.toml"),
    ),
    # cuda-python shares the bare ``v`` tag namespace with cuda-bindings; each scans only its own notes.
    "cuda-python": NotesPackage(
        "cuda_python/releasenotes",
        "v",
        (13, 5, 0),
        ("cuda_python/pyproject.toml", "cuda_python/setup.py"),
    ),
    "cuda-bindings": NotesPackage(
        "cuda_bindings/releasenotes",
        "v",
        (13, 5, 0),
        (
            "cuda_bindings/cuda/",
            "cuda_bindings/pyproject.toml",
            "cuda_bindings/setup.py",
            "cuda_bindings/build_hooks.py",
            "cuda_bindings/MANIFEST.in",
        ),
    ),
}


def format_version(version: Version) -> str:
    return ".".join(str(part) for part in version)


def parse_release_tags(tags: list[str], prefix: str) -> dict[Version, str]:
    """Map each plain ``<prefix>X.Y.Z`` tag in *tags* to its version.

    Pre-release tags (``rc``, ``a``, ``b``), post-release tags and other
    packages' tags do not match.
    """
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)\.(\d+)\.(\d+)$")
    releases = {}
    for tag in tags:
        m = pattern.match(tag)
        if m:
            releases[(int(m.group(1)), int(m.group(2)), int(m.group(3)))] = tag
    return releases


def previous_release(version: Version, candidates: list[Version]) -> Version | None:
    """Return the release that *version*'s notes are relative to, or None.

    The first release of an ``X.Y`` line is its **minor release**.  That is
    normally ``X.Y.0``, but a line may start at ``X.Y.1`` (for example when
    ``X.Y.0`` was only ever a pre-release, and pre-releases are not in
    *candidates*).

    * A minor release is relative to the previous line's minor release.
    * Any other release is relative to the previous release of the same ``X.Y``
      line.
    """
    line = version[:2]
    same_line = [v for v in candidates if v[:2] == line and v < version]
    if same_line:
        return max(same_line)
    minor_releases = {}
    for v in candidates:
        if v[:2] < line:
            minor_releases[v[:2]] = min(v, minor_releases.get(v[:2], v))
    if not minor_releases:
        return None
    return minor_releases[max(minor_releases)]
