# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check that versioned release-notes files exist before releasing.

Usage:
    python check_release_notes.py --git-tag <tag> --component <component>

Exit codes:
    0 — release notes present and non-empty (or .post version, skipped)
    1 — release notes missing or empty
    2 — invalid arguments (including unparsable tag, or component/tag-prefix mismatch)
"""

from __future__ import annotations

import argparse
import os
import re
import sys

COMPONENT_TO_PACKAGE: dict[str, str] = {
    "cuda-core": "cuda_core",
    "cuda-bindings": "cuda_bindings",
    "cuda-pathfinder": "cuda_pathfinder",
    "cuda-python": "cuda_python",
}

# Version characters are restricted to digit-prefixed word chars and dots, so
# malformed inputs like "v../evil" or "v1/2/3" cannot flow into the notes path.
_VERSION_PATTERN = r"\d[\w.]*"

# Each component has exactly one valid tag-prefix form. cuda-bindings and
# cuda-python share the bare "v<version>" namespace (setuptools-scm lookup).
COMPONENT_TO_TAG_RE: dict[str, re.Pattern[str]] = {
    "cuda-bindings": re.compile(rf"^v(?P<version>{_VERSION_PATTERN})$"),
    "cuda-python": re.compile(rf"^v(?P<version>{_VERSION_PATTERN})$"),
    "cuda-core": re.compile(rf"^cuda-core-v(?P<version>{_VERSION_PATTERN})$"),
    "cuda-pathfinder": re.compile(rf"^cuda-pathfinder-v(?P<version>{_VERSION_PATTERN})$"),
}


def parse_version_from_tag(git_tag: str, component: str) -> str | None:
    """Extract the version string from a tag, given the target component.

    Returns None if the tag does not match the component's expected prefix
    or contains characters outside the allowed version set.
    """
    pattern = COMPONENT_TO_TAG_RE.get(component)
    if pattern is None:
        return None
    m = pattern.match(git_tag)
    return m.group("version") if m else None


def is_post_release(version: str) -> bool:
    return ".post" in version


def notes_path(package: str, version: str) -> str:
    return os.path.join(package, "docs", "source", "release", f"{version}-notes.rst")


def check_release_notes(git_tag: str, component: str, repo_root: str = ".") -> list[tuple[str, str]]:
    """Return a list of (path, reason) for missing or empty release notes.

    Returns an empty list when notes are present and non-empty, or when the
    tag is a .post release (no new notes required).
    """
    if component not in COMPONENT_TO_PACKAGE:
        return [("<component>", f"unknown component '{component}'")]

    version = parse_version_from_tag(git_tag, component)
    if version is None:
        return [("<tag>", f"cannot parse version from tag '{git_tag}' for component '{component}'")]

    if is_post_release(version):
        return []

    path = notes_path(COMPONENT_TO_PACKAGE[component], version)
    full = os.path.join(repo_root, path)
    if not os.path.isfile(full):
        return [(path, "missing")]
    if os.path.getsize(full) == 0:
        return [(path, "empty")]
    return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--git-tag", required=True)
    parser.add_argument("--component", required=True, choices=list(COMPONENT_TO_PACKAGE))
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)

    version = parse_version_from_tag(args.git_tag, args.component)
    if version is None:
        print(
            f"ERROR: tag {args.git_tag!r} does not match the expected format for component {args.component!r}.",
            file=sys.stderr,
        )
        return 2

    if is_post_release(version):
        print(f"Post-release tag ({args.git_tag}), skipping release-notes check.")
        return 0

    problems = check_release_notes(args.git_tag, args.component, args.repo_root)
    if not problems:
        print(f"Release notes present for tag {args.git_tag}, component {args.component}.")
        return 0

    print(f"ERROR: missing or empty release notes for tag {args.git_tag}:", file=sys.stderr)
    for path, reason in problems:
        print(f"  - {path} ({reason})", file=sys.stderr)
    print("Add versioned release notes before releasing.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
