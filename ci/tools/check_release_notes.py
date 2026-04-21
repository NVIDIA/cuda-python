# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check that versioned release-notes files exist before releasing.

Usage:
    python check_release_notes.py --git-tag <tag> --component <component>

Exit codes:
    0 — release notes present and non-empty (or .post version, skipped)
    1 — release notes missing or empty
    2 — invalid arguments
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

# Matches tags like "v13.1.0", "cuda-core-v0.7.0", "cuda-pathfinder-v1.5.2"
TAG_RE = re.compile(r"^(?:cuda-\w+-)?v(.+)$")


def parse_version_from_tag(git_tag: str) -> str | None:
    """Extract the bare version string (e.g. '13.1.0') from a git tag."""
    m = TAG_RE.match(git_tag)
    return m.group(1) if m else None


def is_post_release(version: str) -> bool:
    return ".post" in version


def notes_path(package: str, version: str) -> str:
    return os.path.join(package, "docs", "source", "release", f"{version}-notes.rst")


def check_release_notes(git_tag: str, component: str, repo_root: str = ".") -> list[tuple[str, str]]:
    """Return a list of (path, reason) for missing or empty release notes.

    Returns an empty list when notes are present and non-empty.
    """
    version = parse_version_from_tag(git_tag)
    if version is None:
        return [("<tag>", f"cannot parse version from tag '{git_tag}'")]

    if is_post_release(version):
        return []

    package = COMPONENT_TO_PACKAGE.get(component)
    if package is None:
        return [("<component>", f"unknown component '{component}'")]

    path = notes_path(package, version)
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

    version = parse_version_from_tag(args.git_tag)
    if version and is_post_release(version):
        print(f"Post-release tag ({args.git_tag}), skipping release-notes check.")
        return 0

    problems = check_release_notes(args.git_tag, args.component, args.repo_root)
    if not problems:
        print(f"Release notes present for tag {args.git_tag}, component {args.component}.")
        return 0

    print(f"ERROR: missing or empty release notes for tag {args.git_tag}:")
    for path, reason in problems:
        print(f"  - {path} ({reason})")
    print("Add versioned release notes before releasing.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
