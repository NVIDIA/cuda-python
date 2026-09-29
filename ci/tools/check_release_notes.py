# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check that versioned release-notes files exist before releasing.

Usage:
    python -m ci.tools.check_release_notes --git-tag <tag> --component <component>

Exit codes:
    0 — release notes present and non-empty (or .post version, skipped)
    1 — release notes missing or empty
    2 — invalid arguments (including unparsable tag, or component/tag-prefix mismatch)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping

from packaging.version import Version

from . import bindings_config

COMPONENTS: dict[str, tuple[str, str]] = {
    "cuda-core": ("cuda_core", "cuda-core-v"),
    "cuda-bindings": ("cuda_bindings", "v"),
    "cuda-pathfinder": ("cuda_pathfinder", "cuda-pathfinder-v"),
    "cuda-python": ("cuda_python", "v"),
}


def _resolved_bindings_target(data: Mapping[str, object], git_tag: str) -> tuple[str, Version]:
    """Read the release resolver fields consumed by this script."""
    package_root = bindings_config.parse_package_root(data.get("package_root"), "resolved package_root")
    raw_version = data.get("release_version")
    if not isinstance(raw_version, str):
        raise bindings_config.BindingsConfigError("resolved CUDA bindings package has no release_version")
    version = bindings_config.parse_pep440_version(raw_version, "resolved release_version")
    origin = data.get("release_registry_origin")
    if origin not in {"tag", "control"}:
        raise bindings_config.BindingsConfigError("resolved CUDA bindings package has invalid release_registry_origin")
    tag_version = bindings_config.parse_prefixed_version(git_tag, "v")
    matches_tag = (
        tag_version == version
        if origin == "tag"
        else tag_version is not None and tag_version.release == version.release
    )
    if not matches_tag:
        raise bindings_config.BindingsConfigError(
            f"resolved CUDA bindings package does not match release tag {git_tag!r}"
        )
    return package_root, version


def _release_target_from_tag(
    git_tag: str,
    component: str,
    bindings_package: Mapping[str, object] | None = None,
) -> tuple[str, str] | None:
    """Return the release version and source tree selected by a component tag."""
    metadata = COMPONENTS.get(component)
    if metadata is None:
        return None
    package_dir, prefix = metadata
    if component == "cuda-bindings":
        if bindings_package is None:
            package = bindings_config.load_config().match_tag(git_tag)
            if package is None:
                return None
            package_root = package.package_root
            version = package.version_from_tag(git_tag)
        else:
            package_root, version = _resolved_bindings_target(bindings_package, git_tag)
    else:
        version = bindings_config.parse_prefixed_version(git_tag, prefix)
    if version is None:
        return None
    return str(version), package_root if component == "cuda-bindings" else package_dir


def parse_version_from_tag(
    git_tag: str,
    component: str,
    bindings_package: Mapping[str, object] | None = None,
) -> str | None:
    """Extract the version string from a tag, given the target component.

    Returns None if the tag does not match the component's expected prefix,
    contains characters outside the allowed version set, or (for bindings)
    does not select a configured package root.
    """
    target = _release_target_from_tag(git_tag, component, bindings_package)
    return target[0] if target is not None else None


def is_post_release(version: str) -> bool:
    return bindings_config.parse_pep440_version(version, "release version").post is not None


def notes_path(package: str, version: str) -> Path:
    return Path(package, "docs", "source", "release", f"{version}-notes.rst")


def _check_release_target(version: str, package: str, repo_root: Path) -> list[tuple[str | Path, str]]:
    if is_post_release(version):
        return []

    path = notes_path(package, version)
    full = repo_root / path
    if not full.is_file():
        return [(path, "missing")]
    if full.stat().st_size == 0:
        return [(path, "empty")]
    return []


def check_release_notes(
    git_tag: str,
    component: str,
    repo_root: Path = Path("."),
    bindings_package: Mapping[str, object] | None = None,
) -> list[tuple[str | Path, str]]:
    """Return a list of (path, reason) for missing or empty release notes.

    ``path`` is the repo-relative notes path, or a ``<placeholder>`` naming the
    offending argument when the tag or component itself is the problem.

    Returns an empty list when notes are present and non-empty, or when the
    tag is a .post release (no new notes required).
    """
    if component not in COMPONENTS:
        return [("<component>", f"unknown component '{component}'")]

    target = _release_target_from_tag(git_tag, component, bindings_package)
    if target is None:
        return [("<tag>", f"cannot parse version from tag '{git_tag}' for component '{component}'")]
    version, package = target
    return _check_release_target(version, package, repo_root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--git-tag", required=True)
    parser.add_argument("--component", required=True, choices=list(COMPONENTS))
    parser.add_argument("--repo-root", default=Path("."), type=Path)
    parser.add_argument(
        "--bindings-package",
        default="",
        help="normalized CUDA bindings package JSON from the release resolver",
    )
    args = parser.parse_args(argv)

    try:
        bindings_package = json.loads(args.bindings_package) if args.bindings_package else None
        if bindings_package is not None and not isinstance(bindings_package, dict):
            raise ValueError("resolved CUDA bindings package must be a JSON object")
        target = _release_target_from_tag(args.git_tag, args.component, bindings_package)
    except (bindings_config.BindingsConfigError, json.JSONDecodeError, ValueError) as error:
        print(f"ERROR: invalid CUDA bindings configuration: {error}", file=sys.stderr)
        return 2
    if target is None:
        print(
            f"ERROR: tag {args.git_tag!r} does not match the expected format for component {args.component!r}.",
            file=sys.stderr,
        )
        return 2
    version, package = target

    if is_post_release(version):
        print(f"Post-release tag ({args.git_tag}), skipping release-notes check.")
        return 0

    problems = _check_release_target(version, package, args.repo_root)

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
