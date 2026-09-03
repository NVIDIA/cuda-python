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
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Mapping

from packaging.version import Version

from . import bindings_config

COMPONENT_TO_PACKAGE: dict[str, str] = {
    "cuda-core": "cuda_core",
    "cuda-bindings": "cuda_bindings",
    "cuda-pathfinder": "cuda_pathfinder",
    "cuda-python": "cuda_python",
}

COMPONENT_TO_TAG_PREFIX: dict[str, str] = {
    "cuda-bindings": "v",
    "cuda-python": "v",
    "cuda-core": "cuda-core-v",
    "cuda-pathfinder": "cuda-pathfinder-v",
}


def _resolved_bindings_package(
    data: Mapping[str, object], git_tag: str
) -> tuple[bindings_config.BindingsPackage, str, Version]:
    """Validate the release resolver fields consumed by this script."""
    package = bindings_config.package_from_dict(data)
    raw_version = data.get("release_version")
    if not isinstance(raw_version, str):
        raise bindings_config.BindingsConfigError("resolved CUDA bindings package has no release_version")
    version = bindings_config.parse_pep440_version(raw_version, "resolved release_version")
    origin = data.get("release_registry_origin", "tag")
    if origin not in {"tag", "control"}:
        raise bindings_config.BindingsConfigError("resolved CUDA bindings package has invalid release_registry_origin")
    matched = package.scm_version_from_tag(git_tag, fullmatch=origin != "control")
    if matched != version or (origin != "control" and package.version_from_tag(git_tag) != version):
        raise bindings_config.BindingsConfigError(
            f"resolved CUDA bindings package does not match release tag {git_tag!r}"
        )

    package_root = data.get("release_package_root", package.package_root)
    if not isinstance(package_root, str) or not package_root:
        raise bindings_config.BindingsConfigError("resolved CUDA bindings package has invalid release_package_root")
    path = PurePosixPath(package_root)
    if (
        "\\" in package_root
        or PureWindowsPath(package_root).drive
        or path.is_absolute()
        or path.as_posix() != package_root
        or any(part in (".", "..") for part in path.parts)
    ):
        raise bindings_config.BindingsConfigError(
            f"resolved CUDA bindings release_package_root is not repository-relative: {package_root!r}"
        )
    return package, package_root, version


def _release_target_from_tag(
    git_tag: str,
    component: str,
    bindings_package: Mapping[str, object] | None = None,
) -> tuple[str, str] | None:
    """Return the release version and source tree selected by a component tag."""
    prefix = COMPONENT_TO_TAG_PREFIX.get(component)
    if prefix is None:
        return None
    if component == "cuda-bindings":
        if bindings_package is None:
            package = bindings_config.load_config().match_tag(git_tag)
            if package is None:
                return None
            package_root: object = package.package_root
            version = package.version_from_tag(git_tag)
        else:
            package, package_root, version = _resolved_bindings_package(bindings_package, git_tag)
    else:
        version = bindings_config.parse_prefixed_version(git_tag, prefix)
    if version is None:
        return None
    return str(version), package_root if component == "cuda-bindings" else COMPONENT_TO_PACKAGE[component]


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
    if component not in COMPONENT_TO_PACKAGE:
        return [("<component>", f"unknown component '{component}'")]

    target = _release_target_from_tag(git_tag, component, bindings_package)
    if target is None:
        return [("<tag>", f"cannot parse version from tag '{git_tag}' for component '{component}'")]
    version, package = target

    if is_post_release(version):
        return []

    path = notes_path(package, version)
    full = repo_root / path
    if not full.is_file():
        return [(path, "missing")]
    if full.stat().st_size == 0:
        return [(path, "empty")]
    return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--git-tag", required=True)
    parser.add_argument("--component", required=True, choices=list(COMPONENT_TO_PACKAGE))
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
        version = parse_version_from_tag(args.git_tag, args.component, bindings_package)
    except (bindings_config.BindingsConfigError, json.JSONDecodeError, ValueError) as error:
        print(f"ERROR: invalid CUDA bindings configuration: {error}", file=sys.stderr)
        return 2
    if version is None:
        print(
            f"ERROR: tag {args.git_tag!r} does not match the expected format for component {args.component!r}.",
            file=sys.stderr,
        )
        return 2

    if is_post_release(version):
        print(f"Post-release tag ({args.git_tag}), skipping release-notes check.")
        return 0

    problems = check_release_notes(args.git_tag, args.component, args.repo_root, bindings_package)

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
