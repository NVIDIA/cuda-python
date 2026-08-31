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
import json
import re
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Mapping

import bindings_config

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


def _resolved_bindings_line(data: Mapping[str, object]) -> tuple[bindings_config.BindingsLine, str]:
    """Validate the release resolver's normalized line and actual source path."""
    required = (
        "line_id",
        "source_dir",
        "ctk_target",
        "toolkit_version",
        "toolkit_channel",
        "tag_series",
        "allow_alpha_beta_tags",
    )
    missing = [key for key in required if key not in data]
    if missing:
        raise bindings_config.BindingsConfigError(f"resolved CUDA bindings line is missing keys: {', '.join(missing)}")
    string_values = {key: data[key] for key in required if key != "allow_alpha_beta_tags"}
    if any(not isinstance(value, str) for value in string_values.values()):
        raise bindings_config.BindingsConfigError("resolved CUDA bindings line string fields must be strings")
    allow_alpha_beta_tags = data["allow_alpha_beta_tags"]
    if type(allow_alpha_beta_tags) is not bool:
        raise bindings_config.BindingsConfigError("resolved CUDA bindings line allow_alpha_beta_tags must be a boolean")

    line = bindings_config.BindingsLine(
        line_id=str(data["line_id"]),
        source_dir=str(data["source_dir"]),
        ctk_target=str(data["ctk_target"]),
        toolkit_version=str(data["toolkit_version"]),
        toolkit_channel=str(data["toolkit_channel"]),
        tag_series=str(data["tag_series"]),
        allow_alpha_beta_tags=allow_alpha_beta_tags,
    )
    source_dir = data.get("release_source_dir", line.source_dir)
    if not isinstance(source_dir, str) or not source_dir or source_dir != source_dir.strip():
        raise bindings_config.BindingsConfigError(
            "resolved CUDA bindings release_source_dir must be a non-empty, trimmed string"
        )
    path = PurePosixPath(source_dir)
    if (
        "\\" in source_dir
        or PureWindowsPath(source_dir).drive
        or path.is_absolute()
        or path.as_posix() != source_dir
        or any(part in (".", "..") for part in path.parts)
    ):
        raise bindings_config.BindingsConfigError(
            f"resolved CUDA bindings release_source_dir is not repository-relative: {source_dir!r}"
        )
    return line, source_dir


def _release_target_from_tag(
    git_tag: str,
    component: str,
    bindings_line: Mapping[str, object] | None = None,
) -> tuple[str, str] | None:
    """Return the release version and source tree selected by a component tag."""
    pattern = COMPONENT_TO_TAG_RE.get(component)
    if pattern is None:
        return None
    match = pattern.match(git_tag)
    if match is None:
        return None

    version = match.group("version")
    if component == "cuda-bindings":
        if bindings_line is None:
            line = bindings_config.load_config().match_tag(git_tag)
            source_dir = line.source_dir if line is not None else ""
        else:
            line, source_dir = _resolved_bindings_line(bindings_line)
            if not line.matches_tag(git_tag):
                line = None
        if line is None:
            return None
        return version, source_dir
    return version, COMPONENT_TO_PACKAGE[component]


def parse_version_from_tag(
    git_tag: str,
    component: str,
    bindings_line: Mapping[str, object] | None = None,
) -> str | None:
    """Extract the version string from a tag, given the target component.

    Returns None if the tag does not match the component's expected prefix,
    contains characters outside the allowed version set, or (for bindings)
    does not select a configured release line.
    """
    target = _release_target_from_tag(git_tag, component, bindings_line)
    return target[0] if target is not None else None


def is_post_release(version: str) -> bool:
    return ".post" in version


def notes_path(package: str, version: str) -> Path:
    return Path(package, "docs", "source", "release", f"{version}-notes.rst")


def check_release_notes(
    git_tag: str,
    component: str,
    repo_root: Path = Path("."),
    bindings_line: Mapping[str, object] | None = None,
    control_repo_root: Path | None = None,
) -> list[tuple[str | Path, str]]:
    """Return a list of (path, reason) for missing or empty release notes.

    ``path`` is the repo-relative notes path, or a ``<placeholder>`` naming the
    offending argument when the tag or component itself is the problem.

    Returns an empty list when notes are present and non-empty, or when the
    tag is a .post release (no new notes required).
    """
    if component not in COMPONENT_TO_PACKAGE:
        return [("<component>", f"unknown component '{component}'")]

    target = _release_target_from_tag(git_tag, component, bindings_line)
    if target is None:
        return [("<tag>", f"cannot parse version from tag '{git_tag}' for component '{component}'")]
    version, package = target

    if is_post_release(version):
        return []

    path = notes_path(package, version)
    full = repo_root / path
    if not full.is_file():
        if (
            bindings_line is not None
            and bindings_line.get("release_registry_origin") == "control"
            and control_repo_root is not None
            and component in ("cuda-bindings", "cuda-python")
        ):
            if component == "cuda-bindings":
                line, _ = _resolved_bindings_line(bindings_line)
                control_package = line.source_dir
            else:
                control_package = COMPONENT_TO_PACKAGE[component]
            control_path = notes_path(control_package, version)
            control_full = control_repo_root / control_path
            if control_full.is_file():
                if control_full.stat().st_size == 0:
                    return [(control_path, "empty")]
                return []
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
        "--control-repo-root",
        type=Path,
        help="current control checkout used for legacy bare-tag release notes",
    )
    parser.add_argument(
        "--bindings-line",
        default="",
        help="normalized CUDA bindings release-line JSON from the release resolver",
    )
    args = parser.parse_args(argv)

    try:
        bindings_line = json.loads(args.bindings_line) if args.bindings_line else None
        if bindings_line is not None and not isinstance(bindings_line, dict):
            raise ValueError("resolved CUDA bindings line must be a JSON object")
        version = parse_version_from_tag(args.git_tag, args.component, bindings_line)
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

    try:
        problems = check_release_notes(
            args.git_tag,
            args.component,
            args.repo_root,
            bindings_line,
            args.control_repo_root,
        )
    except bindings_config.BindingsConfigError as error:
        print(f"ERROR: invalid CUDA bindings configuration: {error}", file=sys.stderr)
        return 2

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
