# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select a maintenance bindings line's SCM version before its release tag."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import tomllib
from packaging.version import Version

from . import bindings_config

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMIT_PATTERN = re.compile(r"[0-9a-fA-F]{7,64}")


def read_fallback_version(path: Path, ctk_target: str) -> Version:
    try:
        with path.open("rb") as stream:
            value = tomllib.load(stream)["tool"]["setuptools_scm"]["fallback_version"]
    except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"could not read [tool.setuptools_scm].fallback_version from {path}: {error}") from error
    if not isinstance(value, str):
        raise ValueError(f"expected fallback_version in {path} to be a string")
    version = bindings_config.parse_pep440_version(value, f"fallback_version in {path}")
    target = bindings_config.parse_pep440_version(ctk_target, "CTK target")
    if version.dev is None or version.release[:2] != target.release[:2]:
        raise ValueError(f"expected a CUDA {ctk_target} development fallback in {path}, got {value!r}")
    return version


def has_reachable_release(
    repo_root: Path,
    line: bindings_config.BindingsLine,
    minimum_release: Version,
) -> bool:
    process = subprocess.run(
        ["git", "tag", "--merged", "HEAD", "--list"],  # noqa: S607
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        detail = process.stderr.strip() or f"git tag exited with status {process.returncode}"
        raise RuntimeError(f"could not inspect reachable tags for bindings line {line.line_id!r}: {detail}")
    releases = (version for tag in process.stdout.splitlines() if (version := line.version_from_tag(tag)) is not None)
    return any(release >= minimum_release for release in releases)


def pretend_version(
    repo_root: Path,
    commit_sha: str,
    line: bindings_config.BindingsLine,
) -> str | None:
    """Return the pre-tag override, or None once the line has its release tag."""
    if COMMIT_PATTERN.fullmatch(commit_sha) is None:
        raise ValueError(f"expected a 7-64 digit hexadecimal commit SHA, got {commit_sha!r}")
    config_path = repo_root / line.source_dir / "pyproject.toml"
    fallback_version = read_fallback_version(config_path, line.ctk_target)
    if has_reachable_release(repo_root, line, fallback_version):
        return None

    return f"{fallback_version}+g{commit_sha[:7].lower()}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--config", type=Path, default=bindings_config.DEFAULT_CONFIG)
    parser.add_argument("--line-id", required=True)
    parser.add_argument("--sha", required=True)
    args = parser.parse_args(argv)

    line = bindings_config.load_config(args.config, args.repo_root).get_line(args.line_id)
    version = pretend_version(args.repo_root, args.sha, line)
    if version is not None:
        print(version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
