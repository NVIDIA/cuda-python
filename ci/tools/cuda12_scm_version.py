# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select the temporary CUDA 12 SCM version used before its target release tag."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FALLBACK_CONFIG = Path("cuda_bindings_12/pyproject.toml")
FALLBACK_PATTERN = re.compile(r'^fallback_version\s*=\s*"(?P<version>[^"]+)"\s*$', re.MULTILINE)
CUDA12_DEVELOPMENT_PATTERN = re.compile(r"(?P<release>12\.9\.\d+)\.dev\d+")
CUDA12_RELEASE_TAG_PATTERN = re.compile(r"v(?P<release>12\.9\.\d+)")
COMMIT_PATTERN = re.compile(r"[0-9a-fA-F]{7,64}")


def read_fallback_version(path: Path) -> str:
    matches = FALLBACK_PATTERN.findall(path.read_text(encoding="utf-8"))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one fallback_version in {path}, found {len(matches)}")
    version = matches[0]
    if CUDA12_DEVELOPMENT_PATTERN.fullmatch(version) is None:
        raise ValueError(f"expected a CUDA 12.9 development fallback in {path}, got {version!r}")
    return version


def _release_tuple(version: str) -> tuple[int, int, int]:
    major, minor, patch = version.split(".")
    return int(major), int(minor), int(patch)


def has_reachable_cuda12_release(repo_root: Path, minimum_release: tuple[int, int, int]) -> bool:
    process = subprocess.run(
        ["git", "tag", "--merged", "HEAD", "--list", "v12.9.*"],  # noqa: S607
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        detail = process.stderr.strip() or f"git tag exited with status {process.returncode}"
        raise RuntimeError(f"could not inspect reachable CUDA 12 tags: {detail}")
    releases = (
        _release_tuple(match.group("release"))
        for tag in process.stdout.splitlines()
        if (match := CUDA12_RELEASE_TAG_PATTERN.fullmatch(tag)) is not None
    )
    return any(release >= minimum_release for release in releases)


def pretend_version(repo_root: Path, commit_sha: str, fallback_config: Path) -> str | None:
    """Return the pre-tag override, or None once main has a CUDA 12 release tag."""
    if COMMIT_PATTERN.fullmatch(commit_sha) is None:
        raise ValueError(f"expected a 7-64 digit hexadecimal commit SHA, got {commit_sha!r}")
    config_path = fallback_config if fallback_config.is_absolute() else repo_root / fallback_config
    fallback_version = read_fallback_version(config_path)
    match = CUDA12_DEVELOPMENT_PATTERN.fullmatch(fallback_version)
    assert match is not None  # validated by read_fallback_version
    if has_reachable_cuda12_release(repo_root, _release_tuple(match.group("release"))):
        return None

    return f"{fallback_version}+g{commit_sha[:7].lower()}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--sha", required=True)
    parser.add_argument("--fallback-config", type=Path, default=DEFAULT_FALLBACK_CONFIG)
    args = parser.parse_args(argv)

    version = pretend_version(args.repo_root, args.sha, args.fallback_config)
    if version is not None:
        print(version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
