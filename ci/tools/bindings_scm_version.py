# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select a maintenance bindings line's SCM version before its release tag."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import bindings_config

REPO_ROOT = Path(__file__).resolve().parents[2]
FALLBACK_PATTERN = re.compile(r'^fallback_version\s*=\s*"(?P<version>[^"]+)"\s*$', re.MULTILINE)
COMMIT_PATTERN = re.compile(r"[0-9a-fA-F]{7,64}")


def _development_pattern(ctk_target: str) -> re.Pattern[str]:
    return re.compile(rf"(?P<release>{re.escape(ctk_target)}\.\d+)\.dev\d+")


def read_fallback_version(path: Path, ctk_target: str) -> str:
    matches = FALLBACK_PATTERN.findall(path.read_text(encoding="utf-8"))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one fallback_version in {path}, found {len(matches)}")
    version = matches[0]
    if _development_pattern(ctk_target).fullmatch(version) is None:
        raise ValueError(f"expected a CUDA {ctk_target} development fallback in {path}, got {version!r}")
    return version


def _release_tuple(version: str) -> tuple[int, int, int]:
    major, minor, patch = version.split(".")
    return int(major), int(minor), int(patch)


def _tag_release_tuple(line: bindings_config.BindingsLine, tag: str) -> tuple[int, int, int]:
    suffix = tag.removeprefix(line.tag_series).partition(".post")[0]
    patch = re.split("[ab]", suffix, maxsplit=1)[0]
    return _release_tuple(f"{line.ctk_target}.{patch}")


def has_reachable_release(
    repo_root: Path,
    line: bindings_config.BindingsLine,
    minimum_release: tuple[int, int, int],
) -> bool:
    process = subprocess.run(  # noqa: S603
        ["git", "tag", "--merged", "HEAD", "--list", f"{line.tag_series}*"],  # noqa: S607
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        detail = process.stderr.strip() or f"git tag exited with status {process.returncode}"
        raise RuntimeError(f"could not inspect reachable tags for bindings line {line.line_id!r}: {detail}")
    releases = (_tag_release_tuple(line, tag) for tag in process.stdout.splitlines() if line.matches_tag(tag))
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
    match = _development_pattern(line.ctk_target).fullmatch(fallback_version)
    assert match is not None  # validated by read_fallback_version
    if has_reachable_release(repo_root, line, _release_tuple(match.group("release"))):
        return None

    return f"{fallback_version}+g{commit_sha[:7].lower()}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--config", type=Path, default=bindings_config.DEFAULT_CONFIG)
    parser.add_argument("--line-id", required=True)
    parser.add_argument("--sha", required=True)
    args = parser.parse_args(argv)

    line = bindings_config.load_config(args.config).get_line(args.line_id)
    version = pretend_version(args.repo_root, args.sha, line)
    if version is not None:
        print(version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
